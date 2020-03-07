
"""JointVAE

Learning Disentangled Joint Continuous and Discrete Representations
http://arxiv.org/abs/1804.00104
"""

import torch
from torch import nn
from torch.nn import functional as F

import pixyz.distributions as pxd
import pixyz.losses as pxl

from .base import BaseVAE
from ..loss.discrete_kl import CategoricalKullbackLeibler


class EncoderFunction(pxd.Deterministic):
    def __init__(self, channel_num):
        super().__init__(cond_var=["x"], var=["h"], name="f")

        self.enc_x = nn.Sequential(
            nn.Conv2d(channel_num, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
        )

    def forward(self, x):
        h = self.enc_x(x)
        h = h.view(-1, 1024)
        h = self.fc(h)
        return {"h": h}


class ContinuousEncoder(pxd.Normal):
    def __init__(self, z_dim):
        super().__init__(cond_var=["h"], var=["z"], name="q_z")

        self.fc11 = nn.Linear(256, z_dim)
        self.fc12 = nn.Linear(256, z_dim)

    def forward(self, h):
        loc = self.fc11(h)
        scale = F.softplus(self.fc12(h))
        return {"loc": loc, "scale": scale}


class DiscreteEncoder(pxd.RelaxedCategorical):
    def __init__(self, c_dim, temperature):
        super().__init__(cond_var=["h"], var=["c"], name="q_c",
                         temperature=temperature)

        self.fc1 = nn.Linear(256, c_dim)

    def forward(self, h):
        logits = self.fc1(h)
        probs = F.softmax(logits, dim=1)
        probs = torch.clamp(probs, 1e-6, 1 - 1e-6)
        return {"probs": probs}


class JointDecoder(pxd.Bernoulli):
    def __init__(self, channel_num, z_dim, c_dim):
        super().__init__(cond_var=["z", "c"], var=["x"])

        self.fc = nn.Sequential(
            nn.Linear(z_dim + c_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU(),
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, channel_num, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z, c):
        h = self.fc(torch.cat([z, c], dim=1))
        h = h.view(-1, 64, 4, 4)
        probs = self.deconv(h)
        return {"probs": probs}


class JointVAE(BaseVAE):
    def __init__(self, channel_num, z_dim, c_dim, temperature, gamma_z,
                 gamma_c, **kwargs):
        super().__init__()

        self.channel_num = channel_num
        self.z_dim = z_dim
        self.c_dim = c_dim
        self._gamma_z_value = gamma_z
        self._gamma_c_value = gamma_c

        # Distributions
        self.prior_z = pxd.Normal(
            loc=torch.tensor(0.), scale=torch.tensor(1.),
            var=["z"], features_shape=[z_dim])
        self.prior_c = pxd.Categorical(
            probs=torch.ones(c_dim, dtype=torch.float32) / c_dim,
            var=["c"])

        self.encoder_func = EncoderFunction(channel_num)
        self.encoder_z = ContinuousEncoder(z_dim)
        self.encoder_c = DiscreteEncoder(c_dim, temperature)
        self.decoder = JointDecoder(channel_num, z_dim, c_dim)

        self.distributions = nn.ModuleList([
            self.prior_z, self.prior_c, self.encoder_func,
            self.encoder_z, self.encoder_c, self.decoder
        ])

        # Loss
        self.ce = pxl.CrossEntropy(self.encoder_z * self.encoder_c,
                                   self.decoder)
        self.kl_z = pxl.KullbackLeibler(self.encoder_z, self.prior_z)
        self.kl_c = CategoricalKullbackLeibler(
            self.encoder_c, self.prior_c)

        # Coefficient for kl
        self.gamma_z = pxl.Parameter("gamma_z")
        self.gamma_c = pxl.Parameter("gamma_c")

        # Capacity
        self.cap_z = pxl.Parameter("cap_z")
        self.cap_c = pxl.Parameter("cap_c")

    def encode(self, x, mean=False):

        h = self.encoder_func.sample(x, return_all=False)

        if mean:
            z = self.encoder_z.sample_mean(h)
            c = self.encoder_c.sample_mean(h)
            return z, c

        z = self.encoder_z.sample(h, return_all=False)
        c = self.encoder_c.sample(h, return_all=False)
        z.update(c)
        return z

    def decode(self, latent=None, z=None, c=None, mean=False):
        if latent is None:
            latent = {}
            if isinstance(z, dict):
                latent.update(z)
            else:
                latent["z"] = z

            if isinstance(c, dict):
                latent.update(c)
            else:
                latent["c"] = c

        if mean:
            return self.decoder.sample_mean(latent)
        return self.decoder.sample(latent, return_all=False)

    def sample(self, batch_n=1):
        z = self.prior_z.sample(batch_n=batch_n)
        c = self.prior_c.sample(batch_n=batch_n)
        sample = self.decoder.sample_mean({"z": z["z"], "c": c["c"]})
        return sample

    def forward(self, x, return_latent=False):
        latent = self.encode(x)
        sample = self.decode(latent=latent, mean=True)

        if return_latent:
            latent.update({"x": sample})
            return latent
        return sample

    def loss_func(self, x_dict, **kwargs):

        # TODO: update capacity values per epoch
        x_dict.update({"gamma_z": self._gamma_z_value,
                       "gamma_c": self._gamma_c_value,
                       "cap_z": 1, "cap_c": 1})

        # Sample h (surrogate latent variable)
        x_dict = self.encoder_func.sample(x_dict)

        # Cross entropy
        ce_loss = self.ce.eval(x_dict).mean()

        # KL for continuous latent
        kl_z_loss = (
            self.gamma_z * (self.kl_z - self.cap_z).abs()).eval(x_dict).mean()

        # KL for discrete latent
        kl_c_loss = (
            self.gamma_c * (self.kl_c - self.cap_c).abs()).eval(x_dict).mean()

        loss = ce_loss + kl_z_loss + kl_c_loss
        loss_dict = {"loss": loss, "ce_loss": ce_loss, "kl_z_loss": kl_z_loss,
                     "kl_c_loss": kl_c_loss}

        return loss_dict

    @property
    def loss_cls(self):
        return (self.ce + self.gamma_z * (self.kl_z - self.cap_z).abs()
                + self.gamma_c * (self.kl_c - self.cap_c).abs()
                ).expectation(self.encoder_func)
