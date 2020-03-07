
"""Adversarial Autoencoder (AAE)

Unsupervised clustering based on ch.6

Adversarial Autoencoders
http://arxiv.org/abs/1511.05644
"""

import torch
from torch import nn
from torch.nn import functional as F

import pixyz.distributions as pxd
import pixyz.losses as pxl

from .base import BaseVAE
from .dist import Discriminator


class EncoderFunction(pxd.Deterministic):
    def __init__(self, channel_num):
        super().__init__(cond_var=["x"], var=["h"], name="f_e")

        self.conv1 = nn.Conv2d(channel_num, 32, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 4, stride=2, padding=1)
        self.fc1 = nn.Linear(1024, 256)

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = h.view(-1, 1024)
        h = F.relu(self.fc1(h))
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


class DiscreteEncoder(pxd.Categorical):
    def __init__(self, c_dim):
        super().__init__(cond_var=["h"], var=["c"], name="q_c")

        self.fc1 = nn.Linear(256, c_dim)

    def forward(self, h):
        logits = self.fc1(h)
        probs = F.softmax(logits, dim=1)
        probs = torch.clamp(probs, 1e-6, 1 - 1e-6)
        return {"probs": probs}


class JointDecoder(pxd.Bernoulli):
    def __init__(self, channel_num, z_dim, c_dim):
        super().__init__(cond_var=["z", "c"], var=["x"], name="p")

        self.fc1 = nn.Linear(z_dim + c_dim, 256)
        self.fc2 = nn.Linear(256, 1024)
        self.deconv1 = nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(32, channel_num, 4, stride=2,
                                          padding=1)

    def forward(self, z, c):
        h = F.relu(self.fc1(torch.cat([z, c], dim=1)))
        h = F.relu(self.fc2(h))
        h = h.view(-1, 64, 4, 4)
        h = F.relu(self.deconv1(h))
        h = F.relu(self.deconv2(h))
        h = F.relu(self.deconv3(h))
        probs = torch.sigmoid(self.deconv4(h))
        return {"probs": probs}


class AAE(BaseVAE):
    def __init__(self, channel_num, z_dim, c_dim, beta, **kwargs):
        super().__init__()

        # Parameters
        self.channel_num = channel_num
        self.z_dim = z_dim
        self.c_dim = c_dim
        self._beta_value = beta

        # Prior
        self.prior_cont = pxd.Normal(
            loc=torch.tensor(0.), scale=torch.tensor(1.),
            var=["z"], features_shape=[z_dim])
        self.prior_disc = pxd.Categorical(
            probs=torch.ones(c_dim, dtype=torch.float32) / c_dim,
            var=["c"])

        # Encoder
        self.encoder_func = EncoderFunction(channel_num)
        self.encoder_cont = ContinuousEncoder(z_dim)
        self.encoder_disc = DiscreteEncoder(c_dim)

        # Decoder
        self.decoder = JointDecoder(channel_num, z_dim, c_dim)

        self.distributions = nn.ModuleList([
            self.prior_cont, self.prior_disc, self.encoder_func,
            self.encoder_cont, self.encoder_disc, self.decoder,
        ])

        # Loss
        self.ce = pxl.CrossEntropy(self.encoder_cont, self.decoder)
        self.beta = pxl.Parameter("beta")

        # Adversarial loss
        self.disc = Discriminator(z_dim)
        self.adv_js = pxl.AdversarialJensenShannon(
            self.encoder_cont, self.prior_cont, self.disc)

    def encode(self, x, mean=False):

        h = self.encoder_func.sample(x, return_all=False)

        if mean:
            z = self.encoder_cont.sample_mean(h)
            c = self.encoder_disc.sample_mean(h)
            return z, c

        z = self.encoder_cont.sample(h, return_all=False)
        c = self.encoder_disc.sample(h, return_all=False)
        z.update(c)
        return z

    def decode(self, z=None, c=None, latent=None, mean=False):
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
        z = self.prior_cont.sample(batch_n=batch_n)
        c = self.prior_disc.sample(batch_n=batch_n)
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

        optimizer_idx = kwargs["optimizer_idx"]

        # Sample h (surrogate latent) and c (categorical latent)
        x_dict = (self.encoder_disc * self.encoder_func).sample(x_dict)

        if optimizer_idx == 0:
            # VAE loss
            ce_loss = self.ce.eval(x_dict).mean()
            js_loss = self.adv_js.eval(x_dict).mean()
            loss = ce_loss + js_loss
            return {"loss": loss, "ce_loss": ce_loss, "js_loss": js_loss}
        elif optimizer_idx == 1:
            # Discriminator loss
            loss = self.adv_js.eval(x_dict, discriminator=True)
            return {"adv_loss": loss}

    @property
    def loss_cls(self):
        return (self.ce + self.adv_js).expectation(
                    self.encoder_disc * self.encoder_func)

    @property
    def second_optim(self):
        return self.adv_js.d_optimizer
