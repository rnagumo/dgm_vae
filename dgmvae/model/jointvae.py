
"""JointVAE

Learning Disentangled Joint Continuous and Discrete Representations
http://arxiv.org/abs/1804.00104
"""

import torch
from torch import nn, optim
from torch.nn import functional as F

import pixyz.distributions as pxd
import pixyz.losses as pxl

from .base import BaseVAE
from ..loss.discrete_kl import CategoricalKullbackLeibler


class EncoderFunction(pxd.Deterministic):
    def __init__(self, channel_num):
        super().__init__(cond_var=["x"], var=["h"])

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
        super().__init__(cond_var=["h"], var=["z"])

        self.fc11 = nn.Linear(256, z_dim)
        self.fc12 = nn.Linear(256, z_dim)

    def forward(self, h):
        loc = self.fc11(h)
        scale = F.softplus(self.fc12(h))
        return {"loc": loc, "scale": scale}


class DiscreteEncoder(pxd.RelaxedCategorical):
    def __init__(self, c_dim, temperature):
        super().__init__(cond_var=["h"], var=["c"], temperature=temperature)

        self.fc1 = nn.Linear(256, c_dim)

    def forward(self, h):
        logits = self.fc1(h)
        probs = F.softmax(logits, dim=1)
        probs = torch.clamp(probs, 1e-6, 1 - 1e-6)
        return {"probs": probs}


class JointDecoder(pxd.Bernoulli):
    def __init__(self, channel_num, z_dim, c_dim):
        super().__init__(cond_var=["z", "c"], var=["x"])

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


class JointVAE(BaseVAE):
    def __init__(self, device, channel_num, z_dim, c_dim, temperature,
                 gamma_cont, gamma_disc, **kwargs):

        self.device = device
        self.channel_num = channel_num
        self.z_dim = z_dim
        self.c_dim = c_dim

        # Distributions
        self.prior_cont = pxd.Normal(
            loc=torch.tensor(0.), scale=torch.tensor(1.),
            var=["z"], features_shape=[z_dim]).to(device)
        self.prior_disc = pxd.Categorical(
            probs=torch.ones(c_dim, dtype=torch.float32) / c_dim,
            var=["c"]).to(device)

        self.encoder_func = EncoderFunction(channel_num).to(device)
        self.encoder_cont = ContinuousEncoder(z_dim).to(device)
        self.encoder_disc = DiscreteEncoder(c_dim, temperature).to(device)

        self.decoder = JointDecoder(channel_num, z_dim, c_dim).to(device)

        self.distributions = nn.ModuleList([
            self.prior_cont, self.prior_disc, self.encoder_func,
            self.encoder_cont, self.encoder_disc, self.decoder
        ])

        # Loss
        self.ce = pxl.CrossEntropy(self.encoder_cont * self.encoder_disc,
                                   self.decoder)
        self.kl_cont = pxl.KullbackLeibler(self.encoder_cont, self.prior_cont)
        self.kl_disc = CategoricalKullbackLeibler(
            self.encoder_disc, self.prior_disc)

        # Coefficient for kl
        self.gamma_cont = gamma_cont
        self.gamma_disc = gamma_disc

        # Capacity
        self.cap_cont = pxl.Parameter("cap_cont")
        self.cap_disc = pxl.Parameter("cap_disc")

        # Optimizer
        params = self.distributions.parameters()
        self.optimizer = optim.Adam(params)

    def _eval_loss(self, x_dict, **kwargs):

        # TODO: update capacity values per epoch
        x_dict.update({"cap_cont": 1, "cap_disc": 1})

        # Sample h (surrogate latent variable)
        x_dict = self.encoder_func.sample(x_dict)

        # Cross entropy
        ce_loss = self.ce.eval(x_dict).mean()

        # KL for continuous latent
        _kl_cont = self.kl_cont.eval(x_dict).mean()
        _cap_cont = self.cap_cont.eval(x_dict)
        kl_cont_loss = self.gamma_cont * torch.abs(_kl_cont - _cap_cont)

        # KL for discrete latent
        _kl_disc = self.kl_disc.eval(x_dict).mean()
        _cap_disc = self.cap_disc.eval(x_dict)
        kl_disc_loss = self.gamma_disc * torch.abs(_kl_disc - _cap_disc)

        loss = ce_loss + kl_cont_loss + kl_disc_loss
        loss_dict = {"loss": loss, "ce_loss": ce_loss,
                     "kl_cont_loss": kl_cont_loss,
                     "kl_disc_loss": kl_disc_loss}

        return loss, loss_dict

    def reconstruct(self, x, concat=True):

        with torch.no_grad():
            x = x.to(self.device)
            h = self.encoder_func.sample(x, return_all=False)
            z = self.encoder_cont.sample(h, return_all=False)
            c = self.encoder_disc.sample(h, return_all=False)
            x_recon = self.decoder.sample_mean(
                {"z": z["z"], "c": c["c"]}).cpu()

        if concat:
            return torch.cat([z["z"], c["c"]]), x_recon

        return z["z"], c["c"], x_recon

    def sample(self, batch_n=1, concat=True):

        with torch.no_grad():
            z = self.prior_cont.sample(batch_n=batch_n)
            c = self.prior_disc.sample(batch_n=batch_n)
            x = self.decoder.sample_mean({"z": z["z"], "c": c["c"]}).cpu()

        if concat:
            return torch.cat([z["z"], c["c"]]), x

        return z["z"], x
