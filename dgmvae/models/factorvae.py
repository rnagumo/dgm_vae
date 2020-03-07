
"""FactorVAE

Disentangling by Factorising
http://arxiv.org/abs/1802.05983
"""

import torch
from torch import nn

import pixyz.distributions as pxd
import pixyz.losses as pxl

from .base import BaseVAE
from .dist import Decoder, Encoder, Discriminator


class InferenceShuffleDim(pxd.Deterministic):
    def __init__(self, q):
        super().__init__(cond_var=["x_shf"], var=["z"], name="q")

        self.q = q

    def forward(self, x_shf):
        z = self.q.sample({"x": x_shf}, return_all=False)["z"]
        return {"z": z[:, torch.randperm(z.size(1))]}


class FactorVAE(BaseVAE):
    def __init__(self, channel_num, z_dim, beta, gamma, **kwargs):
        super().__init__()

        # Parameters
        self.channel_num = channel_num
        self.z_dim = z_dim
        self._beta_value = beta
        self._gamma_value = gamma

        # Dimension shuffle
        self.prior = pxd.Normal(loc=torch.tensor(0.), scale=torch.tensor(1.),
                                var=["z"], features_shape=[z_dim])
        self.decoder = Decoder(channel_num, z_dim)
        self.encoder = Encoder(channel_num, z_dim)
        self.encoder_shf = InferenceShuffleDim(self.encoder)
        self.distributions = nn.ModuleList(
            [self.prior, self.decoder, self.encoder, self.encoder_shf])

        # Loss
        self.ce = pxl.CrossEntropy(self.encoder, self.decoder)
        self.kl = pxl.KullbackLeibler(self.encoder, self.prior)
        self.beta = pxl.Parameter("beta")
        self.gamma = pxl.Parameter("gamma")

        # Adversarial loss (Total Correlation)
        self.disc = Discriminator(z_dim)
        self.adv_js = pxl.AdversarialKullbackLeibler(
            self.encoder, self.encoder_shf, self.disc)

    def encode(self, x, mean=False):
        if not isinstance(x, dict):
            x = {"x": x}

        if mean:
            return self.encoder.sample_mean(x)
        return self.encoder.sample(x, return_all=False)

    def decode(self, z, mean=False):
        if not isinstance(z, dict):
            z = {"z": z}

        if mean:
            return self.decoder.sample_mean(z)
        return self.decoder.sample(z, return_all=False)

    def sample(self, batch_n=1):
        z = self.prior.sample(batch_n=batch_n)
        sample = self.decoder.sample_mean(z)
        return sample

    def forward(self, x, return_latent=False):
        z = self.encode(x)
        sample = self.decode(z, mean=True)
        if return_latent:
            z.update({"x": sample})
            return z
        return sample

    def loss_func(self, x_dict, **kwargs):

        optimizer_idx = kwargs["optimizer_idx"]

        x = x_dict["x"]
        len_x = x.size(0) // 2
        x_dict.update({"x": x[:len_x], "x_shf": x[len_x:]})
        x_dict.update({"beta": self._beta_value, "gamma": self._gamma_value})

        if optimizer_idx == 0:
            # VAE loss
            ce_loss = self.ce.eval(x_dict).mean()
            kl_loss = (self.beta * self.kl).eval(x_dict).mean()
            tc_loss = (self.gamma * self.adv_js).eval(x_dict)
            loss = ce_loss + kl_loss + tc_loss

            loss_dict = {"loss": loss, "ce_loss": ce_loss, "kl_loss": kl_loss,
                         "tc_loss": tc_loss}
            return loss_dict
        elif optimizer_idx == 1:
            # Discriminator loss
            loss = self.adv_js.eval(x_dict, discriminator=True)
            return {"adv_loss": loss}

    @property
    def loss_cls(self):
        return self.ce + self.beta * self.kl + self.gamma * self.adv_js

    @property
    def second_optim(self):
        return self.adv_js.d_optimizer
