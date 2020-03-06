
"""beta-VAE

β-VAE: LEARNING BASIC VISUAL CONCEPTS WITH A CONSTRAINED VARIATIONAL FRAMEWORK
https://openreview.net/forum?id=Sy2fzU9gl

Understanding disentangling in β-VAE
https://arxiv.org/abs/1804.03599
"""

import torch
from torch import nn

import pixyz.distributions as pxd
import pixyz.losses as pxl

from .base import BaseVAE
from .dist import Encoder, Decoder


class BetaVAE(BaseVAE):
    def __init__(self, channel_num, z_dim, beta, c, **kwargs):
        super().__init__()

        # Parameters
        self.channel_num = channel_num
        self.z_dim = z_dim
        self._beta_value = beta
        self._c_value = c

        # Distributions
        self.prior = pxd.Normal(loc=torch.tensor(0.), scale=torch.tensor(1.),
                                var=["z"], features_shape=[z_dim])
        self.decoder = Decoder(channel_num, z_dim)
        self.encoder = Encoder(channel_num, z_dim)
        self.distributions = nn.ModuleList(
            [self.prior, self.decoder, self.encoder])

        # Loss class
        self.ce = pxl.CrossEntropy(self.encoder, self.decoder)
        self.kl = pxl.KullbackLeibler(self.encoder, self.prior)
        self.beta = pxl.Parameter("beta")
        self.c = pxl.Parameter("c")

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

        # TODO
        x_dict.update({"beta": self._beta_value, "c": self._c_value})

        ce_loss = self.ce.eval(x_dict).mean()
        kl_loss = (self.beta * (self.kl - self.c).abs()).eval(x_dict).mean()
        loss = ce_loss + kl_loss

        return {"loss": loss, "ce_loss": ce_loss, "kl_loss": kl_loss}

    @property
    def loss_cls(self):
        return self.ce + self.beta * (self.kl - self.c).abs()
