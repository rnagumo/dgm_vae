
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
        self.beta = beta
        self.c = c

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

    def encode(self, x, sample=True):
        if sample:
            return self.encoder.sample(x)
        return self.encoder.sample_mean(x)

    def decode(self, z, sample=False):
        if sample:
            return self.decoder.sample(z)
        return self.decoder.sample_mean(z)

    def sample(self, batch_n=1):
        z = self.prior.sample(batch_n=batch_n)
        x = self.decoder.sample_mean(z)
        return x

    def forward(self, x, sample=True, reconstruct=True):
        if sample:
            z = self.encoder.sample(x, return_all=False)
        else:
            z = self.encoder.sample_mean(x, return_all=False)

        if reconstruct:
            return self.decoder.sample_mean(z)
        return z

    def loss_function(self, x_dict, **kwargs):
        ce_loss = self.ce.eval(x_dict).mean()
        _kl_loss = self.kl.eval(x_dict).mean()
        kl_loss = self.beta * torch.abs(_kl_loss - self.c)
        loss = ce_loss + kl_loss

        return {"loss": loss, "ce_loss": ce_loss, "kl_loss": kl_loss}
