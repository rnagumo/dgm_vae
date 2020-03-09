
"""DIP-VAE

Disentangled Inferred Prior-VAE

Variational Inference of Disentangled Latent Concepts from Unlabeled
Observations
http://arxiv.org/abs/1711.00848
"""

import torch
from torch import nn

import pixyz.distributions as pxd
import pixyz.losses as pxl

from .base import BaseVAE
from .dist import Decoder, Encoder
from ..losses.dip_loss import DipLoss


class DIPVAE(BaseVAE):
    def __init__(self, channel_num, z_dim, beta, c, lmd_od, lmd_d, dip_type,
                 **kwargs):
        super().__init__()

        # Parameters
        self.channel_num = channel_num
        self.z_dim = z_dim
        self._beta_value = beta
        self._c_value = c
        self.lmd_od = lmd_od
        self.lmd_d = lmd_d

        # Distributions
        self.prior = pxd.Normal(loc=torch.tensor(0.), scale=torch.tensor(1.),
                                var=["z"], features_shape=[z_dim])
        self.decoder = Decoder(channel_num, z_dim)
        self.encoder = Encoder(channel_num, z_dim)
        self.distributions = [self.prior, self.decoder, self.encoder]

        # Loss class
        self.ce = pxl.CrossEntropy(self.encoder, self.decoder)
        _kl = pxl.KullbackLeibler(self.encoder, self.prior)
        _beta = pxl.Parameter("beta")
        _c = pxl.Parameter("c")
        self.kl = _beta * (_kl - _c).abs()
        self.dip = DipLoss(self.encoder, lmd_od, lmd_d, dip_type)

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

        x_dict.update({"beta": self._beta_value, "c": self._c_value})

        ce_loss = self.ce.eval(x_dict).mean()
        kl_loss = self.kl.eval(x_dict).mean()
        dip_loss = self.dip.eval(x_dict)
        loss = ce_loss + kl_loss + dip_loss
        loss_dict = {"loss": loss, "ce_loss": ce_loss, "kl_loss": kl_loss,
                     "dip_loss": dip_loss}

        return loss_dict

    @property
    def loss_cls(self):
        return self.ce + self.kl + self.dip
