
"""beta-VAE

β-VAE: LEARNING BASIC VISUAL CONCEPTS WITH A CONSTRAINED VARIATIONAL FRAMEWORK
https://openreview.net/forum?id=Sy2fzU9gl

Understanding disentangling in β-VAE
https://arxiv.org/abs/1804.03599
"""

from typing import Dict

import torch
from torch import Tensor

import pixyz.distributions as pxd
import pixyz.losses as pxl

from .base import BaseVAE
from .dist import Encoder, Decoder


class BetaVAE(BaseVAE):
    """beta-VAE.

    Args:
        channel_num (int): Number of input channels.
        z_dim (int): Dimension of latents `z`.
        beta (float): Beta regularization term.
        c (float): Capacity recularization term.
    """

    def __init__(self, channel_num: int, z_dim: int, beta: float, c: float,
                 **kwargs):
        super().__init__()

        # Parameters
        self.channel_num = channel_num
        self.z_dim = z_dim
        self._beta_value = beta
        self._c_value = c

        # Distributions
        self.prior = pxd.Normal(
            loc=torch.zeros(z_dim), scale=torch.ones(z_dim), var=["z"])
        self.decoder = Decoder(channel_num, z_dim)
        self.encoder = Encoder(channel_num, z_dim)
        self.distributions = [self.prior, self.decoder, self.encoder]

        # Loss class
        self.ce = pxl.CrossEntropy(self.encoder, self.decoder)
        self.kl = pxl.KullbackLeibler(self.encoder, self.prior)
        self.beta = pxl.Parameter("beta")
        self.c = pxl.Parameter("c")

    def encode(self, x_dict: Dict[str, Tensor], mean: bool = False, **kwargs
               ) -> Dict[str, Tensor]:
        """Encodes latents given observable x.

        Args:
            x_dict (dict of [str, torch.Tensor]): Dict of Tensor for input
                observations.
            mean (bool, optional): Boolean flag for returning means or samples.

        Returns:
            z_dict (dict of [str, torch.Tensor]): Dict of tensor of encoded
                latents.
        """

        if mean:
            z = self.encoder.sample_mean(x_dict)
            return {"z": z}
        return self.encoder.sample(x_dict, return_all=False)

    def decode(self, z_dict: Dict[str, Tensor], mean: bool = False, **kwargs
               ) -> Dict[str, Tensor]:
        """Decodes observable x given latents.

        Args:
            z_dict (dict of [str, torch.Tensor]): Dict of latents tensors.
            mean (bool, optional): Boolean flag for returning means or samples.

        Returns:
            x_dict (dict of [str, torch.Tensor]): Dict of tensor of decoded
                observations.
        """

        if mean:
            x = self.decoder.sample_mean(z_dict)
            return {"x": x}
        return self.decoder.sample(z_dict, return_all=False)

    def sample(self, batch_n: int) -> Dict[str, Tensor]:
        """Samples observable x from sampled latent z.

        Args:
            batch_n (int): Batch size.

        Returns:
            x_dict (dict of [str, torch.Tensor]): Dict of sampled obsercation
                tensor.
        """

        z = self.prior.sample(batch_n=batch_n)
        x = self.decoder.sample_mean(z)
        return {"x": x}

    def loss_func(self, x: Tensor, **kwargs) -> Dict[str, Tensor]:
        """Calculates loss given observable x.

        Args:
            x (torch.Tensor): Tensor of input observations.

        Returns:
            loss_dict (dict of [str, torch.Tensor]): Dict of calculated losses.
        """

        x_dict = {"x": x, "beta": self._beta_value, "c": self._c_value}

        ce_loss = self.ce.eval(x_dict).mean()
        kl_loss = (self.beta * (self.kl - self.c).abs()).eval(x_dict).mean()
        loss = ce_loss + kl_loss

        return {"loss": loss, "ce_loss": ce_loss, "kl_loss": kl_loss}

    @property
    def loss_str(self):
        return str(self.ce + self.beta * (self.kl - self.c).abs())
