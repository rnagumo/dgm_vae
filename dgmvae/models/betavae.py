
"""beta-VAE

β-VAE: LEARNING BASIC VISUAL CONCEPTS WITH A CONSTRAINED VARIATIONAL FRAMEWORK
https://openreview.net/forum?id=Sy2fzU9gl

Understanding disentangling in β-VAE
https://arxiv.org/abs/1804.03599
"""

from typing import Union, Dict

import torch
from torch import Tensor

import pixyz.distributions as pxd
import pixyz.losses as pxl

from .base import BaseVAE
from .dist import Encoder, Decoder


class BetaVAE(BaseVAE):
    """beta-VAE.

    Attributes:
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

    def encode(self,
               x: Union[Tensor, Dict[str, Tensor]],
               mean: bool = False,
               **kwargs) -> Union[Tensor, Dict[str, Tensor]]:
        """Encodes latent given observable x.

        Args:
            x (torch.Tensor or dict): Tensor or dict or Tensor for input
                observations.
            mean (bool, optional): Boolean flag for returning means or samples.

        Returns:
            z (torch.Tensor or dict): Tensor of encoded latents. `z` is
            `torch.Tensor` if `mean` is `True`, otherwise, dict.
        """

        if not isinstance(x, dict):
            x = {"x": x}

        if mean:
            return self.encoder.sample_mean(x)
        return self.encoder.sample(x, return_all=False)

    def decode(self,
               latent: Union[Tensor, Dict[str, Tensor]],
               mean: bool = False,
               **kwargs) -> Union[Tensor, Dict[str, Tensor]]:
        """Decodes observable x given latents.

        Args:
            latent (torch.Tensor or dict): Tensor or dict of latents.
            mean (bool, optional): Boolean flag for returning means or samples.

        Returns:
            x (torch.Tensor or dict): Tensor of decoded observations. `z` is
            `torch.Tensor` if `mean` is `True`, otherwise, dict.
        """

        if not isinstance(latent, dict):
            latent = {"z": latent}

        if mean:
            return self.decoder.sample_mean(latent)
        return self.decoder.sample(latent, return_all=False)

    def sample(self, batch_n: int = 1, **kwargs) -> Dict[str, Tensor]:
        """Samples observable x from sampled latent z.

        Args:
            batch_n (int, optional): Batch size.

        Returns:
            sample (dict): Dict of sampled tensors.
        """

        z = self.prior.sample(batch_n=batch_n)
        sample = self.decoder.sample_mean(z)
        return sample

    def loss_func(self, x: Tensor, **kwargs) -> Dict[str, Tensor]:
        """Calculates loss given observable x.

        Args:
            x (torch.Tensor): Tensor of input observations.

        Returns:
            loss_dict (dict): Dict of calculated losses.
        """

        x_dict = {"x": x, "beta": self._beta_value, "c": self._c_value}

        ce_loss = self.ce.eval(x_dict).mean()
        kl_loss = (self.beta * (self.kl - self.c).abs()).eval(x_dict).mean()
        loss = ce_loss + kl_loss

        return {"loss": loss, "ce_loss": ce_loss, "kl_loss": kl_loss}

    @property
    def loss_str(self):
        return str(self.ce + self.beta * (self.kl - self.c).abs())
