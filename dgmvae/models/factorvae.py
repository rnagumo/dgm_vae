
"""FactorVAE

Disentangling by Factorising
http://arxiv.org/abs/1802.05983
"""

from typing import Union, Dict

import torch
from torch import Tensor

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
    """Factor VAE.

    Attributes:
        channel_num (int): Number of input channels.
        z_dim (int): Dimension of latents `z`.
        beta (float): Beta regularization term.
        gamma (float): Gamma regularization term.
    """

    def __init__(self, channel_num: int, z_dim: int, beta: float, gamma: float,
                 **kwargs):
        super().__init__()

        # Parameters
        self.channel_num = channel_num
        self.z_dim = z_dim
        self._beta_value = beta
        self._gamma_value = gamma

        # Dimension shuffle
        self.prior = pxd.Normal(
            loc=torch.zeros(z_dim), scale=torch.ones(z_dim), var=["z"])
        self.decoder = Decoder(channel_num, z_dim)
        self.encoder = Encoder(channel_num, z_dim)
        self.encoder_shf = InferenceShuffleDim(self.encoder)
        self.distributions = [self.prior, self.decoder, self.encoder,
                              self.encoder_shf]

        # Loss
        self.ce = pxl.CrossEntropy(self.encoder, self.decoder)
        self.kl = pxl.KullbackLeibler(self.encoder, self.prior)
        self.beta = pxl.Parameter("beta")
        self.gamma = pxl.Parameter("gamma")

        # Adversarial optimizer settings
        if "optimizer_params" in kwargs:
            optimizer_params = {
                "lr": kwargs["optimizer_params"]["lr"],
                "betas": (kwargs["optimizer_params"]["beta1"],
                          kwargs["optimizer_params"]["beta2"]),
            }
        else:
            optimizer_params = {}

        # Adversarial loss (Total Correlation)
        self.disc = Discriminator(z_dim)
        self.adv_js = pxl.AdversarialKullbackLeibler(
            self.encoder, self.encoder_shf, self.disc,
            optimizer_params=optimizer_params)

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

        len_x = x.size(0)
        len_half = x.size(0) // 2

        # `x` and `x_shf` should have the same batch size
        if len_x % 2 == 0:
            x_dict = {"x": x[:len_half], "x_shf": x[len_half:]}
        else:
            x_dict = {"x": x[:len_half], "x_shf": x[len_half + 1:]}

        # Add coeff
        x_dict.update({"beta": self._beta_value, "gamma": self._gamma_value})

        if kwargs["optimizer_idx"] == 0:
            # VAE loss
            ce_loss = self.ce.eval(x_dict).mean()
            kl_loss = (self.beta * self.kl).eval(x_dict).mean()
            tc_loss = (self.gamma * self.adv_js).eval(x_dict)
            loss = ce_loss + kl_loss + tc_loss

            loss_dict = {"loss": loss, "ce_loss": ce_loss, "kl_loss": kl_loss,
                         "tc_loss": tc_loss}
            return loss_dict
        elif kwargs["optimizer_idx"] == 1:
            # Discriminator loss
            loss = self.adv_js.eval(x_dict, discriminator=True)
            return {"adv_loss": loss}

    @property
    def loss_str(self):
        return str(self.ce + self.beta * self.kl + self.gamma * self.adv_js)

    @property
    def second_optim(self):
        return self.adv_js.d_optimizer
