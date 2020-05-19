
"""FactorVAE

Disentangling by Factorising
http://arxiv.org/abs/1802.05983
"""

from typing import Dict

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

    Args:
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

    def encode(self, x_dict: Dict[str, Tensor], mean: bool = False, **kwargs
               ) -> Dict[str, Tensor]:
        """Encodes latents given observable x.

        Args:
            x_dict (dict of [str, torch.Tensor]): Dict of Tensor for input
                observations.
            mean (bool, optional): Boolean flag for returning means or samples.

        Returns:
            latents (dict of [str, torch.Tensor]): Tensor of encoded latents.
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
            x (dict of [str, torch.Tensor]): Tensor of decoded observations.
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
            samples (dict of [str, torch.Tensor]): Dict of sampled tensor.
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
