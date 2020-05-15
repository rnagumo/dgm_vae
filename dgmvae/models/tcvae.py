
"""beta-TCVAE

Isolating Sources of Disentanglement in Variational Autoencoders
http://arxiv.org/abs/1802.04942

code by author
https://github.com/rtqichen/beta-tcvae
"""

from typing import Union, Dict
import math

import torch
from torch import Tensor

import pixyz.distributions as pxd
import pixyz.losses as pxl

from .base import BaseVAE
from .dist import Decoder, Encoder


class TCVAE(BaseVAE):
    """beta-TCVAE (Total Correlation VAE).

    Attributes:
        channel_num (int): Number of input channels.
        z_dim (int): Dimension of latents `z`.
        alpha (float): Alpha regularization term.
        beta (float): Beta regularization term.
        gamma (float): Gamma regularization term.
    """

    def __init__(self, channel_num: int, z_dim: int, alpha: float, beta: float,
                 gamma: float, **kwargs):
        super().__init__()

        # Parameters
        self.channel_num = channel_num
        self.z_dim = z_dim

        self._alpha_value = alpha
        self._beta_value = beta
        self._gamma_value = gamma

        # Distributions
        self.prior = pxd.Normal(
            loc=torch.zeros(z_dim), scale=torch.ones(z_dim), var=["z"])
        self.decoder = Decoder(channel_num, z_dim)
        self.encoder = Encoder(channel_num, z_dim)
        self.distributions = [self.prior, self.decoder, self.encoder]

        # Loss class
        self.ce = pxl.CrossEntropy(self.encoder, self.decoder)
        self.kl = pxl.KullbackLeibler(self.encoder, self.prior)
        self.alpha = pxl.Parameter("alpha")
        self.beta = pxl.Parameter("beta")
        self.gamma = pxl.Parameter("gamma")

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

        x_dict = {"x": x, "alpha": self._alpha_value, "beta": self._beta_value,
                  "gamma": self._gamma_value,
                  "dataset_size": kwargs["dataset_size"]}

        # Sample z from encoder
        x_dict = self.encoder.sample(x_dict)

        # log p(x)
        log_px = self.decoder.get_log_prob(x_dict)

        # log p(z)
        log_pz = self.prior.get_log_prob(x_dict)

        # log q(z|x)
        log_qz_x = self.encoder.get_log_prob_wo_forward(x_dict)

        # Minibatch Weighted Sampling
        # log q(z), size of (z_batch_size, x_batch_size, z_dim)
        _logqz = self.encoder.get_log_prob_wo_forward(
            {"z": x_dict["z"].unsqueeze(1)}, sum_features=False)

        # log NM
        dataset_size = x_dict["dataset_size"]
        batch_size = x_dict["x"].size(0)
        lognm = math.log(dataset_size * batch_size)

        # log prod_j q(z_j) = sum_j log q(z_j)
        log_qz_prodmarginal = (torch.logsumexp(_logqz, 1) - lognm).sum(1)

        # log q(z)
        log_qz = torch.logsumexp(_logqz.sum(2), 1) - lognm

        # Coeff
        alpha = self.alpha.eval(x_dict)
        beta = self.beta.eval(x_dict)
        gamma = self.gamma.eval(x_dict)

        # Reconstruction
        recon = -log_px.mean()

        # Index-Code MI: KL(q(z|x)||q(z))
        mutual_info = alpha * (log_qz_x - log_qz).mean()

        # Total Correlation: KL(q(z)||prod_j q(z_j))
        independence = beta * (log_qz - log_qz_prodmarginal).mean()

        # Dimension-wise KL: sum_j KL(q(z_j)||p(z_j))
        dim_wise_kl = gamma * (log_qz_prodmarginal - log_pz).mean()

        # ELBO loss
        loss = recon + mutual_info + independence + dim_wise_kl

        loss_dict = {"loss": loss, "recon": recon, "mutual_info": mutual_info,
                     "independence": independence, "dim_wise_kl": dim_wise_kl}

        return loss_dict

    @property
    def loss_str(self):
        p_text = ("\\mathbb{E}_{q(z|n)p(n)} \\left[\\log p(n|z)\\right] "
                  "- \\alpha I_q(z;n) "
                  "- \\beta D_{KL}\\left[q(z) || \\prod_j q(z_j) \\right] "
                  "- \\gamma \\sum_j D_{KL} \\left[q(z_j) || p(z_j) \\right]")

        return p_text
