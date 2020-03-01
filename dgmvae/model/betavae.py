
"""beta-VAE

β-VAE: LEARNING BASIC VISUAL CONCEPTS WITH A CONSTRAINED VARIATIONAL FRAMEWORK
https://openreview.net/forum?id=Sy2fzU9gl

Understanding disentangling in β-VAE
https://arxiv.org/abs/1804.03599
"""

import torch

from .base import BaseVAE


class BetaVAE(BaseVAE):
    def __init__(self, channel_num, z_dim, device, beta, c, **kwargs):
        super().__init__(channel_num, z_dim, device, **kwargs)

        self.beta = beta
        self.c = c

    def _eval_loss(self, x_dict, **kwargs):

        ce_loss = self.ce.eval(x_dict).mean()
        kl_loss = self.kl.eval(x_dict).mean()
        loss = ce_loss + self.beta * torch.abs(kl_loss - self.c)
        loss_dict = {"loss": loss.item(), "ce_loss": ce_loss.item(),
                     "kl_loss": kl_loss.item()}

        return loss, loss_dict
