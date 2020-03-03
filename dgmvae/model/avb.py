
"""Adversarial Variational Bayes (AVB)

Adversarial Variational Bayes: Unifying Variational Autoencoders and Generative
Adversarial Networks
http://arxiv.org/abs/1701.04722
"""

import torch

from .base import BaseVAE


class BetaVAE(BaseVAE):
    def __init__(self, channel_num, z_dim, device, beta, c, **kwargs):
        super().__init__(channel_num, z_dim, device, **kwargs)

        self.beta = beta
        self.c = c

    def _eval_loss(self, x_dict, **kwargs):
        raise NotImplementedError
