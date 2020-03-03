
"""Cascade-VAE

Learning Discrete and Continuous Factors of Data via Alternating
Disentanglement
http://arxiv.org/abs/1905.09432

code by authors (TensorFlow)
https://github.com/snu-mllab/DisentanglementICML19
"""

from .base import BaseVAE


class CascadeVAE(BaseVAE):
    def __init__(self, channel_num, z_dim, device, beta, **kwargs):
        super().__init__(channel_num, z_dim, device, beta, **kwargs)

    def _eval_loss(self, x_dict, **kwargs):
        raise NotImplementedError
