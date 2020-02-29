
"""DIP-VAE

Disentangled Inferred Prior-VAE

Variational Inference of Disentangled Latent Concepts from Unlabeled
Observations
http://arxiv.org/abs/1711.00848
"""

import torch

from .base import BaseVAE


class DIPVAE(BaseVAE):
    def __init__(self, channel_num, z_dim, device, lmd_od, lmd_d, dip_type,
                 **kwargs):
        super().__init__(channel_num, z_dim, device)

        self.lmd_od = lmd_od
        self.lmd_d = lmd_d
        self.dip_type = dip_type

        if dip_type not in ["i", "ii"]:
            raise ValueError(f"Inappropriate type is specified: {dip_type}")

    def _eval_loss(self, x_dict, **kwargs):
        raise NotImplementedError
