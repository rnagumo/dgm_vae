
"""DIP-VAE

Disentangled Inferred Prior-VAE

Variational Inference of Disentangled Latent Concepts from Unlabeled
Observations
http://arxiv.org/abs/1711.00848
"""

import torch
from pixyz.losses.losses import Loss

from .base import BaseVAE


def _get_cov_mu(mu):
    """Computes covariance of mu.

    cov(mu) = E[mu mu^T] - E[mu]E[mu]^T

    mu: [batch_size, latent_dim]
    """

    # E[mu mu^T]
    e_mu_mut = (mu.unsqueeze(2) * mu.unsqueeze(1)).mean(dim=0)

    # E[mu]E[mu]^T
    e_mu = mu.sum(dim=0)
    e_mu_e_mut = e_mu.unsqueeze(1) * e_mu.unsqueeze(0)

    return e_mu_mut - e_mu_e_mut


def _get_e_cov(scale):
    """Computes expectation of cov E[Cov_encoder] from scale

    sclae: [batch_size, latent_dim]
    """

    # Cov
    cov = scale.unsqueeze(2) * torch.eye(scale.size(1))

    return cov.sum(dim=0)


class DipLoss(Loss):
    def __init__(self, p, lmd_od, lmd_d, dip_type, **kwargs):
        super().__init__(p, **kwargs)

        if dip_type not in ["i", "ii"]:
            raise ValueError(f"Inappropriate type is specified: {dip_type}")

        self.lmd_od = lmd_od
        self.lmd_d = lmd_d
        self.dip_type = dip_type

    def _get_eval(self, x_dict={}, **kwargs):

        # Compute mu and scale of normal distribution
        params = self.p.get_params(x_dict)

        # Compute covariance
        if self.dip_type == "i":
            cov_dip = _get_cov_mu(params["loc"])
        elif self.dip_type == "ii":
            cov_dip = _get_e_cov(params["scale"]) + _get_cov_mu(params["loc"])

        # Get diagonal and off-diagonal elements
        cov_dip_diag = torch.diagonal(cov_dip)
        cov_dip_off_diag = (cov_dip
                            - cov_dip_diag * torch.eye(cov_dip_diag.size(0)))

        # Calculate moment
        dip = (self.lmd_od * (cov_dip_off_diag ** 2).sum()
               + self.lmd_d * ((cov_dip_diag - 1) ** 2).sum())

        return dip

    @property
    def _symbol(self):
        raise NotImplementedError


class DIPVAE(BaseVAE):
    def __init__(self, channel_num, z_dim, device, lmd_od, lmd_d, dip_type,
                 **kwargs):
        super().__init__(channel_num, z_dim, device)

        self.dip = DipLoss(self.q, lmd_od, lmd_d, dip_type)

    def _eval_loss(self, x_dict, **kwargs):
        raise NotImplementedError
