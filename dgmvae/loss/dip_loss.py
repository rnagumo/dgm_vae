
"""DIP loss"""

import torch
from pixyz.losses.losses import Loss


def _get_cov_mu(mu):
    """Computes covariance of mu.

    cov(mu) = E[mu mu^T] - E[mu]E[mu]^T

    Input
        mu: [batch_size, latent_dim]
    Output
        cov(mu): [latent_dim, latent_dim]
    """

    # E[mu mu^T]
    e_mu_mut = (mu.unsqueeze(2) * mu.unsqueeze(1)).mean(dim=0)

    # E[mu]E[mu]^T
    e_mu = mu.sum(dim=0)
    e_mu_e_mut = e_mu.unsqueeze(1) * e_mu.unsqueeze(0)

    return e_mu_mut - e_mu_e_mut


def _get_e_cov(scale):
    """Computes expectation of cov E[Cov_encoder] from scale

    Input
        scale: [batch_size, latent_dim]
    Output
        cov: [latent_dim, latent_dim]
    """

    # Cov
    cov = scale.unsqueeze(2) * torch.eye(scale.size(1), device=scale.device)

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
        cov_dip_diag = torch.diag(cov_dip)
        eye = torch.eye(cov_dip.size(0), device=cov_dip.device)
        cov_dip_off_diag = cov_dip - cov_dip_diag * eye

        # Calculate loss
        loss = (self.lmd_od * (cov_dip_off_diag ** 2).sum()
                + self.lmd_d * ((cov_dip_diag - 1) ** 2).sum())

        return loss, x_dict

    @property
    def _symbol(self):
        raise NotImplementedError
