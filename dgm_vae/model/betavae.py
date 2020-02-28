
"""beta-VAE

β-VAE: LEARNING BASIC VISUAL CONCEPTS WITH A CONSTRAINED VARIATIONAL FRAMEWORK
https://openreview.net/forum?id=Sy2fzU9gl
"""


from .base import BaseVAE


class BetaVAE(BaseVAE):
    def __init__(self, channel_num, z_dim, device, beta=2, **kwargs):

        if beta == 1:
            raise ValueError(f"Normal VAE: beta={beta}")

        super().__init__(channel_num, z_dim, device, beta, **kwargs)
