
"""beta-TCVAE

Isolating Sources of Disentanglement in Variational Autoencoders
http://arxiv.org/abs/1802.04942
"""


from .base import BaseVAE


class TCVAE(BaseVAE):
    def __init__(self, channel_num, z_dim, device, beta, **kwargs):
        super().__init__(channel_num, z_dim, device, beta, **kwargs)

    def _eval_loss(self, x_dict, **kwargs):
        raise NotImplementedError
