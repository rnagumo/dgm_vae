
"""beta-TCVAE

Isolating Sources of Disentanglement in Variational Autoencoders
http://arxiv.org/abs/1802.04942

code by author
https://github.com/rtqichen/beta-tcvae
"""

import torch

from .base import BaseVAE


class TCVAE(BaseVAE):
    def __init__(self, channel_num, z_dim, device, alpha, beta, gamma,
                 **kwargs):
        super().__init__(channel_num, z_dim, device, **kwargs)

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def _eval_loss(self, x_dict, **kwargs):

        # Sample z from encoder
        x_dict = self.encoder.sample(x_dict)

        # log p(x)
        log_px = self.decoder.get_log_prob(x_dict)

        # log p(z)
        log_pz = self.prior.get_log_prob(x_dict)

        # log q(z|x)
        log_qz_x = self.encoder.get_log_prob(x_dict)

        # Minibatch Weighted Sampling
        # log q(z) size of (z_batch_size, x_batch_size, z_dim)
        x_dict_tmp = {"x": x_dict["x"], "z": x_dict["z"].unsqueeze(1)}
        _logqz = self.encoder.get_log_prob(x_dict_tmp, sum_features=False)

        # log NM
        dataset_size = x_dict["dataset_size"]
        batch_size = x_dict["x"].size(0)
        lognm = torch.log(
            torch.tensor([dataset_size * batch_size], dtype=torch.float32))

        # log \prod q(z_j)
        log_qz_prodmarginal = (torch.logsumexp(_logqz, 1) - lognm).sum(1)

        # log q(z)
        log_qz = torch.logsumexp(_logqz.sum(2), 1) - lognm

        # Calculate ELBO loss
        recon = -log_px.mean()
        mutual_info = self.alpha * (log_qz_x - log_qz).mean()
        independence = self.beta * (log_qz - log_qz_prodmarginal).mean()
        dim_wise_kl = self.gamma * (log_qz_prodmarginal - log_pz).mean()
        loss = recon + mutual_info + independence + dim_wise_kl

        loss_dict = {"loss": loss, "recon": recon, "mutual_info": mutual_info,
                     "independence": independence, "dim_wise_kl": dim_wise_kl}

        return loss, loss_dict
