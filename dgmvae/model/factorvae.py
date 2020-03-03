
"""FactorVAE

Disentangling by Factorising
http://arxiv.org/abs/1802.05983
"""

import collections

import torch
from torch import nn

import pixyz.distributions as pxd
import pixyz.losses as pxl

from .base import BaseVAE


class Discriminator(pxd.Deterministic):
    def __init__(self, z_dim):
        super().__init__(cond_var=["z"], var=["t"])

        self.model = nn.Sequential(
            nn.Linear(z_dim, 1000),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1000, 1),
        )

    def forward(self, z):
        logits = self.model(z)
        probs = torch.sigmoid(logits)
        t = torch.clamp(probs, 1e-6, 1 - 1e-6)
        return {"t": t}


class InferenceShuffleDim(pxd.Deterministic):
    def __init__(self, q):
        super().__init__(cond_var=["x_shf"], var=["z"])

        self.q = q

    def forward(self, x_shf):
        z = self.q.sample({"x": x_shf}, return_all=False)["z"]
        return {"z": z[:, torch.randperm(z.size(1))]}


class FactorVAE(BaseVAE):
    def __init__(self, channel_num, z_dim, device, beta, gamma, **kwargs):
        super().__init__(channel_num, z_dim, device, beta, **kwargs)

        self.gamma = gamma

        # Dimension shuffle
        self.encoder_shf = InferenceShuffleDim(self.encoder).to(device)

        # Discriminator
        self.disc = Discriminator(z_dim).to(device)

        # Loss
        self.tc = pxl.AdversarialKullbackLeibler(
            self.encoder, self.encoder_shf, self.disc)

    def _eval_loss(self, x_dict, **kwargs):

        ce_loss = self.ce.eval(x_dict).mean()
        kl_loss = self.beta * self.kl.eval(x_dict).mean()
        tc_loss = self.gamma * self.tc.eval(x_dict)

        loss = ce_loss + kl_loss + tc_loss
        loss_dict = {"loss": loss.item(), "ce_loss": ce_loss.item(),
                     "kl_loss": kl_loss.item(), "tc_loss": tc_loss.item()}

        return loss, loss_dict

    def run(self, loader, training=True):
        """Overrides super().run() because tc needs to be trained."""

        # Returned value
        loss_dict = collections.defaultdict(float)

        for x in loader:
            if isinstance(x, (tuple, list)):
                x = x[0]
            x = x.to(self.device)
            len_x = x.size(0) // 2

            # Mini-batch size
            minibatch_size = x.size(0)

            # Calculate loss
            if training:
                _batch_loss = self.train({"x": x[:len_x], "x_shf": x[len_x:]})
                _d_loss = self.tc.train({"x": x[:len_x], "x_shf": x[len_x:]})
            else:
                _batch_loss = self.test({"x": x[:len_x], "x_shf": x[len_x:]})
                _d_loss = self.tc.test({"x": x[:len_x], "x_shf": x[len_x:]})

            # Accumulate minibatch loss
            for key in _batch_loss:
                loss_dict[key] += _batch_loss[key] * minibatch_size
            loss_dict["d_loss"] += _d_loss.item() * minibatch_size

        # Devide by data size
        for key in loss_dict:
            loss_dict[key] /= len(loader.dataset)

        return loss_dict
