
"""Adversarial Variational Bayes (AVB)

Adversarial Variational Bayes: Unifying Variational Autoencoders and Generative
Adversarial Networks
http://arxiv.org/abs/1701.04722
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


class AVB(BaseVAE):
    def __init__(self, channel_num, z_dim, device, beta, **kwargs):
        super().__init__(channel_num, z_dim, device, beta, **kwargs)

        self.disc = Discriminator(z_dim).to(device)
        self.adv_js = pxl.AdversarialJensenShannon(
            self.encoder, self.prior, self.disc)

    def _eval_loss(self, x_dict, **kwargs):

        # Calculate loss
        ce_loss = self.ce.eval(x_dict).mean()
        js_loss = self.adv_js.eval(x_dict).mean()
        loss = ce_loss + js_loss

        loss_dict = {"loss": loss.item(), "ce_loss": ce_loss.item(),
                     "js_loss": js_loss.item()}

        return loss, loss_dict

    def run(self, loader, training=True):
        # Returned value
        loss_dict = collections.defaultdict(float)

        for x in loader:
            if isinstance(x, (tuple, list)):
                x = x[0]
            x = x.to(self.device)

            # Mini-batch size
            minibatch_size = x.size(0)

            # Input
            x_dict = {"x": x}

            # Calculate loss
            if training:
                _batch_loss = self.train(x_dict)
                _d_loss = self.adv_js.train(x_dict)
            else:
                _batch_loss = self.test(x_dict)
                _d_loss = self.adv_js.test(x_dict)

            # Accumulate minibatch loss
            for key in _batch_loss:
                loss_dict[key] += _batch_loss[key] * minibatch_size
            loss_dict["d_loss"] += _d_loss.item() * minibatch_size

        # Devide by data size
        for key in loss_dict:
            loss_dict[key] /= len(loader.dataset)

        return loss_dict
