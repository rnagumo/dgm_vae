
"""Adversarial Variational Bayes (AVB)

Adversarial Variational Bayes: Unifying Variational Autoencoders and Generative
Adversarial Networks
http://arxiv.org/abs/1701.04722

Reference
https://github.com/gdikov/adversarial-variational-bayes
http://seiya-kumada.blogspot.com/2018/07/adversarial-variational-bayes.html
https://github.com/LMescheder/AdversarialVariationalBayes
"""

import collections

import torch
from torch import nn, optim
from torch.nn import functional as F

import pixyz.distributions as pxd
import pixyz.losses as pxl

from .base import BaseVAE


class AVBDiscriminator(pxd.Deterministic):
    """T(x, z)"""
    def __init__(self, channel_num, z_dim):
        super().__init__(cond_var=["x", "z"], var=["t"])

        self.disc_x = nn.Sequential(
            nn.Conv2d(channel_num, 32, 4, stride=2, padding=1),
            nn.Conv2d(32, 32, 4, stride=2, padding=1),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.Conv2d(64, 64, 4, stride=2, padding=1),
        )
        self.fc_x = nn.Linear(1024, 256)

        self.disc_z = nn.Sequential(
            nn.Linear(z_dim, 512),
            nn.Linear(512, 512),
            nn.Linear(512, 256),
        )

        self.fc = nn.Linear(512, 1)

    def forward(self, x, z):
        h_x = self.disc_x(x)
        h_x = self.fc_x(h_x.view(-1, 1024))
        h_z = self.disc_z(z)
        logits = self.fc(torch.cat([h_x, h_z], dim=1))
        probs = torch.sigmoid(logits)
        t = torch.clamp(probs, 1e-6, 1 - 1e-6)
        return {"t": t}


class AVBEncoder(pxd.Deterministic):
    """Deterministic encoder z_phi (x, e)"""
    def __init__(self, channel_num, z_dim, e_dim):
        super().__init__(cond_var=["x", "e"], var=["z"])

        self.enc_x = nn.Sequential(
            nn.Conv2d(channel_num, 32, 4, stride=2, padding=1),
            nn.Conv2d(32, 32, 4, stride=2, padding=1),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.Conv2d(64, 64, 4, stride=2, padding=1),
        )
        self.fc_x = nn.Sequential(
            nn.Linear(1024, 256),
            nn.Linear(256, z_dim)
        )
        self.fc_e = nn.Linear(e_dim, z_dim)
        self.fc = nn.Linear(z_dim * 2, z_dim)

    def forward(self, x, e):
        h_x = self.enc_x(x)
        h_x = h_x.view(-1, 1024)
        h_x = F.relu(self.fc_x(h_x))
        h_e = self.fc_e(e)
        z = self.fc(torch.cat([h_x, h_e], dim=1))
        return {"z": z}


class Decoder(pxd.Bernoulli):
    def __init__(self, channel_num, z_dim):
        super().__init__(cond_var=["z"], var=["x"])

        self.fc1 = nn.Linear(z_dim, 256)
        self.fc2 = nn.Linear(256, 1024)
        self.deconv1 = nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(32, channel_num, 4, stride=2,
                                          padding=1)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        h = h.view(-1, 64, 4, 4)
        h = F.relu(self.deconv1(h))
        h = F.relu(self.deconv2(h))
        h = F.relu(self.deconv3(h))
        probs = torch.sigmoid(self.deconv4(h))
        return {"probs": probs}


class AVB(BaseVAE):
    def __init__(self, channel_num, z_dim, e_dim, device, beta, **kwargs):

        self.channel_num = channel_num
        self.z_dim = z_dim
        self.e_dim = e_dim
        self.device = device
        self.beta = beta

        # Distributions
        self.normal = pxd.Normal(loc=torch.tensor(0.), scale=torch.tensor(1.),
                                 var=["e"], features_shape=[e_dim]).to(device)
        self.prior = pxd.Normal(loc=torch.tensor(0.), scale=torch.tensor(1.),
                                var=["z"], features_shape=[z_dim]).to(device)
        self.decoder = Decoder(channel_num, z_dim).to(device)
        self.encoder = AVBEncoder(channel_num, z_dim, e_dim).to(device)
        self.distributions = nn.ModuleList(
            [self.normal, self.prior, self.decoder, self.encoder])

        # Loss
        self.ce = pxl.CrossEntropy(self.encoder, self.decoder)

        # Adversarial loss
        self.disc = AVBDiscriminator(channel_num, z_dim).to(device)
        self.adv_js = pxl.AdversarialJensenShannon(
            self.encoder, self.prior, self.disc)

        # Optimizer
        params = self.distributions.parameters()
        self.optimizer = optim.Adam(params)

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

            # Sample e (noize)
            x_dict = self.normal.sample(x_dict, batch_n=minibatch_size)

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

    def reconstruct(self, x):

        batch_n = x.size(0)

        with torch.no_grad():
            x = x.to(self.device)
            e = self.normal.sample(batch_n=batch_n)
            z = self.encoder.sample({"x": x, "e": e}, return_all=False)
            x_recon = self.decoder.sample_mean(z).cpu()

        return z["z"], x_recon
