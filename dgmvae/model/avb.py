
"""Adversarial Variational Bayes (AVB)

Adversarial Variational Bayes: Unifying Variational Autoencoders and Generative
Adversarial Networks
http://arxiv.org/abs/1701.04722

Reference
https://github.com/gdikov/adversarial-variational-bayes
http://seiya-kumada.blogspot.com/2018/07/adversarial-variational-bayes.html
https://github.com/LMescheder/AdversarialVariationalBayes
"""

import torch
from torch import nn
from torch.nn import functional as F

import pixyz.distributions as pxd
import pixyz.losses as pxl

from .base import BaseVAE
from .dist import Decoder


class AVBDiscriminator(pxd.Deterministic):
    """T(x, z)"""
    def __init__(self, channel_num, z_dim):
        super().__init__(cond_var=["x", "z"], var=["t"], name="d")

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
        super().__init__(cond_var=["x", "e"], var=["z"], name="q")

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


class AVB(BaseVAE):
    def __init__(self, channel_num, z_dim, e_dim, beta, **kwargs):
        super().__init__()

        self.channel_num = channel_num
        self.z_dim = z_dim
        self.e_dim = e_dim
        self._beta_val = beta

        # Distributions
        self.normal = pxd.Normal(loc=torch.tensor(0.), scale=torch.tensor(1.),
                                 var=["e"], features_shape=[e_dim])
        self.prior = pxd.Normal(loc=torch.tensor(0.), scale=torch.tensor(1.),
                                var=["z"], features_shape=[z_dim])
        self.decoder = Decoder(channel_num, z_dim)
        self.encoder = AVBEncoder(channel_num, z_dim, e_dim)
        self.distributions = nn.ModuleList(
            [self.normal, self.prior, self.decoder, self.encoder])

        # Loss
        self.ce = pxl.CrossEntropy(self.encoder, self.decoder)

        # Adversarial loss
        self.disc = AVBDiscriminator(channel_num, z_dim)
        self.adv_js = pxl.AdversarialJensenShannon(
            self.encoder, self.prior, self.disc)

    def encode(self, x, mean=False):
        batch_n = x.size(0)
        e = self.normal.sample(batch_n=batch_n)
        inputs = {"x": x, "e": e["e"]}

        if mean:
            return self.encoder.sample_mean(inputs)
        return self.encoder.sample(inputs, return_all=False)

    def decode(self, z, mean=False):
        if not isinstance(z, dict):
            z = {"z": z}

        if mean:
            return self.decoder.sample_mean(z)
        return self.decoder.sample(z, return_all=False)

    def sample(self, batch_n=1):
        z = self.prior.sample(batch_n=batch_n)
        return self.decoder.sample_mean(z)

    def forward(self, x, reconstruct=True, return_latent=False):
        if reconstruct:
            z = self.encode(x)
            sample = self.decode(z, mean=True)
            if return_latent:
                z.update({"x": sample})
                return z
            return sample

        return self.encode(x, mean=True)

    def loss_func(self, x_dict, **kwargs):

        optimizer_idx = kwargs["optimizer_idx"]

        # Sample e
        batch_n = x_dict["x"].size(0)
        x_dict = self.normal.sample(x_dict, batch_n=batch_n)

        if optimizer_idx == 0:
            # VAE loss
            ce_loss = self.ce.eval(x_dict).mean()
            js_loss = self.adv_js.eval(x_dict).mean()
            loss = ce_loss + js_loss
            return {"loss": loss, "ce_loss": ce_loss, "js_loss": js_loss}
        elif optimizer_idx == 1:
            # Discriminator loss
            loss = self.adv_js.eval(x_dict, discriminator=True)
            return {"loss": loss, "adv_loss": loss}
        else:
            raise ValueError

    @property
    def loss_cls(self):
        return (self.ce + self.adv_js).expectation(self.normal)

    @property
    def second_optim(self):
        return self.adv_js.d_optimizer
