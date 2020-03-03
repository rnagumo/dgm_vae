
"""Adversarial Autoencoder (AAE)

Unsupervised clustering based on ch.6

Adversarial Autoencoders
http://arxiv.org/abs/1511.05644
"""

import collections

import torch
from torch import nn
from torch.nn import functional as F

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


class EncoderFunction(pxd.Deterministic):
    def __init__(self, channel_num):
        super().__init__(cond_var=["x"], var=["h"])

        self.conv1 = nn.Conv2d(channel_num, 32, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 4, stride=2, padding=1)
        self.fc1 = nn.Linear(1024, 256)

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = h.view(-1, 1024)
        h = F.relu(self.fc1(h))
        return {"h": h}


class ContinuousEncoder(pxd.Normal):
    def __init__(self, z_dim):
        super().__init__(cond_var=["h"], var=["z"])

        self.fc11 = nn.Linear(256, z_dim)
        self.fc12 = nn.Linear(256, z_dim)

    def forward(self, h):
        loc = self.fc11(h)
        scale = F.softplus(self.fc12(h))
        return {"loc": loc, "scale": scale}


class DiscreteEncoder(pxd.Categorical):
    def __init__(self, c_dim):
        super().__init__(cond_var=["h"], var=["c"])

        self.fc1 = nn.Linear(256, c_dim)

    def forward(self, h):
        logits = self.fc1(h)
        probs = F.softmax(logits, dim=1)
        probs = torch.clamp(probs, 1e-6, 1 - 1e-6)
        return {"probs": probs}


class JointDecoder(pxd.Bernoulli):
    def __init__(self, channel_num, z_dim, c_dim):
        super().__init__(cond_var=["z", "c"], var=["x"])

        self.fc1 = nn.Linear(z_dim + c_dim, 256)
        self.fc2 = nn.Linear(256, 1024)
        self.deconv1 = nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(32, channel_num, 4, stride=2,
                                          padding=1)

    def forward(self, z, c):
        h = F.relu(self.fc1(torch.cat([z, c], dim=1)))
        h = F.relu(self.fc2(h))
        h = h.view(-1, 64, 4, 4)
        h = F.relu(self.deconv1(h))
        h = F.relu(self.deconv2(h))
        h = F.relu(self.deconv3(h))
        probs = torch.sigmoid(self.deconv4(h))
        return {"probs": probs}


class AAE(BaseVAE):
    def __init__(self, device, channel_num, beta, z_dim, c_dim, **kwargs):
        super().__init__(channel_num, z_dim, device, beta, **kwargs)

        self.c_dim = c_dim

        # Prior
        self.prior_cont = pxd.Normal(
            loc=torch.tensor(0.), scale=torch.tensor(1.),
            var=["z"], features_shape=[z_dim]).to(device)
        self.prior_disc = pxd.Categorical(
            probs=torch.ones(c_dim, dtype=torch.float32) / c_dim,
            var=["c"]).to(device)

        # Encoder
        self.encoder_func = EncoderFunction(channel_num).to(device)
        self.encoder_cont = ContinuousEncoder(z_dim).to(device)
        self.encoder_disc = DiscreteEncoder(c_dim).to(device)

        # Decoder
        self.decoder = JointDecoder(channel_num, z_dim, c_dim).to(device)

        self.distributions = nn.ModuleList([
            self.prior_cont, self.prior_disc, self.encoder_func,
            self.encoder_cont, self.encoder_disc, self.decoder,
        ])

        # Loss
        self.ce = pxl.CrossEntropy(self.encoder_cont, self.decoder)

        # Adversarial loss
        self.disc = Discriminator(z_dim).to(device)
        self.adv_js = pxl.AdversarialJensenShannon(
            self.encoder_cont, self.prior_cont, self.disc)

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

            # Sample h (surrogate latent) and c (categorical latent)
            x_dict = {"x": x}
            x_dict = (self.encoder_disc * self.encoder_func).sample(x_dict)

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

    def reconstruct(self, x, concat=True):

        with torch.no_grad():
            x = x.to(self.device)
            h = self.encoder_func.sample(x, return_all=False)
            z = self.encoder_cont.sample(h, return_all=False)
            c = self.encoder_disc.sample(h, return_all=False)
            x_recon = self.decoder.sample_mean(
                {"z": z["z"], "c": c["c"]}).cpu()

        if concat:
            return torch.cat([z["z"], c["c"]]), x_recon

        return z["z"], c["c"], x_recon

    def sample(self, batch_n=1, concat=True):

        with torch.no_grad():
            z = self.prior_cont.sample(batch_n=batch_n)
            c = self.prior_disc.sample(batch_n=batch_n)
            x = self.decoder.sample_mean({"z": z["z"], "c": c["c"]}).cpu()

        if concat:
            return torch.cat([z["z"], c["c"]]), x

        return z["z"], x
