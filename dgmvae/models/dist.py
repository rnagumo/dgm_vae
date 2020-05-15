
"""Shared distributions"""

import torch
from torch import nn
from torch.nn import functional as F

import pixyz.distributions as pxd
from pixyz.utils import get_dict_values, sum_samples


class Encoder(pxd.Normal):
    def __init__(self, channel_num, z_dim):
        super().__init__(cond_var=["x"], var=["z"], name="q")

        self.conv = nn.Sequential(
            nn.Conv2d(channel_num, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 4, stride=2, padding=1),
            nn.ReLU(),
        )

        self.fc1 = nn.Linear(1024, 256)
        self.fc21 = nn.Linear(256, z_dim)
        self.fc22 = nn.Linear(256, z_dim)

        self.encoded_params = {}

    def forward(self, x):
        h = self.conv(x)
        h = h.view(-1, 1024)
        h = F.relu(self.fc1(h))
        loc = self.fc21(h)
        scale = F.softplus(self.fc22(h))
        self.encoded_params = {"loc": loc, "scale": scale}
        return self.encoded_params

    def get_log_prob_wo_forward(self, x_dict: dict, sum_features: bool = True):
        # Override `set_dist` method
        self._dist = self.distribution_torch_class(**self.encoded_params)

        # Calculate log prob
        x_targets = get_dict_values(x_dict, self._var)
        log_prob = self.dist.log_prob(*x_targets)

        if sum_features:
            log_prob = sum_samples(log_prob)
        return log_prob


class Decoder(pxd.Bernoulli):
    def __init__(self, channel_num, z_dim):
        super().__init__(cond_var=["z"], var=["x"], name="p")

        self.fc = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU(),
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, channel_num, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.view(-1, 64, 4, 4)
        probs = self.deconv(h)
        return {"probs": probs}


class Discriminator(pxd.Deterministic):
    def __init__(self, z_dim):
        super().__init__(cond_var=["z"], var=["t"], name="d")

        self.model = nn.Sequential(
            nn.Linear(z_dim, 1000),
            nn.LeakyReLU(0.2),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2),
            nn.Linear(1000, 1),
        )

    def forward(self, z):
        logits = self.model(z)
        probs = torch.sigmoid(logits)
        t = torch.clamp(probs, 1e-6, 1 - 1e-6)
        return {"t": t}
