
"""Shared distributions"""

from torch import nn
from torch.nn import functional as F

import pixyz.distributions as pxd


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

    def forward(self, x):
        h = self.conv(x)
        h = h.view(-1, 1024)
        h = F.relu(self.fc1(h))
        loc = self.fc21(h)
        scale = F.softplus(self.fc22(h))
        return {"loc": loc, "scale": scale}


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
