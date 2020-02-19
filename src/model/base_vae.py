
"""Base VAE class"""

import tqdm

import torch
from torch import nn, optim
from torch.nn import functional as F

import pixyz.distributions as pxd
import pixyz.losses as pxl


class Encoder(pxd.Normal):
    def __init__(self, channel_num, z_dim):
        super().__init__(cond_var=["x"], var=["z"])

        self.conv1 = nn.Conv2d(channel_num, 32, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 4, stride=2, padding=1)
        self.fc1 = nn.Linear(1024, 256)
        self.fc21 = nn.Linear(256, z_dim)
        self.fc22 = nn.Linear(256, z_dim)

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = h.view(-1, 1024)
        h = F.relu(self.fc1(h))
        loc = self.fc21(h)
        scale = F.softplus(self.fc22(h))
        return {"loc": loc, "scale": scale}


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


class BaseVAE:
    def __init__(self, channel_num, z_dim, device, beta=1, **kwargs):

        self.channel_num = channel_num
        self.z_dim = z_dim
        self.device = device
        self.beta = 1

        # Distributions
        self.prior = pxd.Normal(loc=torch.tensor(0.), scale=torch.tensor(1.),
                                var=["z"], features_shape=[z_dim]).to(device)
        self.decoder = Decoder(channel_num, z_dim).to(device)
        self.encoder = Encoder(channel_num, z_dim).to(device)
        self.distributions = nn.ModuleList(
            [self.prior, self.decoder, self.encoder])

        # Loss class
        self.ce = pxl.CrossEntropy(self.encoder, self.decoder)
        self.kl = pxl.KullbackLeibler(self.encoder, self.prior)

        # Optimizer
        params = self.distributions.parameters()
        self.optimizer = optim.Adam(params)

    def _eval_loss(self, x_dict, **kwargs):

        ce_loss = self.ce.eval(x_dict).mean()
        kl_loss = self.kl.eval(x_dict).mean()
        loss = ce_loss + self.beta * kl_loss
        loss_dict = {"loss": loss.item(), "ce_loss": ce_loss.item(),
                     "kl_loss": kl_loss.item()}

        return loss, loss_dict

    def run(self, loader, training=True):

        # Returned value
        total_loss = 0
        ce_loss = 0
        kl_loss = 0

        for x in tqdm.tqdm(loader):
            if isinstance(x, (tuple, list)):
                x = x[0]
            x = x.to(self.device)

            # Mini-batch size
            minibatch_size = x.size(0)

            # Calculate loss
            if training:
                loss_dict = self.train({"x": x})
            else:
                loss_dict = self.test({"x": x})

            # Log
            total_loss += loss_dict["loss"] * minibatch_size
            ce_loss += loss_dict["ce_loss"] * minibatch_size
            kl_loss += loss_dict["kl_loss"] * minibatch_size

        total_loss /= len(loader.dataset)
        ce_loss /= len(loader.dataset)
        kl_loss /= len(loader.dataset)

        return {"loss": total_loss, "ce_loss": ce_loss, "kl_loss": kl_loss}

    def train(self, x_dict={}, **kwargs):

        # Initialization
        self.distributions.train()
        self.optimizer.zero_grad()

        # Loss evaluation
        loss, loss_dict = self._eval_loss(x_dict, **kwargs)

        # Backprop
        loss.backward()

        # Update
        self.optimizer.step()

        return loss_dict

    def test(self, x_dict={}, **kwargs):

        self.distributions.eval()

        with torch.no_grad():
            _, loss_dict = self._eval_loss(x_dict, **kwargs)

        return loss_dict

    def reconstruction(self, x):

        with torch.no_grad():
            x = x.to(self.device)
            z = self.encoder.sample(x, return_all=False)
            x_recon = self.decoder.sample_mean(z).cpu()

        return torch.cat([x, x_recon])

    def sample(self, batch_n=1):

        with torch.no_grad():
            z = self.prior.sample(batch_n=batch_n)
            x = self.decoder.sample_mean(z).cpu()

        return x
