
"""Adversarial Autoencoder (AAE)

Unsupervised clustering based on ch.6

Adversarial Autoencoders
http://arxiv.org/abs/1511.05644
"""

from typing import Dict, Optional

import torch
from torch import nn, optim, Tensor
from torch.nn import functional as F

import pixyz.distributions as pxd
import pixyz.losses as pxl

from .base import BaseVAE
from .dist import Discriminator


class EncoderFunction(pxd.Deterministic):
    def __init__(self, channel_num):
        super().__init__(cond_var=["x"], var=["h"], name="f_e")

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
        super().__init__(cond_var=["h"], var=["z"], name="q_z")

        self.fc11 = nn.Linear(256, z_dim)
        self.fc12 = nn.Linear(256, z_dim)

    def forward(self, h):
        loc = self.fc11(h)
        scale = F.softplus(self.fc12(h))
        return {"loc": loc, "scale": scale}


class DiscreteEncoder(pxd.Categorical):
    def __init__(self, c_dim):
        super().__init__(cond_var=["h"], var=["c"], name="q_c")

        self.fc1 = nn.Linear(256, c_dim)

    def forward(self, h):
        logits = self.fc1(h)
        probs = F.softmax(logits, dim=1)
        probs = torch.clamp(probs, 1e-6, 1 - 1e-6)
        return {"probs": probs}


class JointDecoder(pxd.Bernoulli):
    def __init__(self, channel_num, z_dim, c_dim):
        super().__init__(cond_var=["z", "c"], var=["x"], name="p")

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
    """Adversarial Autoencoder (AAE).

    Unsupervised clustering based on ch.6

    ref) Adversarial Autoencoders
    http://arxiv.org/abs/1511.05644

    Args:
        channel_num (int): Number of input channels.
        z_dim (int): Dimension of latents `z`.
        c_dim (int): Dimension of latents `c`.
        beta (float): Beta regularization term.
    """

    def __init__(self, channel_num: int, z_dim: int, c_dim: int, beta: float,
                 **kwargs):
        super().__init__()

        # Parameters
        self.channel_num = channel_num
        self.z_dim = z_dim
        self.c_dim = c_dim
        self._beta_value = beta

        # Prior
        self.prior_z = pxd.Normal(
            loc=torch.zeros(z_dim), scale=torch.ones(z_dim), var=["z"])
        self.prior_c = pxd.Categorical(
            probs=torch.ones(c_dim, dtype=torch.float32) / c_dim, var=["c"])

        # Encoder
        self.encoder_func = EncoderFunction(channel_num)
        self.encoder_z = ContinuousEncoder(z_dim)
        self.encoder_c = DiscreteEncoder(c_dim)

        # Decoder
        self.decoder = JointDecoder(channel_num, z_dim, c_dim)

        self.distributions = [self.prior_z, self.prior_c, self.encoder_func,
                              self.encoder_z, self.encoder_c, self.decoder]

        # Loss
        self.ce = pxl.CrossEntropy(self.encoder_z, self.decoder)
        self.beta = pxl.Parameter("beta")

        # Adversarial loss
        self.disc = Discriminator(z_dim)
        self.adv_js = pxl.AdversarialJensenShannon(
            self.encoder_z, self.prior_z, self.disc)

    def encode(self, x_dict: Dict[str, Tensor], mean: bool = False, **kwargs
               ) -> Dict[str, Tensor]:
        """Encodes latents given observable x.

        Args:
            x_dict (dict of [str, torch.Tensor]): Dict of Tensor for input
                observations.
            mean (bool, optional): Boolean flag for returning means or samples.

        Returns:
            z_dict (dict of [str, torch.Tensor]): Dict of tensor of encoded
                latents.
        """

        h = self.encoder_func.sample(x_dict, return_all=False)

        if mean:
            z = self.encoder_z.sample_mean(h)
            c = self.encoder_c.sample_mean(h)
            latent = {"z": z, "c": c}
            return latent

        z = self.encoder_z.sample(h, return_all=False)
        c = self.encoder_c.sample(h, return_all=False)
        z.update(c)
        return z

    def decode(self, z_dict: Dict[str, Tensor], mean: bool = False, **kwargs
               ) -> Dict[str, Tensor]:
        """Decodes observable x given latents.

        Args:
            z_dict (dict of [str, torch.Tensor]): Dict of latents tensors.
            mean (bool, optional): Boolean flag for returning means or samples.

        Returns:
            x_dict (dict of [str, torch.Tensor]): Dict of tensor of decoded
                observations.
        """

        if mean:
            x = self.decoder.sample_mean(z_dict)
            return {"x": x}

        return self.decoder.sample(z_dict, return_all=False)

    def sample(self, batch_n: int) -> Dict[str, Tensor]:
        """Samples observable x from sampled latent z.

        Args:
            batch_n (int): Batch size.

        Returns:
            x_dict (dict of [str, torch.Tensor]): Dict of sampled obsercation
                tensor.
        """

        z = self.prior_z.sample(batch_n=batch_n)
        c = self.prior_c.sample(batch_n=batch_n)
        x = self.decoder.sample_mean({"z": z["z"], "c": c["c"]})
        return {"x": x}

    def loss_func(self, x: Tensor, **kwargs) -> Dict[str, Tensor]:
        """Calculates loss given observable x.

        Args:
            x (torch.Tensor): Tensor of input observations.

        Returns:
            loss_dict (dict of [str, torch.Tensor]): Dict of calculated losses.
        """

        # Input
        x_dict = {"x": x}

        # Select optimizer
        optimizer_idx = kwargs["optimizer_idx"]

        # Sample h (surrogate latent) and c (categorical latent)
        x_dict = (self.encoder_c * self.encoder_func).sample(x_dict)

        if optimizer_idx == 0:
            # VAE loss
            ce_loss = self.ce.eval(x_dict).mean()
            js_loss = self.adv_js.eval(x_dict).mean()
            loss = ce_loss + js_loss
            return {"loss": loss, "ce_loss": ce_loss, "js_loss": js_loss}
        elif optimizer_idx == 1:
            # Discriminator loss
            loss = self.adv_js.eval(x_dict, discriminator=True)
            return {"adv_loss": loss}

    @property
    def loss_str(self) -> str:
        return str((self.ce + self.adv_js).expectation(
                   self.encoder_c * self.encoder_func))

    @property
    def second_optim(self) -> Optional[optim.Optimizer]:
        return self.adv_js.d_optimizer
