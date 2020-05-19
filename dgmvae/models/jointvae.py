
"""JointVAE

Learning Disentangled Joint Continuous and Discrete Representations
http://arxiv.org/abs/1804.00104
"""

from typing import Dict

import torch
from torch import nn, Tensor
from torch.nn import functional as F

import pixyz.distributions as pxd
import pixyz.losses as pxl

from .base import BaseVAE
from ..losses.discrete_kl import CategoricalKullbackLeibler


class EncoderFunction(pxd.Deterministic):
    def __init__(self, channel_num):
        super().__init__(cond_var=["x"], var=["h"], name="f")

        self.enc_x = nn.Sequential(
            nn.Conv2d(channel_num, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
        )

    def forward(self, x):
        h = self.enc_x(x)
        h = h.view(-1, 1024)
        h = self.fc(h)
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


class DiscreteEncoder(pxd.RelaxedCategorical):
    def __init__(self, c_dim, temperature):
        super().__init__(cond_var=["h"], var=["c"], name="q_c",
                         temperature=temperature)

        self.fc1 = nn.Linear(256, c_dim)

    def forward(self, h):
        logits = self.fc1(h)
        probs = F.softmax(logits, dim=1)
        probs = torch.clamp(probs, 1e-6, 1 - 1e-6)
        return {"probs": probs}


class JointDecoder(pxd.Bernoulli):
    def __init__(self, channel_num, z_dim, c_dim):
        super().__init__(cond_var=["z", "c"], var=["x"])

        self.fc = nn.Sequential(
            nn.Linear(z_dim + c_dim, 256),
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

    def forward(self, z, c):
        h = self.fc(torch.cat([z, c], dim=1))
        h = h.view(-1, 64, 4, 4)
        probs = self.deconv(h)
        return {"probs": probs}


class JointVAE(BaseVAE):
    """Joint VAE.

    Args:
        channel_num (int): Number of input channels.
        z_dim (int): Dimension of continuous latents `z`.
        c_dim (int): Dimension of discrete latents `c`.
        temperature (float): Temperature for discrete encoder.
        gamma_z (float): Gamma regularization term for `z`.
        gamma_c (float): Gamma regularization term for `c`.
        cap_z (float): Capacity for `z`.
        cap_c (float): Capacity for `c`.
    """

    def __init__(self, channel_num: int, z_dim: int, c_dim: int,
                 temperature: float, gamma_z: float, gamma_c: float,
                 cap_z: float, cap_c: float, **kwargs):
        super().__init__()

        self.channel_num = channel_num
        self.z_dim = z_dim
        self.c_dim = c_dim
        self._gamma_z_value = gamma_z
        self._gamma_c_value = gamma_c
        self._cap_z_value = cap_z
        self._cap_c_value = cap_c

        # Distributions
        self.prior_z = pxd.Normal(
            loc=torch.zeros(z_dim), scale=torch.ones(z_dim), var=["z"])
        self.prior_c = pxd.Categorical(
            probs=torch.ones(c_dim, dtype=torch.float32) / c_dim, var=["c"])

        self.encoder_func = EncoderFunction(channel_num)
        self.encoder_z = ContinuousEncoder(z_dim)
        self.encoder_c = DiscreteEncoder(c_dim, temperature)
        self.decoder = JointDecoder(channel_num, z_dim, c_dim)

        self.distributions = [self.prior_z, self.prior_c, self.encoder_func,
                              self.encoder_z, self.encoder_c, self.decoder]

        # Loss
        self.ce = pxl.CrossEntropy(self.encoder_z * self.encoder_c,
                                   self.decoder)
        self.kl_z = pxl.KullbackLeibler(self.encoder_z, self.prior_z)
        self.kl_c = CategoricalKullbackLeibler(
            self.encoder_c, self.prior_c)

        # Coefficient for kl
        self.gamma_z = pxl.Parameter("gamma_z")
        self.gamma_c = pxl.Parameter("gamma_c")

        # Capacity
        self.cap_z = pxl.Parameter("cap_z")
        self.cap_c = pxl.Parameter("cap_c")

    def encode(self, x_dict: Dict[str, Tensor], mean: bool = False, **kwargs
               ) -> Dict[str, Tensor]:
        """Encodes latents given observable x.

        Args:
            x_dict (dict of [str, torch.Tensor]): Dict of Tensor for input
                observations.
            mean (bool, optional): Boolean flag for returning means or samples.

        Returns:
            latents (dict of [str, torch.Tensor]): Tensor of encoded latents.
        """

        h = self.encoder_func.sample(x_dict, return_all=False)

        if mean:
            z = self.encoder_z.sample_mean(h)
            c = self.encoder_c.sample_mean(h)
            return {"z": z, "c": c}

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
            x (dict of [str, torch.Tensor]): Tensor of decoded observations.
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
            samples (dict of [str, torch.Tensor]): Dict of sampled tensor.
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

        # TODO: update capacity values per epoch
        x_dict = {
            "x": x,
            "gamma_z": self._gamma_z_value,
            "gamma_c": self._gamma_c_value,
            "cap_z": self._cap_z_value,
            "cap_c": self._cap_c_value,
        }

        # Sample h (surrogate latent variable)
        x_dict = self.encoder_func.sample(x_dict)

        # Cross entropy
        ce_loss = self.ce.eval(x_dict).mean()

        # KL for continuous latent
        kl_z_loss = (
            self.gamma_z * (self.kl_z - self.cap_z).abs()).eval(x_dict).mean()

        # KL for discrete latent
        kl_c_loss = (
            self.gamma_c * (self.kl_c - self.cap_c).abs()).eval(x_dict).mean()

        loss = ce_loss + kl_z_loss + kl_c_loss
        loss_dict = {"loss": loss, "ce_loss": ce_loss, "kl_z_loss": kl_z_loss,
                     "kl_c_loss": kl_c_loss}

        return loss_dict

    @property
    def loss_str(self):
        return str((self.ce + self.gamma_z * (self.kl_z - self.cap_z).abs()
                    + self.gamma_c * (self.kl_c - self.cap_c).abs()
                    ).expectation(self.encoder_func))
