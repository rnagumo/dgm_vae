
"""Adversarial Variational Bayes (AVB)

Adversarial Variational Bayes: Unifying Variational Autoencoders and Generative
Adversarial Networks
http://arxiv.org/abs/1701.04722

Reference
https://github.com/gdikov/adversarial-variational-bayes
http://seiya-kumada.blogspot.com/2018/07/adversarial-variational-bayes.html
https://github.com/LMescheder/AdversarialVariationalBayes
"""

from typing import Dict, Optional

import torch
from torch import nn, optim, Tensor

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
            nn.ReLU(),
            nn.Conv2d(32, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.fc_x = nn.Linear(1024, 256)

        self.disc_z = nn.Sequential(
            nn.Linear(z_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
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
            nn.ReLU(),
            nn.Conv2d(32, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.fc_x = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, z_dim),
            nn.ReLU(),
        )
        self.fc_e = nn.Sequential(
            nn.Linear(e_dim, z_dim),
            nn.ReLU(),
        )
        self.fc = nn.Linear(z_dim * 2, z_dim)

    def forward(self, x, e):
        h_x = self.enc_x(x)
        h_x = h_x.view(-1, 1024)
        h_x = self.fc_x(h_x)
        h_e = self.fc_e(e)
        z = self.fc(torch.cat([h_x, h_e], dim=1))
        return {"z": z}


class AVB(BaseVAE):
    """Adversarial Variational Bayes (AVB).

    Args:
        channel_num (int): Number of input channels.
        z_dim (int): Dimension of latents `z`.
        e_dim (int): Dimension of noize `e`.
        beta (float): Beta regularization term.
    """

    def __init__(self, channel_num: int, z_dim: int, e_dim: int, beta: float,
                 **kwargs):
        super().__init__()

        self.channel_num = channel_num
        self.z_dim = z_dim
        self.e_dim = e_dim
        self._beta_val = beta

        # Distributions
        self.normal = pxd.Normal(
            loc=torch.zeros(z_dim), scale=torch.ones(z_dim), var=["e"])
        self.prior = pxd.Normal(
            loc=torch.zeros(z_dim), scale=torch.ones(z_dim), var=["z"])
        self.decoder = Decoder(channel_num, z_dim)
        self.encoder = AVBEncoder(channel_num, z_dim, e_dim)
        self.distributions = [self.normal, self.prior, self.decoder,
                              self.encoder]

        # Loss
        self.ce = pxl.CrossEntropy(self.encoder, self.decoder)

        # Adversarial loss
        self.disc = AVBDiscriminator(channel_num, z_dim)
        self.adv_js = pxl.AdversarialJensenShannon(
            self.encoder, self.prior, self.disc)

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

        batch_n = x_dict["x"].size(0)
        e = self.normal.sample(batch_n=batch_n)
        x_dict.update({"e": e["e"]})

        if mean:
            z = self.encoder.sample_mean(x_dict)
            return {"z": z}
        return self.encoder.sample(x_dict, return_all=False)

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

        z = self.prior.sample(batch_n=batch_n)
        x = self.decoder.sample_mean(z)
        return {"x": x}

    def loss_func(self, x: Tensor, **kwargs) -> Dict[str, Tensor]:
        """Calculates loss given observable x.

        Args:
            x (torch.Tensor): Tensor of input observations.

        Returns:
            loss_dict (dict of [str, torch.Tensor]): Dict of calculated losses.
        """

        optimizer_idx = kwargs["optimizer_idx"]

        # Sample e
        batch_n = x.size(0)
        x_dict = self.normal.sample({"x": x}, batch_n=batch_n)

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
        return str((self.ce + self.adv_js).expectation(self.normal))

    @property
    def second_optim(self) -> Optional[optim.Optimizer]:
        return self.adv_js.d_optimizer
