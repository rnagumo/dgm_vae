
"""Base VAE class"""

from torch import nn
from pixyz.distributions.distributions import Distribution


class BaseVAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.distributions = nn.ModuleDict([])

    def encode(self, *inputs, mean=True):
        """Encode latent z given observable x"""
        raise NotImplementedError

    def decode(self, *inputs, mean=False):
        """Decode observable x given latent z"""
        raise NotImplementedError

    def sample(self, batch_size, device, **kwargs):
        """Sample observable x' from sampled latent z"""
        raise NotImplementedError

    def forward(self, x, mean=True, reconstruct=True):
        """Reconstruct inputs data"""
        raise NotImplementedError

    def loss_func(self, *inputs, **kwargs):
        """Calculate loss in train/val/test"""
        raise NotImplementedError

    @property
    def loss_cls(self):
        """Return instance of pixyz.losses.Loss class"""
        raise NotImplementedError

    def __str__(self):
        prob_text = []
        func_text = []

        for prob in self.distributions._modules.values():
            if isinstance(prob, Distribution):
                prob_text.append(prob.prob_text)
            else:
                func_text.append(prob.__str__())

        text = ("Distributions (for training): \n "
                " {} \n".format(", ".join(prob_text)))
        if len(func_text) > 0:
            text += ("Deterministic functions (for training): \n "
                     " {} \n".format(", ".join(func_text)))

        text += "Loss function: \n  {} \n".format(str(self.loss_cls))
        return text
