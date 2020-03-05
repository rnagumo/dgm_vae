
"""Base VAE class"""

from torch import nn


class BaseVAE(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, inputs):
        """Encode latent z given observable x"""
        raise NotImplementedError

    def decode(self, inputs):
        """Decode observable x given latent z"""
        raise NotImplementedError

    def sample(self, batch_size, device, **kwargs):
        """Sample data from latent"""
        raise NotImplementedError

    def reconstruct(self, x, **kwargs):
        """Reconstruct observable x' given data x"""
        raise NotImplementedError

    def forward(self, *inputs):
        raise NotImplementedError

    def loss_function(self, *inputs, **kwargs):
        raise NotImplementedError
