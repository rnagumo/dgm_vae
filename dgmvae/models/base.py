
"""Base VAE class"""

from torch import nn
from pixyz.distributions.distributions import Distribution


class BaseVAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.distributions = []

    def forward(self, x, reconstruct=False, return_latent=False):
        # Encode without sampling
        if not reconstruct:
            return self.encode(x, mean=True)

        # Reconstruct image
        latent = self.encode(x)
        obs = self.decode(latent, mean=True)

        # If return_latent=True, return dict of latent and x
        if return_latent:
            latent.update({"x": obs})
            return latent

        # Return tensor of reconstructed image
        return obs

    def encode(self, x, mean=False, **kwargs):
        """Encodes latent given observable x"""
        raise NotImplementedError

    def decode(self, latent, mean=False, **kwargs):
        """Decodes observable x given latent"""
        raise NotImplementedError

    def sample(self, batch_n=1, **kwargs):
        """Samples observable x from sampled latent z"""
        raise NotImplementedError

    def loss_func(self, x, **kwargs):
        """Calculates loss given observable x"""
        raise NotImplementedError

    @property
    def loss_str(self):
        """Returns loss string"""
        raise NotImplementedError

    def __str__(self):
        prob_text = []
        func_text = []

        for prob in self.distributions:
            if isinstance(prob, Distribution):
                prob_text.append(prob.prob_text)
            else:
                func_text.append(prob.__str__())

        text = ("Distributions (for training): \n "
                " {} \n".format(", ".join(prob_text)))
        if len(func_text) > 0:
            text += ("Deterministic functions (for training): \n "
                     " {} \n".format(", ".join(func_text)))

        text += "Loss function: \n  {} \n".format(str(self.loss_str))
        return text

    @property
    def second_optim(self):
        """Returns second optimizer for Adversarial loss"""
        return None
