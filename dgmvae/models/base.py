
"""Base VAE class."""

from typing import Dict, Optional

from torch import nn, optim, Tensor
from pixyz.distributions.distributions import Distribution


class BaseVAE(nn.Module):
    """Base VAE class.

    Attributes:
        distributions (list): List of distributions.
    """
    def __init__(self):
        super().__init__()

        self.distributions = []

    def forward(self, x: Tensor) -> Tensor:
        """Encodes without sampling.

        Args:
            x (torch.Tensor): Tensor of inputs.

        Returns:
            latents (torch.Tensor): Tensor of encoded latents.
        """

        latents = self.encode({"x": x}, mean=True)
        return latents["z"]

    def reconstruct(self, x_dict: Dict[str, Tensor], mean: bool = True
                    ) -> Dict[str, Tensor]:
        """Reconstructs images.

        Args:
            x_dict (dict of [str, torch.Tensor]): Input tensor.
            mean (bool, optional): Boolean flag for returning means or samples.

        Returns:
            obs (dict of [str, torch.Tensor]): Encoded latents and decoded
                observations.
        """

        latents = self.encode(x_dict)
        obs = self.decode(latents, mean=mean)
        obs.update(latents)
        return obs

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
        raise NotImplementedError

    def decode(self, z_dict: Dict[str, Tensor], mean: bool = False, **kwargs
               ) -> Dict[str, Tensor]:
        """Decodes observable x given latents.

        Args:
            z_dict (dict of [str, torch.Tensor]): Dict of latents tensors.
            mean (bool, optional): Boolean flag for returning means or samples.

        Returns:
            x (dict of [str, torch.Tensor]): Tensor of decoded observations.
        """
        raise NotImplementedError

    def sample(self, batch_n: int) -> Dict[str, Tensor]:
        """Samples observable x from sampled latent z.

        Args:
            batch_n (int): Batch size.

        Returns:
            samples (dict of [str, torch.Tensor]): Dict of sampled tensor.
        """
        raise NotImplementedError

    def loss_func(self, x: Tensor, **kwargs) -> Dict[str, Tensor]:
        """Calculates loss given observable x.

        Args:
            x (torch.Tensor): Tensor of input observations.

        Returns:
            loss_dict (dict of [str, torch.Tensor]): Dict of calculated losses.
        """
        raise NotImplementedError

    @property
    def loss_str(self) -> str:
        """Returns loss string."""
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
    def second_optim(self) -> Optional[optim.Optimizer]:
        """Returns second optimizer for Adversarial loss."""
        return None
