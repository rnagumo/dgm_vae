
"""Base VAE class."""

from typing import Union, Dict, Optional

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

        latents = self.encode(x, mean=True)

        if isinstance(latents, tuple):
            return latents[0]
        return latents

    def reconstruct(self, x: Tensor,
                    return_latent: bool = False
                    ) -> Union[Tensor, Dict[str, Tensor]]:
        """Reconstructs images.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            obs (torch.Tensor or dict): Decoded obsercations. If
            `return_latent` is `True`, `obs` is Tensor, otherwise, dict.
        """

        latent = self.encode(x)
        obs = self.decode(latent, mean=True)

        # If `return_latent`=True, return dict of latent and obs
        if return_latent:
            latent.update({"x": obs})
            return latent

        # Return tensor of reconstructed image
        return obs

    def encode(self,
               x: Union[Tensor, Dict[str, Tensor]],
               mean: bool = False,
               **kwargs) -> Union[Tensor, Dict[str, Tensor]]:
        """Encodes latent given observable x.

        Args:
            x (torch.Tensor or dict): Tensor or dict or Tensor for input
                observations.
            mean (bool, optional): Boolean flag for returning means or samples.

        Returns:
            z (torch.Tensor or dict): Tensor of encoded latents. `z` is
            `torch.Tensor` if `mean` is `True`, otherwise, dict.
        """
        raise NotImplementedError

    def decode(self,
               latent: Union[Tensor, Dict[str, Tensor]],
               mean: bool = False,
               **kwargs) -> Union[Tensor, Dict[str, Tensor]]:
        """Decodes observable x given latents.

        Args:
            latent (torch.Tensor or dict): Tensor or dict of latents.
            mean (bool, optional): Boolean flag for returning means or samples.

        Returns:
            x (torch.Tensor or dict): Tensor of decoded observations. `z` is
            `torch.Tensor` if `mean` is `True`, otherwise, dict.
        """
        raise NotImplementedError

    def sample(self, batch_n: int = 1, **kwargs) -> Dict[str, Tensor]:
        """Samples observable x from sampled latent z.

        Args:
            batch_n (int, optional): Batch size.

        Returns:
            sample (dict): Dict of sampled tensors.
        """
        raise NotImplementedError

    def loss_func(self, x: Tensor, **kwargs) -> Dict[str, Tensor]:
        """Calculates loss given observable x.

        Args:
            x (torch.Tensor): Tensor of input observations.

        Returns:
            loss_dict (dict): Dict of calculated losses.
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
