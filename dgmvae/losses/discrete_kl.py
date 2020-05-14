
"""Discrete KL divergence

KL loss for Categorical and RelaxedCategorical

ref) KL divergence in PyTorch
https://pytorch.org/docs/stable/_modules/torch/distributions/kl.html#kl_divergence
"""

from typing import Optional, List, Dict, Tuple

import sympy

import torch
from torch._six import inf

import pixyz
from pixyz.losses.losses import Loss
from pixyz.utils import get_dict_values


def _kl_categorical_categorical(p: torch.distributions.Distribution,
                                q: torch.distributions.Distribution
                                ) -> torch.Tensor:
    """KL divergence between categorical and categorical, KL(p||q).

    Args:
        p (torch.distributions.Distribution): Distribution.
        q (torch.distributions.Distribution): Distribution.

    Returns:
        t (torch.Tensor): Calculated KL divergence.
    """
    t = p.probs * (p.logits - q.logits)
    t[(q.probs == 0).expand_as(t)] = inf
    t[(p.probs == 0).expand_as(t)] = 0
    return t.sum(-1)


class CategoricalKullbackLeibler(Loss):
    """Kullback Leibler divergence for categorical distributions.

    Args:
        p (pixyz.distributions.Distribution): Distribution.
        q (pixyz.distributions.Distribution): Distribution.
        input_var (list, optional): Input variable name.
        dim (int, optional): Aggregate dimension.
    """
    def __init__(self,
                 p: pixyz.distributions.Distribution,
                 q: pixyz.distributions.Distribution,
                 input_var: Optional[List[str]] = None,
                 dim: Optional[int] = None):
        self.dim = dim
        super().__init__(p, q, input_var)

    @property
    def _symbol(self):
        return sympy.Symbol("D_{{KL}} \\left[{}||{} \\right]".format(
            self.p.prob_text, self.q.prob_text))

    def _get_eval(self,
                  x_dict: Dict[str, torch.Tensor],
                  **kwargs) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        if (not hasattr(self.p, 'distribution_torch_class')) \
                or (not hasattr(self.q, 'distribution_torch_class')):
            raise ValueError("Divergence between these two distributions "
                             "cannot be evaluated, got %s and %s."
                             % (self.p.distribution_name,
                                self.q.distribution_name))

        input_dict = get_dict_values(x_dict, self.p.input_var, True)
        self.p.set_dist(input_dict)

        input_dict = get_dict_values(x_dict, self.q.input_var, True)
        self.q.set_dist(input_dict)

        divergence = _kl_categorical_categorical(self.p.dist, self.q.dist)

        if self.dim is not None:
            divergence = torch.sum(divergence, dim=self.dim)
            return divergence, x_dict

        dim_list = list(torch.arange(divergence.dim()))
        divergence = torch.sum(divergence, dim=dim_list[1:])
        return divergence, x_dict
