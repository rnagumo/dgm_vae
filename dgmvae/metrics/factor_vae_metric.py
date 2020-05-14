
"""Implementation of the disentanglement metric from the FactorVAE paper.

Based on "Disentangling by Factorising" (https://arxiv.org/abs/1802.05983).

ref)
https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/evaluation/metrics/factor_vae.py
"""

from typing import Callable, Union, Dict

import numpy as np
from torch import Tensor

from ..datasets.base_data import BaseDataset


def factor_vae_metric(dataset: BaseDataset,
                      repr_fn: Callable[[Tensor], Tensor],
                      batch_size: int = 64,
                      num_train: int = 10000,
                      num_eval: int = 5000,
                      num_var: int = 10000,
                      th_var: float = 0.05) -> Dict[str, Union[int, float]]:
    """Computes factor-VAE metric.

    Args:
        dataset (BaseDataset): Dataset class.
        repr_fn (callable): Function that takes observation as input and
            outputs a representation.
        batch_size (int, optional): Batch size to sample points.
        num_train (int, optional): Number of training data.
        num_eval (int, optional): Number of validation data.
        num_var (int, optional): Number of data for computing global variance.
        th_var (float, optional): Threshold for variance.

    Returns:
        scores_dict (dict): Dictionary including metric score.
    """

    # Compute global variance
    global_var = _compute_variance(dataset, repr_fn, num_var)
    active_dims = (global_var ** 2) > th_var
    if all(~active_dims):
        scores = {
            "train_accuracy": 0.0,
            "eval_accuracy": 0.0,
            "num_active_dims": 0,
        }
        return scores

    # Sample data
    votes_train = _generate_batch(
        dataset, repr_fn, batch_size, num_train, global_var, active_dims)
    votes_eval = _generate_batch(
        dataset, repr_fn, batch_size, num_eval, global_var, active_dims)

    # Majority-vote classifier
    classifier = np.argmax(votes_train, axis=0)
    other_index = np.arange(votes_train.shape[1])

    # Evaluate
    scores_dict = {
        "train_accuracy": (np.sum(votes_train[classifier, other_index])
                           / np.sum(votes_train)),
        "eval_accuracy": (np.sum(votes_eval[classifier, other_index])
                          / np.sum(votes_eval)),
        "num_active_dims": sum(global_var > 0),
    }
    return scores_dict


def _compute_variance(dataset: BaseDataset,
                      repr_fn: Callable[[Tensor], Tensor],
                      num_var: int) -> np.ndarray:
    """Computes variances for each dimension of the representation.

    Args:
        dataset (BaseDataset): Dataset class.
        repr_fn (callable): Function that takes observation as input and
            outputs a representation.
        num_var (int): Number of variables.

    Returns:
        votes (np.ndarray): Array of majority votes.
    """

    data, _ = dataset.sample_batch(num_var)
    reprs = repr_fn(data)
    return reprs.var(0)


def _generate_batch(dataset: BaseDataset,
                    repr_fn: Callable[[Tensor], Tensor],
                    batch_size: int,
                    num_points: int,
                    global_var: np.ndarray,
                    active_dims: np.ndarray) -> np.ndarray:
    """Generates batch sample.

    Args:
        dataset (BaseDataset): Dataset class.
        repr_fn (callable): Function that takes observation as input and
            outputs a representation.
        batch_size (int): Batch size for calculation.
        num_points (int): Number of data points.
        global_var (np.ndarray): Array of global variance.
        active_dims (np.ndarray): Array of active dimensions.
    """

    votes = np.zeros((dataset.num_factors, global_var.shape[0]))
    for _ in range(num_points):
        # Select random coordinate to keep fixed
        factor_index = dataset.sample_factor_index()

        # Sample data with fixed factor
        data, factor = dataset.sample_fixed_batch(batch_size, factor_index)

        # Represent latents
        reprs = repr_fn(data)

        # Compute local variance
        local_var = reprs.var(0)
        argmin = np.argmin(local_var[active_dims] / global_var[active_dims])

        # Count (factor index, minimum variance latents) pairs
        votes[factor_index, argmin] += 1

    return votes
