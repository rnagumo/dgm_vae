
"""Interventional Robustness Score.

Based on the paper https://arxiv.org/abs/1811.00007.

ref)
https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/evaluation/metrics/irs.py
"""

from typing import Callable, Dict, Union

import numpy as np
from torch import Tensor

from ..datasets.base_data import BaseDataset
from .util_funcs import generate_repr_factor_batch, discretize_target


def irs(dataset: BaseDataset,
        repr_fn: Callable[[Tensor], Tensor], 
        batch_size: int = 16,
        num_points: int = 10000,
        num_bins: int = 20) -> Dict[str, float]:
    """Interventional Robustness Score.

    Args:
        dataset (BaseDataset): Dataset class.
        repr_fn (callable): Function that takes observation as input and
            outputs a representation.
        batch_size (int, optional): Batch size to sample points.
        num_points (int, optional): Number of samples.
        num_bins (int, optional): Number of bins for discretization.

    Returns:
        scores_dict (dict): Dictionary including metric score.
    """

    # Sample dataset
    mus, ys = generate_repr_factor_batch(
        dataset, repr_fn, batch_size, num_points)

    # Discretize true factors
    ys_discrete = discretize_target(ys, num_bins)

    # Drop constant dim
    active_mus = mus[:, mus.var(0) > 0]

    # Compute IRS score
    if not active_mus.any():
        irs_score = 0.0
    else:
        irs_score = compute_irs_score(ys_discrete, active_mus)["avg_score"]

    scores_dict = {
        "irs": irs_score,
        "num_active_dims": np.sum(mus.var(0) > 0),
    }
    return scores_dict


def compute_irs_score(gen_factors: np.ndarray,
                      latents: np.ndarray,
                      diff_quantile: float = 0.99
                      ) -> Dict[str, Union[float, np.ndarray]]:
    """Computes IRS (Interventional Robustness Score) score given dataset.

    Args:
        gen_factors (np.array): array of shape (num_samples, num_gen_factors)
        latents (np.array): array of shape (num_samples, num_latents)
        diff_quantile (float, optional): quantile of diffs to select (1.0 in
            paper).

    Returns:
        scores_dict (dict): Metrics dict.
    """

    num_gen = gen_factors.shape[1]
    num_lat = latents.shape[1]

    # For each generative factor, compute EMPIDA metric
    cum_deviations = np.zeros([num_lat, num_gen])
    for i in range(num_gen):
        unique_factors = np.unique(gen_factors[:, i], axis=0)
        num_distinct_factors = unique_factors.shape[0]

        for k in range(num_distinct_factors):
            # Compute E[Z | g_i], E[Z | g_i, g_j]
            match = gen_factors[:, i] == unique_factors[k]
            e_loc = np.mean(latents[match, :], axis=0)

            # PIDA: Difference of each value within that group of constant g_i
            # to its mean (d(a, b): L1 norm)
            pida = np.abs(latents[match, :] - e_loc)

            # Maximal PIDA
            max_pida = np.percentile(pida, q=diff_quantile * 100, axis=0)
            cum_deviations[:, i] += max_pida

        # Expected MPIDA
        cum_deviations[:, i] /= num_distinct_factors

    # IRS: Normalize value of each latent dimension with its maximal deviation
    max_deviations = np.max(np.abs(latents - latents.mean(0)), axis=0)
    normalized_deviations = cum_deviations / max_deviations[:, np.newaxis]
    irs_matrix = 1.0 - normalized_deviations

    # Compute numerical metrics
    disentanglement_scores = irs_matrix.max(axis=1)

    if np.sum(max_deviations) > 0.0:
        avg_score = np.average(disentanglement_scores, weights=max_deviations)
    else:
        avg_score = np.mean(disentanglement_scores)

    scores_dict = {
        "disentanglement_scores": disentanglement_scores,
        "avg_score": avg_score,
        "parents": irs_matrix.argmax(axis=1),
        "IRS_matrix": irs_matrix,
        "max_deviations": max_deviations,
    }
    return scores_dict
