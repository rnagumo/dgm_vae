
"""Mutual Information Gap from the beta-TC-VAE paper.

Based on "Isolating Sources of Disentanglement in Variational Autoencoders"
(https://arxiv.org/pdf/1802.04942.pdf).

ref)
https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/evaluation/metrics/mig.py
"""

import numpy as np
from .util_funcs import (generate_repr_factor_batch, discretize_target,
                         discrete_mutual_info, discrete_entropy)


def mig(dataset, repr_fn, batch_size=16, num_points=10000, num_bins=20):
    """Computes Mutual Information Gap.

    Args:
        dataset (BaseDataset): Dataset class.
        repr_fn: Function that takes observation as input and outputs a
            representation.
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
    mus_discrete = discretize_target(mus, num_bins)

    # Discrete mutual information and entropy
    mi = discrete_mutual_info(mus_discrete, ys)
    sorted_mi = np.sort(mi, axis=0)[::-1]
    entropy = discrete_entropy(ys)

    # Compute MI gap of top 2 rows
    scores_dict = {
        "discrete_mig":
            np.mean(np.divide(sorted_mi[0] - sorted_mi[1], entropy)),
    }
    return scores_dict
