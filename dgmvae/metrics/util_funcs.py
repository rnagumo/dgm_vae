
"""Utils."""

import numpy as np
import sklearn


def generate_repr_factor_batch(dataset, repr_fn, batch_size, num_points):

    reprs = []
    factors = []
    for i in range(num_points // batch_size + 1):
        # Calculate batch size
        batch_iter = min(num_points - batch_size * i, batch_size)

        # Sample fixed factor and observations
        factor_index = dataset.sample_factor_index()
        data, targets = dataset.sample_fixed_batch(batch_iter, factor_index)

        # Representation
        rep = repr_fn(data)

        # Add repr and target to list
        reprs.append(rep)
        factors.append(targets)

    return np.vstack(reprs), np.vstack(factors)


def discretize_target(target, num_bins):
    discretized = np.zeros_like(target)
    for i in range(target.shape[0]):
        discretized[i] = np.digitize(
            target[i], np.histogram(target[i], num_bins)[1][:-1])

    return discretized


def discrete_mutual_info(mus, ys):
    """Discrete Mutual Information for all code-factor pairs.

    Args:
        mus: array, (num_samples, num_codes)
        ys: array, (num_samples, num_factors)
    """
    num_codes = mus.shape[1]
    num_factors = ys.shape[1]

    mi = np.zeros((num_codes, num_factors))
    for i in range(num_codes):
        for j in range(num_factors):
            mi[i, j] = sklearn.metrics.mutual_info_score(ys[:, j], mus[:, i])
    return mi


def discrete_entropy(ys):
    """Discrete Mutual Information for all code-factor pairs.

    Args:
        ys: array, (num_samples, num_factors)
    """
    num_factors = ys.shape[1]
    h = np.zeros(num_factors)
    for i in range(num_factors):
        h[i] = sklearn.metrics.mutual_info_score(ys[:, i], ys[:, i])
    return h
