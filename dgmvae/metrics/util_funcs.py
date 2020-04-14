
"""Utils."""

import numpy as np
import sklearn


def generate_repr_factor_batch(dataset, repr_fn, batch_size, num_points):
    """Computes Mutual Information Gap.

    Args:
        dataset (BaseDataset): Dataset class.
        repr_fn: Function that takes observation as input and outputs a
            representation.
        batch_size (int, optional): Batch size to sample points.
        num_points (int, optional): Number of samples.

    Returns:
        reprs (np.array): Represented latents (num_points, num_latents)
        factors (np.array): True factors (num_points, num_factors)
    """

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
    """Discretizes targets.

    Args:
        target (array like): Targets of shape (num_points, num_latents).
        num_bins (int): Number of bins.

    Returns:
        discretized (np.array): Discretized targets of shape
            (num_points, num_latents).
    """

    discretized = np.zeros_like(target)
    for i in range(target.shape[0]):
        discretized[i] = np.digitize(
            target[i], np.histogram(target[i], num_bins)[1][:-1])

    return discretized


def discrete_mutual_info(mus, ys):
    """Discrete Mutual Information for all code-factor pairs.

    Args:
        mus (array like): Mean representation vector of shape
            (num_samples, num_codes).
        ys (array like): True factor vector of shape
            (num_samples, num_factors).

    Returns:
        mi (np.array): MI matrix of shape (num_codes, num_factors).
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
        ys (array like): Vector of shape (num_samples, num_factors).

    Returns:
        h (np.array): Entropy vector of shape (num_factors).
    """

    num_factors = ys.shape[1]
    h = np.zeros(num_factors)
    for i in range(num_factors):
        h[i] = sklearn.metrics.mutual_info_score(ys[:, i], ys[:, i])
    return h
