
"""Utils."""

import numpy as np


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
