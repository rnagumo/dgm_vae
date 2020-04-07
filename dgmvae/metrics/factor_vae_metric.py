
"""Implementation of the disentanglement metric from the FactorVAE paper.

Based on "Disentangling by Factorising" (https://arxiv.org/abs/1802.05983).

ref)
https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/evaluation/metrics/factor_vae.py
"""

import numpy as np


def factor_vae_metric(dataset, repr_fn, random_state=0, batch_size=64,
                      num_train=10000, num_eval=1000, num_var=10000):
    """Computes factor-VAE metric."""

    # Compute global variance
    global_var = _compute_variance(dataset, repr_fn, num_var)
    if all(global_var == 0):
        scores = {
            "train_accuracy": 0.0,
            "eval_accuracy": 0.0,
            "num_active_dims": 0,
        }
        return scores

    # Sample data
    votes_train = _generate_batch(
        dataset, repr_fn, batch_size, num_train, global_var)
    votes_eval = _generate_batch(
        dataset, repr_fn, batch_size, num_eval, global_var)

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


def _compute_variance(dataset, repr_fn, num_var):
    data, _ = dataset.sample_batch(num_var)
    reprs = repr_fn(data)
    return reprs.var(0)


def _generate_batch(dataset, repr_fn, batch_size, num_points, global_var):

    votes = np.zeros(dataset.num_factors, global_var.shape[0])
    for _ in range(num_points):
        factor_index, argmin = _generate_sample(
            dataset, repr_fn, batch_size, global_var)
        votes[factor_index, argmin]

    return votes


def _generate_sample(dataset, repr_fn, batch_size, global_var):

    # Select random coordinate to keep fixed
    factor_index = dataset.sample_factor_index()

    # Sample data with fixed factor
    data, factor = dataset.sample_fixed_batch(batch_size, factor_index)

    # Represent latents
    reprs = repr_fn(data)

    # Compute local variance
    local_var = reprs.var(0)
    active_dims = global_var > 0
    argmin = np.argmin(local_var[active_dims] / global_var[active_dims])

    return factor_index, argmin
