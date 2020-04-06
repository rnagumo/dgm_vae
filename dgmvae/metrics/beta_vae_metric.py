
"""Implementation of the disentanglement metric from the BetaVAE paper.

Based on "beta-VAE: Learning Basic Visual Concepts with a Constrained
Variational Framework" (https://openreview.net/forum?id=Sy2fzU9gl).
"""

import numpy as np
from sklearn import linear_model as lm


def beta_vae_metric(dataset, repr_fn, random_state=0, batch_size=64,
                    num_train=10000, num_eval=5000):
    """Compute beta-VAE metric with sklearn."""

    # Sample data
    x_train, y_train = _generate_batch(dataset, repr_fn, batch_size, num_train)
    x_eval, y_eval = _generate_batch(dataset, repr_fn, batch_size, num_eval)

    # Train model
    model = lm.LogisticRegression(random_state=random_state)
    model.fit(x_train, y_train)

    # Evaluate
    score_dict = {
        "train_accuracy": model.score(x_train, y_train),
        "eval_accuracy": model.score(x_eval, y_eval),
    }
    return score_dict


def _generate_batch(dataset, repr_fn, batch_size, num_points):
    """Generates batch sample of true factors and encoded representations."""

    features = []
    labels = []
    for i in range(num_points):
        _ftr, _idx = _generate_sample(dataset, repr_fn, batch_size)
        features.append(_ftr)
        labels.append(_idx)
    return np.stack(features), np.stack(labels).ravel()


def _generate_sample(dataset, repr_fn, batch_size):
    """Samples and encodes observations"""

    # Select random coordinate to keep fixed
    factor_index = dataset.sample_factor_index()

    # Sample data with fixed factor
    data1, data2, targets1, targets2 = dataset.sample_paired_batch(
        batch_size, factor_index)

    # Represent latents
    repr1 = repr_fn(data1)
    repr2 = repr_fn(data2)

    # Compute feature vector based on differences in representation
    feature_vector = (repr1 - repr2).abs().mean(0)
    return feature_vector, factor_index
