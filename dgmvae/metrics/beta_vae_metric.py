
"""Implementation of the disentanglement metric from the BetaVAE paper.

Based on "beta-VAE: Learning Basic Visual Concepts with a Constrained
Variational Framework" (https://openreview.net/forum?id=Sy2fzU9gl).
"""

from typing import Callable, Dict, Tuple

import numpy as np
from sklearn import linear_model as lm
from torch import Tensor

from ..datasets.base_data import BaseDataset


def beta_vae_metric(dataset: BaseDataset,
                    repr_fn: Callable[[Tensor], Tensor],
                    random_state: int = 0,
                    batch_size: int = 64,
                    num_train: int = 10000,
                    num_eval: int = 5000) -> Dict[str, float]:
    """Computes beta-VAE metric with sklearn.

    Args:
        dataset (BaseDataset): Dataset class.
        repr_fn (callable): Function that takes observation as input and
            outputs a representation.
        random_state (int, optional): Random seed for LogisticRegression model.
        batch_size (int, optional): Batch size to sample points.
        num_train (int, optional): Number of training data.
        num_eval (int, optional): Number of validation data.

    Returns:
        scores_dict (dict): Dictionary including metric score.
    """

    # Sample data
    x_train, y_train = _generate_batch(dataset, repr_fn, batch_size, num_train)
    x_eval, y_eval = _generate_batch(dataset, repr_fn, batch_size, num_eval)

    # Train model
    model = lm.LogisticRegression(random_state=random_state)
    model.fit(x_train, y_train)

    # Evaluate
    scores_dict = {
        "train_accuracy": model.score(x_train, y_train),
        "eval_accuracy": model.score(x_eval, y_eval),
    }
    return scores_dict


def _generate_batch(dataset: BaseDataset,
                    repr_fn: Callable[[Tensor], Tensor],
                    batch_size: int,
                    num_points: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generates batch sample of true factors and encoded representations.

    Args:
        dataset (BaseDataset): Dataset class.
        repr_fn (callable): Function that takes observation as input and
            outputs a representation.
        batch_size (int): Batch size to sample points.
        num_points (int): Number of data.

    Returns:
        features (np.ndarray): Represented features.
        labels (np.ndarray): Sampled factor indices.
    """

    features = []
    labels = []
    for i in range(num_points):
        _ftr, _idx = _generate_sample(dataset, repr_fn, batch_size)
        features.append(_ftr)
        labels.append(_idx)
    return np.stack(features), np.stack(labels).ravel()


def _generate_sample(dataset: BaseDataset,
                     repr_fn: Callable[[Tensor], Tensor],
                     batch_size: int) -> Tuple[Tensor, Tensor]:
    """Samples and encodes observations.

    Args:
        dataset (BaseDataset): Dataset class.
        repr_fn (callable): Function that takes observation as input and
            outputs a representation.
        batch_size (int): Batch size to sample points.

    Returns:
        feature_vector (torch.Tensor): Represented features.
        factor_index (torch.Tensor): Sampled factor indices.
    """

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
