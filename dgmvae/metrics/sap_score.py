
"""Implementation of the SAP (Separated Attribute Predictability) score.

Based on "Variational Inference of Disentangled Latent Concepts from Unlabeled
Observations" (https://openreview.net/forum?id=H1kG7GZAW), Section 3.

ref)
https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/evaluation/metrics/sap_score.py
"""

from typing import Callable, Dict, Union

import numpy as np
from sklearn import svm
from torch import Tensor

from ..datasets.base_data import BaseDataset
from .util_funcs import generate_repr_factor_batch


def sap_score(dataset: BaseDataset,
              repr_fn: Callable[[Tensor], Tensor],
              batch_size=16, num_points=5000,
              continuous=False) -> Dict[str, float]:
    """Computes SAP (Separated Attribute Predictability) score.

    Args:
        dataset (BaseDataset): Dataset class.
        repr_fn (callable): Function that takes observation as input and
            outputs a representation.
        batch_size (int, optional): Batch size to sample points.
        num_points (int, optional): Number of samples.
        continuous (bool, optional): Boolean flag which specifies that latent
            variables are continuous or not.

    Returns:
        scores_dict (dict): Dictionary including metric score.
    """

    # Sample data
    mus, ys = generate_repr_factor_batch(
        dataset, repr_fn, batch_size, num_points)

    # Compute score matrix
    score_matrix = compute_score_matrix(mus, ys, continuous)

    # Compute difference of top two score
    scores_dict = {
        "sap_score": compute_avg_diff_top_two(score_matrix),
    }
    return scores_dict


def compute_score_matrix(mus: Union[np.ndarray, Tensor],
                         ys: Union[np.ndarray, Tensor],
                         continuous: bool) -> np.ndarray:
    """Computes score matrix.

    Args:
        mus (np.ndarray or torch.Tensor): Predicted mean vector, size
            `(num_samples, num_latents)`.
        ys (np.ndarray or torch.Tensor): True factors, size
            `(num_samples, num_factors)`.
        continuous (bool): Boolean flag for specifying factors are continuous
            or discrete.

    Returns:
        score_matrix (np.ndarray): Calculated score matrix, size
        `(num_latents, num_factors)`.
    """

    num_latents = mus.shape[1]
    num_factors = ys.shape[1]
    score_matrix = np.zeros((num_latents, num_factors))
    for i in range(num_latents):
        for j in range(num_factors):
            mu_i = mus[:, i]
            y_j = ys[:, j]
            if continuous:
                # Continuous variable
                cov_mu_i_y_j = np.cov(mu_i, y_j, ddof=1)
                cov_mu_y = cov_mu_i_y_j[0, 1] ** 2
                var_mu = cov_mu_i_y_j[0, 0]
                var_y = cov_mu_i_y_j[1, 1]
                if var_mu > 1e-12:
                    score_matrix[i, j] = cov_mu_y / (var_mu * var_y)
                else:
                    score_matrix[i, j] = 0.
            else:
                # Discrete variable
                classifier = svm.LinearSVC(C=0.01, class_weight="balanced")
                classifier.fit(mu_i[:, np.newaxis], y_j)
                y_j_pred = classifier.predict(mu_i[:, np.newaxis])
                score_matrix[i, j] = np.mean(y_j == y_j_pred)

    return score_matrix


def compute_avg_diff_top_two(matrix: np.ndarray) -> float:
    """Computes difference between top two scores.

    Args:
        matrix (np.ndarray): Score matrix, size `(num_latents, num_factors)`.

    Returns:
        diff_score (float): Calculated difference of top two scores.
    """

    sorted_matrix = np.sort(matrix, axis=0)
    return np.mean(sorted_matrix[-1, :] - sorted_matrix[-2, :])
