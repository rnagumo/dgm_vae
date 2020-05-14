
"""Implementation of Disentanglement, Completeness and Informativeness.

Based on "A Framework for the Quantitative Evaluation of Disentangled
Representations" (https://openreview.net/forum?id=By-7dz-AZ).
"""

from typing import Callable, Dict, Tuple, Union

import numpy as np
import scipy
from sklearn import ensemble
from torch import Tensor

from ..datasets.base_data import BaseDataset
from .util_funcs import generate_repr_factor_batch


def dci(dataset: BaseDataset,
        repr_fn: Callable[[Tensor], Tensor],
        batch_size: int = 16,
        num_train: int = 10000,
        num_test: int = 5000) -> Dict[str, float]:
    """Computes Disentanglement, Completeness and Informativeness.

    Args:
        dataset (BaseDataset): Dataset class.
        repr_fn (callable): Function that takes observation as input and
            outputs a representation.
        batch_size (int, optional): Batch size to sample points.
        num_train (int, optional): Number of training data.
        num_test (int, optional): Number of validation data.

    Returns:
        scores_dict (dict): Dictionary including metric score.
    """

    # Sample data
    mus_train, ys_train = generate_repr_factor_batch(
        dataset, repr_fn, batch_size, num_train)
    mus_test, ys_test = generate_repr_factor_batch(
        dataset, repr_fn, batch_size, num_test)

    # Compute importance matrix
    importance_matrix, train_err, test_err = compute_importance_gbt(
        mus_train, ys_train, mus_test, ys_test)

    # Results
    scores_dict = {
        "informativeness_train": train_err,
        "informativeness_test": test_err,
        "disentanglement": disentanglement(importance_matrix),
        "completeness": completeness(importance_matrix),
    }
    return scores_dict


def compute_importance_gbt(x_train: Union[np.ndarray, Tensor],
                           y_train: Union[np.ndarray, Tensor],
                           x_test: Union[np.ndarray, Tensor],
                           y_test: Union[np.ndarray, Tensor]
                           ) -> Tuple[np.ndarray, float, float]:
    """Computes importance matrix with sklear gbt classifier.

    Args:
        x_train (array): Array or Tensor, size `(num_train, num_codes)`.
        y_train (array): Array or Tensor, size `(num_train, num_factors)`.
        x_test (array): Array or Tensor, size `(num_test, num_codes)`.
        y_test (array): Array or Tensor, size `(num_test, num_factors)`.

    Returns:
        importance_matrix (np.ndarray): Importance matrix,
            `(num_codes, num_factors)`.
        train_loss (float): train Informativeness.
        test_loss (float): test Informativeness.
    """

    num_codes = x_train.shape[1]
    num_factors = y_train.shape[1]
    importance_matrix = np.zeros((num_codes, num_factors))

    train_loss = []
    test_loss = []
    for i in range(num_factors):
        model = ensemble.GradientBoostingClassifier()
        model.fit(x_train, y_train[:, i])
        importance_matrix[:, i] = np.abs(model.feature_importances_)
        train_loss.append(np.mean(model.predict(x_train) == y_train[:, i]))
        test_loss.append(np.mean(model.predict(x_test) == y_test[:, i]))

    return importance_matrix, np.mean(train_loss), np.mean(test_loss)


def disentanglement(importance_matrix: np.ndarray) -> float:
    """Computes disentanglement.

    Args:
        importance_matrix (np.ndarray): Importance matrix.

    Returns:
        disentanglemtn_score (float): Calculated metric.
    """

    # Compute score for each code
    per_code = 1 - scipy.stats.entropy(importance_matrix.T + 1e-11,
                                       base=importance_matrix.shape[1])

    if importance_matrix.sum() == 0:
        importance_matrix = np.ones_like(importance_matrix)
    code_importance = importance_matrix.sum(axis=1) / importance_matrix.sum()

    return np.sum(per_code * code_importance)


def completeness(importance_matrix: np.ndarray) -> float:
    """Computes completeness.

    Args:
        importance_matrix (np.ndarray): Importance matrix.

    Returns:
        completeness_score (float): Calculated metric.
    """

    # Compute score for each factor
    per_factor = 1 - scipy.stats.entropy(importance_matrix + 1e-11,
                                         base=importance_matrix.shape[0])

    if importance_matrix.sum() == 0:
        importance_matrix = np.ones_like(importance_matrix)
    factor_importance = importance_matrix.sum(axis=0) / importance_matrix.sum()

    return np.sum(per_factor * factor_importance)
