import numpy as np
from typing import Union, List


def mean_squared_error(y_true: Union[np.ndarray, List],
                       y_pred: Union[np.ndarray, List]):
    """
    Evaluates the MSE given ground truth and predictions

    Args:
        y_true: ground truth labels
        y_pred: predicted labels

    Returns:
        the mean squared error, i.e. the arithmetic
        mean of the square of y_true - y_pred
    """
    # to allow list types usage
    if isinstance(y_true, list):
        y_true = np.asarray(y_true)

    if isinstance(y_pred, list):
        y_pred = np.asarray(y_pred)

    return np.mean((y_true - y_pred)**2)


def covariance_matrix(X: np.ndarray):
    """
    Computes the covariance matrix of X
    Args:
        X: array-like

    Returns:
        the covariance matrix
    """
    n_samples, _ = X.shape
    # subtract from X its mean
    dc = X - X.mean(axis=0)
    # eventually retrieve 1 / n_samples * dc^T dc
    return (dc.T @ dc) / (n_samples - 1)




