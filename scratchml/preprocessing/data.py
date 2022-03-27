import numpy as np


def standardize(X: np.ndarray):
    """Standardizes the given dataset X"""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    return (X - mean) / std


def normalize(X: np.ndarray,
              axis: int = -1,
              order: int = 2):
    """
    Normalizes the given dataset X
    Args:
        X:
            the dataset to normalize
        axis:
            the axis along with to compute the norm
        order:
            the order of the norm

    Returns:
        the normalized dataset using
        the l_<order> norm
    """
    norm = np.linalg.norm(X, order, axis)
    l_norm = np.atleast_1d(norm)
    l_norm[l_norm == 0] = 1
    return X / np.expand_dims(l_norm, axis)
