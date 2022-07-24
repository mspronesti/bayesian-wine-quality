import numpy as np
from scipy.spatial.distance import cdist


def linear_kernel(xi, xj):
    """
    Computes the linear kernel between xi and xj
    Args:
        xi: ndarray of shape (n_samples, n_feats)
        xj: ndarray of shape (n_samples, n_feats)

    Returns:
        kernel matrix of shape (xi.shape[0], xj.shape[0])
    """
    return xi.T @ xj


def polynomial_kernel(xi, xj, gamma=None, d=0, c=0, csi=0):
    """
    Computes the Polynomial kernel between xi and xj
    K(xi, xj) = (<xi, xj> + c)^d + csi

    Args:
        xi: ndarray of shape (n_samples, n_feats)
        xj: ndarray of shape (n_samples, n_feats)
        gamma: float, default=None
              If None, defaults to 1.0 / n_features.
        d:  degree, default 0
        c:  float, default 0
        csi: shift term, default 0

    Returns:
        kernel matrix of shape (xi.shape[0], xj.shape[0])
    """
    k = xi.T @ xj
    if gamma is None:
        gamma = 1. / xi.shape[1]

    return (gamma * k + c) ** d + csi


def rbf_kernel(xi, xj, gamma=None, csi=0):
    """
    Compute the Gaussian kernel between xi and xj

    K(xi, xj) = e^(-gamma ||xi-xj||^2)
    Args:
        xi: ndarray of shape (n_samples, n_feats)
        xj: ndarray of shape (n_samples, n_feats)
        gamma: float, default=None
            If None, defaults to 1.0 / n_features.
        csi: shift term, default 0

    Returns:
        kernel matrix of shape (xi.shape[0], xj.shape[0])
    """
    if gamma is None:
        gamma = 1. / xi.shape[1]

    dist = cdist(xi, xj) ** 2
    return np.exp(-gamma * dist) + csi
