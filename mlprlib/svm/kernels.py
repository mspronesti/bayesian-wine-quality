import numpy as np


def linear_kernel(xi, xj):
    """
    Computes the linear kernel between xi and xj
            K(xi, xj) = xi^T xj

    Parameters
    ----------
    xi:
        ndarray of shape (n_feats, n_samples)
    xj:
        ndarray of shape (n_feats, n_samples)

    Returns
    -------
        kernel matrix of shape (xi.shape[0], xj.shape[0])
    """
    return xi.T @ xj


def polynomial_kernel(xi, xj, gamma=None, d=3, c=0, csi=0):
    """
    Computes the Polynomial kernel between xi and xj
        K(xi, xj) = (<xi, xj> + c)^d + csi

    Parameters
    ----------
    xi:
        ndarray of shape (n_feats, n_samples)
    xj:
        ndarray of shape (n_feats, n_samples)
    gamma:
        float, default=None
        If None, defaults to 1.0 / n_samples.
    d:
        degree, default 0.
    c:
        float, default 0
    csi:
        shift term, default 0

    Returns
    -------
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

    Parameters
    ----------
    xi:
        ndarray of shape (n_feats, n_samples)
    xj:
        ndarray of shape (n_feats, n_samples)
    gamma:
        float, default=None
        If None, defaults to 1.0 / n_samples.
    csi:
        shift term, default 0

    Returns
    -------
         Kernel matrix of shape (xi.shape[1], xj.shape[1])
    """
    if gamma is None:
        gamma = 1. / xi.shape[1]

    kern = np.zeros([xi.shape[1], xj.shape[1]])
    for i in range(xi.shape[1]):
        for j in range(xj.shape[1]):
            norm = ((xi[:, i] - xj[:, j])**2).sum()
            kern[i, j] = np.exp(-gamma * norm) + csi
    return kern