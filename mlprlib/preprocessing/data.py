from typing import Union

import numpy as np
from scipy.stats import norm


def standardize(X: np.ndarray, mean: float = None, std: float = None):
    """
    Standardizes the given dataset X
    and retrieves the Gaussian z score

        z_score = (X - μ) / σ

    Allows specifying a mean and a standard deviation
    in order to apply the z_score on the test set using
    the mean and the variance of the training set

    Parameters
    ----------
    X:
        ndarray, data to normalize.

    mean:
        float, the mean to subtract to center data.
        If None, use mean of input data.

    std:
        float, the standard deviation to divide centered
        data from.
        If None, use std dev of input data.

    Returns
    -------
        Gaussian Z score of the input data
    """
    if mean is None:
        mean = X.mean(axis=0)
    if std is None:
        std = X.std(axis=0)
    return (X - mean) / std


def normalize(X: np.ndarray,
              axis: int = -1,
              order: Union[str, int] = 2):
    """
    Normalizes the given dataset X
    Args:
        X:
            the dataset to normalize
        axis:
            the axis along with to compute the norm
        order:
            the order of the norm, must respect
            numpy.linalg.norm `ord` attribute
            specifications.
            Default 2 (norm l2)

    Returns:
        the normalized dataset using
        the l_<order> norm
    """
    norm = np.linalg.norm(X, order, axis)
    l_norm = np.atleast_1d(norm)
    l_norm[l_norm == 0] = 1
    return X / np.expand_dims(l_norm, axis)


def cumulative_feature_rank(X: np.ndarray, X_ref: np.ndarray = None):
    """
    Transforms features, computing the cumulative
    inverse of the rank function
        y = phi^-1 { r(x) }

    being
        r(x) = (sum^{N}_{i=0} I[x < xi] + 1)/(N+2)

    the rank of a feature x over the training set

    where:
    - phi is the inverse of the cumulative distribution
      function  (percent point function, p.p.f) of the standard
      normal distribution
    - xi: the value of the considered feature
       for the i-th sample
    - N: the number of samples
    - I: the samples matrix

    Args:
        X:
            np.ndarray of shape (n_feats, n_samples)

        X_ref:
            np.ndarray of shape
                (n_feats, n_samples)
            containing the set used as a reference.
            If None, X is used.
            Default None

    Returns:
        the rank wrt to the training set
    """
    n_feats, n_samples = X.shape
    transformed = np.empty([n_feats, n_samples])

    X_ref = X if X_ref is None else X_ref
    for i in range(n_samples):
        # compute rank applying the definition
        # 1. is a trick to cast it to a float
        # to avoid numerical errors with following division
        rank = 1. + (X[:, i].reshape([n_feats, 1]) < X_ref).sum(axis=1)
        rank /= (n_samples + 2.)
        transformed[:, i] = norm.ppf(rank)
    return transformed
