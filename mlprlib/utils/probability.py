import numpy as np
from typing import Union, List


def mean_squared_error(
        y_true: Union[np.ndarray, List], y_pred: Union[np.ndarray, List]
):
    """
    Evaluates the MSE given the ground truth and the
    predicted labels

    Parameters
    ----------
    y_true: ground truth labels
    y_pred: predicted labels

    Returns
    -------
        the mean squared error, i.e. the arithmetic
        mean of the square of y_true - y_pred
    """
    # to allow list types usage
    if isinstance(y_true, list):
        y_true = np.asarray(y_true)

    if isinstance(y_pred, list):
        y_pred = np.asarray(y_pred)

    return np.mean((y_true - y_pred) ** 2)


def covariance_matrix(X: np.ndarray):
    """
    Computes the covariance matrix of the
    data matrix X

    Parameters
    ----------
    X: ndarray, data matrix

    Returns
    -------
        ndarray, covariance matrix
    """
    n_samples, _ = X.shape
    # subtract from X its mean
    dc = X - X.mean(axis=0)
    # eventually retrieve 1 / n_samples * dc^T dc
    return (dc.T @ dc) / (n_samples - 1)


def within_class_covariance(X, y):
    """
    Computes the within class covariance of data.

    Parameters
    ----------
    X : np.ndarray
    y : np.ndarray

    Returns
    -------
        within class covariance, ndarray
    """
    n_samples, n_feats = X.shape
    sw = np.zeros((n_samples, n_samples))

    for i in range(len(y)):
        selected = X[:, y == i]
        sw += np.cov(selected, bias=True) * float(selected.shape[1])
    return sw / float(n_feats)


def between_class_covariance(X, y):
    """
    Computes the between class covariance of data.

    Parameters
    ----------
    X : np.ndarray
    y : np.ndarray

    Returns
    -------
        between class covariance, ndarray
    """
    n_samples, n_feats = X.shape
    sb = np.zeros((n_samples, n_samples))

    mu = np.row_stack(X.mean(axis=1))
    for i in range(len(y)):
        selected = X[:, y == i]
        muc = np.row_stack(selected.mean(axis=1))
        muc -= mu
        sb += float(selected.shape[1]) * np.dot(muc, muc.T)
    return sb / float(n_feats)


def normal_pdf(X: np.ndarray, mu: float = 0., sigma: float = 1.):
    """
    Computes the Gaussian probability density function, defined as

       N(x ; μ, σ) = 1 / sqrt(2π σ^2) exp [ - .5 (x - μ)^2 / σ^2)

    Parameters
    ----------
    X: input parameters
    mu: mean of the distribution, default 0
    sigma: variance of the distribution, default 1

    Returns
    -------
        ndarray probability of each sample
    """
    k = 1 / np.sqrt(2 * np.pi * sigma)
    up = .5 * (X - mu) ** 2 / sigma
    return k * np.exp(up)


def normal_logpdf(X: np.ndarray, mu: float = 0., sigma: float = 1.):
    """
    Computes the log Gaussian probability density function, i.e.

        log N(x ; μ, σ) = - .5 [ log(2π) - log(σ^2) - (X - μ)^2 / σ^2]

    Parameters
    ----------
    X: input parameters
    mu: mean of the distribution, default 0
    sigma: variance of the distribution, default 1

    Returns
    -------
        log probability of each sample
    """
    return -.5 * (np.log(2 * np.pi) - np.log(sigma) - (X - mu) ** 2 / sigma)


def multivariate_normal_logpdf(X: np.ndarray, mu: np.ndarray, cov: np.ndarray):
    """
    Computes the log probability density function of a multivariate
    gaussian distribution, defined as

        log N(X; μ,Σ) = - .5 [ M log(2π) - log|Σ| - (X - μ)^T Σ^-1 (X - μ) ]

    being
    - M the number of samples
    - |.| the determinant of matrix "."
    - .^-1 the inverse matrix of "."

    Parameters
    ----------
    X: input matrix
    mu: mean vector
    cov: covariance matrix

    Returns
    -------
        Multivariate normal log density function
    """
    M = X.shape[0]
    # log det sigma contains, computed in a robust
    # and efficient way, the logarithm of the determinant
    # of the covariance matrix, i.e. log(| cov |)
    _, log_det_sigma = np.linalg.slogdet(cov)
    cov_inv = np.linalg.inv(cov)

    # quadratic term
    quad_term = (X - mu).T @ cov_inv @ (X - mu)

    if X.shape[1] == 1:
        logN = - .5 * (M * np.log(2 * np.pi) - log_det_sigma - quad_term)
    else:
        logN = - .5 * (M * np.log(2 * np.pi) - log_det_sigma - np.diagonal(quad_term))
    return logN
