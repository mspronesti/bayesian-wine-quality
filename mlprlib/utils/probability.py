import numpy as np
from typing import Union, List


def mean_squared_error(
    y_true: Union[np.ndarray, List], y_pred: Union[np.ndarray, List]
):
    """
    Evaluates the MSE given the ground truth and the
    predicted labels

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

    return np.mean((y_true - y_pred) ** 2)


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


def normal_pdf(x: np.ndarray, mu: float = 0., sigma: float = 1.):
    """
    Gaussian probability density function
    Args:
        x: input parameters
        mu: mean of the distribution (default 0)
        sigma: variance of the distribution (default 1)

    Returns:
        ndarray probability of each sample
    """
    k = 1 / np.sqrt(2 * np.pi * sigma)
    up = .5 * (x - mu) ** 2 / sigma
    return k * np.exp(up)


def normal_logpdf(x: np.ndarray, mu: float = 0., sigma: float = 1.):
    """
    Log Gaussian probability density function.

    Args:
        x: input parameters
        mu: mean of the distribution (default 0)
        sigma: variance of the distribution (default 1)

    Returns:
        log probability of each sample
    """
    return -.5 * (np.log(2 * np.pi) - np.log(sigma) - (x - mu)**2 / sigma)


def multivariate_normal_logpdf(x: np.ndarray, mu: np.ndarray, cov: np.ndarray):
    """
    Multivariate Gaussian distribution probability function

    Args:
        x: input matrix
        mu: mean vector
        cov: covariance matrix

    Returns:

    """
    M = x.shape[0]
    _, logSigma = np.linalg.slogdet(cov)
    cov_inv = np.linalg.inv(cov)

    if x.shape[1] == 1:
        logN = - .5 * M * np.log(2 * np.pi) - .5 * logSigma - .5 * (x - mu).T @ cov_inv @ (x - mu)
    else:
        logN = - .5 * M * np.log(2 * np.pi) - .5 * logSigma - .5 * np.diagonal((x - mu).T @ cov_inv @ (x - mu))
    return logN

