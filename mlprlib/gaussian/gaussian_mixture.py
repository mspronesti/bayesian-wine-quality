import numpy as np
from .._base import Estimator


def gmm_logpdf(X, gmm):
    """
    Log probability density function for
    Guassian Mixtures

    Args:
        X:
        gmm:

    Returns:

    """
    pass


def eig_constraint(cov, psi):
    """

    Args:
        cov: covariance matrix
        psi:

    Returns:

    """
    U, s, _ = np.lianlg.svd(cov)
    s[s < psi] = psi
    return U @ (s.reshape(s.size, 1) * U.T)


def EM_estimation():
    """Expectation Maximization algorithm"""
    pass


def LGB_estimation():
    """Linde-Buzo-Gray algorithm"""
    pass


class GaussianMixture(Estimator):
    def __init__(self, n_classes=2, alpha=.1, psi=None, covariance_type='full'):
        """

        Args:
            n_classes:
                number of classes
            alpha:

            psi:

            covariance_type: string,
                describes the type of covariance. Must be one of
                {'tied', 'diag', 'full}. Default 'full'
        """
        self.n_classes = n_classes
        self.alpha = alpha
        self.psi = psi
        self.covariance_type = covariance_type
        self.gmm_estimates = {}

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

