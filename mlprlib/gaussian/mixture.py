import numpy as np
from .._base import Estimator
from ..utils.probability import multivariate_normal_logpdf
from scipy.special import logsumexp


def gmm_logpdf(X, gmm):
    """
    Log probability density function for
    Gaussian Mixtures Models

    Args:
        X:
        gmm:

    Returns:

    """
    S = np.zeros([len(gmm), X.shape[1]])
    for g in range(len(gmm)):
        S[g, :] = multivariate_normal_logpdf(X, gmm[g][1], gmm[g][2] + np.log(gmm[g][0]))

    marginal_log_density = logsumexp(S, axis=0)
    log_gamma = S - marginal_log_density
    gamma = np.exp(log_gamma)
    return marginal_log_density, gamma


def eig_constraint(cov, psi):
    """
    Computes the constraints on the eigenvalues

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
