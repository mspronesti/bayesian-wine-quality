import numpy as np
from scipy.special import logsumexp

from .._base import Estimator
from ..utils.probability import (
    covariance_matrix,
    multivariate_normal_logpdf
)


class GaussianClassifier(Estimator):
    """Multivariate Gaussian classifier"""

    def __init__(self):
        """
        Constructs a Gaussian Classifier object
        -----
        estimates contains, for each label
          - label
          - mean
          - covariance
          - probability
        """
        self.estimates = []
        self.posterior = None

    def fit(self, X, y):
        """
        Fits the Gaussian Classifier

        Args:
            X: ndarray, training samples
            y: ndarray, training labels

        Returns:
            fitted GaussianClassifier instance
        """
        return self._fit_helper(X, y, covariance_matrix)

    def _fit_helper(self, X, y, cov_fn: callable):
        """cov_fn is the function  used to compute the
         covariance matrix. It must only accept a ndarray"""
        self.estimates = []
        y_un, counts = np.unique(y, return_counts=True)
        for label, count in zip(y_un, counts):
            # extract from the train set
            # the rows having correct label
            mat = X[y == label, :]
            cov = cov_fn(mat)
            estimate = (
                label,
                np.mean(mat, 0),
                cov,
                count / X.shape[0]
            )
            self.estimates.append(estimate)
        return self

    def predict(self, X):
        """
        Predicts from provided unseen data X
        computing class-conditional log probabilities
        for each class

        Args:
            X: ndarray, test samples

        Returns:
            predicted labels
        """
        scores = []
        for label, mu, cov, prob in self.estimates:
            distro = multivariate_normal_logpdf(X.T, mu.reshape(-1, 1), cov)
            scores.append(distro + np.log(prob))

        joint_mat = np.hstack([value.reshape(-1, 1) for value in scores])
        logsum = logsumexp(joint_mat, axis=1)
        self.posterior = joint_mat - logsum.reshape(1, -1).T
        return np.argmax(self.posterior, axis=1)


class NaiveBayes(GaussianClassifier):
    """Naive Bayes classifier"""

    def fit(self, X, y):
        """
        Fits the Naive Bayes classifier

        Args:
            X: ndarray, training samples
            y: ndarray, training labels

        Returns:
            fitted NaiveBayes instance
        """
        return self._fit_helper(X, y, lambda m: np.diag(np.var(m, 0)))


class TiedGaussian(GaussianClassifier):
    """Tied Gaussian Classifier"""

    def fit(self, X, y):
        super().fit(X, y)
        # compute the tied covariance matrix
        # averaging all covariance matrices
        tied_cov = 1. / y.shape[0] * sum([cov * np.sum(y == label) for label, _, cov, _ in self.estimates])
        self.estimates = [(label, mu, tied_cov, prob) for label, mu, _, prob in self.estimates]
        return self
