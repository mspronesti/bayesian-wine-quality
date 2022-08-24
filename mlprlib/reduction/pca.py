import numpy as np

from .._base import Transformer, NotFittedError
from ..utils import covariance_matrix


class PCA(Transformer):
    """Principal Component Analysis algorithm for dimensionality reduction"""
    def __init__(self, n_components: int = None):
        """

        Parameters
        ----------
        n_components: number of features for
                dimensionality reduction.
                Default None
        """
        self.n_components = n_components
        self.P = None

    def fit(self, X, y=None, *, use_svd: bool = False):
        """
        Fits the model given the training
        data X

        Parameters
        ----------
        X:
            ndarray, training set
        y:
            Not used. Only kept for compatibility
            with base class.

        use_svd:
            bool, whether to use the svd solver or not.
            Default False.

        Returns
        -------
            a fitted PCA instance
        """
        if self.n_components is None:
            n_components = X.shape[0] - 1
        else:
            n_components = self.n_components

        cov = covariance_matrix(X)
        if use_svd:
            eigenvectors, _, _ = np.linalg.svd(cov)
            self.P = eigenvectors[:, 0:n_components]
        else:
            _, eigenvectors = np.linalg.eigh(cov)
            self.P = eigenvectors[:, ::-1][:, 0:n_components]

        # extract the top n_components eigenvectors
        return self

    def transform(self, X, y=None):
        """
        Transform the data with the P matrix
        computed already, projecting to training
        matrix.

        Parameters
        ----------
        X:
            ndarray, data to be transformed
        y:
            ignored

        Returns
        -------
            the transformed data
        """
        if X is None or self.P is None:
            raise NotFittedError("This PCA instance has not been fitted."
                                 "Call fit before calling transform")
        return (self.P.T @ X.T).T
