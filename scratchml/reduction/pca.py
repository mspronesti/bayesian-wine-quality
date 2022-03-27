import numpy as np

from .._base import Transformer
from ..utils import covariance_matrix
from .._exceptions import NotFittedError


class PCA(Transformer):
    """Principal Component Analysis algorithm for dimensionality reduction"""
    def __init__(self, n_components: int = 2):
        """
        Args:
            n_components: the number of features
                for dimensionality reduction
        """
        self.n_components = n_components
        self.P = None

    def fit(self, X, y=None):
        """
        Fits the model given the dataset X
        Args:
            X: train dataset
            y: ignored

        Returns:
            the fitted instance
        """
        if self.n_components is None:
            n_components = X.shape[0] - 1
        else:
            n_components = self.n_components

        cov = covariance_matrix(X)
        _, eigenvectors = np.linalg.eig(cov)

        # extract the top n_components eigenvectors
        self.P = eigenvectors[:, ::-1][:, 0:n_components]
        return self

    def transform(self, X, y=None):
        """
        Transform the data with the P computed already
        Args:
            X: the input dataset to be transformed
            y: ignored

        Returns:
            the transformed data
        """
        if X is None or self.P is None:
            raise NotFittedError("This PCA instance has not been fitted."
                                 "Call fit before calling transform")
        return (self.P.T @ X.T).T

    def __str__(self):
        return f"PCA(n_components={self.n_components})"
