import numpy as np

from .._base import Transformer, NotFittedError
from ..preprocessing import normalize, standardize
from ..utils import covariance_matrix


class LDA(Transformer):
    """Linear Discriminant Analysis algorithm for reduction"""
    def __init__(self, n_components: int = 2):
        self.n_components = n_components
        self.U = None
        self.X = None
        self.y = None
        self.mean = None

    def transform(self, X, y=None):
        if self.X is None or self.U is None:
            raise NotFittedError("This PCA instance has not been fitted."
                                 "Call fit before calling transform")
        return (self.U.T @ (X - self.mean.T)).T

    def fit(self, X, y=None):
        if self.n_components is None:
            self.n_components = X.shape[0]
        self.X = X.T
        self.y = y

        self.mean = self.X.mean(1).reshape((-1, 1))
        # TODO: to be completed

    def __str__(self):
        return f"LDA(n_components={self.n_components}"