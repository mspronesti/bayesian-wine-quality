import scipy

from .._base import Transformer, NotFittedError

from ..utils import (
    within_class_covariance,
    between_class_covariance
)


class LDA(Transformer):
    """Linear Discriminant Analysis algorithm for reduction"""
    def __init__(self, n_components: int = 2):
        """

        Parameters
        ----------
        n_components: number of features of output data
        """
        self.n_components = n_components
        self.U = None

    def fit(self, X, y=None):
        """
        Fits the LDA instance with given data X

        Parameters
        ----------
        X: ndarray, data matrix
        y: ignored

        Returns
        -------
            fitted LDA instance
        """
        if self.n_components is None:
            self.n_components = X.shape[0]

        sw = within_class_covariance(X, y)
        sb = between_class_covariance(X, y)
        _, U = scipy.linalg.eigh(sb, sw)
        self.U = U[:, ::-1][:, :self.n_components]

    def transform(self, X, y=None):
        """
        Transforms input data applying linear
        discriminant analysis

        Parameters
        ----------
        X: ndarray, data to be transformed
        y: ignored

        Returns
        -------
            transformed data
        """
        if self.U is None:
            raise NotFittedError("This LDA instance has not been fitted."
                                 "Call fit before calling transform")
        return (self.U.T @ X).T
