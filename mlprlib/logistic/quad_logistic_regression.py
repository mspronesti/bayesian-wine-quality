import numpy as np
from .logistic_regression import LogisticRegression


class QuadLogisticRegression(LogisticRegression):
    """Quadratic Logistic Regression using LBFGS solver"""
    def __init__(self, l_scaler=1.):
        super().__init__(l_scaler)

    @staticmethod
    def _map_to_quad_space(X: np.ndarray):
        """
        Maps the given data to the quadratic
        feature space

        Parameters
        ----------
        X: ndarray, data matrix to be mapped

        Returns
        -------
            ndarray matrix of size

                (n_samples, n_features^2 + n_features)

            where, for each sample, we have a row of mapped
            features in the quadratic space
        """
        n_samples, n_feats = X.shape
        # X_mapped contains, for each sample, the mapped
        # feats, i.e. an array of size n_feats (n_feats + 1)
        X_mapped = np.empty([n_samples, n_feats ** 2 + n_feats])

        for i in range(n_samples):
            # extract features
            # (i.e. each row)
            x_i = X[i, :].reshape([n_feats, 1])
            # compute the mapping using
            # a quadratic kernel
            mat = x_i @ x_i.T
            # flatten to get a 1d array
            mat = mat.flatten("F")
            X_mapped[i, :] = np.vstack([mat.reshape([n_feats ** 2, 1]), x_i])[:, 0]
        return X_mapped

    def fit(self, X, y, initial_guess=None):
        """
        Fits the Quadratic Logistic Regression model
        using the training data

        Parameters
        ----------
        X:
            ndarray, training data matrix

        y:
            ndarray, target values

        initial_guess:
            the starting point for the optimization
            procedure via L-BFGS-B algorithm.
            If None, it uses an array of zeros

        Returns
        -------
            Fitted QuadraticLogisticRegression model
        """
        X_2 = QuadLogisticRegression._map_to_quad_space(X)
        return super().fit(X_2, y)

    def predict(self, X, return_proba=False):
        X_2 = QuadLogisticRegression._map_to_quad_space(X)
        return super().predict(X_2, return_proba)
