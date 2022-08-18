import numpy as np
from .logistic_regression import LogisticRegression


class QuadLogisticRegression(LogisticRegression):
    """Quadratic Logistic Regression using LBFGS solver"""
    def __init__(self, norm_scaler=1.):
        super().__init__(norm_scaler)

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
            (n_samples^2 + n_samples, n_features)
        """
        r, n = X.shape
        mapped = np.empty([r * r + r, n])
        for i in range(n):
            x_i = X[:, i].reshape([r, 1])
            mat = x_i @ x_i.T
            mat = mat.flatten("F")
            mapped[:, i] = np.vstack([mat.reshape([r ** 2, 1]), x_i])[:, 0]
        return mapped

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
        super().fit(X_2, y)
