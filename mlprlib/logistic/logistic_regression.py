import numpy as np
from scipy.optimize import fmin_l_bfgs_b

from .._base import Estimator, NotFittedError


class LogisticRegression(Estimator):
    """Linear Logistic Regression using LBFGS solver"""
    def __init__(self, norm_scaler=1.):
        """
        Creates a Logistic Regression classifier

        Parameters
        ----------
        norm_scaler:
                float, coefficient to multiply the norm
                in the formulation of the optimization
                problem.
                Default 1.0
        """
        self.norm_scaler = norm_scaler
        self.w = None
        self.b = None

    def fit(self, X, y, initial_guess=None):
        """
        Fits the Linear Logistic Regression Model
        with the given training data

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
            fitted LogisticRegression object
        """
        if initial_guess is None:
            initial_guess = np.zeros(X.shape[1] + 1)

        def objective_function(v):
            w, b = v[:-1].reshape(-1, 1), v[-1]
            # regularization term
            regular = self.norm_scaler / 2 * np.linalg.norm(w.T, 2) ** 2
            s = w.T @ X.T + b
            body = y * np.log1p(np.exp(-s)) + (1 - y) * np.log1p(np.exp(s))
            return regular + np.sum(body) / y.shape[0]

        m, _, _ = fmin_l_bfgs_b(objective_function,
                                initial_guess,
                                approx_grad=True)

        self.w = m[:-1]
        self.b = m[-1]
        return self

    def predict(self, X, return_proba=False):
        """
        Predicts the unseen data X

        Parameters
        ----------
        X:
            ndarray, unseen data matrix

        return_proba:
            bool, whether to return the score.
            Default False

        Returns
        -------
            predicted labels and score, depending
            on `return_proba`
        """
        if self.w is None:
            raise NotFittedError("This LogisticRegression object"
                                 "is not fitted yet. Call fit before"
                                 " predict")

        score = self.w @ X.T + self.b
        y_pred = score > 0

        if return_proba:
            return y_pred, score
        else:
            return y_pred
