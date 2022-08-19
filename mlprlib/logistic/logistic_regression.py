import numpy as np
from scipy.optimize import fmin_l_bfgs_b

from .._base import Estimator, NotFittedError


class LogisticRegression(Estimator):
    """Linear Logistic Regression using LBFGS solver"""

    def __init__(self, l_scaler: float = 1., pi_t: float = .5):
        """
        Creates a Logistic Regression classifier

        Parameters
        ----------
        l_scaler:
                float, coefficient to multiply the norm
                in the formulation of the optimization
                problem.
                Default 1.0

        pi_t :
            float, prior probability for the true class.
            Default 0.5
        """
        self.l_scaler = l_scaler
        self.pi_t = pi_t
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
            """
            Objective function for the L-BFGS algorithm

                R(w) = l/2 * |w|^2 + pi_T / nt * sum_{i | zi=1} { log (σ(zi*si)) }
                              + (1 - pi_T) / nf * sum_{i|zi=-1} { log (σ(zi*si)) }

            Where
                - l is the norm scaler
                - |.| is the quadratic norm
                - nt is the number of positive samples
                - nf is the number of negative samples
                - pi_T is the prior for the true class
                - σ(.) is the sigmoid function
                - zi = 1 for the pos class, -1 for the neg class
                - si = w^T * x_i + b

            Therefore we optimize an objective made of
            - a regularization term
            - the sum over the positive samples of the log-sigmoid of "- s", scaled by the prior over
              the number of positive samples
            - the sum over the negative samples of the log sigmoid of "s", scaled by 1-prior over
              the number of negative samples
            """
            w, b = v[:-1], v[-1]
            # regularization term
            regular = self.l_scaler / 2 * np.linalg.norm(w.T, 2) ** 2
            # positive sample
            pos_f = X.T[:, y == 1]
            # negative samples
            neg_f = X.T[:, y == 0]

            nt = pos_f.shape[1]
            nf = neg_f.shape[1]

            s_pos = w.T @ pos_f + b
            s_neg = w.T @ neg_f + b

            sum_pos = np.sum(np.log1p(np.exp(- s_pos)))
            sum_neg = np.sum(np.log1p(np.exp(s_neg)))

            return regular + (1 - self.pi_t)/nf * sum_neg + self.pi_t / nt * sum_pos

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


