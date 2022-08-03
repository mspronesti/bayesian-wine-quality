import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from scipy.special import logsumexp

from .._base import Estimator, NotFittedError


class LogisticRegression(Estimator):
    def __init__(self, norm_scaler=1.):
        """
        Creates a Logistic Regression classifier

        Args:
            norm_scaler:
                coefficient to multiply the norm
                in the formulation of the optimization
                problem
        """
        self.norm_scaler = norm_scaler
        self.w = None
        self.b = None
        # set dynamically from labels
        # unique size
        self.multiclass: bool = False

    def fit(self, X, y):
        n_classes = np.unique(y).shape[0]
        if n_classes == 2:
            self._fit_binary(X, y)
            self.multiclass = False
        else:
            self._fit_multiclass(X, y)
            self.multiclass = True

    def predict(self, X):
        if self.w is None:
            raise NotFittedError("This LogisticRegression object"
                                 "is not fitted yet. Call fit before"
                                 " predict")

        sc = self.w @ X.T + self.b
        if self.multiclass:
            return np.argmax(sc, axis=0)
        else:
            return sc > 0

    def _fit_binary(self, X, y):
        def objective_function(arr):
            w, b = arr[:-1].reshape(-1, 1), arr[-1]
            regular = self.norm_scaler / 2 * np.linalg.norm(w.T, 2) ** 2
            s = w.T @ X.T + b
            body = y * np.log1p(np.exp(-s)) + (1 - y) * np.log1p(np.exp(s))
            return regular + np.sum(body) / y.shape[0]

        m, _, _ = fmin_l_bfgs_b(objective_function,
                                np.zeros(X.shape[1] + 1),
                                approx_grad=True)
        self.w = m[:-1]
        self.b = m[-1]

    def _fit_multiclass(self, X, y):
        T = np.array((y.reshape(-1, 1) == np.unique(y)), dtype="int32")
        sh = (np.unique(y).shape[0], X.shape[1] + 1)

        def objective_function(arr):
            arr = arr.reshape(sh)
            w, b = arr[:, :-1], arr[:, -1].reshape(-1, 1)
            regular = self.norm_scaler / 2 * np.sum(w * w)
            s = w @ X.T
            ls = logsumexp(s + b)
            ly = s + b - ls
            return regular - np.sum(T * ly.T) / X.shape[0]

        init_w = np.zeros(sh)
        m, _, _ = fmin_l_bfgs_b(objective_function, init_w,
                                approx_grad=True)

        m = m.reshape(sh)
        self.w = m[:, :-1]
        self.b = m[:, -1].reshape(-1, 1)
