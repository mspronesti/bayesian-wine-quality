import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from .kernels import rbf_kernel, polynomial_kernel, linear_kernel


class SVClassifier:
    def __int__(self, kernel='linear', k=1, c=1):
        pass

    def fit(self, X, y):
        """
        Fits the Support Vector Classifier
        Args:
            X:
            y:

        Returns:

        """
        # TODO: to be implemented
        return self

    def predict(self, X):
        """
        Predicts given the unseen data X
        Args:
            X: unseen data of shape (n_samples, n_feats)

        Returns:
            predicted labels
        """
        pass

    def __str__(self):
        pass
