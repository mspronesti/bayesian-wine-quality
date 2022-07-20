from abc import ABC, abstractmethod

import numpy as np


class Estimator(ABC):
    """Base class for classifiers, regressors and cluster methods"""
    @abstractmethod
    def fit(self, X, y):
        """Fits the estimator instance"""
        raise NotImplemented("Must have implemented this.")

    @abstractmethod
    def predict(self, X):
        """Labels the test dataset X"""
        raise NotImplemented("Must have implemented this.")

    def fit_predict(self, X, y):
        """Fits and predicts the model at once using the entire
         dataset"""
        self.fit(X, y)
        return self.predict(X)

    @abstractmethod
    def score(self, X, y_true):
        """Provides a score for the given estimator according
        to some metric"""
        raise NotImplemented("Must have implemented this.")


class Classifier(Estimator, ABC):
    """Abstract class for classifiers"""
    def score(self, X, y_true):
        """
        Retrieves the "classic" accuracy score expressed
        as the sum of the wrong predicted values over the
        total

        Args:
            X: the data to predict
            y_true: the ground truth labels

        Returns:
            the accuracy score
        """
        y_pred = self.predict(X)
        assert len(y_pred) == len(y_true)
        # retrieve the fraction of wrongly predicted values
        return np.count_nonzero(y_true - y_pred, axis=1) / len(y_pred)


class Transformer(ABC):
    """Base class for Transformer methods"""
    @abstractmethod
    def fit(self, X, y=None):
        """Fits the .... instance"""
        raise NotImplemented("Must have implemented this.")

    @abstractmethod
    def transform(self, X, y=None):
        """Transforms the data"""
        raise NotImplemented("Must have implemented this.")

    def fit_transform(self, X, y=None):
        """Fit to data, then transform it"""
        if y is None:
            self.fit(X).transform(X)
        else:
            self.fit(X, y).transform(X, y)
