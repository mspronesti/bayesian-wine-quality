from abc import ABC, abstractmethod


class NotFittedError(Exception):
    """Exception class for not fitted instances"""
    def __init__(self, message=""):
        super().__init__(message)


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


class Transformer(ABC):
    """Base class for Transformer methods"""

    @abstractmethod
    def fit(self, X, y=None):
        """Fits the Transformer instance"""
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
