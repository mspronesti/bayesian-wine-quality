import numpy as np
from typing import List

from .._base import Transformer, Estimator
from .split import KFold


class CrossValidator:
    def __init__(self, n_folds: int = 5):
        """
        Creates a cross validator object

        Parameters
        ----------
        n_folds:
            int, number of folds for k-fold
            split.
            Default 5
        """
        self._n_folds = n_folds
        self._scores = None

    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            model: Estimator,
            transformers: List[Transformer] = None,
            *,
            shuffle: bool = True,
            seed: int = 0):
        """
        Fits the cross validator.
        After calling fit, the scores are available to
        get retrieved calling scores().

        Parameters
        ----------
        X:
            ndarray, data to apply cross validation on.
            Expected of shape (n_samples, n_feats)

        y:
            ndarray, target values

        model:
            mlprlib.Estimator, the model to evaluate
            using this cross validator

        transformers:
            a list of mlprlib.Transformer, to be applied
            to data after split

        shuffle:
            Bool, whether to shuffle data.
            Default False.

        seed:
            int, random seed. Ignored if shuffle == False.
            default 0.

        Returns
        -------

        """

        scores = np.zeros([X.shape[0], ])

        kfold = KFold(n_folds=self.n_folds, shuffle=shuffle, seed=seed)
        for idx_train, idx_test in kfold.split(X):
            X_train, X_test = X[idx_train], X[idx_test]
            y_train, y_test = y[idx_train], y[idx_test]

            if transformers is not None:
                # apply cascade of chosen
                # transformations on data
                for transformer in transformers:
                    # all these transformations must be
                    # applied using current training data
                    # as reference not to introduce any
                    # data leakage
                    t = transformer.fit(X_train)
                    X_train = t.transform(X_train)
                    X_test = t.transform(X_test)

            # now fit
            model.fit(X_train, y_train)
            _, score = model.predict(X_test, return_proba=True)
            scores[idx_test] = score

        self._scores = scores

    @property
    def scores(self) -> np.ndarray:
        return self._scores

    @property
    def n_folds(self) -> int:
        return self._n_folds
