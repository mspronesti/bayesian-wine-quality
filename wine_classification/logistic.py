"""
Using the Linear Logistic Regression with linear and quadratic
kernels with a 5-fold cross validation approach to
classify wines and analyze the performances these methods yield.
"""
import numpy as np

from mlprlib.dataset import (
    load_wine_train
)

from mlprlib.model_selection import (
    train_test_split,
    k_fold_split
)

from mlprlib.preprocessing import standardize
from mlprlib.metrics import min_detection_cost_fun

from mlprlib.utils import Writer
from mlprlib.logistic import (
    LogisticRegression,
    QuadLogisticRegression
)


def split_data_lr():
    """
    Applies a Linear Logistic Regression
    to the wine dataset after a single split
    for a set of  lambdas:

        lambdas = [0, 1e-6, 1e-4, 1e-2, 1, 100]
    """
    pass


def split_data_qlr():
    """
    Applies a Quadratic Logistic Regression
    to the wine dataset after a single split
    for a set of lambdas:

        lambdas = [0, 1e-6, 1e-4, 1e-2, 1, 100]
    """
    pass


if __name__ == '__main__':
    # number of folds for cross validation
    n_folds = 5

    # load the dataset in the shape (n_samples, n_feats)
    X, y = load_wine_train(feats_first=False)
    X_gauss = np.load('../results/gaussian_feats.npy').T
    X_std = standardize(X)

    X_train, X_val, y_train, y_val = train_test_split(X_gauss, y, test_size=.2)

    lr = QuadLogisticRegression(1e-6)
    lr.fit(X_train, y_train)

    _, score = lr.predict(X_val, return_proba=True)
    min_dfc, _ = min_detection_cost_fun(score, y_val)

    print(min_dfc)
