import numpy as np
from mlprlib.logistic import LogisticRegression


def calibrate(X, y, n_folds, pi=.5, cfn=1., cfp=1., seed=0):
    """
    Calibrates scores using a Linear Logistic
    Regression Model

    Parameters
    ----------
    X:
        ndarray, data matrix

    y:
        ndarray, target values

    n_folds:
        the `k` folds to be used for the
        cross validation

    pi:
        float, class prior probability for the True case

    cfn:
        float, cost of the false negative error

    cfp:
        float, cost of the false positive error

    seed:
        the random seed for numpy

    Returns
    -------

    """
    np.random.seed(seed)
    # TODO: to be written
