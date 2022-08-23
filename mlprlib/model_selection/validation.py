import numpy as np
from mlprlib.logistic import LogisticRegression


def calibrate(X, y, n_folds, pi=.5, cfn=1., cfp=1., seed=0):
    """
    Calibrates scores using a Linear Logistic
    Regression Model

    Parameters
    ----------
    X:
        ndarray, data matrix of shape
            (n_samples, n_feats)

    y:
        ndarray, target values

    n_folds:
        the `k` folds to be used for the
        cross validation
        Default 5

    pi:
        float, class prior probability for the True case.
        Default 0.5

    cfn:
        float, cost of the false negative error.
        Default 1.

    cfp:
        float, cost of the false positive error.
        Default 1.

    seed:
        int, the random seed for numpy

    Returns
    -------

    """
    np.random.seed(seed)
    # TODO: to be written


def joint_evaluation(y_train, y_test, l, **args):
    """
    Evaluate the fusion of N models using a Linear Logistic
    Regression

    Parameters
    ----------
    y_train:
        ndarray, the training labels

    y_test:
        ndarray, the test labels

    l:
        float, the lambda parameter of the logistic regression.
        It's the norm multiplier.

    args:


    Returns
    -------
        score:
            the score retrieved by the logistic regression

        act_dcf:
            the actual normalized bayes risk

        min_dcf:
            the minimum detection cost (min bayes risk)
    """

    pass

