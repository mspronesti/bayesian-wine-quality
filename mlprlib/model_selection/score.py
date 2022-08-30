import numpy as np
from mlprlib.logistic import LogisticRegression

from mlprlib.metrics import (
    min_detection_cost_fun,
    normalized_bayes_risk,
    detection_cost_fun,
    confusion_matrix
)

from mlprlib.model_selection import KFold


def calibrate(scores_train,
              scores_test,
              y_train,
              y_test,
              l: float = .1,
              *,
              pi: float = .5,
              cfn: float = 1.,
              cfp: float = 1.
              ):
    """
    Calibrates scores using a Linear Logistic
    Regression Model

    Parameters
    ----------
    scores_train:
        ndarray, loglikelihood ratios
        outputted by the model on training set

    scores_test:
        ndarray, loglikelihood ratios
        outputted by the model on test set

    y_train:
        ndarray, training target values

    y_test:
        ndarray, testing target values

    l:
        float, the lambda parameter of the linear
        logistic regression. It's the norm multiplier.
        Default 0.1

    pi:
        float, class prior probability for the True case.
        Default 0.5

    cfn:
        float, cost of the false negative error.
        Default 1.

    cfp:
        float, cost of the false positive error.
        Default 1.

    Returns
    -------
        estimated and calibration dcf 
    """
    lr = LogisticRegression(l_scaler=l)
    lr.fit(scores_train, y_train)

    _, score = lr.predict(scores_test, return_proba=True)
    # the calibration score is obtained from
    # the score of the LR subtracted by the theoretical
    # threshold
    calib_score = score - np.log(pi / (1 - pi))
    # dfc of calibration, i.e. the dfc computed with
    # the calibration score
    dcf_cal = detection_cost_fun(calib_score, y_test, pi, cfn, cfp)

    # optimal threshold
    _, opt_threshold = min_detection_cost_fun(scores_train, y_train, pi, cfn, cfp)
    y_pred = (scores_test > opt_threshold).astype(np.int32)
    cm = confusion_matrix(y_test, y_pred)

    # estimate actual dcf using the optimal threshold
    dcf_est = normalized_bayes_risk(cm, pi, cfn, cfp)

    return dcf_cal, dcf_est


def kfold_calibrate(llr, y, l, n_folds: int = 5, pi=.5, cfn=1, cfp=1, seed=0, just_cal=True):
    """
    Calibrates scores using a Linear Logistic Regression in a
     K-fold cross validation

    Parameters
    ----------
    llr:
        ndarray, scores to calibrate
    y:
        ndarray, true labels
    l:
        float, lambda scaler of the Linear Logistic Regression.

    n_folds:
        int, number of folds for the CV.
        Default 5.

    pi:
        float, class prior probability for the True case.
        Default 0.5

    cfn:
        float, cost of the false negative error.
        Default 1

    cfp:
        float, cost of the false negative error.
        Default 1

    seed:
        int, random seed for numpy.
        Default 0

    Returns
    -------
        estimated and calibrated dcf
    """
    n_samples = len(llr)
    lr = LogisticRegression(l_scaler=l)

    scores_cal = np.zeros([n_samples, ])
    opt_th_decisions = np.zeros(n_samples)

    llr = llr.reshape([llr.shape[0], 1])

    kfold = KFold(n_folds=n_folds, seed=seed)
    for idx_train, idx_test in kfold.split(llr):
        X_train, X_test = llr[idx_train], llr[idx_test]
        y_train, y_test = y[idx_train], y[idx_test]

        lr.fit(X_train, y_train)

        _, score = lr.predict(X_test, return_proba=True)
        scores_cal[idx_test] = score

        if not just_cal:
            _, opt_t = min_detection_cost_fun(X_train.reshape(X_train.shape[0], ),
                                              y_train, pi, cfn, cfp)
            opt_th_decisions[idx_test] = 1. * (X_test.reshape([X_test.shape[0], ]) > opt_t)

    # subtract the theoretical threshold
    scores_cal -= np.log(pi / (1 - pi))
    # compute actual dcf for calibrated score
    act_dcf_cal = detection_cost_fun(scores_cal, y, pi, cfn, cfp)

    if just_cal:
        dcf_est = 0
    else:
        cm = confusion_matrix(y, opt_th_decisions)
        dcf_est = normalized_bayes_risk(cm, pi, cfn, cfp)
    return act_dcf_cal, dcf_est


def joint_eval(*scores,
               y_train,
               y_test,
               l: float,
               pi: float = .5,
               cfn: float = 1,
               cfp: float = 1,
               ):
    """
    Evaluate the fusion of `n_models` models using a Linear Logistic
    Regression

    Parameters
    ----------
    scores:
        variadic containing train and test samples for each model.
        Must be of size (2 * n_models, n_samples)

    y_train:
        ndarray, the training labels

    y_test:
        ndarray, the test labels

    l:
        float, the lambda parameter of the logistic regression.
        It's the norm multiplier.

    pi:
        float, class prior probability for the True case.
        Default 0.5

    cfn:
        float, cost of the false negative error.
        Default 1

    cfp:
        float, cost of the false negative error.
        Default 1

    Returns
    -------
        score:
            the score retrieved by the logistic regression

        act_dcf:
            the actual normalized bayes risk

        min_dcf:
            the minimum detection cost (min bayes risk)
    """
    n_scores = len(scores)
    n_models = len(scores[0])

    if n_scores <= 1:
        raise ValueError(
            "joint evaluation requires at least two"
            " two models. Got n_models=%s." % n_models
        )

    if n_scores != 2 * n_models:
        raise ValueError(
            "joint evaluation requires train and test scores"
            " for each  models. Got %s scores." % n_scores
        )
    # iterate over training and testing scores
    X_train = np.zeros([n_models, scores[0].shape])
    X_test = np.zeros([n_models, scores[0].shape])
    for s in range(0, n_scores, 2):
        train_score = scores[s]
        test_score = scores[s+1]
        X_train[:, s] = train_score
        X_test[:, s+1] = test_score

    lr = LogisticRegression(l_scaler=l)
    lr.fit(X_train, y_train)
    _, score = lr.predict(X_test, return_proba=True)

    min_dfc = min_detection_cost_fun(score, y_test, pi, cfn, cfp)
    act_dfc = detection_cost_fun(score, y_test, pi, cfn, cfp)
    return (
        min_dfc,
        act_dfc,
        score
    )


def k_fold_joint_eval(*scores,
                      y,
                      l: float = 0.,
                      n_folds: int = 5,
                      pi: float = .5,
                      cfn: float = 1,
                      cfp: float = 1,
                      seed: int = 0
                      ):
    """
    Evaluates the fusion of N models
    given their scores, using a linear logistic
    regression and a k-fold cross validation

    Parameters
    ----------
    scores:
        variadic, an array-like of an array-like
        of scores. Must be of shape
            (n_models, n_samples)

    y:
        ndarray, the true labels

    l:
        float, the lambda scaler of the linear logistic
        regression.
        Default 0

    n_folds:
        int, the number of folds for cross validation.
        Default 5.

    pi:
        float, class prior probability for the True case.
        Default 0.5

    cfn:
        float, cost of the false negative error.
        Default 1

    cfp:
        float, cost of the false negative error.
        Default 1

    seed:
        int, random seed for numpy.
        Default 0.

    Returns
    -------
        min_dcf:
            the minimum detection cost (min bayes risk)

        act_dcf:
            the actual normalized bayes risk
    """
    n_scores = len(scores)
    if n_scores <= 1:
        raise ValueError(
            "joint evaluation requires at least two"
            " two models. Got scores for %s models." % n_scores
        )

    # define the Linear Logistic Regression
    # for calibrating the scores
    lr = LogisticRegression(l_scaler=l)

    # define log likelihood ratios
    n_samples = len(scores[0])
    llr = np.zeros([n_samples, ])

    kfold = KFold(n_folds=n_folds, seed=seed)
    for idx_train, idx_test in kfold.split(scores[0]):
        X_train = np.zeros([idx_train.shape[0], n_scores])
        X_test = np.zeros([idx_test.shape[0], n_scores])

        for i, _ in enumerate(scores):
            X_train[:, i] = scores[i][idx_train]
            X_test[:, i] = scores[i][idx_test]
            y_train, _ = y[idx_train], y[idx_test]

            lr.fit(X_train, y_train)
            _, score = lr.predict(X_test, return_proba=True)
            llr[idx_test] = score

    min_dcf, _ = min_detection_cost_fun(llr, y, pi, cfn, cfp)
    act_dcf = detection_cost_fun(llr, y, pi, cfn, cfp)
    return min_dcf, act_dcf
