import numpy as np


def confusion_matrix(y, y_pred):
    """
    Computes the confusion matrix with given
    predictions

    Args:
        y:  true labels (ndarray)
        y_pred: predicted labels (ndarray)

    Returns:
        ndarray confusion matrix
    """
    n_labels = len(np.unique(y_pred))
    cm = np.zeros((n_labels, n_labels))

    for i in range(n_labels):
        for j in range(n_labels):
            cm[i, j] = sum((y_pred == i) & (y == j))
    return cm


def roc_curve(llr, y):
    """
    Computes the ROC curve given
    Args:
        llr:
        y: true labels

    Returns:

    """
    # return fpr, tpr
    pass


def plot_roc_curve():
    pass


def accuracy_score(y, y_pred):
    return np.sum(y_pred == y)/len(y)


def f_score(y, y_pred):
    cm = confusion_matrix(y, y_pred)
    return cm[0, 0] / (cm[0, 0] * .5 * (cm[0, 1] + cm[1, 0]))


def bayes_risk(cm, pi=.5, cfn=1, cfp=1):
    """
    Computes the un-normalized empirical decision cost function
    defined as (in the binary case)
       DCFu(Cfn, Cfp, π ) = π Cfn * Pfn + (1 − π ) * Cfp * Pfp

    Notice that empirical bayes risk B_emp and un-normalized empirical
    decision cost are the same thing.

    Args:
        cm: confusion matrix
        pi: class prior probability for the True case
        cfn: cost of the false negative error
        cfp: cost of the false positive error

    Returns:
        the un-normalized decision cost function (also called Bayesian risk)
    """
    fnr = cm[0, 1] / (cm[0, 1] + cm[1, 1])
    fpr = cm[1, 0] / (cm[1, 0] + cm[0, 0])
    return pi * cfn * fnr + (1 - pi) * cfp * fpr


def normalized_bayes_risk(cm, pi=.5, cfn=1, cfp=1):
    """
    Computes the normalized empirical decision cost function
    defined as (in the binary case)
       DCF(Cfn, Cfp, π ) = 1/K * [π Cfn * Pfn + (1 − π ) * Cfp * Pfp]
    being K equal to
       K = min(π * Cfn, (1 − π) * Cfp)

    Args:
        cm: confusion matrix
        pi: class prior probability for the True case
        cfn: cost of the false negative error
        cfp: cost of the false positive error

    Returns:
        the normalized decision cost function
    """
    dcf_u = bayes_risk(cm, pi, cfn, cfp)
    return dcf_u / min(pi * cfn, (1 - pi) * cfp)


def optimal_bayes_decision(llr, pi, cfn, cfp):
    """
    Computes the optimal Bayes decision ratio given
    the likelihood ratio

    Args:
        llr: likelihood ratio
        pi: prior probability
        cfn: cost of false negative error
        cfp: cost of false positive error

    Returns:
        the optimal Bayes decision ratio
    """
    threshold = np.log(pi * cfn / ((1 - pi) * cfp))
    return llr > - threshold


def min_dcf(llr, pi, cfn, cfp, y):
    """
    Computes the minimum decision cost function

    Args:
        llr: likelihood ratio
        pi: prior probability
        cfn: cost of false negative error
        cfp: cost of false positive error
        y: true labels

    Returns:

    """
    pass
