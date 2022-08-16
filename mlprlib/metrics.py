import numpy as np
import matplotlib.pyplot as plt


def confusion_matrix(y, y_pred):
    """
    Computes the confusion matrix with given
    predictions and true labels

    Parameters
    ----------
    y: true labels (ndarray)
    y_pred: predicted labels (ndarray)

    Returns
    -------
        confusion matrix as ndarray of
        size (n_labels, n_labels)
    """
    n_labels = len(np.unique(y))
    cm = np.zeros((n_labels, n_labels))

    for i in range(n_labels):
        for j in range(n_labels):
            cm[i, j] = sum((y_pred == i) & (y == j))
    return cm


def roc_curve(llr, y):
    """
    Computes the ROC curve given the
    likelihood ratio and the true labels

    Parameters
    ----------
    llr: likelihood ratio
    y: true labels

    Returns
    -------
        fpr:
            Increasing false positive rates such that
            element i is the false positive rate of
            predictions with score >= thresholds[i]

        tpr:
            Increasing true positive rates such that
            element i is the true positive rate of
            predictions with score >= thresholds[i].
    """
    # TODO: not tested
    sorted_llr = np.sort(llr)
    fpr = np.array(sorted_llr.shape[0])

    for i in sorted_llr:
        cm = confusion_matrix(llr > i, y)
        fpr[i] = cm[1, 0] / (cm[0, 0] + cm[1, 0])

    return fpr, 1 - fpr


def plot_roc_curve(llr, y, label=None):
    fpr, tpr = roc_curve(llr, y)
    plt.plot(fpr, tpr, label=label)


def accuracy_score(y, y_pred):
    return np.sum(y_pred == y) / len(y)


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

    Parameters
    ----------
    cm: confusion matrix
    pi: class prior probability for the True case
    cfn: cost of the false negative error
    cfp: cost of the false positive error

    Returns
    -------
        The un-normalized decision cost function (also called Bayesian risk)
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

    Parameters
    ----------
    cm: confusion matrix
    pi: class prior probability for the True case
    cfn: cost of the false negative error
    cfp: cost of the false positive error

    Returns
    -------
        the normalized decision cost function
    """
    dcf_u = bayes_risk(cm, pi, cfn, cfp)
    return dcf_u / min(pi * cfn, (1 - pi) * cfp)


def optimal_bayes_decision(llr, pi=.5, cfn=1, cfp=1):
    """
    Computes the optimal Bayes decision ratio given
    the likelihood ratio as using the following threshold

       threshold = log( pi / (1 - pi) * Cfn / Cfp )

    Parameters
    ----------
    llr: likelihood ratio
    pi: prior probability
    cfn: cost of the false negative error
    cfp: cost of the false positive error

    Returns
    -------
        the optimal Bayes decision ratio
    """
    threshold = np.log(pi * cfn / ((1 - pi) * cfp))
    # retrieves a np.ndarray of bools instead of 0s and 1s
    # alternative (for readability)
    # >> opt_bayes = np.zeros(llr.shape[0])
    # >> opt_bayes[llr > - threshold] = 1
    # >> return opt_bayes
    return llr > - threshold


def detection_cost_fun(llr, y, pi=.5, cfn=1, cfp=1):
    """
    Computes the actual detection cost function,
    defined as (in the binary case)

        DCF(Cfn, Cfp, π ) = π Cfn * Pfn + (1 − π ) * Cfp * Pfp

    from the likelihood ratio. Just calls optimal_bayes_decision
    and the normalized bayes risk given the likelihood ratio instead
    of the confusion matrix

    Parameters
    ----------
    llr: likelihood ratio
    y: true labels
    pi: prior probability
    cfn: cost of the false negative error
    cfp: cost of the false positive error

    Returns
    -------
        actual minimal detection cost
    """
    opt_decision = optimal_bayes_decision(llr, pi, cfn, cfp)
    cm = confusion_matrix(y, opt_decision)

    return normalized_bayes_risk(cm, pi, cfn, cfp)


def min_detection_cost_fun(llr, y, pi=.5, cfn=1, cfp=1):
    """
    Computes the minimum detection cost function

    Parameters
    ----------
    llr: likelihood ratio
    y: true labels
    pi: prior probability
    cfn: cost of the false negative error
    cfp: cost of the false positive error

    Returns
    -------
        The minimum decision cost function and the
        optimal threshold
    """
    min_dcf = float('inf')
    opt_threshold = 0

    for t in np.sort(llr, kind="mergesort"):
        pred = (llr > t).astype(int)
        cm = confusion_matrix(y, pred)
        dcf = normalized_bayes_risk(cm, pi, cfn, cfp)
        # instead of computing all the DCFs and applying
        # min(.), the minimum is computed in the same loop
        # to reduce the complexity
        if dcf < min_dcf:
            min_dcf = dcf
            opt_threshold = t

    return min_dcf, opt_threshold
