import numpy as np
import matplotlib.pyplot as plt

from mlprlib.dataset import (
    load_wine_train,
    load_wine_test
)
from mlprlib.preprocessing import StandardScaler

# most performing models
from mlprlib.svm import SVClassifier
from mlprlib.gaussian import GaussianMixture
from mlprlib.logistic import QuadLogisticRegression

from mlprlib.model_selection import (
    calibrate,
    joint_eval
)

from mlprlib.metrics import (
    min_detection_cost_fun,
    detection_cost_fun,
    roc_curve
)

from mlprlib.utils import Writer


def save_roc_plot(scores, y, fig_name: str, labels: list):
    plt.figure()
    plt.grid(b=True)
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    for llr in scores:
        fpr, tpr = roc_curve(llr, y)
        plt.plot(fpr, tpr)
    plt.legend(labels)
    plt.savefig("../report/assets/%s.png" % fig_name)


def save_bayes_err_plot(scores, labels:list, fig_name: str):
    prior_log_odds = np.linspace(-3, 3, 21)

    plt.figure()
    for i, s in enumerate(scores):
        # dash alternately
        # the plots for an easier
        # reading of the plot
        if i % 2 == 1:
            plt.plot(prior_log_odds, s, label=labels[i], linestyle='dashed')
        else:
            plt.plot(prior_log_odds, s, label=labels[i])

        plt.ylim([0, 1.1])
        plt.xlim([-3, 3])
        plt.legend()
        plt.xlabel(r"$\log \frac{\tilde{\pi}}{1 - \tilde{\pi}}$")
        plt.ylabel("DCF")
        plt.savefig("../report/assets/%s.png" % fig_name)


def logodds_scores(llr, y):
    prior_log_odds = np.linspace(-3, 3, 21)

    dcf = np.zeros(prior_log_odds.shape[0])
    min_dcf = np.zeros(prior_log_odds.shape[0])

    for i, p in enumerate(prior_log_odds):
        pi_tilde = 1 / (1 + np.exp(-p))
        dcf[i] = detection_cost_fun(llr, y, pi_tilde)
        min_dcf[i], _ = min_detection_cost_fun(llr, y, pi_tilde)

    return dcf, min_dcf


if __name__ == '__main__':
    X_train, y_train = load_wine_train(feats_first=False)
    X_test, y_test = load_wine_test(feats_first=False)

    # most performing models
    svc = SVClassifier(kernel='rbf', C=10,
                       gamma=np.exp(-2), pi_t=.5, csi=1)
    gmm = GaussianMixture(n_components=8)
    qlr = QuadLogisticRegression(l_scaler=0)

    models = {
        svc: "SVC",
        gmm: "GMM",
        qlr: "QLR"
    }

    sc = StandardScaler().fit(X_train)
    X_train = sc.transform(X_train)
    X_test = sc.transform(X_test)

    # best lambdas for
    # svc, qlr, gmm,
    # svc + qlr, svc + gmm,
    # qvc + qlr + gmm
    # computed in ../calibration.py
    lambdas = [0, 1, 0]
    lambdas_joint = [0, 0, 0]

    pi, cfn, cfp = (.5, 1, 1)

    writer = Writer("../results/best_eval.txt")
    writer("Best models")
    writer("-------------\n")
    # analyze best models
    for i, (model, name) in enumerate(models.items()):
        writer('\n' + name)
        writer("----")

        model.fit(X_train, y_train)
        _, score_test = model.predict(X_test, return_proba=True)
        np.save("../results/llr%s_eval.npy" % name, score_test)

        act_dcf = detection_cost_fun(score_test, y_test, pi, cfn, cfp)
        score_train = np.load("../results/llr%s.npy" % name)

        dcf_cal, dcf_est = calibrate(score_train, score_test,
                                     y_train, y_test,
                                     lambdas[i],
                                     pi=pi, cfn=cfn, cfp=cfp)
        writer(
            "act dcf: %s \ndcf_calibrated: %s \ndcf_estimated: %s"
            % (act_dcf, dcf_cal, dcf_est)
        )

    # joint models
    llr_svc = np.load("../results/llrSVC.npy")
    llr_svc_eval = np.load("../results/llrSVC_eval.npy")

    llr_gmm = np.load("../results/llrGMM.npy")
    llr_gmm_eval = np.load("../results/llrGMM_eval.npy")

    llr_qlr = np.load("../results/llrQLR.npy")
    llr_qlr_eval = np.load("../results/llrQLR_eval.npy")

    writer("\n\nSVC + QLR")
    writer("--------------")
    min_dcf, act_dcf, llr_svc_qlr = \
        joint_eval(*[llr_svc, llr_svc_eval,
                     llr_qlr, llr_qlr_eval],
                   y_train=y_train,
                   y_test=y_test,
                   l=lambdas_joint[0],
                   pi=pi, cfn=cfn, cfp=cfp)

    writer("min dcf: %s \nactual dcf: %s" % (min_dcf, act_dcf))

    # SVC + GMM
    writer("\n\nSVC + GMM")
    writer("--------------")
    min_dcf, act_dcf, llr_svc_gmm = \
        joint_eval(*[llr_svc, llr_svc_eval,
                     llr_gmm, llr_gmm_eval],
                   y_train=y_train,
                   y_test=y_test,
                   l=lambdas_joint[1],
                   pi=pi, cfn=cfn, cfp=cfp)

    writer("min dcf: %s\n actual dcf: %s" % (min_dcf, act_dcf))

    # SVC + GMM + QLR
    writer("\n\nSVC + GMM + QLR")
    writer("--------------")
    min_dcf, act_dcf, llr_svc_gmm_qlr = \
        joint_eval(*[llr_svc, llr_svc_eval,
                     llr_qlr, llr_qlr_eval,
                     llr_gmm, llr_gmm_eval],
                   y_train=y_train,
                   y_test=y_test,
                   l=lambdas_joint[2],
                   pi=pi, cfn=cfn, cfp=cfp)

    writer("min dcf: %s\n actual dcf: %s" % (min_dcf, act_dcf))
    writer.destroy()

    # plot ROC of single models
    llrs_single = [llr_svc_eval, llr_qlr_eval, llr_gmm_eval]
    save_roc_plot(llrs_single, y_test, "roc_single_models", ["SVC", "QLR", "GMM"])

    # plot ROC of joint models
    llrs_joint = [llr_svc_qlr, llr_svc_gmm, llr_svc_gmm_qlr]
    save_roc_plot(llrs_joint, y_test, "roc_joint_models", ["SVC+QLR", "SVC+GMM", "SVC+QLR+GMM"])

    # store bayesian plot for single models
    dcf_svc, min_dcf_svc = logodds_scores(llr_svc_eval, y_test)
    dcf_gmm, min_dcf_gmm = logodds_scores(llr_gmm_eval, y_test)
    dcf_qlr, min_dcf_qlr = logodds_scores(llr_qlr_eval, y_test)

    scores = [
        dcf_svc,
        min_dcf_svc,
        dcf_qlr,
        min_dcf_qlr,
        dcf_gmm,
        min_dcf_gmm
    ]

    labels = [
        "SVC - act DCF",
        "SVC - min DCF",
        "QLR - act DCF",
        "QLR - min DCF",
        "GMM - act DCF",
        "GMM - min DCF"
    ]

    save_bayes_err_plot(scores, labels, "bayes_error")

    # store bayes plots for joint models
    dcf_svc_qlr, min_dcf_svc_qlr = logodds_scores(llr_svc_qlr, y_test)
    dcf_svc_gmm, min_dcf_svc_gmm = logodds_scores(llr_svc_gmm, y_test)
    dcf_svc_gmm_qlr, min_dcf_svc_gmm_qlr = logodds_scores(llr_svc_gmm_qlr, y_test)

    scores = [
        dcf_svc_qlr,
        min_dcf_svc_qlr,
        dcf_svc_gmm,
        min_dcf_svc_gmm,
        dcf_svc_gmm_qlr,
        min_dcf_svc_gmm_qlr
    ]

    labels = [
        "SVC & QLR - act DCF",
        "SVC & QLR- min DCF",
        "SVC & GMM - act DCF",
        "SVC & GMM - min DCF",
        "SVC & QLR & GMM - act DCF",
        "SVC & QLR & GMM - min DCF"
    ]

    save_bayes_err_plot(scores, labels, "bayes_error_joint")

    # store bayes plot of single + fusion
    scores = [
        dcf_svc,
        dcf_qlr,
        dcf_svc_gmm_qlr,
        min_dcf_svc_gmm_qlr
    ]

    labels = [
        "SVC  - act DCF",
        "QLR - act DCF",
        "SVC & QLR & GMM - act DCF",
        "SVC & QLR & GMM - min DCF"
    ]

    save_bayes_err_plot(scores, labels, "bayes_error_joint2")

    # store bayes plot of single + fusion but with GMM
    scores = [
        dcf_svc,
        dcf_gmm,
        dcf_svc_gmm_qlr,
        min_dcf_svc_gmm_qlr
    ]

    labels = [
        "SVC  - act DCF",
        "GMM - act DCF",
        "SVC & QLR & GMM - act DCF",
        "SVC & QLR & GMM - min DCF"
    ]

    save_bayes_err_plot(scores, labels, "bayes_error_joint3")
