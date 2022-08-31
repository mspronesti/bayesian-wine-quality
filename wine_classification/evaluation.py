import numpy as np

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

from mlprlib.metrics import detection_cost_fun

from mlprlib.utils import Writer

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
    lambdas_joint = [.001, .001, 0]

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
    min_dcf, act_dcf, _ = joint_eval(*[llr_svc, llr_svc_eval,
                                       llr_qlr, llr_qlr_eval],
                                     y_train=y_train,
                                     y_test=y_test,
                                     l=lambdas_joint[0],
                                     pi=pi, cfn=cfn, cfp=cfp)

    writer("min dcf: %s \nactual dcf: %s" % (min_dcf, act_dcf))

    # SVC + GMM
    writer("\n\nSVC + GMM")
    writer("--------------")
    min_dcf, act_dcf, _ = joint_eval(*[llr_svc, llr_svc_eval,
                                       llr_gmm, llr_gmm_eval],
                                     y_train=y_train,
                                     y_test=y_test,
                                     l=lambdas_joint[1],
                                     pi=pi, cfn=cfn, cfp=cfp)

    writer("min dcf: %s\n actual dcf: %s" % (min_dcf, act_dcf))

    # SVC + GMM + QLR
    writer("\n\nSVC + GMM + QLR")
    writer("--------------")
    min_dcf, act_dcf, _ = joint_eval(*[llr_svc, llr_svc_eval,
                                       llr_qlr, llr_qlr_eval,
                                       llr_gmm, llr_gmm_eval],
                                     y_train=y_train,
                                     y_test=y_test,
                                     l=lambdas_joint[2],
                                     pi=pi, cfn=cfn, cfp=cfp)

    writer("min dcf: %s\n actual dcf: %s" % (min_dcf, act_dcf))
    writer.destroy()
