"""
Evaluate the scores and calibrated scores for the
best 3 models and their fusions
"""
import numpy as np
from tqdm import tqdm

from mlprlib.model_selection import (
    kfold_calibrate,
    k_fold_joint_eval
)

from mlprlib.metrics import detection_cost_fun
from mlprlib.dataset import (
    load_wine_train
)

# import the 3 best performing models
from mlprlib.svm import SVClassifier
from mlprlib.gaussian import GaussianMixture
from mlprlib.logistic import QuadLogisticRegression

# standard scaler for preprocessing
from mlprlib.preprocessing import StandardScaler
# cross validation
from mlprlib.model_selection import CrossValidator

# list of lambdas used for the LR
l_list = [0, 1e-6, 1e-4, 1e-1, 1]


def scores_kfold_calibrate(llr, y, n_folds=5, pi=.5, cfn=1, cfp=1):
    act_dcf = detection_cost_fun(llr, y, pi, cfn, cfp)

    best_lambda = 0
    act_dcf_est = 0.
    min_dcf_cal = 1.
    for _lambda in l_list:
        just_cal = _lambda != l_list[-1]
        act_dcf_cal, act_dcf_est = kfold_calibrate(llr, y, _lambda,
                                                   n_folds, pi, cfn, cfp, just_cal=just_cal)

        if act_dcf_cal < min_dcf_cal:
            min_dcf_cal = act_dcf_cal
            best_lambda = _lambda

    print("actual dcf: %s" % act_dcf)
    print("act dcf calibrated (LR): %s with best lambda: %s" % (min_dcf_cal, best_lambda))
    print("actual dcf (with estimated threshold): %s\n" % act_dcf_est)


def scores_kfold_fusion(scores_list, y, n_folds):
    min_dcf_ = 1.
    min_act_dcf = 1.
    best_lambda = 0

    for _lambda in l_list:
        min_dcf, act_dcf = k_fold_joint_eval(
            *scores_list, y=y, n_folds=n_folds, l=_lambda
        )
        if min_dcf < min_dcf_:
            min_dcf_ = min_dcf
            min_act_dcf = act_dcf
            best_lambda = _lambda

    print("min dcf: %s" % min_dcf_)
    print("act dcf: %s" % min_act_dcf)
    print("best lambda: %s" % best_lambda)


if __name__ == '__main__':
    n_folds = 5
    X, y = load_wine_train(feats_first=False)

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
    transformers = [StandardScaler()]

    for model, name in models.items():
        # progress_bar.set_description(
        #     "MODEL: %s" % type(model).__name__
        # )
        # cross validator
        print(name)
        print("----")
        cv = CrossValidator(n_folds=5)
        cv.fit(X, y, model, transformers)
        scores = cv.scores

        np.save("../results/llr%s.npy" % name, scores)
        scores_kfold_calibrate(scores, y, n_folds)

    llr_svm = np.load("../results/llrSVC.npy")
    llr_qlr = np.load("../results/llrQLR.npy")
    llr_gmm = np.load("../results/llrGMM.npy")

    # SVM + QLR
    scores_kfold_fusion([llr_svm, llr_qlr], y, n_folds)
    # SVM + GMM
    scores_kfold_fusion([llr_svm, llr_gmm], y, n_folds)
    # SVM + GMM + QLR
    scores_kfold_fusion([llr_svm, llr_qlr, llr_gmm], y, n_folds)


