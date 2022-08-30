"""
Using the Support Vector Classifier (SVC) with linear, quadratic and
Gaussian (RBF) kernels with a 5-fold cross validation approach to
classify wines and analyze the performances these methods yield.
"""
import numpy as np
from tqdm import tqdm

from mlprlib.dataset import (
    load_wine_train,
    load_wine_test
)

from mlprlib.model_selection import (
    train_test_split,
    CrossValidator
)

from mlprlib.preprocessing import (
    standardize,
    StandardScaler,
    GaussianScaler
)
from mlprlib.metrics import min_detection_cost_fun

from mlprlib.utils import Writer
from mlprlib.svm import SVClassifier

# list of C values for the support vector classifier
C_list = [.1, 1., 10.]


# def balance_bounds_svc(X, y, C, pi_t=.5):
#     """
#     Retrieves the bounds to be used in
#     the L-BFGS-B optimization algorithm
#     in the Support Vector classifier to
#     balance classes
#     """
#     # number of true samples
#     nt = sum(y == 1) / y.shape[0]
#     # init bounds
#     # the expected shape is
#     # (n_samples, 2) where we want to have
#     # [0, C0]
#     # [0, C1]
#     #  ...
#     # [0,  Cn]
#     # being Ci = Ct if yi == 1 else Cf
#     bounds = np.zeros([X.shape[0], 2])
#     Ct = C * pi_t / nt
#     Cf = C * (1 - pi_t) / (1 - nt)
#     bounds[y == 1, 1] = Ct
#     bounds[y == 0, 1] = Cf
#     return bounds


def single_split_svm(writer, kernel: str, X, y, **kwargs):
    """
    Splits given data in train and validation set and
     uses the Support Vector Classifier on it, using the given
     kernel and searching on a list of `C`

     for the following values:
         [.1, 1, 10]
    """
    writer("Single Split")
    writer("----------------")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.2)

    progress_bar = tqdm(C_list)
    for c in progress_bar:
        for pi_t in [None, .5]:
            # if pi_T is None, don't re-balance
            balance = pi_t is not None
            progress_bar.set_description(
                "KERNEL %s | C: %f | balance: %s | single split"
                % (kernel, c, balance)
            )
            svc = SVClassifier(C=c, kernel=kernel, pi_t=pi_t, **kwargs)
            svc.fit(X_train, y_train)

            _, score = svc.predict(X_val, return_proba=True)
            min_dcf, _ = min_detection_cost_fun(score, y_val)
            writer("C: %s | %f | balance %s" % (c, min_dcf, balance))


def k_fold_svm(writer, k, kernel: str,
               gauss: bool,
               std: bool,
               X, y, **kwargs):

    writer("5-fold cross validation")
    writer("----------------")

    transformers = []
    if gauss:
        transformers.append(GaussianScaler())
    elif std:
        transformers.append(StandardScaler())

    cv = CrossValidator(n_folds=k)
    progress_bar = tqdm(C_list)
    for c in progress_bar:
        for pi_t in [None, .5]:
            # if pi_T is None, don't re-balance
            balance = pi_t is not None
            progress_bar.set_description(
                "KERNEL %s | C: %f | balance: %s | 5-fold"
                % (kernel, c, balance)
            )
            svc = SVClassifier(C=c, kernel=kernel, pi_t=pi_t, **kwargs)
            cv.fit(X, y, svc, transformers)
            scores = cv.scores

            min_dcf, _ = min_detection_cost_fun(scores, y)
            writer("C: %s | %f | balance %s" % (c, min_dcf, balance))


def svm_eval(writer, kernel: str, data_t: str,
             X, y, X_ts, y_ts, **kwargs):
    """
    Uses SVC on test data using the given
     kernel and searching on a list of `C`
     for the following values:
         [.1, 1, 10]
    """
    progress_bar = tqdm(C_list)
    for c in progress_bar:
        for pi_t in [None, .5]:
            balance = pi_t is not None
            progress_bar.set_description(
                "KERNEL: %s | C: %s | data: %s | balance: %s"
                % (kernel, c, data_t, balance)
            )
            svc = SVClassifier(C=c, kernel=kernel, pi_t=pi_t, **kwargs)
            svc.fit(X, y)
            _, score = svc.predict(X_ts, return_proba=True)

            min_dcf, _ = min_detection_cost_fun(score, y_ts)
            writer("C: %s | %f" % (c, min_dcf))


if __name__ == '__main__':
    # number of folds for cross validation
    n_folds = 5

    # load the training dataset in the shape (n_samples, n_feats)
    X, y = load_wine_train(feats_first=False)
    # load the test data in the shape (n_samples, n_feats)
    X_test, y_test = load_wine_test(feats_first=False)

    # Standardized data
    X_std = standardize(X)
    # standardize test data using training data mean and variance
    X_test_std = standardize(X_test, X.mean(axis=0), X.std(axis=0))

    # Gaussianised data
    X_gauss = np.load('../results/gaussian_feats.npy').T
    X_test_gauss = np.load('../results/gaussian_feats_test.npy').T

    # writer = Writer("../results/svc_results.txt")
    #
    # # try the 3 kernels with different
    # # combinations of data both with and without
    # # balancing features
    # writer("kernel: linear")
    # writer("----------------")
    # # linear kernel on raw, gaussianized and std data
    # # with and without feats balancing, with 5-folds and single split
    # writer("Raw data")
    # single_split_svm(writer, 'linear', X, y)
    # k_fold_svm(writer, n_folds, 'linear', gauss=False, std=False, X=X, y=y)
    #
    # writer("\nGaussianized data")
    # single_split_svm(writer, 'linear', X, y)
    # k_fold_svm(writer, n_folds, 'linear', gauss=True, std=False, X=X, y=y)
    #
    # writer("\nStandardized data")
    # single_split_svm(writer, 'linear', X, y)
    # k_fold_svm(writer, n_folds, 'linear', gauss=False, std=True, X=X, y=y)
    #
    # # polynomial svm (quadratic, d=2) on std data
    # # with and without feats balancing, with 5-folds and single split
    # writer("\n\n----------------")
    # writer("kernel : poly (degree=2)")
    # writer("----------------")
    # writer("Standardized data")
    # single_split_svm(writer, 'poly', X_std, y, gamma=1, degree=2, coef=1)
    # k_fold_svm(writer, n_folds, 'poly', gauss=False, std=True, X=X, y=y,
    #            # kwargs for the kernel
    #            gamma=1, degree=2, coef=1)
    #
    # # RBF svm on std data with 2 values of gamma
    # # with and without feats balancing, with 5-folds and single split
    # writer("\n\n----------------")
    # writer("kernel : RBF (log gamma = -1)")
    # writer("----------------")
    # writer("Standardized data")
    # # log gamma = -1
    # single_split_svm(writer, 'rbf', X_gauss, y, gamma=np.exp(-1), csi=1)
    # k_fold_svm(writer, n_folds, 'rbf', gauss=False, std=True,
    #            X=X, y=y, gamma=np.exp(-1), csi=1)
    #
    # # log gamma = -2
    # writer("\n\n----------------")
    # writer("kernel : RBF (log gamma = -2)")
    # writer("----------------")
    # writer("\nStandardized data")
    # single_split_svm(writer, 'rbf', X_gauss, y, gamma=np.exp(-2), csi=1)
    # k_fold_svm(writer, n_folds, 'rbf', gauss=False, std=True, X=X, y=y,
    #            gamma=np.exp(-2), csi=1)
    #
    # writer.destroy()

    #############################
    # Evaluation
    #############################
    writer = Writer("../results/svc_results_eval.txt")
    writer("----------------")
    writer("kernel : linear")
    writer("----------------")
    writer("Raw data")
    svm_eval(writer, 'linear', 'raw', X, y, X_test, y_test)
    writer("\nGaussianized data")
    svm_eval(writer, 'linear', 'gauss', X_gauss, y, X_test_gauss, y_test)
    writer("\nStandardized data")
    svm_eval(writer, 'linear', 'std', X_std, y, X_test_std, y_test)

    writer("\n\n----------------")
    writer("kernel : poly (degree=2)")
    writer("----------------")
    writer("Standardized data")
    svm_eval(writer, 'poly', 'std', X_std, y, X_test_std, y_test,
             # poly kernel kwargs
             gamma=1, degree=2, coef=1)

    writer("\n\n----------------")
    writer("kernel : RBF (log gamma = -2)")
    writer("----------------")
    writer("Standardized data")
    svm_eval(writer, 'rbf', 'std', X_std, y, X_test_std, y_test,
             # poly kernel kwargs
             gamma=np.exp(-2), csi=1)

    writer("\n\n----------------")
    writer("kernel : RBF (log gamma = -2)")
    writer("----------------")
    writer("Standardized data")
    svm_eval(writer, 'rbf', 'std', X_std, y, X_test_std, y_test,
             # poly kernel kwargs
             gamma=np.exp(-1), csi=1)
    writer.destroy()
