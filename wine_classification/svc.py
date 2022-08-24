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
    KFold
)

from mlprlib.preprocessing import standardize
from mlprlib.metrics import min_detection_cost_fun

from mlprlib.utils import Writer
from mlprlib.svm import SVClassifier

# list of C values for the support vector classifier
C_list = [.1, 1., 10.]


def single_split_svm(writer, kernel: str, X, y, **kwargs):
    """
    Splits given data in train and validation set and
     uses the Support Vector Classifier on it, using the given
     kernel and searching on a list of `C`

     for the following values:
         [.1, 1, 10]
    """
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.2)
    for c in tqdm(C_list, desc="KERNEL %s" % kernel):
        svc = SVClassifier(C=c, kernel=kernel, **kwargs)
        svc.fit(X_train, y_train)
        _, score = svc.predict(X_val, return_proba=True)
        min_dcf, _ = min_detection_cost_fun(score, y_val)
        writer("C: %s | %f" % (c, min_dcf))


def k_fold_svm(writer, k, kernel: str, X, y, **kwargs):
    scores = np.empty((0,), dtype=np.float64)
    # labels = np.empty((0,), int)

    svc = SVClassifier(C=.1, kernel=kernel, **kwargs)

    kf = KFold(k, shuffle=True)
    # kf.split(.) is a generator, producing the folds
    for idx_train, idx_test in kf.split(X):
        X_train, X_val = X[idx_train], X[idx_test]
        y_train, y_val = y[idx_train], y[idx_test]
        svc.fit(X_train, y_train)
        _, score = svc.predict(X_val, return_proba=True)
        # evaluate score and store labels used for
        # the samples of that ith fold
        scores = np.append(scores, score)
        # labels = np.append(labels, y_val)

    min_dcf, _ = min_detection_cost_fun(scores, y)
    print(min_dcf)
    # np.random.seed(0)
    # idx = np.random.permutation(X.shape[0])
    # k = n_folds
    # start_index = 0
    # elements = int(X.shape[0] / k)
    #
    # llr = np.zeros([X.shape[0], ])
    #
    # svc = SVClassifier(C=.1, kernel=kernel, **kwargs)
    # for _ in range(k):
    #     # Define training and test partitions
    #     if start_index + elements > X.shape[0]:
    #         end_index = X.shape[0]
    #     else:
    #         end_index = start_index + elements
    #
    #     idxTrain = np.concatenate((idx[0:start_index], idx[end_index:]))
    #     idxTest = idx[start_index:end_index]
    #
    #     DTR = X[idxTrain, :]
    #     LTR = y[idxTrain]
    #
    #     DTE = X[idxTest, :]
    #     # LTE = y[idxTest]
    #
    #     svc.fit(DTR, LTR)
    #     _, llr[idxTest] = svc.predict(DTE, return_proba=True)
    #
    #     start_index += elements
    # min_dcf, _ = min_detection_cost_fun(llr, y)
    # print(min_dcf)


def svm_eval(writer, kernel: str, data_t: str, X, y, X_ts, y_ts, **kwargs):
    """
    Uses SVC on test data using the given
     kernel and searching on a list of `C`
     for the following values:
         [.1, 1, 10]
    """
    for c in tqdm(C_list, desc="KERNEL %s, data %s" % (kernel, data_t)):
        svc = SVClassifier(C=c, kernel=kernel, **kwargs)
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
    # writer("Train-Test Split")
    # writer("----------------")
    #
    # # linear svm on raw, gauss and std data
    # writer("kernel : linear")
    # writer("----------------")
    # writer("Raw data")
    # single_split_svm(writer, 'linear', X, y)
    # writer("Gaussianized data")
    # single_split_svm(writer, 'linear', X_gauss, y)
    # writer("Standardized data")
    # single_split_svm(writer, 'linear', X_std, y)
    #
    # # polynomial svm on std data
    # writer("----------------")
    # writer("kernel : poly")
    # writer("----------------")
    # writer("Standardized data")
    # single_split_svm(writer, 'poly', X_std, y, gamma=1, degree=2, coef=1)
    #
    # # RBF svm on std data
    # writer("----------------")
    # writer("kernel : RBF")
    # writer("----------------")
    # writer("Standardized data")
    # # log gamma = -2
    # single_split_svm(writer, 'rbf', X_gauss, y, gamma=np.exp(-2), csi=1)
    #
    # writer.destroy()

    writer = Writer("../results/svc_results_eval.txt")
    writer("kernel : linear")
    writer("----------------")
    writer("Raw data")
    svm_eval(writer, 'linear', 'raw', X, y, X_test, y_test)
    writer("Gaussianized data")
    svm_eval(writer, 'linear', 'gauss', X_gauss, y, X_test_gauss, y_test)
    writer("Standardized data")
    svm_eval(writer, 'linear', 'std', X_std, y, X_test_std, y_test)
    writer.destroy()


