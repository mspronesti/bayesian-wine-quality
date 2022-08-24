"""
Using the Gaussian Mixture Model with different number of
components with a 5-fold cross validation approach to
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
from mlprlib.gaussian import GaussianMixture

n_components_list = [2, 4, 8, 16]


def k_fold_gmm(writer, cov_t: str, data_t: str, X, y):
    # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.2)
    for n_comp in tqdm(n_components_list, desc="Cov %s | Data %s" % (cov_t, data_t)):
        scores = np.empty((0,), dtype=np.float64)
        kf = KFold(n_folds, shuffle=False)

        for idx_train, idx_test in kf.split(X):
            X_train, X_val = X[idx_train], X[idx_test]
            y_train, y_val = y[idx_train], y[idx_test]

            gmm = GaussianMixture(n_comp, cov_type=cov_t)
            gmm.fit(X_train, y_train)
            _, score = gmm.predict(X_val, return_proba=True)
            scores = np.append(scores, score)

        # evaluate minDCF
        min_dcf, _ = min_detection_cost_fun(scores, y)
        # gmm = GaussianMixture(n_comp, cov_type=cov_t)
        # np.random.seed(0)
        # idx = np.random.permutation(X.shape[0])
        # k = n_folds
        # start_index = 0
        # elements = int(X.shape[0] / k)
        #
        # llr = np.zeros([X.shape[0], ])
        # for _ in range(n_folds):
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
        #     gmm.fit(DTR, LTR)
        #     _, llr[idxTest] = gmm.predict(DTE, return_proba=True)
        #
        #     start_index += elements
        # min_dcf, _ = min_detection_cost_fun(llr, y)
        writer("n_components: %s | %f" % (n_comp, min_dcf))


if __name__ == '__main__':
    # number of folds for cross validation
    n_folds = 5

    # load the dataset in the shape (n_samples, n_feats)
    X, y = load_wine_train(feats_first=False)
    X_gauss = np.load('../results/gaussian_feats.npy').T
    X_std = standardize(X)

    # writing LR results
    writer = Writer("../results/gmm_results.txt")

    writer("----------------")
    writer("GMM Cov Type : Full")
    writer("----------------")
    writer("Gaussianized data")
    k_fold_gmm(writer, 'full', 'gauss', X_gauss, y)
    writer("Standardized data")
    k_fold_gmm(writer, 'full', 'std', X_std, y)

    writer("\n----------------")
    writer("GMM Cov Type : Diag ")
    writer("----------------")
    writer("Gaussianized data")
    k_fold_gmm(writer, 'diag', 'gauss', X_gauss, y)
    writer("Standardized data")
    k_fold_gmm(writer, 'diag', 'std', X_std, y)

    writer("\n----------------")
    writer("GMM Type : Full-Tied")
    writer("----------------")
    writer("Gaussianized data")
    k_fold_gmm(writer, 'full-tied', 'gauss', X_gauss, y)
    writer("Standardized data")
    k_fold_gmm(writer, 'full-tied', 'std', X_std, y)

    writer("\n----------------")
    writer("GMM Type : Diag-Tied")
    writer("----------------")
    writer("Gaussianized data")
    k_fold_gmm(writer, 'diag-tied', 'gauss', X_gauss, y)
    writer("Standardized data")
    k_fold_gmm(writer, 'diag-tied', 'std', X_std, y)
