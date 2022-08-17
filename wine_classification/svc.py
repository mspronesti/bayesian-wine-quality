"""
Using the Support Vector Classifier (SVC) with linear, quadratic and
Gaussian (RBF) kernels with a 5-fold cross validation approach to
classify wines and analyze the performances these methods yield.
"""
import numpy as np

from mlprlib.dataset import (
    load_wine_train
)

from mlprlib.model_selection import (
    train_test_split,
    k_fold_split
)

from mlprlib.preprocessing import standardize
from mlprlib.metrics import min_detection_cost_fun

from mlprlib.utils import Writer
from mlprlib.svm import SVClassifier

# list of C values for the support vector classifier
C_list = [.1, 1., 10.]


def split_data_svm(writer, kernel: str, X, y, **kwargs):
    """
    Uses SVC on Raw Data using the given
     kernel and searching on a list of `C`
     for the following values:
         [.1, 1, 10]
    """
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.2)
    for c in C_list:
        svc = SVClassifier(C=c, kernel=kernel, **kwargs)
        svc.fit(X_train, y_train)
        _, score = svc.predict(X_val, return_proba=True)
        min_dcf, _ = min_detection_cost_fun(score, y_val)
        writer("C: %s | %f" % (c, min_dcf))


if __name__ == '__main__':
    # number of folds for cross validation
    n_folds = 5

    # load the dataset in the shape (n_samples, n_feats)
    X, y = load_wine_train(feats_first=False)
    X_gauss = np.load('../results/gaussian_feats.npy').T
    X_std = standardize(X)

    writer = Writer("../results/svc_results.txt")
    writer("Train-Test Split")
    writer("----------------")

    # linear svm on raw, gauss and std data
    writer("kernel : linear")
    writer("----------------")
    writer("Raw data")
    split_data_svm(writer, 'linear', X, y)
    writer("Gaussianized data")
    split_data_svm(writer, 'linear', X_gauss, y)
    writer("Standardized data")
    split_data_svm(writer, 'linear', X_std, y)

    # polynomial svm on std data
    writer("----------------")
    writer("kernel : poly")
    writer("----------------")
    writer("Standardized data")
    split_data_svm(writer, 'poly', X_std, y, gamma=1, degree=2, coef=1)

    # RBF svm on std data
    writer("----------------")
    writer("kernel : RBF")
    writer("----------------")
    writer("Standardized data")
    # log gamma = -2
    split_data_svm(writer, 'rbf', X_gauss, y, gamma=np.exp(-2), csi=1)

    writer.destroy()
