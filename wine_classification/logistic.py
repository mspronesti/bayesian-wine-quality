"""
Using the Linear Logistic Regression with linear and quadratic
kernels with a 5-fold cross validation approach to
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

from mlprlib.preprocessing import standardize, StandardScaler, GaussianScaler
from mlprlib.metrics import min_detection_cost_fun
from mlprlib.reduction import PCA

from mlprlib.utils import Writer
from mlprlib.logistic import (
    LogisticRegression,
    QuadLogisticRegression
)

l_list = [0, 1e-6, 1e-4, 1e-2, 1, 100]


def lr_lambda_search(writer, lr_t: str, data_t: str,
            X_train, y_train, X_test, y_test):
    lr = QuadLogisticRegression() if lr_t == 'quadratic' \
        else LogisticRegression()

    progress_bar = tqdm(l_list)
    for l in progress_bar:
        progress_bar.set_description(
            "LR %s | lambda: %f | Data %s | single split" % (lr_t, l, data_t)
        )
        # set l_scaler
        lr.l_scaler = l
        # fit and evaluate score and minDCF
        lr.fit(X_train, y_train)
        _, score = lr.predict(X_test, return_proba=True)
        min_dcf, _ = min_detection_cost_fun(score, y_test)
        writer("lambda: %s | %f" % (l, min_dcf))


def split_data_lr(writer, lr_t: 'str', data_t: 'str', X, y):
    """
    Applies a Linear or Quadratic Logistic Regression
    to the wine dataset after a single split
    for a set of  lambdas:

        lambdas = [0, 1e-6, 1e-4, 1e-2, 1, 100]
    """
    writer("----------------")
    writer(" Single split ")
    writer("----------------")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.2)
    lr_lambda_search(writer, lr_t, data_t, X_train, y_train, X_val, y_val)


def k_fold_lr(writer, lr_t: str,
              data_t: str,
              X, y,
              n_folds=5,
              *,
              std=False,
              gauss=False,
              use_pca=False,
              # ignored if "use_pace" is False
              n_components=None):
    writer("----------------")
    writer("5 fold cross validation")
    writer("----------------")

    # define a cross validator object
    # which is going to use the KFold
    # class and apply a pipeline of transformation
    # on data after the split
    cv = CrossValidator(n_folds=n_folds)

    # define the pipeline of transformers
    # we want to apply for the CV
    transformers = []
    if std:
        transformers.append(StandardScaler())
    elif gauss:
        transformers.append(GaussianScaler())

    if use_pca:
        transformers.append(PCA(n_components=n_components))

    lr = QuadLogisticRegression() if lr_t == 'quadratic' \
        else LogisticRegression()

    progress_bar = tqdm(l_list)
    for l in progress_bar:
        progress_bar.set_description(
            "LR %s | lambda: %f | Data %s | k fold" % (lr_t, l, data_t)
        )
        # set l_scaler
        lr.l_scaler = l
        cv.fit(X, y, lr, transformers)
        scores = cv.scores
        min_dcf, _ = min_detection_cost_fun(scores, y)
        writer("lambda: %s | %f" % (l, min_dcf))


if __name__ == '__main__':
    # number of folds for cross validation
    n_folds = 5

    # load the dataset in the shape (n_samples, n_feats)
    X, y = load_wine_train(feats_first=False)
    X_gauss = np.load('../results/gaussian_feats.npy').T
    X_std = standardize(X)

    # writing LR results
    writer = Writer("../results/lr_results.txt")

    writer("----------------")
    writer("LR Type : linear")
    writer("----------------")
    writer("Raw data")
    split_data_lr(writer, 'linear', 'raw', X, y)

    k_fold_lr(writer, 'linear', 'raw', X, y, n_folds)
    writer("Gaussianized data")
    split_data_lr(writer, 'linear', 'gauss', X_gauss, y)
    k_fold_lr(writer, 'linear', 'gauss', X, y, n_folds, gauss=True)

    writer("Standardized data")
    split_data_lr(writer, 'linear', 'std', X_std, y)
    k_fold_lr(writer, 'linear', 'std', X, y, n_folds, std=True)

    writer("\n----------------")
    writer("LR type : quadratic")
    writer("----------------")
    writer("Raw data")
    split_data_lr(writer, 'quadratic', 'raw', X, y)
    k_fold_lr(writer, 'quadratic', 'raw', X, y, n_folds)

    writer("Gaussianized data")
    split_data_lr(writer, 'quadratic', 'gauss', X_gauss, y)
    k_fold_lr(writer, 'quadratic', 'gauss', X, y, n_folds, gauss=True)

    writer("Standardized data")
    split_data_lr(writer, 'quadratic', 'std', X_std, y)
    k_fold_lr(writer, 'quadratic', 'std', X, y, n_folds, std=True)

    writer.destroy()

    # load the test data in the shape (n_samples, n_feats)
    X_test, y_test = load_wine_test(feats_first=False)
    # standardize test data using training data mean and variance
    X_test_std = standardize(X_test, X.mean(axis=0), X.std(axis=0))

    # Gaussianised data
    X_gauss = np.load('../results/gaussian_feats.npy').T
    X_test_gauss = np.load('../results/gaussian_feats_test.npy').T

