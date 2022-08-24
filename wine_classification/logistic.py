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
    k_fold_split
)

from mlprlib.preprocessing import standardize
from mlprlib.metrics import min_detection_cost_fun

from mlprlib.utils import Writer
from mlprlib.logistic import (
    LogisticRegression,
    QuadLogisticRegression
)

l_list = [0, 1e-6, 1e-4, 1e-2, 1, 100]


def split_data_lr(writer, lr_t: 'str', data_t: 'str', X, y):
    """
    Applies a Linear or Quadratic Logistic Regression
    to the wine dataset after a single split
    for a set of  lambdas:

        lambdas = [0, 1e-6, 1e-4, 1e-2, 1, 100]
    """
    lr = QuadLogisticRegression() if lr_t == 'quadratic' \
        else LogisticRegression()

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.2)
    for l in tqdm(l_list, desc="LR %s | Data %s" % (lr_t, data_t)):
        # set l_scaler
        lr.l_scaler = l
        # fit and evaluate score and minDCF
        lr.fit(X_train, y_train)
        _, score = lr.predict(X_val, return_proba=True)
        min_dcf, _ = min_detection_cost_fun(score, y_val)
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
    writer("Gaussianized data")
    split_data_lr(writer, 'linear', 'gauss', X_gauss, y)
    writer("Standardized data")
    split_data_lr(writer, 'linear', 'std', X_std, y)

    writer("\n----------------")
    writer("LR type : quadratic")
    writer("----------------")
    writer("Raw data")
    split_data_lr(writer, 'quadratic', 'raw', X, y)
    writer("Gaussianized data")
    split_data_lr(writer, 'quadratic', 'gauss', X_gauss, y)
    writer("Standardized data")
    split_data_lr(writer, 'quadratic', 'std', X_std, y)

    writer.destroy()

    # load the test data in the shape (n_samples, n_feats)
    X_test, y_test = load_wine_test(feats_first=False)
    # standardize test data using training data mean and variance
    X_test_std = standardize(X_test, X.mean(axis=0), X.std(axis=0))

    # Gaussianised data
    X_gauss = np.load('../results/gaussian_feats.npy').T
    X_test_gauss = np.load('../results/gaussian_feats_test.npy').T

    # write LR results for test
    # TODO: to be written
