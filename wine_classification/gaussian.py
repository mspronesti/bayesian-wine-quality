"""
Using the Gaussian classifiers and Naive Bayes
with a 5-fold cross validation approach to
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
from mlprlib.gaussian import (
    GaussianClassifier,
    TiedGaussian,
    NaiveBayes,
    TiedNaiveBayes
)

from mlprlib.reduction import PCA


if __name__ == '__main__':
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

    gc = GaussianClassifier()
    nb = NaiveBayes()
    tg = TiedGaussian()
    tnb = TiedNaiveBayes()

    X_train, X_val, y_train, y_val = train_test_split(X_gauss, y, test_size=.2)

    pca = PCA(n_components=10).fit(X_gauss)
    X_train = pca.transform(X_train)
    X_val = pca.transform(X_val)

    gc.fit(X_train, y_train)
    nb.fit(X_train, y_train)
    tg.fit(X_train, y_train)
    tnb.fit(X_train, y_train)

    _, score = gc.predict(X_val, return_proba=True)
    min_dcf, _ = min_detection_cost_fun(score, y_val)
    print(min_dcf)

    _, score = nb.predict(X_val, return_proba=True)
    min_dcf, _ = min_detection_cost_fun(score, y_val)
    print(min_dcf)

    _, score = tg.predict(X_val, return_proba=True)
    min_dcf, _ = min_detection_cost_fun(score, y_val)
    print(min_dcf)

    _, score = tnb.predict(X_val, return_proba=True)
    min_dcf, _ = min_detection_cost_fun(score, y_val)
    print(min_dcf)
