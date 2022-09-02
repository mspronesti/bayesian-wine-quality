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
    CrossValidator,
    KFold
)

from mlprlib.preprocessing import (
    standardize,
    StandardScaler,
    GaussianScaler,
)

from mlprlib.metrics import min_detection_cost_fun

from mlprlib.utils import Writer
from mlprlib.gaussian import (
    GaussianClassifier,
    TiedGaussian,
    NaiveBayes,
    TiedNaiveBayes
)

from mlprlib.reduction import PCA


def gaussian_classifiers_eval(
        writer, models,
        X_train, y_train,
        X_test, y_test,
        *,
        use_pca=False,
        # ignored if "use_pace" is False
        n_components=None
):
    """This function is used both for evaluation
    and to evaluate a train-test split"""
    if use_pca:
        pca = PCA(n_components=n_components)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

    progress_bar = tqdm(models)
    for model in progress_bar:
        progress_bar.set_description(
            "%s | single split" % type(model).__name__
        )
        model.fit(X_train, y_train)
        _, score = model.predict(X_test, return_proba=True)
        min_dcf, _ = min_detection_cost_fun(score, y_test)
        writer("model: %s \t| dcf: %f"
               % (type(model).__name__, min_dcf))


def single_split_gauss(writer, models, X, y,
                       *,
                       use_pca=False,
                       # ignored if "use_pace" is False
                       n_components=None
                       ):
    """
    Splits given data in train and validation set and
     uses the Gaussian classifiers on it, using the given
     covariance
    """
    writer("----------------")
    writer("Single split")
    writer("----------------")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.2)
    gaussian_classifiers_eval(writer, models, X_train, y_train, X_val, y_val,
                              use_pca=use_pca, n_components=n_components)


def k_fold_gauss(writer, models, X, y,
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

    progress_bar = tqdm(models)
    for model in progress_bar:
        progress_bar.set_description(
            "%s | 5-fold cross val" % type(model).__name__
        )
        cv.fit(X, y, model, transformers)
        # acquire the scores after cv
        scores = cv.scores
        min_dcf, _ = min_detection_cost_fun(scores, y)
        writer("model: %s \t| dcf: %f"
               % (type(model).__name__, min_dcf))


if __name__ == '__main__':
    n_folds = 5

    # load the training dataset in the shape (n_samples, n_feats)
    X, y = load_wine_train(feats_first=False)

    # Gaussianised data
    # (not used for KFold CV)
    X_gauss = np.load('../results/gaussian_feats.npy').T

    models = [
        GaussianClassifier(),
        NaiveBayes(),
        TiedGaussian(),
        TiedNaiveBayes()
    ]

    writer = Writer("../results/gaussian_results.txt")

    writer("----------------")
    writer("Raw data")
    writer("----------------")
    single_split_gauss(writer, models, X, y)
    k_fold_gauss(writer, models, X, y, n_folds)

    writer("\n----------------")
    writer("Gaussian data")
    writer("----------------")
    single_split_gauss(writer, models, X_gauss, y)
    k_fold_gauss(writer, models, X, y, n_folds, gauss=True)

    writer("\n----------------")
    writer("Gaussian data, PCA(n_components=10)")
    writer("----------------")
    single_split_gauss(writer, models, X_gauss, y, use_pca=True, n_components=10)
    k_fold_gauss(writer, models, X, y, gauss=True, use_pca=True, n_components=10)

    writer("\n----------------")
    writer("Gaussian data, PCA(n_components=9)")
    writer("----------------")
    single_split_gauss(writer, models, X_gauss, y, use_pca=True, n_components=9)
    k_fold_gauss(writer, models, X, y, gauss=True, use_pca=True, n_components=9)

    writer.destroy()

    ##########################
    # evaluation on test set
    ##########################
    X_test, y_test = load_wine_test(feats_first=False)
    gs = GaussianScaler().fit(X)

    X_gauss = gs.transform(X)
    # transform using X (as reference) as reference
    # i.e. the scaler is already fitted here!
    X_test_gauss = gs.transform(X_test)

    writer = Writer("../results/gaussian_results_eval.txt")
    writer("----------------")
    writer("Raw data")
    writer("----------------")
    gaussian_classifiers_eval(writer, models, X, y, X_test, y_test)

    writer("\n----------------")
    writer("Gaussian data")
    writer("----------------")
    gaussian_classifiers_eval(writer, models, X_gauss, y, X_test_gauss, y_test)

    writer("\n----------------")
    writer("Gaussian data, PCA(n_components=10)")
    writer("----------------")
    gaussian_classifiers_eval(writer, models, X_gauss, y, X_test_gauss, y_test,
                              use_pca=True, n_components=10)

    writer("\n----------------")
    writer("Gaussian data, PCA(n_components=9)")
    writer("----------------")
    gaussian_classifiers_eval(writer, models, X_gauss, y, X_test_gauss, y_test,
                              use_pca=True, n_components=9)

    writer.destroy()

