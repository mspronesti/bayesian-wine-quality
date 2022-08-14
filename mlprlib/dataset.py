import numpy as np

features = ["Fixed acidity",
            "Volatile acidity",
            "Citric acidity",
            "Residual sugar",
            "Chlorides",
            "Free sulfur dioxide",
            "Total sulfur dioxide",
            "Density",
            "pH",
            "Sulphates",
            "Alcohol"]

classes = ["Bad quality", "Good quality"]

n_feats = len(features)
n_classes = len(classes)


def load_from(file_path: str, *, feats_first: bool = True):
    """
    Loads dataset from a .txt file formatted as follows

        <feat11> ... <feat1D> <label1>
        <feat21> ... <feat2D> <label2>
        ...
        <featN1> ... <feat ND> <labelN>

    Where N is the number of samples and D is the number
    of features.

    Parameters
    ----------
    file_path:
        string, the path of the txt file

    feats_first:
        if True, the returned data array is a
        ndarray of shape

            (n_feats, n_samples)

        i.e. [[feat11 feat21 ... featN1]
               ...
              [featD1 featD2 ... featDN]

        where row `i` contains the value of
        the i-th feature each sample.

        Otherwise, retrieves a ndarray of shape

            (n_samples, n_feats)

        where row `i` contains the features of the i-th
        samples. This is the usual way datasets are loaded.

    Returns
    -------
        an ndarray of size (n_feats, n_samples) or
        (n_samples, n_feats), depending on feats_first
    """
    data = np.loadtxt(
        file_path,
        delimiter=","
    )

    # for readability only
    # extract all columns but
    # the last one
    samples = data[:, :-1]
    # extract the last column
    # as integer
    labels = data[:, -1].astype(np.int32)

    if feats_first:
        return samples.T, labels
    else:
        return samples, labels


def load_wine_train(*, feats_first=True):
    return load_from('../data/Train.txt', feats_first=feats_first)


def load_wine_test(*, feats_first=True):
    return load_from('../data/Test.txt', feats_first=feats_first)
