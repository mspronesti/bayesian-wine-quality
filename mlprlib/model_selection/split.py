import numpy as np


def train_test_split(X, y, test_size: float = .25, seed: int = 0):
    """
    Splits the dataset in two chunks for training
    and testing according to the `train_size` parameter

    Parameters
    ----------
    X: ndarray, data matrix of samples
        in the shape
            (n_samples, n_feats)

    y:
        ndarray, training labels

    test_size: float, percentage of samples used
                for test or validation

    seed:
        int, random seed for the split

    Returns
    -------
        X_train: training samples
        X_test: test samples
        y_train: training labels
        y_test: test labels
    """
    np.random.seed(seed)
    n_train = int(X.shape[0] * (1. - test_size))

    idx = np.random.permutation(X.shape[0])
    idx_train = idx[0:n_train]
    idx_test = idx[n_train:]

    return X[idx_train, :], X[idx_test, :], y[idx_train], y[idx_test]


def k_fold_split(X, y,
                 n_folds: int = 5,
                 *,
                 shuffle: bool = False,
                 seed: int = 0):
    """
    Splits the input dataset in `n_folds` splits

    Parameters
    ----------
    X: ndarray, data matrix of samples
        in the shape
            (n_samples, n_feats)

    y: ndarray targets

    n_folds:
        int, number of folds to use for the cross
        validation. Default 5

    shuffle:
        bool, whether to shuffle the indices before
        the k_fold split.
        Default False

    seed: int, random seed.
        Ignored if shuffle is False.
        Default 0

    Returns
    -------
        a generator of data samples and labels
         split into `n_folds`
    """
    if shuffle:
        np.random.seed(seed)
        idx = np.random.permutation(X.shape[0])
    else:
        idx = np.arange(X.shape[0])
    # split indices in n_folds groups
    # (almost) equally
    chunks = np.array_split(idx, n_folds)
    for i in range(n_folds):
        # i-th fold is for test,
        # the remaining n_folds-1 are for training
        X_train = np.vstack([X[el] for idx, el in enumerate(chunks) if idx != i])
        y_train = np.hstack([y[el] for idx, el in enumerate(chunks) if idx != i])
        X_test = X[chunks[i]]
        y_test = y[chunks[i]]
        yield X_train, y_train, X_test, y_test


def leave_one_out_split(X, y,
                        *,
                        shuffle: bool = False,
                        seed: int = 0):
    """
    Splits the input dataset using the LOOCV
    approach, i.e. creating folds leaving only one
    sample in the validation set

    Parameters
    ----------
    X: ndarray, data matrix of samples
        in the shape
            (n_samples, n_feats)

    y:
        ndarray of labels

    shuffle:
        bool, whether to shuffle the indices before
        the k_fold split.
        Default False

    seed: int, random seed.
        Ignored if shuffle is False.
        Default 0

    Returns
    -------
        a generator of data samples and labels
         split into `n_folds`
    """
    return k_fold_split(X, y, len(y), shuffle=shuffle, seed=seed)
