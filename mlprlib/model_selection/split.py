import numpy as np


def train_test_split(X, y, train_size=.8, seed=0):
    """
    Splits the dataset in two chunks for training
    and testing according to the `train_size` parameter

    Parameters
    ----------
    X: ndarray, training samples
    y: ndarray, training labels
    train_size: percentage of samples used
                for test or validation
    seed: random seed for the split

    Returns
    -------
        X_train: training samples
        X_test: test samples
        y_train: training labels
        y_test: test labels
    """
    np.random.seed(seed)
    n_train = int(X.shape[0] * train_size)

    idx = np.random.permutation(X.shape[0])
    idx_train = idx[0:n_train]
    idx_test = idx[n_train:]

    return X[idx_train, :], X[idx_test, :], y[idx_train], y[idx_test]


def k_fold_split(X, y, n_folds, seed=0):
    """
    Splits the input dataset in `n_folds` splits

    Parameters
    ----------
    X: ndarray samples
    y: ndarray targets
    n_folds: int
    seed: random seed

    Returns
    -------

    """
    np.random.seed(seed)
    idx = np.random.permutation(X.shape[0])
    chunks = np.array_split(idx, n_folds)
    for i in range(n_folds):
        x_tr = np.vstack([X[el] for j, el in enumerate(chunks) if j != i])
        y_tr = np.hstack([y[el] for j, el in enumerate(chunks) if j != i])
        x_ts = X[chunks[i]]
        y_ts = y[chunks[i]]
        yield x_tr, y_tr, x_ts, y_ts


def leave_one_out_split(X, y, seed=0):
    """
    Splits the input dataset using the LOOCV
    approach, i.e. creating folds leaving only one
    sample in the validation set

    Parameters
    ----------
    X: ndarray of samples
    y: ndarray of labels
    seed: random seed

    Returns
    -------

    """
    return k_fold_split(X, y, len(y), seed)
