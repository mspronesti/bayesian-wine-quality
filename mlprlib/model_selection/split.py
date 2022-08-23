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


def k_fold_indices(X,
                   n_folds: int = 5,
                   *,
                   shuffle: bool = False,
                   seed: int = 0):
    """
    Retrieves the indices for the K-fold
    cross validation, according to the chosen
    `n_fold` parameter

    Parameters
    ----------
    X: ndarray, data matrix of samples
        in the shape
            (n_samples, n_feats)

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
        a generator of indexes,
        split into `n_folds`
    """
    if n_folds <= 1:
        raise ValueError(
            "k-fold cross-validation requires at least one"
            " train/test split by setting n_splits=2 or more,"
            " got n_splits=%s." % n_folds
        )

    if shuffle:
        np.random.seed(seed)
        idx = np.random.permutation(X.shape[0])
    else:
        idx = np.arange(X.shape[0])
    # split indices in n_folds groups
    # (almost) equally
    chunks = np.array_split(idx, n_folds)
    for i in range(n_folds):
        yield (
            np.hstack([chunks[j] for j in range(n_folds) if j != i]),
            chunks[i]
        )


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
    for idx_train, idx_test in k_fold_indices(X, n_folds, shuffle=shuffle, seed=seed):
        X_train, X_test = X[idx_train], X[idx_test]
        y_train, y_test = y[idx_train], y[idx_test]
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


class KFold:
    def __init__(self,
                 n_folds: int = 5,
                 *,
                 shuffle: bool = False,
                 seed: int = 0
                 ):
        """
        Constructs a KFold object

        Parameters
        ----------
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
        """
        if n_folds <= 1:
            raise ValueError(
                "k-fold cross-validation requires at least one"
                " train/test split by setting n_splits=2 or more,"
                " got n_splits=%s." % n_folds
            )

        self.n_folds = n_folds
        self.shuffle = shuffle
        self.seed = seed

    def split(self, X: np.ndarray):
        """
        Generate indices to split data into training and test set.

        Parameters
        ----------
        X:
            ndarray of samples, of shape
                (n_samples, n_features)

        Returns
        -------

        """
        return k_fold_indices(X,
                              self.n_folds,
                              shuffle=self.shuffle,
                              seed=self.seed)
