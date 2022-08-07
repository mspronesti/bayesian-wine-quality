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


def load_from(file_path: str, *, return_X_y=True):
    data = np.loadtxt(file_path, delimiter=",")
    # for readability only
    samples = data[:, :-1]
    labels = data[:, -1]

    if return_X_y:
        return samples, labels
    else:
        return data


def load_wine_train(*, return_X_y=True):
    return load_from('../data/Train.txt', return_X_y=return_X_y)


def load_wine_test(*, return_X_y=True):
    return load_from('../data/Test.txt', return_X_y=return_X_y)
