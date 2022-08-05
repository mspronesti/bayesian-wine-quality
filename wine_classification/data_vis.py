import matplotlib.pyplot as plt
import numpy as np
from mlprlib.dataset import load_wine, classes, features


def data_histogram (X, y, label: str = ""):

    pass


if __name__ == '__main__':
    train_dataset_path = 'data/Train.txt'
    test_dataset_path = 'data/Test.txt'

    X_train, y_train = load_wine(train_dataset_path)
    X_test, y_test = load_wine(test_dataset_path)

    print("ones are %d" % np.sum(y_train))
    print("zeros are %d" % (len(y_train) - np.sum(y_train)))
