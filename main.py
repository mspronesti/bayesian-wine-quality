import numpy as np
from sklearn.ensemble import RandomForestClassifier


n_feats = 11
n_classes = 2

def load_wine(fileName):
    samples = []
    targets = []

    with open(fileName) as f:
        for line in f:
            try:
                line = line.split(",")
                sample = np.array(line[0:n_feats], dtype=float).reshape((n_feats, 1))
                samples.append(sample)
                targets.append(int(line[n_feats]))
            except:
                pass

    data_matrix = np.array(samples).reshape((len(samples), n_feats)).T
    class_labels = np.array(targets)

    return data_matrix, class_labels


if __name__ == '__main__':
    DATASET = 'data/Train.txt'
    X, y = load_wine(DATASET)

    print("ones are %d" % np.sum(y))
    print("zeros are %d" % (len(y) - np.sum(y)))


