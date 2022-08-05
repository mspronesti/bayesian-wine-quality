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

    data_matrix = np.array(samples).reshape((len(samples), n_feats))
    class_labels = np.array(targets)

    return data_matrix, class_labels

