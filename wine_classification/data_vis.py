import matplotlib.pyplot as plt
import numpy as np
from mlprlib.dataset import load_wine_train, classes, features
import seaborn as sb
from mlprlib.preprocessing import cumulative_feature_rank


def feats_histogram(X, y, label: str = ''):
    for i, feat in enumerate(features):
        plt.figure()
        plt.xlabel(feat)
        plt.title("%s distribution" % feat)
        plt.hist(X[i, y == 0], bins=20, density=True, alpha=.5, ec='black')
        plt.hist(X[i, y == 1], bins=20, density=True, alpha=.5, ec='black')
        plt.legend(classes)
        plt.savefig("../report/assets/" + label + str(i) + ".png")
        plt.close()


def feat_heatmap(X, label: str = ""):
    plt.figure()
    sb.heatmap(np.corrcoef(X))
    plt.savefig(f'../report/assets/{label}.png')


if __name__ == "__main__":
    X_train, y_train = load_wine_train()

    print("ones are %d" % np.sum(y_train))
    print("zeros are %d" % (len(y_train) - np.sum(y_train)))

    # plot histograms of raw features
    feats_histogram(X_train, y_train, "raw_hist")

    X_gauss = cumulative_feature_rank(X_train)
    feats_histogram(X_gauss, y_train, "gauss_hist")

    # features correlations using heatmap
    feat_heatmap(X_gauss, "gauss_feat_heat")
    feat_heatmap(X_gauss[:, y_train == 0], "gauss_feat_heat0")
    feat_heatmap(X_gauss[:, y_train == 1], "gauss_feat_heat1")
    feat_heatmap(X_gauss, "raw_feat_heat")
    feat_heatmap(X_gauss[:, y_train == 0], "raw_feat_heat0")
    feat_heatmap(X_gauss[:, y_train == 1], "raw_feat_heat1")

    np.save("../results/gaussian_feats.npy", X_gauss)
