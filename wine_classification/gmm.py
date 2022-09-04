"""
Using the Gaussian Mixture Model with different number of
components with a 5-fold cross validation approach to
classify wines and analyze the performances these methods yield.
"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from mlprlib.dataset import (
    load_wine_train,
    load_wine_test
)

from mlprlib.model_selection import CrossValidator

from mlprlib.preprocessing import (
    StandardScaler,
    GaussianScaler
)

from mlprlib.metrics import min_detection_cost_fun

from mlprlib.utils import Writer
from mlprlib.gaussian import GaussianMixture

n_components_list = [2, 4, 8, 16]


def k_fold_gmm(writer, cov_t: str, data_t: str, X, y, pi=.5):
    transformers = []
    if data_t == 'gauss':
        transformers.append(GaussianScaler())
    elif data_t == 'std':
        transformers.append(StandardScaler())

    dcf_scores = []
    progress_bar = tqdm(n_components_list)
    for n_comp in progress_bar:
        progress_bar.set_description(
            "Cov %s | n_components %s | Data %s | pi %s"
            % (cov_t, n_comp, data_t, pi)
        )
        cv = CrossValidator(n_folds=n_folds)
        gmm = GaussianMixture(n_components=n_comp, cov_type=cov_t)
        cv.fit(X, y, gmm, transformers)

        scores = cv.scores
        # evaluate minDCF
        min_dcf, _ = min_detection_cost_fun(scores, y, pi=pi)

        writer("n_components: %s | %f" % (n_comp, min_dcf))
        dcf_scores.append(min_dcf)
    return dcf_scores


def gmm_eval(writer,
             cov_t: str,
             data_t: str,
             X_train, y_train,
             X_test, y_test,
             pi=.5
             ):
    progress_bar = tqdm(n_components_list)
    for n_comp in progress_bar:
        progress_bar.set_description(
            "Cov %s | n_components %s | Data %s | pi %s"
            % (cov_t, n_comp, data_t, pi)
        )
        gmm = GaussianMixture(n_components=n_comp, cov_type=cov_t)
        gmm.fit(X_train, y_train)
        _, score = gmm.predict(X_test, return_proba=True)
        min_dcf, _ = min_detection_cost_fun(score, y_test, pi=pi)
        writer("n_components: %s | %f" % (n_comp, min_dcf))


def save_plots(scores_gauss, scores_std, fig_name):
    plt.figure()
    plt.plot(n_components_list, scores_gauss, marker='o')
    plt.plot(n_components_list, scores_std, marker='o')
    plt.xlabel("number of components")
    plt.ylabel("min DCF")
    plt.legend(["Gaussianized", "Z-normalized"])
    plt.savefig("../report/assets/gmm/%s.png" % fig_name)


def save_barplots(scores_pi, fig_name):
    """
    scores_pi is a dictionary containing the scores for the
    3 values of pi tested, i.e. [.1, .5, .9]
    """
    width = .25
    fig, ax = plt.subplots()

    x = np.arange(len(n_components_list))
    ax.bar(x, scores_pi[.1], width, label=r"$\tilde{\pi}=0.1$")
    ax.bar(x + width, scores_pi[.5], width, label=r"$\tilde{\pi}=0.5$")
    ax.bar(x + 2*width, scores_pi[.9], width, label=r"$\tilde{\pi}=0.9$")
    ax.set_xlabel("number of components")
    ax.set_ylabel("min DCF")

    ax.set_xticks(x + width)
    ax.set_xticklabels(n_components_list)
    ax.legend()
    plt.savefig("../report/assets/gmm/%s.png" % fig_name)


if __name__ == '__main__':
    # number of folds for cross validation
    n_folds = 5

    # load the dataset in the shape (n_samples, n_feats)
    X, y = load_wine_train(feats_first=False)

    # writing GMM results
    writer = Writer("../results/gaussian_results.txt")

    # data structures to plot bar plots
    # for the different priors
    # (for full GMM only)
    dcf_gauss_pi, dcf_std_pi = {}, {}
    # loop over priors
    for pi in [.1, .5, .9]:
        writer("************* pi = %s *************\n" % pi)
        writer("----------------")
        writer("GMM Cov Type : Full")
        writer("----------------")
        writer("Gaussianized data")
        dcf_gauss = k_fold_gmm(writer, 'full', 'gauss', X, y, pi)
        writer("Standardized data")
        dcf_std = k_fold_gmm(writer, 'full', 'std', X, y, pi)

        # save the results
        dcf_gauss_pi[pi] = dcf_gauss
        dcf_std_pi[pi] = dcf_std

        # save plots for full-cov gmm
        save_plots(dcf_gauss, dcf_std, "gmm_full_%s" % pi)

        writer("\n----------------")
        writer("GMM Cov Type : Diag ")
        writer("----------------")
        writer("Gaussianized data")
        dcf_gauss = k_fold_gmm(writer, 'diag', 'gauss', X, y, pi)
        writer("Standardized data")
        dcf_std = k_fold_gmm(writer, 'diag', 'std', X, y, pi)

        # save plots for diag-cov gmm
        save_plots(dcf_gauss, dcf_std, "gmm_diag_%s" % pi)

        writer("\n----------------")
        writer("GMM Type : Full-Tied")
        writer("----------------")
        writer("Gaussianized data")
        dcf_gauss = k_fold_gmm(writer, 'full-tied', 'gauss', X, y, pi)
        writer("Standardized data")
        dcf_std = k_fold_gmm(writer, 'full-tied', 'std', X, y, pi)

        # save plots for tied full-cov gmm
        save_plots(dcf_gauss, dcf_std, "gmm_full_tied_%s" % pi)

        writer("\n----------------")
        writer("GMM Type : Diag-Tied")
        writer("----------------")
        writer("Gaussianized data")
        dcf_gauss = k_fold_gmm(writer, 'diag-tied', 'gauss', X, y, pi)
        writer("Standardized data")
        dcf_std = k_fold_gmm(writer, 'diag-tied', 'std', X, y, pi)

        # save plots for the tied diag-cov gmm
        save_plots(dcf_gauss, dcf_std, "gmm_diag_tied_%s" % pi)
        writer('\n')
    writer.destroy()

    # save bar plots
    save_barplots(dcf_gauss_pi, "gmm_gauss_pi")
    save_barplots(dcf_std_pi, "gmm_std_pi")

    ##########################
    # evaluation on test set
    ##########################
    X_test, y_test = load_wine_test(feats_first=False)
    gs = GaussianScaler().fit(X)
    sc = StandardScaler().fit(X)

    X_std = sc.transform(X)
    # transform using the mean and cov
    # computed in the fit using X (train)
    # as reference
    X_test_std = sc.transform(X_test)

    X_gauss = gs.transform(X)
    # transform using X (as reference) as reference
    # i.e. the scaler is already fitted here!
    X_test_gauss = gs.transform(X_test)

    writer = Writer("../results/gmm_results_eval.txt")
    for pi in [.1, .5, .9]:
        writer("--------- pi = %s ----------\n" % pi)
        writer("----------------")
        writer("GMM Cov Type : Full")
        writer("----------------")
        writer("Gaussianized data")
        gmm_eval(writer, 'full', 'gauss', X_gauss, y, X_test_gauss, y_test, pi)
        writer("Standardized data")
        gmm_eval(writer, 'full', 'std', X_std, y, X_test_std, y_test, pi)

        writer("----------------")
        writer("GMM Cov Type : Diag")
        writer("----------------")
        writer("Gaussianized data")
        gmm_eval(writer, 'diag', 'gauss', X_gauss, y, X_test_gauss, y_test, pi)
        writer("Standardized data")
        gmm_eval(writer, 'diag', 'std', X_std, y, X_test_std, y_test, pi)

        writer("----------------")
        writer("GMM Cov Type : Full-Tied")
        writer("----------------")
        writer("Gaussianized data")
        gmm_eval(writer, 'full-tied', 'gauss', X_gauss, y, X_test_gauss, y_test, pi)
        writer("Standardized data")
        gmm_eval(writer, 'full-tied', 'std', X_std, y, X_test_std, y_test, pi)

        writer("----------------")
        writer("GMM Cov Type : Diag-Tied")
        writer("----------------")
        writer("Gaussianized data")
        gmm_eval(writer, 'diag-tied', 'gauss', X_gauss, y, X_test_gauss, y_test, pi)
        writer("Standardized data")
        gmm_eval(writer, 'diag-tied', 'std', X_std, y, X_test_std, y_test, pi)
        writer('\n')

    writer.destroy()
