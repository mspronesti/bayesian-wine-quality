import numpy as np
from .._base import Estimator
from ..utils.probability import multivariate_normal_logpdf
from scipy.special import logsumexp
from typing import List


def gmm_logpdf(X, gmm: List):
    """
    Joint log densities  for
    Gaussian Mixtures Models

    Parameters
    ----------
    X:
        ndarray, data matrix

    gmm:
        list of components of the GMM

    Returns
    -------
        marginal_log_density:
            GMM marginal log density

        gamma:
           the cluster posterior, also known as "responsibility".
           It can be seen as a probability of a data point belonging
           to a cluster c
    """
    S = np.zeros([len(gmm), X.shape[1]])
    for g in range(len(gmm)):
        S[g, :] = multivariate_normal_logpdf(X, gmm[g][1], gmm[g][2] + np.log(gmm[g][0]))

    # marginal densities
    marginal_log_density = logsumexp(S, axis=0)
    # log posterior
    log_gamma = S - marginal_log_density
    # posterior
    gamma = np.exp(log_gamma)
    return marginal_log_density, gamma


def cov_eig_constraint(cov: np.ndarray, eig_bound: float = .01):
    """
    Computes the constraints on the eigenvalues
    given the covariance matrix, to avoid degenerate
    solutions

    Parameters
    ----------
    cov:
        covariance matrix

    eig_bound:
        constraint threshold.
        It's the maximum value we admit
        for the eigenvalues

    Returns
    -------
        bounded reconstructed covariance
    """
    U, s, _ = np.lianlg.svd(cov)
    # check and set on psi
    s[s < eig_bound] = eig_bound
    # recompute the covariance from the new
    # s matrix and the unitary array
    return U @ (s.reshape(s.size, 1) * U.T)


def em_estimation(X: np.ndarray,
                  gmm: List,
                  psi: float = .01,
                  tol: float = 1e-6,
                  *,
                  tied: bool = False,
                  diag: bool = False):
    """
    Expectation Maximization (EM) Algorithm.

    E-Step: Estimate the distribution of the hidden variables
            given the data and the current parameters
    M-Step: Maximize the joint distribution of the data and the
            hidden variable

    Parameters
    ----------
    X:
        ndarray, transposed data matrix
        in the shape
            (n_feats, n_samples)

    gmm:
        current parameters

    psi:
        float, the eigenvalues bound to avoid
        degenerate solutions.
        Default 0.01

    tol:
        float, accepted precision for convergence of GMM estimation.
        Default 1e-6

    tied:
        bool, whether to apply the tied hypothesis and add the
        constraints at the end, using the tied covariance.
        Default False

    diag:
        bool, whether to diagonalize the covariance.
        Default False

    Returns
    -------
        computed parameters.
    """
    n_params = len(gmm)
    # NOTICE: X is expected transposed
    n_feats, n_samples = X.shape
    curr_params = gmm.copy()
    ll_current = np.NaN

    while True:
        # E-step
        ll_previous = ll_current
        # compute marginals and responsibility
        marginals, gamma = gmm_logpdf(X, curr_params)

        ll_current = marginals.sum() / n_samples

        # evaluate stop condition
        if np.abs(ll_previous - ll_current) < tol:
            return curr_params

        # M-step
        Z = np.sum(gamma, axis=1)

        for g in range(n_params):
            # compute stats
            F = np.sum(gamma[g] * X, axis=1)
            S = (gamma[g] * X) @ X.T

            mean = (F / Z[g]).reshape(X.shape[0], 1)
            cov = S / Z[g] - mean @ mean.T
            w = Z[g] / sum(Z)

            if diag:
                cov *= np.eye(cov.shape[0])

            # bound covariance
            cov = cov_eig_constraint(cov, psi)
            curr_params[g] = (w, mean, cov)

        if tied:
            tied_cov = np.zeros(n_feats, n_feats)
            for g in range(n_params):
                tied_cov += Z[g] * curr_params[g][2]
            tied_cov /= n_samples
            # TODO: vedere se far così è uguale
            #  tied_cov = (Z * curr_params[:, 2])/n_samples
            tied_cov = cov_eig_constraint(tied_cov, psi)
            for g in range(n_params):
                mu = curr_params[g][1]
                w = curr_params[g][0]
                curr_params[g] = (w, mu, tied_cov)


def lbg_estimation(X: np.ndarray,
                   n_components: int = 2,
                   alpha: float = .1,
                   psi: float = .01,
                   tol: float = 1e-6,
                   *,
                   diag: bool = False,
                   tied: bool = False):
    """
    Linde-Buzo-Gray algorithm.
    Given a G-components GMM,
     - split the components as follows:

            mu_+ = mu + eps
            mu_- = mu - eps

       to obtain a 2G components GMM. `eps` is the displacement
       factor.

     - run the EM algorithm until convergence for the
       2G components
     - iterate until the desired number of Gaussians is reached

    Parameters
    ----------
    X: ndarray, data matrix.
        Expected in the shape
            (n_feats, n_samples)

    n_components:
        int, the number of components
        of the GMM.
        Default 2

    alpha:
        float, the displacement factor.
        Default 0.1

    psi:
        float, the eigenvalues bound to avoid
        degenerate solutions.
        Default 0.01

    tol:
        float, accepted precision for convergence of GMM estimation.
        Default 1e-6

    tied:
        bool, whether to apply the tied hypothesis and add the
        constraints at the end, using the tied covariance.
        Default False

    diag:
        bool, whether to diagonalize the covariance.
        Default False

    Returns
    -------

    """
    n = X.shape[0]

    mean = X.mean(axis=1).reshape([n, 1])
    cov = np.cov(X)
    cov = cov_eig_constraint(cov)

    gmm_1 = [(1.0, mean, cov)]

    for _ in range(int(np.log2(n_components))):
        gmm = []
        for p in gmm_1:
            w = p[0] / 2
            cov = p[2]

            U, s, _ = np.linalg.svd(cov)
            d = U[:, 0:1] * s[0] ** 0.5 * alpha
            mean = p[1]

            gmm.append((w, mean + d, cov))
            gmm.append((w, mean - d, cov))

        gmm_1 = em_estimation(X, gmm, psi, tol, diag=diag, tied=tied)

    return gmm_1


class GaussianMixture(Estimator):
    def __init__(self,
                 n_components: int = 2,
                 alpha: float = .1,
                 psi: float = .01,
                 *,
                 cov_type: str = 'full',
                 tol: float = 1e-6) -> None:
        """
        Constructs a Gaussian Mixture Model

        Parameters
        ----------
        n_components:
            int, the number of components of the gmm.
            Default 2

        alpha:
            float, the displacement factor for the LGB
            estimation.
            Default 0.1

        psi:
            float, the eigenvalues bound to avoid
            degenerate solutions.
            Default 0.01

        cov_type:
            str, describes the type of covariance. Must be one of
                {'full', tied', 'diag'}.
                - 'full': each component has its own covariance matrix.
                - 'tied': all components share the same covariance matrix.
                - 'diag': each component has its own diagonal covariance matrix.
                Default 'full'

        tol:
            float, the convergence threshold. EM iterations will stop
            when the lower bound average gain is below this threshold.
            Default 1e-6
        """
        self.n_components = n_components
        self.alpha = alpha
        self.psi = psi
        self.cov_type = cov_type
        self.tol = tol
        self.gmm_estimates = {}

    def fit(self, X, y):
        n_labels = np.unique(y)
        for label in n_labels:
            # extract all elements having
            # current label
            x_ = X[y == label, :].T
            posterior = x_.shape[1] / X.shape[0]
            params = lbg_estimation(x_, )
            self.gmm_estimates[label] = params

    def predict(self, X, return_proba=False):

        # pred = ratio >= 0
        if return_proba:
            # return y_pred, score
            pass
        # return y_pred

