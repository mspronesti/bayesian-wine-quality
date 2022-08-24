import numpy as np
from .._base import Estimator
from ..utils.probability import multivariate_normal_logpdf
from scipy.special import logsumexp
from typing import List


def gmm_logpdf(X, gmm: np.ndarray):
    """
    Joint log densities  for
    Gaussian Mixtures Models

    Parameters
    ----------
    X:
        ndarray, data matrix in the shape
            (n_feats, n_samples)

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
    n_feats, n_samples = X.shape
    n_params = len(gmm)

    # data structure for the log joints
    S = np.empty(shape=(n_params, n_samples))

    for g in range(len(gmm)):
        mean = gmm[g][1]
        cov = gmm[g][2]
        ll = multivariate_normal_logpdf(X, mean, cov)
        S[g, :] = ll + np.log(gmm[g][0])

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
    U, s, _ = np.linalg.svd(cov)
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

    curr_params = np.array(gmm, dtype=object)
    ll_current = np.NaN

    while True:
        # E-step
        ll_previous = ll_current
        # compute marginals and responsibility
        marginals, gamma = gmm_logpdf(X, curr_params)

        ll_current = sum(marginals) / n_samples

        # evaluate stop condition
        if np.abs(ll_current - ll_previous) < tol:
            return curr_params

        # M-step
        Z = np.sum(gamma, axis=1)
        for g in range(n_params):
            # compute stats
            F = np.sum(gamma[g] * X, axis=1)
            S = (gamma[g] * X) @ X.T

            mean = (F / Z[g]).reshape(n_feats, 1)
            cov = S / Z[g] - mean @ mean.T
            w = Z[g] / sum(Z)

            if diag:
                # only keep diagonal
                cov *= np.eye(cov.shape[0])

            # bound covariance
            cov = cov_eig_constraint(cov, psi)
            curr_params[g] = [w, mean, cov]

        if tied:
            tied_cov = sum(Z * curr_params[:, 2]) / n_samples
            # tied_cov = np.zeros([n_feats, n_feats])
            # for g in range(n_params):
            #     tied_cov += Z[g] * curr_params[g][2]
            # tied_cov /= n_samples
            tied_cov = cov_eig_constraint(tied_cov, psi)
            # replace covariance with the tied one
            # computed above
            curr_params[:, 2].fill(tied_cov)


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

    # initialize parameters
    # this list of lists is going
    # to be converted into an ndarray
    gmm_1 = [(1.0, mean, cov)]

    for _ in range(int(np.log2(n_components))):
        gmm = []
        for param in gmm_1:
            w = param[0] / 2
            mean = param[1]
            cov = param[2]

            U, s, _ = np.linalg.svd(cov)
            d = U[:, 0:1] * s[0] ** 0.5 * alpha

            gmm.append((w, mean + d, cov))
            gmm.append((w, mean - d, cov))

        gmm_1 = em_estimation(X, gmm, psi, tol, diag=diag, tied=tied)

    return gmm_1


class GaussianMixture(Estimator):
    _valid_cov_types = ['full', 'diag', 'full-tied', 'diag-tied']

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
                {'full', 'diag', 'full-tied', 'diag-tied'}.
                - 'full' : use full covariance.
                - 'diag' : use diagonal covariance to reduce the number
                           of parameters to estimate.
                - 'full-tied': use the same full covariance for all GMM
                            components
                - 'diag-tied': use the same diag covariance for all GMM
                            components
                Default 'full'

        tol:
            float, the convergence threshold. EM iterations will stop
            when the lower bound average gain is below this threshold.
            Default 1e-6
        """
        if cov_type == 'full':
            self.diag, self.tied = (False, False)
        elif cov_type == 'diag':
            self.diag, self.tied = (True, False)
        elif cov_type == 'full-tied':
            self.diag, self.tied = (False, True)
        elif cov_type == 'diag-tied':
            self.diag, self.tied = (True, True)
        else:
            raise ValueError("Unknown covariance type %s."
                             "Valid types are %s"
                             % (cov_type, self._valid_cov_types))

        self.n_components = n_components
        self.alpha = alpha
        self.psi = psi
        self.tol = tol
        # gmm_estimates is going to contain
        # for each label, the params
        self.gmm_estimates = {}

    def fit(self, X, y):
        """
        Fits the Gaussian Mixture Model
        given the training data

        Parameters
        ----------
        X: ndarray, training data matrix
        y: ndarray, target values

        Returns
        -------
            fitted GaussianMixture instance
        """
        n_labels = np.unique(y)
        for label in n_labels:
            # extract all elements having
            # current label and transpose it
            # (as the LGB algorithm expects it)
            x_ = X[y == label, :].T
            params = lbg_estimation(x_,
                                    self.n_components,
                                    alpha=self.alpha,
                                    psi=self.psi,
                                    tol=self.tol,
                                    diag=self.diag,
                                    tied=self.tied)

            self.gmm_estimates[label] = params

        return self

    def predict(self, X, return_proba=False):
        log_densities = []
        for label in self.gmm_estimates:
            gm = self.gmm_estimates[label]
            marginals, _ = gmm_logpdf(X.T, gm)
            log_densities.append(marginals)

        # Notice: this is correct, but just because we
        # assume 2 labels. Replace with an argmax for
        # multiclass
        score = log_densities[1] - log_densities[0]
        y_pred = (score >= 0).astype(np.int32)
        if return_proba:
            return y_pred, score

        return y_pred

    def __str__(self):
        return "Tied" * self.tied + \
               "Diag" * self.diag + \
               f"GMM(n_components={self.n_components}, " \
               f"alpha={self.alpha})"
