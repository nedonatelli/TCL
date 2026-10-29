"""
Maximum Likelihood Estimation and Information Theory.

This module provides tools for maximum likelihood estimation, Fisher
information computation, and Cramer-Rao bound analysis.

References
----------
- S. M. Kay, "Fundamentals of Statistical Signal Processing:
  Estimation Theory," Prentice Hall, 1993.
- H. L. Van Trees, "Detection, Estimation, and Modulation Theory,"
  Wiley, 2001.
"""

import warnings
from typing import Any, Callable, NamedTuple, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pytcl.mathematical_functions.basic_matrix.special_matrices import (
    commutation_matrix,
)


class MLResult(NamedTuple):
    """Result of maximum likelihood estimation.

    Attributes
    ----------
    theta : ndarray
        Estimated parameters.
    log_likelihood : float
        Log-likelihood at the estimate.
    fisher_info : ndarray
        Fisher information matrix at the estimate.
    covariance : ndarray
        Estimated covariance of the parameters. This is the inverse Fisher
        information wherever the parameterization is unconstrained. It is not
        for the multivariate Gaussian, whose ``vec(Sigma)`` block carries
        ``d^2`` entries for a symmetric matrix with only ``d(d+1)/2`` free
        ones -- there the sampling covariance is rank-deficient by
        construction and the inverse Fisher matrix would describe a different,
        unconstrained estimator (gh-20).
    n_iter : int
        Number of iterations (for iterative methods).
    converged : bool
        Whether the optimization converged.
    """

    theta: NDArray[np.floating]
    log_likelihood: float
    fisher_info: NDArray[np.floating]
    covariance: NDArray[np.floating]
    n_iter: int
    converged: bool


class CRBResult(NamedTuple):
    """Result of Cramer-Rao bound computation.

    Attributes
    ----------
    crb_matrix : ndarray
        Cramer-Rao bound matrix (inverse Fisher information).
    variances : ndarray
        Diagonal elements (variance bounds for each parameter).
    std_bounds : ndarray
        Square root of variances (standard deviation bounds).
    fisher_info : ndarray
        Fisher information matrix used.
    is_efficient : ndarray or None
        Boolean mask indicating which estimators achieve the bound.
    """

    crb_matrix: NDArray[np.floating]
    variances: NDArray[np.floating]
    std_bounds: NDArray[np.floating]
    fisher_info: NDArray[np.floating]
    is_efficient: Optional[NDArray[np.bool_]]


# =============================================================================
# Fisher Information
# =============================================================================


def fisher_information_numerical(
    log_likelihood: Callable[[np.ndarray[Any, Any]], float],
    theta: ArrayLike,
    h: float = 1e-5,
) -> np.ndarray[Any, Any]:
    """
    Compute Fisher information matrix numerically.

    Uses the negative expected Hessian of the log-likelihood.

    Parameters
    ----------
    log_likelihood : callable
        Log-likelihood function L(theta).
    theta : array_like
        Parameter vector at which to evaluate.
    h : float, optional
        Step size for numerical differentiation. Default 1e-5.

    Returns
    -------
    fisher : ndarray
        Fisher information matrix of shape (n_params, n_params).

    Examples
    --------
    >>> data = np.array([-1.0, 0.0, 1.0])
    >>> def log_lik(theta):
    ...     return -0.5 * np.sum((data - theta[0])**2 / theta[1])
    >>> theta = np.array([0.0, 1.0])
    >>> F = fisher_information_numerical(log_lik, theta)
    >>> bool(np.allclose(F, [[3, 0], [0, 2]], atol=1e-4))
    True

    Notes
    -----
    For a single observation, the Fisher information is:
        I(theta) = -E[d^2 log p(x|theta) / d theta^2]

    This function approximates the Hessian using central differences.
    """
    theta = np.asarray(theta, dtype=np.float64)
    n = len(theta)
    fisher = np.zeros((n, n))

    # Compute Hessian using central differences
    for i in range(n):
        for j in range(i, n):
            # Four-point formula for mixed partials
            theta_pp = theta.copy()
            theta_pm = theta.copy()
            theta_mp = theta.copy()
            theta_mm = theta.copy()

            theta_pp[i] += h
            theta_pp[j] += h
            theta_pm[i] += h
            theta_pm[j] -= h
            theta_mp[i] -= h
            theta_mp[j] += h
            theta_mm[i] -= h
            theta_mm[j] -= h

            hess_ij = (
                log_likelihood(theta_pp)
                - log_likelihood(theta_pm)
                - log_likelihood(theta_mp)
                + log_likelihood(theta_mm)
            ) / (4 * h * h)

            # Fisher info is negative expected Hessian
            fisher[i, j] = -hess_ij
            fisher[j, i] = -hess_ij

    return fisher


def fisher_information_gaussian(
    jacobian: ArrayLike,
    noise_cov: ArrayLike,
) -> NDArray[np.floating]:
    """
    Fisher information for Gaussian measurement model.

    For the model y = h(theta) + noise, where noise ~ N(0, R),
    the Fisher information is J^T R^{-1} J.

    Parameters
    ----------
    jacobian : array_like
        Jacobian matrix dh/dtheta of shape (m, n).
    noise_cov : array_like
        Measurement noise covariance R of shape (m, m).

    Returns
    -------
    fisher : ndarray
        Fisher information matrix of shape (n, n).

    Examples
    --------
    >>> H = np.array([[1, 0], [0, 1], [1, 1]])  # 3 measurements, 2 params
    >>> R = np.eye(3) * 0.1
    >>> F = fisher_information_gaussian(H, R)

    Notes
    -----
    This is the standard result for linear Gaussian models and
    provides the information content of measurements about parameters.
    """
    J = np.asarray(jacobian, dtype=np.float64)
    R = np.asarray(noise_cov, dtype=np.float64)

    R_inv = np.linalg.inv(R)
    return J.T @ R_inv @ J


def fisher_information_exponential_family(
    sufficient_stats: Callable[
        [np.ndarray[Any, Any], np.ndarray[Any, Any]], np.ndarray[Any, Any]
    ],
    theta: ArrayLike,
    data: ArrayLike,
) -> np.ndarray[Any, Any]:
    """
    Fisher information for exponential family distributions.

    For exponential family: p(x|theta) = h(x) exp(eta(theta)^T T(x) - A(theta))
    The Fisher information equals the covariance of sufficient statistics.

    Parameters
    ----------
    sufficient_stats : callable
        Function T(x, theta) returning sufficient statistics.
    theta : array_like
        Natural parameters.
    data : array_like
        Observed data of shape (n_samples, ...).
    Returns
    -------
    fisher : ndarray
        Fisher information matrix.

    Notes
    -----
    For exponential families, I(theta) = Var[T(X)] = d^2 A(theta) / d theta^2,
    where A(theta) is the log-partition function.

    Examples
    --------
    >>> def suff_stats(x, theta):
    ...     return np.array([x, x**2])  # Mean and second moment
    >>> data = np.random.normal(0, 1, 100)
    >>> theta = np.array([0.0, 1.0])
    >>> F = fisher_information_exponential_family(suff_stats, theta, data)
    """
    theta = np.asarray(theta, dtype=np.float64)
    data = np.asarray(data, dtype=np.float64)

    # Compute sufficient statistics for all data
    T = np.array([sufficient_stats(x, theta) for x in data])

    # Fisher info is covariance of sufficient statistics
    return np.cov(T.T)


def observed_fisher_information(
    log_likelihood: Callable[[np.ndarray[Any, Any]], float],
    theta: ArrayLike,
    h: float = 1e-5,
) -> np.ndarray[Any, Any]:
    """
    Compute observed Fisher information (negative Hessian).

    Unlike the expected Fisher information, this uses the actual
    observed data rather than the expectation.

    Parameters
    ----------
    log_likelihood : callable
        Log-likelihood function.
    theta : array_like
        Parameter estimate.
    h : float, optional
        Step size for numerical differentiation.

    Returns
    -------
    observed_fisher : ndarray
        Observed Fisher information matrix.

    Notes
    -----
    The observed Fisher information is often more accurate for
    finite samples and is asymptotically equivalent to the
    expected Fisher information.

    Examples
    --------
    >>> data = np.array([1.2, 0.8, 1.1, 0.9, 1.0])
    >>> def log_lik(theta):
    ...     return -0.5 * np.sum((data - theta[0])**2 / theta[1]) - 2.5 * np.log(theta[1])
    >>> theta = np.array([1.0, 0.1])
    >>> F_obs = observed_fisher_information(log_lik, theta)
    """
    return fisher_information_numerical(log_likelihood, theta, h)


# =============================================================================
# Cramer-Rao Bound
# =============================================================================


def cramer_rao_bound(
    fisher_info: ArrayLike,
    estimator_variance: Optional[ArrayLike] = None,
) -> CRBResult:
    """
    Compute Cramer-Rao lower bound on estimator variance.

    Parameters
    ----------
    fisher_info : array_like
        Fisher information matrix of shape (n, n).
    estimator_variance : array_like, optional
        Actual estimator variance for efficiency check.

    Returns
    -------
    result : CRBResult
        Named tuple with CRB matrix, variances, and efficiency info.
        For a singular ``fisher_info`` the bound is ``inf`` on every
        parameter an unidentified direction touches, NaN on their cross
        terms, and the inverse on the identified subspace elsewhere (the
        unidentified directions are eigenvalues of ``D^-1/2 F D^-1/2``,
        ``D = abs(diag(F))``, at or below ``n * eps`` times the largest).

    Warns
    -----
    RuntimeWarning
        If ``fisher_info`` is not positive definite. Versions up to 2.11.1 returned
        ``pinv(fisher_info)``, a finite bound on parameters with no
        information.

    Examples
    --------
    >>> F = np.array([[10, 0], [0, 5]])  # Fisher info
    >>> result = cramer_rao_bound(F)
    >>> result.variances  # Minimum achievable variances
    array([0.1, 0.2])

    Notes
    -----
    The Cramer-Rao bound states that for any unbiased estimator:
        Var(theta_hat) >= I(theta)^{-1}

    The bound is achieved by efficient estimators, such as the MLE
    for exponential family distributions.
    """
    F = np.asarray(fisher_info, dtype=np.float64)

    crb, unidentified = _covariance_from_information(F, F.shape[0] * _EPS)
    if np.any(unidentified):
        warnings.warn(
            "cramer_rao_bound: Fisher information is not positive definite "
            f"({_scaled_eigenvalue_range(F)}); the likelihood is flat or a "
            "saddle in at least one direction. The bound is infinite on "
            f"parameters {np.flatnonzero(unidentified).tolist()}.",
            RuntimeWarning,
            stacklevel=2,
        )

    variances = np.diag(crb)
    std_bounds = np.sqrt(np.maximum(variances, 0))

    # Check efficiency if estimator variance provided
    is_efficient = None
    if estimator_variance is not None:
        est_var = np.asarray(estimator_variance, dtype=np.float64)
        if est_var.ndim == 1:
            # Compare diagonal elements
            is_efficient = np.isclose(est_var, variances, rtol=0.1)
        else:
            # Compare full matrices
            is_efficient = np.isclose(np.diag(est_var), variances, rtol=0.1)

    return CRBResult(
        crb_matrix=crb,
        variances=variances,
        std_bounds=std_bounds,
        fisher_info=F,
        is_efficient=is_efficient,
    )


def cramer_rao_bound_biased(
    fisher_info: ArrayLike,
    bias_gradient: ArrayLike,
) -> NDArray[np.floating]:
    """
    Cramer-Rao bound for biased estimators.

    Parameters
    ----------
    fisher_info : array_like
        Fisher information matrix.
    bias_gradient : array_like
        Gradient of the bias with respect to theta.

    Returns
    -------
    crb : ndarray
        Cramer-Rao bound matrix for the biased estimator.

    Notes
    -----
    For a biased estimator with bias b(theta), the CRB becomes:
        Var(theta_hat) >= (I + db/dtheta) I^{-1} (I + db/dtheta)^T

    Examples
    --------
    >>> F = np.array([[10.0, 0], [0, 5.0]])  # Fisher info
    >>> db = np.array([[0.1, 0], [0, 0.2]])  # Bias gradient
    >>> crb_biased = cramer_rao_bound_biased(F, db)
    >>> crb_biased.shape
    (2, 2)
    """
    F = np.asarray(fisher_info, dtype=np.float64)
    db = np.asarray(bias_gradient, dtype=np.float64)

    n = F.shape[0]
    I_plus_db = np.eye(n) + db

    F_inv = np.linalg.inv(F)
    return I_plus_db @ F_inv @ I_plus_db.T


def efficiency(
    estimator_variance: ArrayLike,
    crb: ArrayLike,
) -> NDArray[np.floating]:
    """
    Compute estimator efficiency relative to CRB.

    Parameters
    ----------
    estimator_variance : array_like
        Variance of the estimator (scalar or diagonal).
    crb : array_like
        Cramer-Rao bound (scalar or diagonal).

    Returns
    -------
    eff : ndarray
        Efficiency values in [0, 1]. Value of 1 means efficient.

    Examples
    --------
    >>> var_est = np.array([0.12, 0.25])
    >>> crb = np.array([0.1, 0.2])
    >>> efficiency(var_est, crb)
    array([0.83333333, 0.8       ])
    """
    var = np.asarray(estimator_variance, dtype=np.float64)
    crb = np.asarray(crb, dtype=np.float64)

    # Efficiency = CRB / actual variance
    return np.where(var > 0, crb / var, 0.0)


# =============================================================================
# Maximum Likelihood Estimation
# =============================================================================


_EPS = float(np.finfo(np.float64).eps)

# Multiplier on the four-point Hessian's per-entry error model in
# mle_newton_raphson. On 600 data-driven regressions, k = 2 flagged none whose
# inv(fisher) was within 1% and left none unflagged that was over 10% off;
# k = 4 flagged 4 within 1%, k = 1 left 7 over 10% unflagged.
_HESSIAN_NOISE_K = 2.0


def _information_scaling(info: NDArray[np.floating]) -> NDArray[np.floating]:
    """``D^-1/2`` as a vector, ``D = abs(diag(info))`` with 1 where zero."""
    d = np.abs(np.diagonal(info))
    return 1.0 / np.sqrt(np.where(d > 0, d, 1.0))


def _scaled_eigenvalue_range(info: NDArray[np.floating]) -> str:
    """Warning text: extreme eigenvalues of ``D^-1/2 info D^-1/2``."""
    sc = _information_scaling(info)
    scaled = np.linalg.eigvalsh(info * np.outer(sc, sc))
    return (
        "eigenvalues of the diagonally scaled information: smallest "
        f"{scaled[0]:.3g}, largest {scaled[-1]:.3g}"
    )


def _covariance_from_information(
    info: NDArray[np.floating],
    rel_tol: float,
    *,
    noise_bound: Optional[float] = None,
) -> tuple[NDArray[np.floating], NDArray[np.bool_]]:
    """Invert an information matrix on the subspace it identifies.

    Identifiability is judged on the scaled matrix
    ``C = D^-1/2 info D^-1/2``, ``D = abs(diag(info))`` (1 where a diagonal
    entry is zero), so it does not depend on the parameters' units: a
    Fisher matrix ``D [[1, .3], [.3, 1]] D`` with ``D = diag(1e9, 0.1)``
    (condition number 1.1e20) is fully identified.

    Parameters
    ----------
    info : ndarray
        Symmetric information matrix, shape (n, n).
    rel_tol : float
        Eigenvalues of ``C`` at or below ``rel_tol * max(largest
        eigenvalue of C, 0)`` are unidentified directions.
    noise_bound : float, optional
        Absolute uncertainty of the eigenvalues of ``C``, e.g. the 2-norm
        of ``D^-1/2 E D^-1/2`` for an entrywise error bound ``E`` on a
        numerically differentiated ``info``. Eigenvalues at or below it are
        also unidentified. Default None: ``info`` is exact.

    Returns
    -------
    cov : ndarray
        ``inv(info)`` when no direction is unidentified. Otherwise
        ``D^-1/2 pinv(C) D^-1/2`` with ``pinv`` taken on the identified
        eigen-subspace of ``C``, and ``inf`` variance and NaN
        cross-covariance for every parameter an unidentified direction
        touches.
    unidentified : ndarray of bool
        Per-parameter mask of the parameters an unidentified direction
        touches.

    Raises
    ------
    ValueError
        If ``noise_bound`` is NaN, which a comparison would otherwise drop.
    """
    s = _information_scaling(info)
    eigvals, eigvecs = np.linalg.eigh(info * np.outer(s, s))
    threshold = rel_tol * max(eigvals[-1], 0.0)
    if noise_bound is not None:
        if np.isnan(noise_bound):
            raise ValueError("noise_bound is NaN")
        threshold = max(threshold, noise_bound)
    flat = eigvals <= threshold
    if not np.any(flat):
        return np.linalg.inv(info), np.zeros(len(eigvals), dtype=bool)
    V = eigvecs[:, ~flat]
    cov = np.outer(s, s) * ((V / eigvals[~flat]) @ V.T)
    Vf = eigvecs[:, flat]
    touched = np.diag(Vf @ Vf.T) > 1e-12
    cov[touched, :] = np.nan
    cov[:, touched] = np.nan
    cov[touched, touched] = np.inf
    return cov, touched


def mle_newton_raphson(
    log_likelihood: Callable[[np.ndarray[Any, Any]], float],
    score: Callable[[np.ndarray[Any, Any]], np.ndarray[Any, Any]],
    theta_init: ArrayLike,
    hessian: Optional[Callable[[np.ndarray[Any, Any]], np.ndarray[Any, Any]]] = None,
    max_iter: int = 100,
    tol: float = 1e-8,
    h: float = 1e-5,
) -> MLResult:
    """
    Maximum likelihood estimation using Newton-Raphson.

    Parameters
    ----------
    log_likelihood : callable
        Log-likelihood function L(theta).
    score : callable
        Score function (gradient of log-likelihood).
    theta_init : array_like
        Initial parameter guess.
    hessian : callable, optional
        Hessian function. If None, computed numerically.
    max_iter : int, optional
        Maximum iterations. Default 100.
    tol : float, optional
        Convergence tolerance. Default 1e-8.
    h : float, optional
        Step size for numerical Hessian.

    Returns
    -------
    result : MLResult
        MLE result with estimate, Fisher info, and covariance.
        ``fisher_info`` is the observed information ``-H`` exactly as
        computed, never regularized.

    Warns
    -----
    RuntimeWarning
        If the observed information is not positive definite, to within
        its accuracy, at the solution (a flat direction or a saddle). The
        test is on ``C = D^-1/2 F D^-1/2``, ``F`` the observed information
        and ``D = abs(diag(F))``, so it does not depend on the parameters'
        units. With a caller-supplied ``hessian`` an eigenvalue of ``C`` at
        or below ``n * eps`` times the largest counts as zero
        (``np.linalg.matrix_rank``'s convention). The numerical Hessian is
        far less accurate: the four-point stencil's error in entry
        ``(i, j)`` is modelled as ``2 (eps abs(L) / h**2 + eps max(abs(t_i),
        abs(t_j), 1) / h abs(F_ij))``, where ``L`` is the log-likelihood at
        the solution, and an eigenvalue of ``C`` within the 2-norm of that
        error (scaled like ``C``) also counts as zero. The roundoff is set
        mainly by ``abs(L)``, which a real likelihood's residuals and
        normalizing constant keep well above zero. If ``L`` or ``F`` is not
        finite there is no error model, and the result is v2.11.1's.

        Measured rates of the numerical-Hessian rule, each on the stated
        random sample only:

        * 300 unidentified Gaussian-location models (sample sizes 10 to
          1e5, location up to 1e3): 7 not flagged (v2.11.1: 154).
        * 300 identified diagonal quadratics, curvatures 1e-6 to 1e6, kept
          only where the numerical Fisher diagonal was within 1% of the
          truth: 1 newly flagged.
        * 600 identified linear regressions (2 to 4 parameters, regressor
          scales 1e-2 to 1e2, 10 to 1e4 samples): 50 newly flagged (8%).
          v2.11.1's variances on those were within 1% of exact on 6,
          within 10% on 21, within 100% on all 50. One unflagged case was
          more than 10% off.
        * 1500 identified correlated quadratics (2 to 4 parameters,
          correlation condition number up to 1e3, scales 1e-3 to 1e3):
          226 newly flagged (15%). v2.11.1 was within 1% on 19, within 10%
          on 97, within 100% on 217. Two unflagged cases were more than
          10% off.

        A newly flagged identified model gets ``inf`` variance; supply
        ``hessian`` where that matters. Earlier versions added a hidden
        ridge of ``abs(min eigenvalue) + 1e-6`` to ``fisher_info``,
        reported ``converged=True``, and returned a covariance of ``1e6`` on
        the flat direction that looked like a measurement. Now
        ``converged`` is False, ``fisher_info`` is unridged, and
        ``covariance`` is the inverse on the identified subspace of ``C``,
        scaled back, with ``inf`` variance (and NaN cross-covariance) for
        every parameter that the unidentified directions touch. An
        identified model's covariance is ``inv(fisher_info)``. v2.11.1
        tested the unscaled matrix and warned on badly scaled identified
        Fisher matrices (with an analytic ``hessian``, 1002 of 3000 random
        ones with scales from 1e-9 to 1e9); those now return
        ``inv(fisher_info)``, whose diagonal was within ``2.8e-15 *
        cond(C)`` relative of exact on that sample.

    Examples
    --------
    >>> data = np.array([1.0, 2.0, 3.0])
    >>> def log_lik(theta):
    ...     return -0.5 * np.sum((data - theta[0])**2)
    >>> def score(theta):
    ...     return np.array([np.sum(data - theta[0])])
    >>> result = mle_newton_raphson(log_lik, score, np.array([0.0]))
    >>> float(np.round(result.theta[0], 6))  # MLE is the sample mean
    2.0

    Notes
    -----
    Newton-Raphson update: theta_{n+1} = theta_n - H^{-1} @ score
    where H is the Hessian of the log-likelihood.
    """
    theta = np.asarray(theta_init, dtype=np.float64).copy()
    n_params = len(theta)

    converged = False

    def numerical_hessian(t: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        H = np.zeros((n_params, n_params))
        for i in range(n_params):
            for j in range(i, n_params):
                t_pp = t.copy()
                t_pm = t.copy()
                t_mp = t.copy()
                t_mm = t.copy()
                t_pp[i] += h
                t_pp[j] += h
                t_pm[i] += h
                t_pm[j] -= h
                t_mp[i] -= h
                t_mp[j] += h
                t_mm[i] -= h
                t_mm[j] -= h
                H[i, j] = (
                    log_likelihood(t_pp)
                    - log_likelihood(t_pm)
                    - log_likelihood(t_mp)
                    + log_likelihood(t_mm)
                ) / (4 * h * h)
                H[j, i] = H[i, j]
        return H

    hess_func = hessian if hessian is not None else numerical_hessian

    for iteration in range(max_iter):
        g = score(theta)
        H = hess_func(theta)

        # Newton step
        try:
            delta = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(H, -g, rcond=None)[0]

        theta_new = theta + delta

        # Check convergence
        if np.linalg.norm(delta) < tol * (1 + np.linalg.norm(theta)):
            converged = True
            theta = theta_new
            break

        theta = theta_new

    # Compute final Fisher information
    fisher = -hess_func(theta)

    ll_value = float(log_likelihood(theta))
    rel_tol = n_params * _EPS
    noise_bound = None
    # A non-finite log-likelihood or Fisher matrix has no error model; the
    # test then reduces to v2.11.1's result (a NaN Fisher is never flagged).
    if hessian is None and np.isfinite(ll_value) and np.all(np.isfinite(fisher)):
        t = np.maximum(np.abs(theta), 1.0)
        E = _HESSIAN_NOISE_K * (
            _EPS * abs(ll_value) / h**2
            + _EPS * np.maximum.outer(t, t) / h * np.abs(fisher)
        )
        sc = _information_scaling(fisher)
        noise_bound = float(np.linalg.norm(E * np.outer(sc, sc), 2))
    cov, unidentified = _covariance_from_information(
        fisher, rel_tol, noise_bound=noise_bound
    )
    if np.any(unidentified):
        warnings.warn(
            "mle_newton_raphson: observed Fisher information is not positive "
            "definite to within its accuracy "
            f"({_scaled_eigenvalue_range(fisher)}); the likelihood is "
            "flat or a saddle in at least one direction. No ridge is "
            "applied: fisher_info is reported unmodified, the covariance is "
            "infinite on the unidentified parameters, and converged is False.",
            RuntimeWarning,
            stacklevel=2,
        )
        converged = False

    return MLResult(
        theta=theta,
        log_likelihood=ll_value,
        fisher_info=fisher,
        covariance=cov,
        n_iter=iteration + 1,
        converged=converged,
    )


def mle_scoring(
    log_likelihood: Callable[[np.ndarray[Any, Any]], float],
    score: Callable[[np.ndarray[Any, Any]], np.ndarray[Any, Any]],
    fisher_info: Callable[[np.ndarray[Any, Any]], np.ndarray[Any, Any]],
    theta_init: ArrayLike,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> MLResult:
    """
    Maximum likelihood estimation using Fisher scoring.

    Uses expected Fisher information instead of observed Hessian,
    which can be more stable.

    Parameters
    ----------
    log_likelihood : callable
        Log-likelihood function.
    score : callable
        Score function (gradient).
    fisher_info : callable
        Fisher information function.
    theta_init : array_like
        Initial parameter guess.
    max_iter : int, optional
        Maximum iterations.
    tol : float, optional
        Convergence tolerance.

    Returns
    -------
    result : MLResult
        MLE result. ``covariance`` is ``inv(fisher_info(theta))`` when that
        is nonsingular; otherwise it is the inverse on the identified
        subspace, with ``inf`` variance and NaN cross-covariance on every
        parameter an unidentified direction touches (an eigenvalue of
        ``D^-1/2 F D^-1/2``, ``D = abs(diag(F))``, at or below ``n * eps``
        times the largest).

    Warns
    -----
    RuntimeWarning
        If the Fisher information at the solution is not positive definite
        (a flat direction); ``converged`` is then False, as in
        ``mle_newton_raphson``. Versions up to 2.11.1 returned ``pinv(F)``,
        a finite covariance on parameters with no information.

    Notes
    -----
    Fisher scoring update: theta_{n+1} = theta_n + I(theta_n)^{-1} @ score
    This is equivalent to Newton-Raphson when I(theta) = -E[H].

    Examples
    --------
    >>> data = np.array([1.0, 1.1, 0.9, 1.2, 0.8])
    >>> def log_lik(theta):
    ...     return -0.5 * len(data) * np.log(2*np.pi) - np.sum((data - theta[0])**2) / 2
    >>> def score(theta):
    ...     return np.array([np.sum(data - theta[0])])
    >>> def fisher(theta):
    ...     return np.array([[len(data)]])
    >>> result = mle_scoring(log_lik, score, fisher, np.array([0.0]))
    """
    theta = np.asarray(theta_init, dtype=np.float64).copy()

    converged = False

    for iteration in range(max_iter):
        g = score(theta)
        F = fisher_info(theta)

        # Fisher scoring step
        try:
            delta = np.linalg.solve(F, g)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(F, g, rcond=None)[0]

        theta_new = theta + delta

        # Check convergence
        if np.linalg.norm(delta) < tol * (1 + np.linalg.norm(theta)):
            converged = True
            theta = theta_new
            break

        theta = theta_new

    F = fisher_info(theta)
    cov, unidentified = _covariance_from_information(
        np.asarray(F, dtype=np.float64), len(theta) * _EPS
    )
    if np.any(unidentified):
        warnings.warn(
            "mle_scoring: Fisher information is not positive definite at the "
            f"solution ({_scaled_eigenvalue_range(F)}); the likelihood is flat "
            "or a saddle in at least one direction. The covariance is "
            "infinite on parameters "
            f"{np.flatnonzero(unidentified).tolist()}, and converged is False.",
            RuntimeWarning,
            stacklevel=2,
        )
        converged = False

    return MLResult(
        theta=theta,
        log_likelihood=float(log_likelihood(theta)),
        fisher_info=F,
        covariance=cov,
        n_iter=iteration + 1,
        converged=converged,
    )


def mle_gaussian(
    data: ArrayLike,
    estimate_mean: bool = True,
    estimate_variance: bool = True,
) -> MLResult:
    """
    Closed-form MLE for Gaussian distribution.

    Parameters
    ----------
    data : array_like
        Observed data of shape (n_samples,) or (n_samples, n_features).
    estimate_mean : bool, optional
        Whether to estimate mean. Default True.
    estimate_variance : bool, optional
        Whether to estimate variance. Default True.

    Returns
    -------
    result : MLResult
        MLE result with mean and/or variance estimates.

    Examples
    --------
    >>> rng = np.random.default_rng(42)
    >>> data = rng.normal(5, 2, 1000)
    >>> result = mle_gaussian(data)
    >>> np.round(result.theta, 2)  # [mean, variance]
    array([4.94, 3.91])
    >>> bool(np.allclose(result.theta, [np.mean(data), np.var(data)]))
    True
    """
    data = np.asarray(data, dtype=np.float64)

    if data.ndim == 1:
        n = len(data)
        mean_mle = np.mean(data) if estimate_mean else 0.0
        var_mle = np.var(data) if estimate_variance else 1.0

        theta = []
        if estimate_mean:
            theta.append(mean_mle)
        if estimate_variance:
            theta.append(var_mle)
        theta = np.array(theta)

        # Log-likelihood
        log_lik = -n / 2 * np.log(2 * np.pi * var_mle) - np.sum(
            (data - mean_mle) ** 2
        ) / (2 * var_mle)

        # Fisher information
        n_params = len(theta)
        fisher = np.zeros((n_params, n_params))
        idx = 0
        if estimate_mean:
            fisher[idx, idx] = n / var_mle
            idx += 1
        if estimate_variance:
            fisher[idx, idx] = n / (2 * var_mle**2)

        cov = np.linalg.inv(fisher)

    else:
        # Multivariate case
        n, d = data.shape
        mean_mle = np.mean(data, axis=0)
        cov_mle = np.cov(data.T, ddof=0)

        theta = np.concatenate([mean_mle, cov_mle.flatten()])

        # Log-likelihood
        sign, logdet = np.linalg.slogdet(cov_mle)
        centered = data - mean_mle
        cov_inv = np.linalg.inv(cov_mle)
        log_lik = (
            -n * d / 2 * np.log(2 * np.pi)
            - n / 2 * logdet
            - 0.5 * np.sum(centered @ cov_inv * centered)
        )

        # Fisher information for (mu, vec(Sigma)). Both blocks are exact and
        # the cross terms vanish -- the mean and covariance MLEs of a Gaussian
        # are independent.
        #
        #     I_mu    = n * Sigma^-1
        #     I_Sigma = (n/2) * (Sigma^-1 kron Sigma^-1)
        #
        # These used to be `np.eye(n_params) * n` and `np.eye(n_params) / n`,
        # which did not depend on the data at all (gh-20).
        identity = np.eye(d * d)
        fisher = np.zeros((len(theta), len(theta)))
        fisher[:d, :d] = n * cov_inv
        fisher[d:, d:] = 0.5 * n * np.kron(cov_inv, cov_inv)

        # The parameter covariance is *not* the inverse of that Fisher matrix.
        # vec(Sigma) carries d^2 entries for a symmetric matrix with only
        # d(d+1)/2 free ones, so the sampling covariance of the estimator is
        # rank-deficient by construction:
        #
        #     Cov(vec(Sigma_hat)) = (I + K)(Sigma kron Sigma) / n
        #
        # with K the commutation matrix. Inverting the Fisher block instead
        # would give (2/n)(Sigma kron Sigma), which is full rank and wrong --
        # it describes an unconstrained matrix estimator that does not exist
        # here. Both expressions were checked against Monte Carlo.
        cov = np.zeros((len(theta), len(theta)))
        cov[:d, :d] = cov_mle / n
        cov[d:, d:] = (
            (identity + commutation_matrix(d, d)) @ np.kron(cov_mle, cov_mle) / n
        )

    return MLResult(
        theta=theta,
        log_likelihood=float(log_lik),
        fisher_info=fisher,
        covariance=cov,
        n_iter=1,
        converged=True,
    )


# =============================================================================
# Information Criteria
# =============================================================================


def aic(log_likelihood: float, n_params: int) -> float:
    """
    Akaike Information Criterion.

    Parameters
    ----------
    log_likelihood : float
        Log-likelihood at the MLE.
    n_params : int
        Number of parameters.

    Returns
    -------
    aic : float
        AIC value (lower is better).

    Notes
    -----
    AIC = -2 * log_likelihood + 2 * n_params

    Examples
    --------
    >>> log_lik = -100.0
    >>> n_params = 3
    >>> aic(log_lik, n_params)
    206.0
    """
    return -2 * log_likelihood + 2 * n_params


def bic(log_likelihood: float, n_params: int, n_samples: int) -> float:
    """
    Bayesian Information Criterion.

    Parameters
    ----------
    log_likelihood : float
        Log-likelihood at the MLE.
    n_params : int
        Number of parameters.
    n_samples : int
        Number of samples.

    Returns
    -------
    bic : float
        BIC value (lower is better).

    Notes
    -----
    BIC = -2 * log_likelihood + n_params * log(n_samples)

    Examples
    --------
    >>> log_lik = -100.0
    >>> bic(log_lik, n_params=3, n_samples=100)
    213.81551055796427
    """
    return -2 * log_likelihood + n_params * np.log(n_samples)


def aicc(log_likelihood: float, n_params: int, n_samples: int) -> float:
    """
    Corrected Akaike Information Criterion.

    Parameters
    ----------
    log_likelihood : float
        Log-likelihood at the MLE.
    n_params : int
        Number of parameters.
    n_samples : int
        Number of samples.

    Returns
    -------
    aicc : float
        AICc value (lower is better).

    Notes
    -----
    AICc adds a correction for small sample sizes:
    AICc = AIC + 2*k*(k+1)/(n-k-1)

    Examples
    --------
    >>> log_lik = -50.0
    >>> aicc(log_lik, n_params=3, n_samples=20)
    107.5
    """
    k = n_params
    n = n_samples
    if n - k - 1 <= 0:
        return np.inf
    return aic(log_likelihood, k) + 2 * k * (k + 1) / (n - k - 1)


__all__ = [
    "MLResult",
    "CRBResult",
    "fisher_information_numerical",
    "fisher_information_gaussian",
    "fisher_information_exponential_family",
    "observed_fisher_information",
    "cramer_rao_bound",
    "cramer_rao_bound_biased",
    "efficiency",
    "mle_newton_raphson",
    "mle_scoring",
    "mle_gaussian",
    "aic",
    "bic",
    "aicc",
]
