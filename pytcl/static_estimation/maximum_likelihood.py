"""
Maximum Likelihood Estimation and Information Theory.

This module provides tools for maximum likelihood estimation, Fisher
information computation, and Cramer-Rao bound analysis.

References
----------
.. [1] S. M. Kay, "Fundamentals of Statistical Signal Processing:
       Estimation Theory," Prentice Hall, 1993.
.. [2] H. L. Van Trees, "Detection, Estimation, and Modulation Theory,"
       Wiley, 2001.
"""

from typing import Any, Callable, NamedTuple, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray


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
        Estimated covariance (inverse Fisher information).
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
    >>> def log_lik(theta):
    ...     return -0.5 * np.sum((data - theta[0])**2 / theta[1])
    >>> theta = np.array([0.0, 1.0])
    >>> F = fisher_information_numerical(log_lik, theta)

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
    h: float = 1e-5,
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
    h : float, optional
        Step size for numerical differentiation.

    Returns
    -------
    fisher : ndarray
        Fisher information matrix.

    Notes
    -----
    For exponential families, I(theta) = Var[T(X)] = d^2 A(theta) / d theta^2,
    where A(theta) is the log-partition function.
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

    # CRB is inverse of Fisher information
    try:
        crb = np.linalg.inv(F)
    except np.linalg.LinAlgError:
        # Use pseudoinverse for singular Fisher info
        crb = np.linalg.pinv(F)

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
    array([0.833, 0.8])
    """
    var = np.asarray(estimator_variance, dtype=np.float64)
    crb = np.asarray(crb, dtype=np.float64)

    # Efficiency = CRB / actual variance
    return np.where(var > 0, crb / var, 0.0)


# =============================================================================
# Maximum Likelihood Estimation
# =============================================================================


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

    Examples
    --------
    >>> def log_lik(theta):
    ...     return -0.5 * np.sum((data - theta[0])**2)
    >>> def score(theta):
    ...     return np.array([np.sum(data - theta[0])])
    >>> result = mle_newton_raphson(log_lik, score, np.array([0.0]))

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

    # Ensure positive definiteness
    eigvals = np.linalg.eigvalsh(fisher)
    if np.any(eigvals <= 0):
        # Add regularization
        fisher = fisher + np.eye(n_params) * (abs(min(eigvals)) + 1e-6)

    try:
        cov = np.linalg.inv(fisher)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(fisher)

    return MLResult(
        theta=theta,
        log_likelihood=float(log_likelihood(theta)),
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
        MLE result.

    Notes
    -----
    Fisher scoring update: theta_{n+1} = theta_n + I(theta_n)^{-1} @ score
    This is equivalent to Newton-Raphson when I(theta) = -E[H].
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

    # Final Fisher information and covariance
    F = fisher_info(theta)
    try:
        cov = np.linalg.inv(F)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(F)

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
    >>> data = np.random.normal(5, 2, 1000)
    >>> result = mle_gaussian(data)
    >>> result.theta  # [mean, variance]
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

        # Fisher info (simplified diagonal approximation)
        n_params = len(theta)
        fisher = np.eye(n_params) * n
        cov = np.eye(n_params) / n

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
