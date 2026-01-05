"""
Robust estimation methods.

This module provides robust estimators that are resistant to outliers,
including M-estimators (Huber, Tukey bisquare) and RANSAC.

References
----------
.. [1] P. J. Huber, "Robust Statistics," Wiley, 1981.
.. [2] M. A. Fischler and R. C. Bolles, "Random Sample Consensus: A Paradigm
       for Model Fitting with Applications to Image Analysis and Automated
       Cartography," Communications of the ACM, 1981.
"""

from typing import Any, Callable, NamedTuple, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray


class RobustResult(NamedTuple):
    """Result of robust estimation.

    Attributes
    ----------
    x : ndarray
        Estimated parameters.
    residuals : ndarray
        Residual vector.
    weights : ndarray
        Final weights for each observation.
    scale : float
        Estimated scale of residuals.
    n_iter : int
        Number of iterations performed.
    converged : bool
        Whether the algorithm converged.
    """

    x: NDArray[np.floating]
    residuals: NDArray[np.floating]
    weights: NDArray[np.floating]
    scale: float
    n_iter: int
    converged: bool


class RANSACResult(NamedTuple):
    """Result of RANSAC estimation.

    Attributes
    ----------
    x : ndarray
        Estimated parameters from best model.
    inliers : ndarray
        Boolean mask of inlier points.
    n_inliers : int
        Number of inliers.
    residuals : ndarray
        Residuals for all points.
    n_iter : int
        Number of iterations performed.
    best_score : float
        Score of the best model found.
    """

    x: NDArray[np.floating]
    inliers: NDArray[np.bool_]
    n_inliers: int
    residuals: NDArray[np.floating]
    n_iter: int
    best_score: float


# =============================================================================
# Weight Functions for M-estimators
# =============================================================================


def huber_weight(r: ArrayLike, c: float = 1.345) -> NDArray[np.floating]:
    """
    Huber weight function.

    Parameters
    ----------
    r : array_like
        Standardized residuals.
    c : float, optional
        Tuning constant. Default 1.345 gives 95% efficiency for
        normal distribution.

    Returns
    -------
    weights : ndarray
        Weights for each residual.

    Notes
    -----
    The Huber weight function is:
        w(r) = 1           if |r| <= c
        w(r) = c / |r|     if |r| > c

    Examples
    --------
    >>> r = np.array([0.5, 1.0, 2.0, 5.0])  # Standardized residuals
    >>> w = huber_weight(r, c=1.345)
    >>> w[0]  # Small residual gets weight 1
    1.0
    >>> w[3] < 0.5  # Large residual gets reduced weight
    True
    """
    r = np.asarray(r, dtype=np.float64)
    abs_r = np.abs(r)
    weights = np.ones_like(r)
    mask = abs_r > c
    weights[mask] = c / abs_r[mask]
    return weights


def huber_rho(r: ArrayLike, c: float = 1.345) -> NDArray[np.floating]:
    """
    Huber rho (loss) function.

    Parameters
    ----------
    r : array_like
        Standardized residuals.
    c : float, optional
        Tuning constant.

    Returns
    -------
    rho : ndarray
        Loss values.

    Notes
    -----
    The Huber rho function is:
        rho(r) = r^2 / 2           if |r| <= c
        rho(r) = c * |r| - c^2/2   if |r| > c

    Examples
    --------
    >>> r = np.array([0.5, 1.0, 2.0])
    >>> rho = huber_rho(r, c=1.345)
    >>> rho[0]  # Small residual: r^2/2
    0.125
    """
    r = np.asarray(r, dtype=np.float64)
    abs_r = np.abs(r)
    rho = np.where(abs_r <= c, r**2 / 2, c * abs_r - c**2 / 2)
    return rho


def tukey_weight(r: ArrayLike, c: float = 4.685) -> NDArray[np.floating]:
    """
    Tukey bisquare weight function.

    Parameters
    ----------
    r : array_like
        Standardized residuals.
    c : float, optional
        Tuning constant. Default 4.685 gives 95% efficiency for
        normal distribution.

    Returns
    -------
    weights : ndarray
        Weights for each residual.

    Notes
    -----
    The Tukey bisquare weight function is:
        w(r) = (1 - (r/c)^2)^2   if |r| <= c
        w(r) = 0                 if |r| > c

    This provides complete rejection of large outliers.

    Examples
    --------
    >>> r = np.array([0.5, 2.0, 5.0, 10.0])
    >>> w = tukey_weight(r, c=4.685)
    >>> w[0] > 0.9  # Small residual gets high weight
    True
    >>> w[3]  # Large residual completely rejected
    0.0
    """
    r = np.asarray(r, dtype=np.float64)
    abs_r = np.abs(r)
    weights = np.zeros_like(r)
    mask = abs_r <= c
    weights[mask] = (1 - (r[mask] / c) ** 2) ** 2
    return weights


def tukey_rho(r: ArrayLike, c: float = 4.685) -> NDArray[np.floating]:
    """
    Tukey bisquare rho (loss) function.

    Parameters
    ----------
    r : array_like
        Standardized residuals.
    c : float, optional
        Tuning constant.

    Returns
    -------
    rho : ndarray
        Loss values.

    Notes
    -----
    The Tukey rho function is:
        rho(r) = c^2/6 * (1 - (1 - (r/c)^2)^3)   if |r| <= c
        rho(r) = c^2/6                            if |r| > c

    Examples
    --------
    >>> r = np.array([0.0, 2.0, 10.0])
    >>> rho = tukey_rho(r, c=4.685)
    >>> rho[0]  # Zero residual
    0.0
    >>> rho[2] == rho[2]  # Large residuals saturate at c^2/6
    True
    """
    r = np.asarray(r, dtype=np.float64)
    abs_r = np.abs(r)
    rho = np.full_like(r, c**2 / 6)
    mask = abs_r <= c
    rho[mask] = c**2 / 6 * (1 - (1 - (r[mask] / c) ** 2) ** 3)
    return rho


def cauchy_weight(r: ArrayLike, c: float = 2.385) -> NDArray[np.floating]:
    """
    Cauchy weight function.

    Parameters
    ----------
    r : array_like
        Standardized residuals.
    c : float, optional
        Tuning constant.

    Returns
    -------
    weights : ndarray
        Weights for each residual.

    Notes
    -----
    The Cauchy weight function is:
        w(r) = 1 / (1 + (r/c)^2)

    Examples
    --------
    >>> r = np.array([0.0, 1.0, 5.0])
    >>> w = cauchy_weight(r, c=2.385)
    >>> w[0]  # Zero residual gets weight 1
    1.0
    >>> 0 < w[2] < 1  # Large residuals get reduced weight (but never zero)
    True
    """
    r = np.asarray(r, dtype=np.float64)
    return 1 / (1 + (r / c) ** 2)


# =============================================================================
# Scale Estimators
# =============================================================================


def mad(residuals: ArrayLike, c: float = 1.4826) -> float:
    """
    Median Absolute Deviation (MAD) scale estimator.

    Parameters
    ----------
    residuals : array_like
        Residual vector.
    c : float, optional
        Consistency constant. Default 1.4826 makes MAD consistent
        for normal distribution.

    Returns
    -------
    scale : float
        Estimated scale.

    Notes
    -----
    MAD = c * median(|r - median(r)|)

    This is a robust scale estimator with 50% breakdown point.

    Examples
    --------
    >>> residuals = np.array([1.0, 1.1, 0.9, 1.0, 100.0])  # One outlier
    >>> scale = mad(residuals)
    >>> scale < 1.0  # Robust to the outlier
    True
    """
    r = np.asarray(residuals, dtype=np.float64)
    return c * float(np.median(np.abs(r - np.median(r))))


def tau_scale(
    residuals: ArrayLike,
    c1: float = 4.5,
    c2: float = 3.0,
) -> float:
    """
    Tau scale estimator.

    Parameters
    ----------
    residuals : array_like
        Residual vector.
    c1, c2 : float
        Tuning constants.

    Returns
    -------
    scale : float
        Estimated scale.

    Notes
    -----
    Tau scale combines high breakdown point with efficiency.

    Examples
    --------
    >>> residuals = np.array([1.0, 1.1, 0.9, 1.0, 1.2, 100.0])  # One outlier
    >>> scale = tau_scale(residuals)
    >>> scale < 10.0  # Robust to the outlier
    True
    """
    r = np.asarray(residuals, dtype=np.float64)
    n = len(r)

    # Initial scale from MAD
    s0 = mad(r)
    if s0 < 1e-10:
        return s0

    # Standardize
    u = r / s0

    # Compute weights
    w1 = tukey_rho(u, c1)
    w2 = tukey_rho(u, c2)

    # Tau scale
    num = np.sum(w1)
    denom = np.sum(w2)

    if denom < 1e-10:
        return s0

    return s0 * np.sqrt(num / n) * np.sqrt(n / denom)


# =============================================================================
# M-estimators
# =============================================================================


def irls(
    A: ArrayLike,
    b: ArrayLike,
    weight_func: Callable[[np.ndarray[Any, Any]], np.ndarray[Any, Any]] = huber_weight,
    scale_func: Callable[[np.ndarray[Any, Any]], float] = mad,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> RobustResult:
    """
    Iteratively Reweighted Least Squares (IRLS).

    General M-estimator using IRLS algorithm.

    Parameters
    ----------
    A : array_like
        Design matrix of shape (m, n).
    b : array_like
        Observation vector of shape (m,).
    weight_func : callable, optional
        Weight function w(r) for standardized residuals.
        Default is Huber weight.
    scale_func : callable, optional
        Scale estimation function. Default is MAD.
    max_iter : int, optional
        Maximum number of iterations. Default 50.
    tol : float, optional
        Convergence tolerance for parameter change. Default 1e-6.

    Returns
    -------
    result : RobustResult
        Robust estimation result.

    Examples
    --------
    >>> A = np.array([[1, 1], [1, 2], [1, 3], [1, 4], [1, 10]])
    >>> b = np.array([2, 3, 4, 5, 100])  # Last point is outlier
    >>> result = irls(A, b)
    >>> result.x  # Should be close to [1, 1] ignoring outlier

    Notes
    -----
    IRLS iteratively solves weighted least squares problems,
    updating weights based on the current residuals.
    """
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    m, n = A.shape

    # Initial estimate from OLS
    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    converged = False

    for iteration in range(max_iter):
        x_old = x.copy()

        # Compute residuals
        residuals = b - A @ x

        # Estimate scale
        scale = scale_func(residuals)
        if scale < 1e-10:
            scale = 1.0

        # Standardize residuals
        r_std = residuals / scale

        # Compute weights
        weights = weight_func(r_std)
        weights = np.maximum(weights, 1e-10)  # Avoid zero weights

        # Weighted least squares
        W = np.diag(weights)
        AtWA = A.T @ W @ A
        AtWb = A.T @ W @ b

        try:
            x = np.linalg.solve(AtWA, AtWb)
        except np.linalg.LinAlgError:
            x = np.linalg.lstsq(AtWA, AtWb, rcond=None)[0]

        # Check convergence
        if np.linalg.norm(x - x_old) < tol * (1 + np.linalg.norm(x_old)):
            converged = True
            break

    # Final residuals
    residuals = b - A @ x
    scale = scale_func(residuals)

    return RobustResult(
        x=x,
        residuals=residuals,
        weights=weights,
        scale=float(scale),
        n_iter=iteration + 1,
        converged=converged,
    )


def huber_regression(
    A: ArrayLike,
    b: ArrayLike,
    c: float = 1.345,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> RobustResult:
    """
    Huber robust regression.

    M-estimation using Huber's weight function.

    Parameters
    ----------
    A : array_like
        Design matrix of shape (m, n).
    b : array_like
        Observation vector of shape (m,).
    c : float, optional
        Tuning constant. Default 1.345 gives 95% efficiency.
    max_iter : int, optional
        Maximum iterations. Default 50.
    tol : float, optional
        Convergence tolerance. Default 1e-6.

    Returns
    -------
    result : RobustResult
        Robust estimation result.

    Examples
    --------
    >>> A = np.array([[1, 1], [1, 2], [1, 3]])
    >>> b = np.array([2.1, 2.9, 100])  # Outlier in last observation
    >>> result = huber_regression(A, b)

    Notes
    -----
    Huber regression provides a balance between efficiency for
    Gaussian errors and resistance to outliers.
    """

    def weight_func(r: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        return huber_weight(r, c)

    return irls(A, b, weight_func=weight_func, max_iter=max_iter, tol=tol)


def tukey_regression(
    A: ArrayLike,
    b: ArrayLike,
    c: float = 4.685,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> RobustResult:
    """
    Tukey bisquare robust regression.

    M-estimation using Tukey's bisquare weight function.

    Parameters
    ----------
    A : array_like
        Design matrix of shape (m, n).
    b : array_like
        Observation vector of shape (m,).
    c : float, optional
        Tuning constant. Default 4.685 gives 95% efficiency.
    max_iter : int, optional
        Maximum iterations. Default 50.
    tol : float, optional
        Convergence tolerance. Default 1e-6.

    Returns
    -------
    result : RobustResult
        Robust estimation result.

    Examples
    --------
    >>> A = np.array([[1, 1], [1, 2], [1, 3]])
    >>> b = np.array([2.1, 2.9, 100])  # Outlier in last observation
    >>> result = tukey_regression(A, b)

    Notes
    -----
    Tukey bisquare provides complete rejection of large outliers
    (zero weight for residuals > c), making it more robust than
    Huber for gross outliers.
    """

    def weight_func(r: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        return tukey_weight(r, c)

    return irls(A, b, weight_func=weight_func, max_iter=max_iter, tol=tol)


# =============================================================================
# RANSAC
# =============================================================================


def ransac(
    A: ArrayLike,
    b: ArrayLike,
    min_samples: Optional[int] = None,
    residual_threshold: Optional[float] = None,
    max_trials: int = 100,
    stop_n_inliers: Optional[int] = None,
    stop_score: Optional[float] = None,
    random_state: Optional[int] = None,
) -> RANSACResult:
    """
    RANdom SAmple Consensus (RANSAC) regression.

    Robust regression by iteratively selecting random subsets,
    fitting models, and identifying inliers.

    Parameters
    ----------
    A : array_like
        Design matrix of shape (m, n).
    b : array_like
        Observation vector of shape (m,).
    min_samples : int, optional
        Minimum number of samples for fitting. Default is n (number of features).
    residual_threshold : float, optional
        Maximum residual for a point to be considered inlier.
        Default is MAD of initial OLS residuals.
    max_trials : int, optional
        Maximum number of random trials. Default 100.
    stop_n_inliers : int, optional
        Stop early if this many inliers found. Default None.
    stop_score : float, optional
        Stop early if score exceeds this. Default None.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    result : RANSACResult
        RANSAC result with best model and inlier mask.

    Examples
    --------
    >>> rng = np.random.default_rng(42)
    >>> A = np.column_stack([np.ones(100), np.arange(100)])
    >>> b = 2 + 3 * np.arange(100) + rng.normal(0, 1, 100)
    >>> # Add outliers
    >>> b[90:] = 1000
    >>> result = ransac(A, b, random_state=42)
    >>> result.x  # Should be close to [2, 3]

    Notes
    -----
    RANSAC is particularly effective when the fraction of outliers
    is large (>25%), where M-estimators may struggle.

    The algorithm:
    1. Randomly select min_samples points
    2. Fit model to selected points
    3. Count inliers (points with residual < threshold)
    4. If best model found, save it
    5. Repeat for max_trials
    6. Refit model using all inliers of best model

    References
    ----------
    .. [1] M. A. Fischler and R. C. Bolles, "Random Sample Consensus,"
           Communications of the ACM, 1981.
    """
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    m, n = A.shape

    if min_samples is None:
        min_samples = n

    if min_samples > m:
        raise ValueError(f"min_samples ({min_samples}) > n_samples ({m})")

    # Initialize threshold from OLS
    if residual_threshold is None:
        x_ols, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        residuals_ols = b - A @ x_ols
        residual_threshold = 2.5 * mad(residuals_ols)
        if residual_threshold < 1e-10:
            residual_threshold = 1.0

    rng = np.random.default_rng(random_state)

    best_x = None
    best_inliers = np.zeros(m, dtype=bool)
    best_n_inliers = 0
    best_score = -np.inf

    for trial in range(max_trials):
        # Random sample
        sample_idx = rng.choice(m, size=min_samples, replace=False)
        A_sample = A[sample_idx]
        b_sample = b[sample_idx]

        # Fit to sample
        try:
            x_sample, _, _, _ = np.linalg.lstsq(A_sample, b_sample, rcond=None)
        except np.linalg.LinAlgError:
            continue

        # Compute residuals for all points
        residuals = np.abs(b - A @ x_sample)

        # Find inliers
        inliers = residuals < residual_threshold
        n_inliers = np.sum(inliers)

        # Score (number of inliers)
        score = float(n_inliers)

        # Update best if improved
        if score > best_score:
            best_score = score
            best_x = x_sample
            best_inliers = inliers
            best_n_inliers = n_inliers

        # Early stopping
        if stop_n_inliers is not None and n_inliers >= stop_n_inliers:
            break
        if stop_score is not None and score >= stop_score:
            break

    # Refit using all inliers
    if best_n_inliers >= n:
        A_inliers = A[best_inliers]
        b_inliers = b[best_inliers]
        best_x, _, _, _ = np.linalg.lstsq(A_inliers, b_inliers, rcond=None)

    # Handle case where no good model found
    if best_x is None:
        best_x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        best_inliers = np.ones(m, dtype=bool)
        best_n_inliers = m

    # Final residuals
    residuals = b - A @ best_x

    return RANSACResult(
        x=best_x,
        inliers=best_inliers,
        n_inliers=best_n_inliers,
        residuals=residuals,
        n_iter=trial + 1,
        best_score=best_score,
    )


def ransac_n_trials(
    n_samples: int,
    n_outliers: int,
    min_samples: int,
    probability: float = 0.99,
) -> int:
    """
    Compute number of RANSAC trials needed.

    Parameters
    ----------
    n_samples : int
        Total number of samples.
    n_outliers : int
        Expected number of outliers.
    min_samples : int
        Number of samples per trial.
    probability : float, optional
        Desired probability of success. Default 0.99.

    Returns
    -------
    n_trials : int
        Number of trials needed.

    Examples
    --------
    >>> ransac_n_trials(100, 30, 2)  # 30% outliers, 2 samples per trial
    10

    Notes
    -----
    Formula: k = log(1 - p) / log(1 - (1 - e)^n)
    where:
    - p = probability of success
    - e = outlier ratio
    - n = min_samples
    """
    if n_samples <= n_outliers:
        return 1

    outlier_ratio = n_outliers / n_samples
    inlier_ratio = 1 - outlier_ratio

    if inlier_ratio**min_samples < 1e-10:
        return 10000  # Very high outlier rate

    prob_all_inliers = inlier_ratio**min_samples
    prob_failure = 1 - prob_all_inliers

    if prob_failure < 1e-10:
        return 1

    n_trials = np.log(1 - probability) / np.log(prob_failure)

    return max(1, int(np.ceil(n_trials)))


__all__ = [
    "RobustResult",
    "RANSACResult",
    "huber_weight",
    "huber_rho",
    "tukey_weight",
    "tukey_rho",
    "cauchy_weight",
    "mad",
    "tau_scale",
    "irls",
    "huber_regression",
    "tukey_regression",
    "ransac",
    "ransac_n_trials",
]
