"""
Unscented Kalman Filter (UKF) implementation.

The UKF uses the unscented transform to propagate the mean and covariance
through nonlinear functions without requiring Jacobian computation.
"""

from typing import Any, Callable, NamedTuple, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pytcl.dynamic_estimation.kalman.linear import KalmanPrediction, KalmanUpdate


class SigmaPoints(NamedTuple):
    """Sigma points and weights for unscented transform.

    Attributes
    ----------
    points : ndarray
        Sigma points, shape (2n+1, n).
    Wm : ndarray
        Weights for mean computation.
    Wc : ndarray
        Weights for covariance computation.
    """

    points: NDArray[np.floating]
    Wm: NDArray[np.floating]
    Wc: NDArray[np.floating]


def sigma_points_merwe(
    x: ArrayLike,
    P: ArrayLike,
    alpha: float = 1e-3,
    beta: float = 2.0,
    kappa: float = 0.0,
) -> SigmaPoints:
    """
    Generate sigma points using Van der Merwe's scaled method.

    Parameters
    ----------
    x : array_like
        State mean, shape (n,).
    P : array_like
        State covariance, shape (n, n).
    alpha : float, optional
        Spread of sigma points (default: 1e-3).
        Small values keep points close to mean.
    beta : float, optional
        Prior knowledge of distribution (default: 2.0 for Gaussian).
    kappa : float, optional
        Secondary scaling parameter (default: 0, or 3-n).

    Returns
    -------
    result : SigmaPoints
        Sigma points and weights.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([0.0, 0.0])
    >>> P = np.eye(2)
    >>> sp = sigma_points_merwe(x, P)
    >>> sp.points.shape
    (5, 2)

    Notes
    -----
    Generates 2n+1 sigma points:
        X_0 = x
        X_i = x + sqrt((n+lambda)*P)_i,  i = 1..n
        X_{i+n} = x - sqrt((n+lambda)*P)_i,  i = 1..n

    where lambda = alpha^2 * (n + kappa) - n.

    References
    ----------
    .. [1] Van der Merwe, R., "Sigma-Point Kalman Filters for Probabilistic
           Inference in Dynamic State-Space Models", PhD Thesis, 2004.
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    P = np.asarray(P, dtype=np.float64)

    n = len(x)
    lambda_ = alpha**2 * (n + kappa) - n

    # Compute matrix square root
    try:
        sqrt_P = np.linalg.cholesky((n + lambda_) * P)
    except np.linalg.LinAlgError:
        # Fall back to eigendecomposition for near-singular P
        eigvals, eigvecs = np.linalg.eigh(P)
        eigvals = np.maximum(eigvals, 1e-10)
        sqrt_P = eigvecs @ np.diag(np.sqrt((n + lambda_) * eigvals))

    # Generate sigma points
    points = np.zeros((2 * n + 1, n), dtype=np.float64)
    points[0] = x

    for i in range(n):
        points[i + 1] = x + sqrt_P[:, i]
        points[n + i + 1] = x - sqrt_P[:, i]

    # Compute weights
    Wm = np.full(2 * n + 1, 1 / (2 * (n + lambda_)), dtype=np.float64)
    Wm[0] = lambda_ / (n + lambda_)

    Wc = Wm.copy()
    Wc[0] = lambda_ / (n + lambda_) + (1 - alpha**2 + beta)

    return SigmaPoints(points=points, Wm=Wm, Wc=Wc)


def sigma_points_julier(
    x: ArrayLike,
    P: ArrayLike,
    kappa: float = 0.0,
) -> SigmaPoints:
    """
    Generate sigma points using Julier's original method.

    Parameters
    ----------
    x : array_like
        State mean.
    P : array_like
        State covariance.
    kappa : float, optional
        Scaling parameter (default: 0, typical: 3-n).

    Returns
    -------
    result : SigmaPoints
        Sigma points and weights.

    Notes
    -----
    Julier's method is a special case of Merwe's with alpha=1, beta=0.
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    P = np.asarray(P, dtype=np.float64)

    n = len(x)

    # Compute matrix square root
    try:
        sqrt_P = np.linalg.cholesky((n + kappa) * P)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(P)
        eigvals = np.maximum(eigvals, 1e-10)
        sqrt_P = eigvecs @ np.diag(np.sqrt((n + kappa) * eigvals))

    # Generate sigma points
    points = np.zeros((2 * n + 1, n), dtype=np.float64)
    points[0] = x

    for i in range(n):
        points[i + 1] = x + sqrt_P[:, i]
        points[n + i + 1] = x - sqrt_P[:, i]

    # Weights (same for mean and covariance in Julier's method)
    W = np.full(2 * n + 1, 1 / (2 * (n + kappa)), dtype=np.float64)
    W[0] = kappa / (n + kappa)

    return SigmaPoints(points=points, Wm=W, Wc=W)


def unscented_transform(
    sigmas: NDArray[np.floating],
    Wm: NDArray[np.floating],
    Wc: NDArray[np.floating],
    noise_cov: Optional[ArrayLike] = None,
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Compute mean and covariance from transformed sigma points.

    Parameters
    ----------
    sigmas : ndarray
        Transformed sigma points, shape (2n+1, m).
    Wm : ndarray
        Weights for mean.
    Wc : ndarray
        Weights for covariance.
    noise_cov : array_like, optional
        Additive noise covariance.

    Returns
    -------
    mean : ndarray
        Weighted mean, shape (m,).
    cov : ndarray
        Weighted covariance, shape (m, m).
    """
    # Weighted mean
    mean = np.sum(Wm[:, np.newaxis] * sigmas, axis=0)

    # Weighted covariance (vectorized: avoids loop over sigma points)
    residuals = sigmas - mean
    # Compute weighted outer products in one operation: (W * residuals)^T @ residuals
    weighted_residuals = np.sqrt(np.abs(Wc))[:, np.newaxis] * residuals
    # Handle negative weights (e.g., from Merwe scaling) by adjusting sign
    cov = weighted_residuals.T @ weighted_residuals
    # Correct for any negative weights (subtract their contribution twice to flip sign)
    neg_mask = Wc < 0
    if np.any(neg_mask):
        neg_residuals = residuals[neg_mask]
        neg_weights = -Wc[neg_mask]
        for i, (w, r) in enumerate(zip(neg_weights, neg_residuals)):
            cov -= 2 * w * np.outer(r, r)

    if noise_cov is not None:
        cov += np.asarray(noise_cov, dtype=np.float64)

    # Ensure symmetry
    cov = (cov + cov.T) / 2

    return mean, cov


def ukf_predict(
    x: ArrayLike,
    P: ArrayLike,
    f: Callable[[NDArray[Any]], NDArray[Any]],
    Q: ArrayLike,
    alpha: float = 1e-3,
    beta: float = 2.0,
    kappa: float = 0.0,
) -> KalmanPrediction:
    """
    Unscented Kalman filter prediction step.

    Parameters
    ----------
    x : array_like
        Current state estimate, shape (n,).
    P : array_like
        Current state covariance, shape (n, n).
    f : callable
        Nonlinear state transition function f(x) -> x_next.
    Q : array_like
        Process noise covariance, shape (n, n).
    alpha : float, optional
        UKF spread parameter.
    beta : float, optional
        UKF distribution parameter.
    kappa : float, optional
        UKF scaling parameter.

    Returns
    -------
    result : KalmanPrediction
        Predicted state and covariance.

    Examples
    --------
    >>> import numpy as np
    >>> def f(x):
    ...     return np.array([x[0] + x[1], x[1]])
    >>> x = np.array([0.0, 1.0])
    >>> P = np.eye(2) * 0.1
    >>> Q = np.eye(2) * 0.01
    >>> pred = ukf_predict(x, P, f, Q)

    See Also
    --------
    ukf_update : UKF measurement update.
    sigma_points_merwe : Sigma point generation.
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)

    # Generate sigma points
    sp = sigma_points_merwe(x, P, alpha, beta, kappa)

    # Transform sigma points through dynamics
    sigmas_f = np.array([f(s) for s in sp.points], dtype=np.float64)

    # Compute predicted mean and covariance
    x_pred, P_pred = unscented_transform(sigmas_f, sp.Wm, sp.Wc, Q)

    return KalmanPrediction(x=x_pred, P=P_pred)


def ukf_update(
    x: ArrayLike,
    P: ArrayLike,
    z: ArrayLike,
    h: Callable[[NDArray[Any]], NDArray[Any]],
    R: ArrayLike,
    alpha: float = 1e-3,
    beta: float = 2.0,
    kappa: float = 0.0,
) -> KalmanUpdate:
    """
    Unscented Kalman filter update step.

    Parameters
    ----------
    x : array_like
        Predicted state estimate, shape (n,).
    P : array_like
        Predicted state covariance, shape (n, n).
    z : array_like
        Measurement, shape (m,).
    h : callable
        Nonlinear measurement function h(x) -> z.
    R : array_like
        Measurement noise covariance, shape (m, m).
    alpha : float, optional
        UKF spread parameter.
    beta : float, optional
        UKF distribution parameter.
    kappa : float, optional
        UKF scaling parameter.

    Returns
    -------
    result : KalmanUpdate
        Updated state, covariance, and innovation statistics.

    See Also
    --------
    ukf_predict : UKF prediction step.
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    P = np.asarray(P, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64).flatten()
    R = np.asarray(R, dtype=np.float64)

    m = len(z)

    # Generate sigma points
    sp = sigma_points_merwe(x, P, alpha, beta, kappa)

    # Transform sigma points through measurement function
    sigmas_h = np.array([h(s) for s in sp.points], dtype=np.float64)

    # Predicted measurement mean and covariance
    z_pred, S = unscented_transform(sigmas_h, sp.Wm, sp.Wc, R)

    # Cross-covariance between state and measurement (vectorized)
    x_residuals = sp.points - x
    z_residuals = sigmas_h - z_pred
    # Weighted cross-covariance: sum of Wc[i] * outer(x_res[i], z_res[i])
    Pxz = (sp.Wc[:, np.newaxis] * x_residuals).T @ z_residuals

    # Kalman gain
    K = np.linalg.solve(S.T, Pxz.T).T

    # Innovation
    y = z - z_pred

    # Updated state and covariance
    x_upd = x + K @ y
    P_upd = P - K @ S @ K.T

    # Ensure symmetry
    P_upd = (P_upd + P_upd.T) / 2

    # Likelihood
    det_S = np.linalg.det(S)
    if det_S > 0:
        mahal_sq = y @ np.linalg.solve(S, y)
        likelihood = np.exp(-0.5 * mahal_sq) / np.sqrt((2 * np.pi) ** m * det_S)
    else:
        likelihood = 0.0

    return KalmanUpdate(
        x=x_upd,
        P=P_upd,
        y=y,
        S=S,
        K=K,
        likelihood=likelihood,
    )


def ckf_spherical_cubature_points(
    n: int,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Generate cubature points for Cubature Kalman Filter.

    Parameters
    ----------
    n : int
        State dimension.

    Returns
    -------
    points : ndarray
        Unit cubature points, shape (2n, n).
    weights : ndarray
        Cubature weights.

    Notes
    -----
    The CKF uses a third-degree spherical-radial cubature rule with
    2n points at ±sqrt(n) along each axis.
    """
    points = np.zeros((2 * n, n), dtype=np.float64)
    sqrt_n = np.sqrt(n)

    for i in range(n):
        points[i, i] = sqrt_n
        points[n + i, i] = -sqrt_n

    weights = np.full(2 * n, 1 / (2 * n), dtype=np.float64)

    return points, weights


def ckf_predict(
    x: ArrayLike,
    P: ArrayLike,
    f: Callable[[NDArray[Any]], NDArray[Any]],
    Q: ArrayLike,
) -> KalmanPrediction:
    """
    Cubature Kalman filter prediction step.

    The CKF uses spherical-radial cubature for numerical integration,
    which is more accurate than the UKF for high-dimensional states.

    Parameters
    ----------
    x : array_like
        Current state estimate.
    P : array_like
        Current state covariance.
    f : callable
        Nonlinear state transition function.
    Q : array_like
        Process noise covariance.

    Returns
    -------
    result : KalmanPrediction
        Predicted state and covariance.

    References
    ----------
    .. [1] Arasaratnam, I. and Haykin, S., "Cubature Kalman Filters",
           IEEE Trans. Automatic Control, 2009.
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)

    n = len(x)

    # Matrix square root
    try:
        sqrt_P = np.linalg.cholesky(P)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(P)
        eigvals = np.maximum(eigvals, 1e-10)
        sqrt_P = eigvecs @ np.diag(np.sqrt(eigvals))

    # Generate cubature points
    unit_pts, weights = ckf_spherical_cubature_points(n)
    cubature_pts = x + (sqrt_P @ unit_pts.T).T

    # Transform through dynamics
    transformed = np.array([f(pt) for pt in cubature_pts], dtype=np.float64)

    # Predicted mean
    x_pred = np.sum(weights[:, np.newaxis] * transformed, axis=0)

    # Predicted covariance (vectorized)
    residuals = transformed - x_pred
    # All CKF weights are equal and positive, so vectorization is straightforward
    weighted_residuals = np.sqrt(weights)[:, np.newaxis] * residuals
    P_pred = weighted_residuals.T @ weighted_residuals + Q

    P_pred = (P_pred + P_pred.T) / 2

    return KalmanPrediction(x=x_pred, P=P_pred)


def ckf_update(
    x: ArrayLike,
    P: ArrayLike,
    z: ArrayLike,
    h: Callable[[NDArray[Any]], NDArray[Any]],
    R: ArrayLike,
) -> KalmanUpdate:
    """
    Cubature Kalman filter update step.

    Parameters
    ----------
    x : array_like
        Predicted state estimate.
    P : array_like
        Predicted state covariance.
    z : array_like
        Measurement.
    h : callable
        Nonlinear measurement function.
    R : array_like
        Measurement noise covariance.

    Returns
    -------
    result : KalmanUpdate
        Updated state and covariance.
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    P = np.asarray(P, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64).flatten()
    R = np.asarray(R, dtype=np.float64)

    n = len(x)
    m = len(z)

    # Matrix square root
    try:
        sqrt_P = np.linalg.cholesky(P)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(P)
        eigvals = np.maximum(eigvals, 1e-10)
        sqrt_P = eigvecs @ np.diag(np.sqrt(eigvals))

    # Generate cubature points
    unit_pts, weights = ckf_spherical_cubature_points(n)
    cubature_pts = x + (sqrt_P @ unit_pts.T).T

    # Transform through measurement function
    transformed = np.array([h(pt) for pt in cubature_pts], dtype=np.float64)

    # Predicted measurement
    z_pred = np.sum(weights[:, np.newaxis] * transformed, axis=0)

    # Innovation covariance (vectorized)
    z_residuals = transformed - z_pred
    weighted_z_residuals = np.sqrt(weights)[:, np.newaxis] * z_residuals
    S = weighted_z_residuals.T @ weighted_z_residuals + R

    # Cross-covariance (vectorized)
    x_residuals = cubature_pts - x
    Pxz = (weights[:, np.newaxis] * x_residuals).T @ z_residuals

    # Kalman gain
    K = np.linalg.solve(S.T, Pxz.T).T

    # Innovation
    y = z - z_pred

    # Update
    x_upd = x + K @ y
    P_upd = P - K @ S @ K.T
    P_upd = (P_upd + P_upd.T) / 2

    # Likelihood
    det_S = np.linalg.det(S)
    if det_S > 0:
        mahal_sq = y @ np.linalg.solve(S, y)
        likelihood = np.exp(-0.5 * mahal_sq) / np.sqrt((2 * np.pi) ** m * det_S)
    else:
        likelihood = 0.0

    return KalmanUpdate(
        x=x_upd,
        P=P_upd,
        y=y,
        S=S,
        K=K,
        likelihood=likelihood,
    )


__all__ = [
    "SigmaPoints",
    "sigma_points_merwe",
    "sigma_points_julier",
    "unscented_transform",
    "ukf_predict",
    "ukf_update",
    "ckf_spherical_cubature_points",
    "ckf_predict",
    "ckf_update",
]
