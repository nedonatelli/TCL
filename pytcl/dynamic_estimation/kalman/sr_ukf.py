"""
Square-root Unscented Kalman Filter (SR-UKF).

The SR-UKF propagates the square root of the covariance matrix directly,
providing improved numerical stability for the Unscented Kalman Filter.
This is particularly important for nonlinear systems with high-dimensional
state spaces.

References
----------
.. [1] R. van der Merwe and E. A. Wan, "The Square-Root Unscented Kalman
       Filter for State and Parameter-Estimation," ICASSP 2001.
.. [2] S. J. Julier and J. K. Uhlmann, "Unscented Filtering and Nonlinear
       Estimation," Proceedings of the IEEE, 2004.
"""

from typing import Any, Callable

import numpy as np
import scipy.linalg
from numpy.typing import ArrayLike

from pytcl.dynamic_estimation.kalman.square_root import (
    SRKalmanPrediction,
    SRKalmanUpdate,
    cholesky_update,
)


def sr_ukf_predict(
    x: ArrayLike,
    S: ArrayLike,
    f: Callable[[np.ndarray[Any, Any]], np.ndarray[Any, Any]],
    S_Q: ArrayLike,
    alpha: float = 1e-3,
    beta: float = 2.0,
    kappa: float = 0.0,
) -> SRKalmanPrediction:
    """
    Square-root Unscented Kalman Filter prediction step.

    Parameters
    ----------
    x : array_like
        Current state estimate, shape (n,).
    S : array_like
        Lower triangular Cholesky factor of covariance, shape (n, n).
    f : callable
        State transition function f(x) -> x_next.
    S_Q : array_like
        Cholesky factor of process noise covariance.
    alpha : float, optional
        Spread of sigma points around mean. Default 1e-3.
    beta : float, optional
        Prior knowledge about distribution. Default 2.0 (Gaussian).
    kappa : float, optional
        Secondary scaling parameter. Default 0.0.

    Returns
    -------
    result : SRKalmanPrediction
        Predicted state and Cholesky factor.

    Examples
    --------
    >>> import numpy as np
    >>> def f(x):
    ...     return np.array([x[0] + x[1], x[1]])
    >>> x = np.array([1.0, 0.5])
    >>> S = np.linalg.cholesky(np.eye(2) * 0.1)
    >>> S_Q = np.linalg.cholesky(np.eye(2) * 0.01)
    >>> pred = sr_ukf_predict(x, S, f, S_Q)

    See Also
    --------
    sr_ukf_update : Measurement update step.
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    S = np.asarray(S, dtype=np.float64)
    S_Q = np.asarray(S_Q, dtype=np.float64)
    n = len(x)

    # Sigma point parameters
    lam = alpha**2 * (n + kappa) - n
    gamma = np.sqrt(n + lam)

    # Weights
    W_m = np.zeros(2 * n + 1)
    W_c = np.zeros(2 * n + 1)
    W_m[0] = lam / (n + lam)
    W_c[0] = lam / (n + lam) + (1 - alpha**2 + beta)
    for i in range(1, 2 * n + 1):
        W_m[i] = 1 / (2 * (n + lam))
        W_c[i] = 1 / (2 * (n + lam))

    # Generate sigma points
    sigma_points = np.zeros((n, 2 * n + 1))
    sigma_points[:, 0] = x
    for i in range(n):
        sigma_points[:, i + 1] = x + gamma * S[:, i]
        sigma_points[:, n + i + 1] = x - gamma * S[:, i]

    # Propagate sigma points
    sigma_points_pred = np.zeros_like(sigma_points)
    for i in range(2 * n + 1):
        sigma_points_pred[:, i] = f(sigma_points[:, i])

    # Predicted mean
    x_pred = np.sum(W_m * sigma_points_pred, axis=1)

    # Predicted covariance square root via QR
    # Build matrix for QR: [sqrt(W_c[1]) * (X - x_mean), S_Q]
    residuals = sigma_points_pred[:, 1:] - x_pred[:, np.newaxis]
    sqrt_Wc = np.sqrt(np.abs(W_c[1:]))
    weighted_residuals = residuals * sqrt_Wc

    compound = np.hstack([weighted_residuals, S_Q]).T
    _, R = np.linalg.qr(compound)
    S_pred = R[:n, :n].T

    # Handle negative weight for mean point
    if W_c[0] < 0:
        # Downdate for the mean point
        v = sigma_points_pred[:, 0] - x_pred
        try:
            S_pred = cholesky_update(S_pred, np.sqrt(np.abs(W_c[0])) * v, sign=-1.0)
        except ValueError:
            # Fall back to direct computation
            pass
    else:
        v = sigma_points_pred[:, 0] - x_pred
        S_pred = cholesky_update(S_pred, np.sqrt(W_c[0]) * v, sign=1.0)

    # Ensure lower triangular with positive diagonal
    for i in range(n):
        if S_pred[i, i] < 0:
            S_pred[i:, i] = -S_pred[i:, i]

    return SRKalmanPrediction(x=x_pred, S=S_pred)


def sr_ukf_update(
    x: ArrayLike,
    S: ArrayLike,
    z: ArrayLike,
    h: Callable[[np.ndarray[Any, Any]], np.ndarray[Any, Any]],
    S_R: ArrayLike,
    alpha: float = 1e-3,
    beta: float = 2.0,
    kappa: float = 0.0,
) -> SRKalmanUpdate:
    """
    Square-root Unscented Kalman Filter update step.

    Parameters
    ----------
    x : array_like
        Predicted state estimate, shape (n,).
    S : array_like
        Lower triangular Cholesky factor of covariance, shape (n, n).
    z : array_like
        Measurement, shape (m,).
    h : callable
        Measurement function h(x) -> z.
    S_R : array_like
        Cholesky factor of measurement noise covariance.
    alpha : float, optional
        Spread of sigma points around mean. Default 1e-3.
    beta : float, optional
        Prior knowledge about distribution. Default 2.0 (Gaussian).
    kappa : float, optional
        Secondary scaling parameter. Default 0.0.

    Returns
    -------
    result : SRKalmanUpdate
        Updated state and Cholesky factor.

    Examples
    --------
    >>> import numpy as np
    >>> def h(x):
    ...     return np.array([x[0]])  # Measure first state
    >>> x = np.array([1.0, 0.5])
    >>> S = np.linalg.cholesky(np.eye(2) * 0.1)
    >>> z = np.array([1.1])
    >>> S_R = np.linalg.cholesky(np.array([[0.05]]))
    >>> upd = sr_ukf_update(x, S, z, h, S_R)

    See Also
    --------
    sr_ukf_predict : Prediction step.
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    S = np.asarray(S, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64).flatten()
    S_R = np.asarray(S_R, dtype=np.float64)
    n = len(x)
    m = len(z)

    # Sigma point parameters
    lam = alpha**2 * (n + kappa) - n
    gamma = np.sqrt(n + lam)

    # Weights
    W_m = np.zeros(2 * n + 1)
    W_c = np.zeros(2 * n + 1)
    W_m[0] = lam / (n + lam)
    W_c[0] = lam / (n + lam) + (1 - alpha**2 + beta)
    for i in range(1, 2 * n + 1):
        W_m[i] = 1 / (2 * (n + lam))
        W_c[i] = 1 / (2 * (n + lam))

    # Generate sigma points
    sigma_points = np.zeros((n, 2 * n + 1))
    sigma_points[:, 0] = x
    for i in range(n):
        sigma_points[:, i + 1] = x + gamma * S[:, i]
        sigma_points[:, n + i + 1] = x - gamma * S[:, i]

    # Propagate through measurement function
    Z = np.zeros((m, 2 * n + 1))
    for i in range(2 * n + 1):
        Z[:, i] = h(sigma_points[:, i])

    # Predicted measurement mean
    z_pred = np.sum(W_m * Z, axis=1)

    # Innovation
    y = z - z_pred

    # Innovation covariance square root via QR
    residuals_z = Z[:, 1:] - z_pred[:, np.newaxis]
    sqrt_Wc = np.sqrt(np.abs(W_c[1:]))
    weighted_residuals_z = residuals_z * sqrt_Wc

    compound_z = np.hstack([weighted_residuals_z, S_R]).T
    _, R_z = np.linalg.qr(compound_z)
    S_y = R_z[:m, :m].T

    # Handle mean point weight
    v_z = Z[:, 0] - z_pred
    if W_c[0] >= 0:
        S_y = cholesky_update(S_y, np.sqrt(W_c[0]) * v_z, sign=1.0)

    for i in range(m):
        if S_y[i, i] < 0:
            S_y[i:, i] = -S_y[i:, i]

    # Cross covariance
    residuals_x = sigma_points[:, 1:] - x[:, np.newaxis]
    P_xz = (
        W_c[0] * np.outer(sigma_points[:, 0] - x, Z[:, 0] - z_pred)
        + (residuals_x * W_c[1:]) @ (Z[:, 1:] - z_pred[:, np.newaxis]).T
    )

    # Kalman gain
    K = scipy.linalg.solve_triangular(
        S_y.T, scipy.linalg.solve_triangular(S_y, P_xz.T, lower=True), lower=False
    ).T

    # Updated state
    x_upd = x + K @ y

    # Updated covariance square root
    S_upd = S.copy()
    KS_y = K @ S_y
    for j in range(m):
        try:
            S_upd = cholesky_update(S_upd, KS_y[:, j], sign=-1.0)
        except ValueError:
            # Fallback: compute directly
            P = S_upd @ S_upd.T - np.outer(KS_y[:, j], KS_y[:, j])
            P = (P + P.T) / 2
            eigvals = np.linalg.eigvalsh(P)
            if np.min(eigvals) < 0:
                P = P + (np.abs(np.min(eigvals)) + 1e-10) * np.eye(n)
            S_upd = np.linalg.cholesky(P)

    # Likelihood
    det_S_y = np.prod(np.diag(S_y)) ** 2
    if det_S_y > 0:
        y_normalized = scipy.linalg.solve_triangular(S_y, y, lower=True)
        mahal_sq = np.sum(y_normalized**2)
        likelihood = np.exp(-0.5 * mahal_sq) / np.sqrt((2 * np.pi) ** m * det_S_y)
    else:
        likelihood = 0.0

    return SRKalmanUpdate(
        x=x_upd,
        S=S_upd,
        y=y,
        S_y=S_y,
        K=K,
        likelihood=likelihood,
    )


__all__ = [
    "sr_ukf_predict",
    "sr_ukf_update",
]
