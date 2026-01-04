"""
Square-root Kalman filter implementations.

This module provides numerically stable Kalman filter variants that
propagate the square root (Cholesky factor) of the covariance matrix
instead of the covariance itself. This improves numerical stability
and guarantees positive semi-definiteness of the covariance.

Implementations include:
- Square-root Kalman filter (Cholesky-based)

For U-D factorization filters, see :mod:`pytcl.dynamic_estimation.kalman.ud_filter`.
For square-root UKF, see :mod:`pytcl.dynamic_estimation.kalman.sr_ukf`.
"""

from typing import NamedTuple, Optional

import numpy as np
import scipy.linalg
from numpy.typing import ArrayLike, NDArray


class SRKalmanState(NamedTuple):
    """State of a square-root Kalman filter.

    Attributes
    ----------
    x : ndarray
        State estimate.
    S : ndarray
        Lower triangular Cholesky factor of covariance (P = S @ S.T).
    """

    x: NDArray[np.floating]
    S: NDArray[np.floating]


class SRKalmanPrediction(NamedTuple):
    """Result of square-root Kalman filter prediction step.

    Attributes
    ----------
    x : ndarray
        Predicted state estimate.
    S : ndarray
        Lower triangular Cholesky factor of predicted covariance.
    """

    x: NDArray[np.floating]
    S: NDArray[np.floating]


class SRKalmanUpdate(NamedTuple):
    """Result of square-root Kalman filter update step.

    Attributes
    ----------
    x : ndarray
        Updated state estimate.
    S : ndarray
        Lower triangular Cholesky factor of updated covariance.
    y : ndarray
        Innovation (measurement residual).
    S_y : ndarray
        Lower triangular Cholesky factor of innovation covariance.
    K : ndarray
        Kalman gain.
    likelihood : float
        Measurement likelihood (for association).
    """

    x: NDArray[np.floating]
    S: NDArray[np.floating]
    y: NDArray[np.floating]
    S_y: NDArray[np.floating]
    K: NDArray[np.floating]
    likelihood: float


def cholesky_update(
    S: NDArray[np.floating], v: NDArray[np.floating], sign: float = 1.0
) -> NDArray[np.floating]:
    """
    Rank-1 Cholesky update/downdate.

    Computes the Cholesky factor of P ± v @ v.T given S where P = S @ S.T.

    Parameters
    ----------
    S : ndarray
        Lower triangular Cholesky factor, shape (n, n).
    v : ndarray
        Vector for rank-1 update, shape (n,).
    sign : float
        +1 for update (addition), -1 for downdate (subtraction).

    Returns
    -------
    S_new : ndarray
        Updated lower triangular Cholesky factor.

    Notes
    -----
    Uses the efficient O(n²) algorithm from [1].

    References
    ----------
    .. [1] P. E. Gill, G. H. Golub, W. Murray, and M. A. Saunders,
           "Methods for modifying matrix factorizations,"
           Mathematics of Computation, vol. 28, pp. 505-535, 1974.

    Examples
    --------
    >>> import numpy as np
    >>> S = np.linalg.cholesky(np.eye(2))
    >>> v = np.array([0.5, 0.5])
    >>> S_updated = cholesky_update(S, v, sign=1.0)
    >>> P_updated = S_updated @ S_updated.T
    >>> np.allclose(P_updated, np.eye(2) + np.outer(v, v))
    True
    """
    S = np.asarray(S, dtype=np.float64).copy()
    v = np.asarray(v, dtype=np.float64).flatten().copy()
    n = len(v)

    if sign > 0:
        # Cholesky update
        for k in range(n):
            r = np.sqrt(S[k, k] ** 2 + v[k] ** 2)
            c = r / S[k, k]
            s = v[k] / S[k, k]
            S[k, k] = r
            if k < n - 1:
                S[k + 1 :, k] = (S[k + 1 :, k] + s * v[k + 1 :]) / c
                v[k + 1 :] = c * v[k + 1 :] - s * S[k + 1 :, k]
    else:
        # Cholesky downdate
        for k in range(n):
            r_sq = S[k, k] ** 2 - v[k] ** 2
            if r_sq < 0:
                raise ValueError("Downdate would make matrix non-positive definite")
            r = np.sqrt(r_sq)
            c = r / S[k, k]
            s = v[k] / S[k, k]
            S[k, k] = r
            if k < n - 1:
                S[k + 1 :, k] = (S[k + 1 :, k] - s * v[k + 1 :]) / c
                v[k + 1 :] = c * v[k + 1 :] - s * S[k + 1 :, k]

    return S


def qr_update(
    S_x: NDArray[np.floating],
    S_noise: NDArray[np.floating],
    F: Optional[NDArray[np.floating]] = None,
) -> NDArray[np.floating]:
    """
    QR-based covariance square root update.

    Computes the Cholesky factor of F @ P @ F.T + Q given S_x (where P = S_x @ S_x.T)
    and S_noise (where Q = S_noise @ S_noise.T).

    Parameters
    ----------
    S_x : ndarray
        Lower triangular Cholesky factor of state covariance, shape (n, n).
    S_noise : ndarray
        Lower triangular Cholesky factor of noise covariance, shape (n, n).
    F : ndarray, optional
        State transition matrix, shape (n, n). If None, uses identity.

    Returns
    -------
    S_new : ndarray
        Lower triangular Cholesky factor of the updated covariance.

    Notes
    -----
    Uses QR decomposition for numerical stability. The compound matrix
    [F @ S_x, S_noise].T is QR decomposed, and R.T gives the new Cholesky factor.

    Examples
    --------
    >>> import numpy as np
    >>> S_x = np.linalg.cholesky(np.eye(2) * 0.1)
    >>> S_noise = np.linalg.cholesky(np.eye(2) * 0.01)
    >>> F = np.array([[1, 1], [0, 1]])
    >>> S_new = qr_update(S_x, S_noise, F)
    """
    S_x = np.asarray(S_x, dtype=np.float64)
    S_noise = np.asarray(S_noise, dtype=np.float64)
    n = S_x.shape[0]

    if F is not None:
        F = np.asarray(F, dtype=np.float64)
        FS = F @ S_x
    else:
        FS = S_x

    # Stack the matrices: [F @ S_x; S_noise]
    compound = np.vstack([FS.T, S_noise.T])

    # QR decomposition
    _, R = np.linalg.qr(compound)

    # The upper triangular R gives us the new Cholesky factor
    # Take absolute values on diagonal to ensure positive
    S_new = R[:n, :n].T
    for i in range(n):
        if S_new[i, i] < 0:
            S_new[i:, i] = -S_new[i:, i]

    return S_new


def srkf_predict(
    x: ArrayLike,
    S: ArrayLike,
    F: ArrayLike,
    S_Q: ArrayLike,
    B: Optional[ArrayLike] = None,
    u: Optional[ArrayLike] = None,
) -> SRKalmanPrediction:
    """
    Square-root Kalman filter prediction step.

    Parameters
    ----------
    x : array_like
        Current state estimate, shape (n,).
    S : array_like
        Lower triangular Cholesky factor of current covariance, shape (n, n).
        Satisfies P = S @ S.T.
    F : array_like
        State transition matrix, shape (n, n).
    S_Q : array_like
        Lower triangular Cholesky factor of process noise, shape (n, n).
        Satisfies Q = S_Q @ S_Q.T.
    B : array_like, optional
        Control input matrix, shape (n, m).
    u : array_like, optional
        Control input, shape (m,).

    Returns
    -------
    result : SRKalmanPrediction
        Named tuple with predicted state x and Cholesky factor S.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([0.0, 1.0])
    >>> S = np.linalg.cholesky(np.eye(2) * 0.1)
    >>> F = np.array([[1, 1], [0, 1]])
    >>> S_Q = np.linalg.cholesky(np.array([[0.25, 0.5], [0.5, 1.0]]))
    >>> pred = srkf_predict(x, S, F, S_Q)
    >>> pred.x
    array([1., 1.])

    See Also
    --------
    srkf_update : Measurement update step.
    kf_predict : Standard Kalman filter prediction.
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    S = np.asarray(S, dtype=np.float64)
    F = np.asarray(F, dtype=np.float64)
    S_Q = np.asarray(S_Q, dtype=np.float64)

    # Predicted state
    x_pred = F @ x

    # Add control input if provided
    if B is not None and u is not None:
        B = np.asarray(B, dtype=np.float64)
        u = np.asarray(u, dtype=np.float64).flatten()
        x_pred = x_pred + B @ u

    # Predicted covariance square root using QR update
    S_pred = qr_update(S, S_Q, F)

    return SRKalmanPrediction(x=x_pred, S=S_pred)


def srkf_update(
    x: ArrayLike,
    S: ArrayLike,
    z: ArrayLike,
    H: ArrayLike,
    S_R: ArrayLike,
) -> SRKalmanUpdate:
    """
    Square-root Kalman filter update step.

    Uses the Potter square-root filter formulation for the measurement update.

    Parameters
    ----------
    x : array_like
        Predicted state estimate, shape (n,).
    S : array_like
        Lower triangular Cholesky factor of predicted covariance, shape (n, n).
    z : array_like
        Measurement, shape (m,).
    H : array_like
        Measurement matrix, shape (m, n).
    S_R : array_like
        Lower triangular Cholesky factor of measurement noise, shape (m, m).
        Satisfies R = S_R @ S_R.T.

    Returns
    -------
    result : SRKalmanUpdate
        Named tuple with updated state, Cholesky factor, innovation, etc.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([1.0, 1.0])
    >>> S = np.linalg.cholesky(np.array([[0.35, 0.5], [0.5, 1.1]]))
    >>> z = np.array([1.2])
    >>> H = np.array([[1, 0]])
    >>> S_R = np.linalg.cholesky(np.array([[0.1]]))
    >>> upd = srkf_update(x, S, z, H, S_R)

    See Also
    --------
    srkf_predict : Prediction step.
    kf_update : Standard Kalman filter update.
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    S = np.asarray(S, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64).flatten()
    H = np.asarray(H, dtype=np.float64)
    S_R = np.asarray(S_R, dtype=np.float64)

    m = len(z)

    # Innovation
    y = z - H @ x

    # Innovation covariance square root via QR
    # S_y such that S_y @ S_y.T = H @ P @ H.T + R
    HS = H @ S
    compound = np.vstack([HS.T, S_R.T])
    _, R_y = np.linalg.qr(compound)
    S_y = R_y[:m, :m].T
    for i in range(m):
        if S_y[i, i] < 0:
            S_y[i:, i] = -S_y[i:, i]

    # Kalman gain: K = P @ H.T @ S_inv where S = S_y @ S_y.T
    # K = S @ S.T @ H.T @ inv(S_y @ S_y.T)
    # Use triangular solves for efficiency
    PHt = S @ S.T @ H.T
    K = scipy.linalg.solve_triangular(
        S_y.T, scipy.linalg.solve_triangular(S_y, PHt.T, lower=True), lower=False
    ).T

    # Updated state
    x_upd = x + K @ y

    # Updated covariance square root
    # P_upd = P - K @ S_y @ S_y.T @ K.T
    # Use sequential rank-1 downdates
    S_upd = S.copy()
    KS_y = K @ S_y
    for j in range(m):
        S_upd = cholesky_update(S_upd, KS_y[:, j], sign=-1.0)

    # Compute likelihood
    det_S_y = np.prod(np.diag(S_y)) ** 2  # det(S_y @ S_y.T) = det(S_y)^2
    if det_S_y > 0:
        # Mahalanobis distance using triangular solve
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


def srkf_predict_update(
    x: ArrayLike,
    S: ArrayLike,
    z: ArrayLike,
    F: ArrayLike,
    S_Q: ArrayLike,
    H: ArrayLike,
    S_R: ArrayLike,
    B: Optional[ArrayLike] = None,
    u: Optional[ArrayLike] = None,
) -> SRKalmanUpdate:
    """
    Combined square-root Kalman filter prediction and update.

    Parameters
    ----------
    x : array_like
        Current state estimate.
    S : array_like
        Cholesky factor of current covariance.
    z : array_like
        Measurement.
    F : array_like
        State transition matrix.
    S_Q : array_like
        Cholesky factor of process noise.
    H : array_like
        Measurement matrix.
    S_R : array_like
        Cholesky factor of measurement noise.
    B : array_like, optional
        Control input matrix.
    u : array_like, optional
        Control input.

    Returns
    -------
    result : SRKalmanUpdate
        Updated state and Cholesky factor.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([0.0, 1.0])
    >>> S = np.linalg.cholesky(np.eye(2) * 0.1)
    >>> F = np.array([[1, 1], [0, 1]])
    >>> S_Q = np.linalg.cholesky(np.eye(2) * 0.01)
    >>> H = np.array([[1, 0]])
    >>> S_R = np.linalg.cholesky(np.array([[0.1]]))
    >>> z = np.array([1.05])
    >>> result = srkf_predict_update(x, S, z, F, S_Q, H, S_R)
    """
    pred = srkf_predict(x, S, F, S_Q, B, u)
    return srkf_update(pred.x, pred.S, z, H, S_R)


# =============================================================================
# Backward compatibility: Re-export from submodules
# =============================================================================

# Square-root UKF (now in sr_ukf.py)
from pytcl.dynamic_estimation.kalman.sr_ukf import (  # noqa: E402
    sr_ukf_predict,
    sr_ukf_update,
)

# U-D factorization filter (now in ud_filter.py)
from pytcl.dynamic_estimation.kalman.ud_filter import (  # noqa: E402
    UDState,
    ud_factorize,
    ud_predict,
    ud_reconstruct,
    ud_update,
    ud_update_scalar,
)

__all__ = [
    # Square-root KF types
    "SRKalmanState",
    "SRKalmanPrediction",
    "SRKalmanUpdate",
    # Utilities
    "cholesky_update",
    "qr_update",
    # Square-root KF
    "srkf_predict",
    "srkf_update",
    "srkf_predict_update",
    # U-D factorization (re-exported for backward compatibility)
    "UDState",
    "ud_factorize",
    "ud_reconstruct",
    "ud_predict",
    "ud_update_scalar",
    "ud_update",
    # Square-root UKF (re-exported for backward compatibility)
    "sr_ukf_predict",
    "sr_ukf_update",
]
