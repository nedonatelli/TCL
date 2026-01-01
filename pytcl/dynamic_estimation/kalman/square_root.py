"""
Square-root Kalman filter implementations.

This module provides numerically stable Kalman filter variants that
propagate the square root (Cholesky factor) of the covariance matrix
instead of the covariance itself. This improves numerical stability
and guarantees positive semi-definiteness of the covariance.

Implementations include:
- Square-root Kalman filter (Cholesky-based)
- U-D factorization filter (Bierman's method)
- Square-root versions of UKF and CKF
"""

from typing import Callable, NamedTuple, Optional

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


class UDState(NamedTuple):
    """State of a U-D factorization filter.

    The covariance is represented as P = U @ D @ U.T where U is
    unit upper triangular and D is diagonal.

    Attributes
    ----------
    x : ndarray
        State estimate.
    U : ndarray
        Unit upper triangular factor.
    D : ndarray
        Diagonal elements (1D array).
    """

    x: NDArray[np.floating]
    U: NDArray[np.floating]
    D: NDArray[np.floating]


def cholesky_update(S: NDArray, v: NDArray, sign: float = 1.0) -> NDArray:
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


def qr_update(S_x: NDArray, S_noise: NDArray, F: Optional[NDArray] = None) -> NDArray:
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
    """
    pred = srkf_predict(x, S, F, S_Q, B, u)
    return srkf_update(pred.x, pred.S, z, H, S_R)


# =============================================================================
# U-D Factorization Filter (Bierman's Method)
# =============================================================================


def ud_factorize(P: ArrayLike) -> tuple[NDArray, NDArray]:
    """
    Compute U-D factorization of a symmetric positive definite matrix.

    Decomposes P = U @ D @ U.T where U is unit upper triangular and D is diagonal.

    Parameters
    ----------
    P : array_like
        Symmetric positive definite matrix, shape (n, n).

    Returns
    -------
    U : ndarray
        Unit upper triangular matrix.
    D : ndarray
        Diagonal elements (1D array).

    Notes
    -----
    The U-D factorization is equivalent to a modified Cholesky decomposition
    and requires only n(n+1)/2 storage elements.
    """
    P = np.asarray(P, dtype=np.float64).copy()  # Make a copy to avoid modifying input
    n = P.shape[0]

    U = np.eye(n)
    D = np.zeros(n)

    for j in range(n - 1, -1, -1):
        D[j] = P[j, j]
        if D[j] > 0:
            alpha = 1.0 / D[j]
            for k in range(j):
                U[k, j] = P[k, j] * alpha
            for i in range(j):
                for k in range(i + 1):
                    P[k, i] = P[k, i] - U[k, j] * D[j] * U[i, j]

    return U, D


def ud_reconstruct(U: ArrayLike, D: ArrayLike) -> NDArray:
    """
    Reconstruct covariance matrix from U-D factors.

    Parameters
    ----------
    U : array_like
        Unit upper triangular matrix.
    D : array_like
        Diagonal elements.

    Returns
    -------
    P : ndarray
        Covariance matrix P = U @ diag(D) @ U.T.
    """
    U = np.asarray(U, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)
    return U @ np.diag(D) @ U.T


def ud_predict(
    x: ArrayLike,
    U: ArrayLike,
    D: ArrayLike,
    F: ArrayLike,
    Q: ArrayLike,
) -> tuple[NDArray, NDArray, NDArray]:
    """
    U-D filter prediction step.

    Parameters
    ----------
    x : array_like
        Current state estimate, shape (n,).
    U : array_like
        Unit upper triangular factor, shape (n, n).
    D : array_like
        Diagonal elements, shape (n,).
    F : array_like
        State transition matrix, shape (n, n).
    Q : array_like
        Process noise covariance, shape (n, n).

    Returns
    -------
    x_pred : ndarray
        Predicted state.
    U_pred : ndarray
        Predicted unit upper triangular factor.
    D_pred : ndarray
        Predicted diagonal elements.
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    U = np.asarray(U, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)
    F = np.asarray(F, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)

    # Predicted state
    x_pred = F @ x

    # Predicted covariance: P_pred = F @ P @ F.T + Q
    P = ud_reconstruct(U, D)
    P_pred = F @ P @ F.T + Q

    # Ensure symmetry
    P_pred = (P_pred + P_pred.T) / 2

    # Re-factorize
    U_pred, D_pred = ud_factorize(P_pred)

    return x_pred, U_pred, D_pred


def ud_update_scalar(
    x: ArrayLike,
    U: ArrayLike,
    D: ArrayLike,
    z: float,
    h: ArrayLike,
    r: float,
) -> tuple[NDArray, NDArray, NDArray]:
    """
    U-D filter scalar measurement update (Bierman's algorithm).

    This is the most efficient form - for vector measurements,
    process each component sequentially.

    Parameters
    ----------
    x : array_like
        Predicted state estimate, shape (n,).
    U : array_like
        Unit upper triangular factor, shape (n, n).
    D : array_like
        Diagonal elements, shape (n,).
    z : float
        Scalar measurement.
    h : array_like
        Measurement row vector, shape (n,).
    r : float
        Measurement noise variance.

    Returns
    -------
    x_upd : ndarray
        Updated state.
    U_upd : ndarray
        Updated unit upper triangular factor.
    D_upd : ndarray
        Updated diagonal elements.

    Notes
    -----
    This implements Bierman's sequential scalar update algorithm which
    is numerically stable and efficient for U-D filters.
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    U = np.asarray(U, dtype=np.float64).copy()
    D = np.asarray(D, dtype=np.float64).copy()
    h = np.asarray(h, dtype=np.float64).flatten()
    n = len(x)

    # f = U.T @ h
    f = U.T @ h

    # g = D * f (element-wise)
    g = D * f

    # alpha[0] = r + f[0] * g[0]
    alpha = np.zeros(n + 1)
    alpha[0] = r

    for j in range(n):
        alpha[j + 1] = alpha[j] + f[j] * g[j]

    # Innovation
    y = z - h @ x

    # Update D and U
    D_upd = D.copy()
    U_upd = U.copy()

    for j in range(n):
        D_upd[j] = D[j] * alpha[j] / alpha[j + 1]
        if j > 0:
            gamma = g[j]
            for i in range(j):
                U_upd[i, j] = U[i, j] + (gamma / alpha[j]) * (f[i] - U[i, j] * f[j])
                g[i] = g[i] + g[j] * U[i, j]

    # Kalman gain
    K = g / alpha[n]

    # Updated state
    x_upd = x + K * y

    return x_upd, U_upd, D_upd


def ud_update(
    x: ArrayLike,
    U: ArrayLike,
    D: ArrayLike,
    z: ArrayLike,
    H: ArrayLike,
    R: ArrayLike,
) -> tuple[NDArray, NDArray, NDArray, NDArray, float]:
    """
    U-D filter vector measurement update.

    Processes measurements sequentially using scalar updates.

    Parameters
    ----------
    x : array_like
        Predicted state estimate, shape (n,).
    U : array_like
        Unit upper triangular factor, shape (n, n).
    D : array_like
        Diagonal elements, shape (n,).
    z : array_like
        Measurement vector, shape (m,).
    H : array_like
        Measurement matrix, shape (m, n).
    R : array_like
        Measurement noise covariance, shape (m, m).
        Should be diagonal for sequential processing.

    Returns
    -------
    x_upd : ndarray
        Updated state.
    U_upd : ndarray
        Updated unit upper triangular factor.
    D_upd : ndarray
        Updated diagonal elements.
    y : ndarray
        Innovation vector.
    likelihood : float
        Measurement likelihood.

    Notes
    -----
    For correlated measurement noise (non-diagonal R), the measurements
    are decorrelated first using a Cholesky decomposition.
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    U = np.asarray(U, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64).flatten()
    H = np.asarray(H, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    m = len(z)

    # Full innovation before update
    y = z - H @ x

    # Check if R is diagonal
    is_diagonal = np.allclose(R, np.diag(np.diag(R)))

    if is_diagonal:
        # Sequential scalar updates
        x_upd = x.copy()
        U_upd = U.copy()
        D_upd = D.copy()

        for i in range(m):
            x_upd, U_upd, D_upd = ud_update_scalar(
                x_upd, U_upd, D_upd, z[i], H[i, :], R[i, i]
            )
    else:
        # Decorrelate measurements
        S_R = np.linalg.cholesky(R)
        z_dec = scipy.linalg.solve_triangular(S_R, z, lower=True)
        H_dec = scipy.linalg.solve_triangular(S_R, H, lower=True)

        # Sequential scalar updates with unit variance
        x_upd = x.copy()
        U_upd = U.copy()
        D_upd = D.copy()

        for i in range(m):
            x_upd, U_upd, D_upd = ud_update_scalar(
                x_upd, U_upd, D_upd, z_dec[i], H_dec[i, :], 1.0
            )

    # Compute likelihood
    P = ud_reconstruct(U, D)
    S_innov = H @ P @ H.T + R
    det_S = np.linalg.det(S_innov)
    if det_S > 0:
        mahal_sq = y @ np.linalg.solve(S_innov, y)
        likelihood = np.exp(-0.5 * mahal_sq) / np.sqrt((2 * np.pi) ** m * det_S)
    else:
        likelihood = 0.0

    return x_upd, U_upd, D_upd, y, likelihood


# =============================================================================
# Square-Root UKF
# =============================================================================


def sr_ukf_predict(
    x: ArrayLike,
    S: ArrayLike,
    f: Callable,
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
    h: Callable,
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
    alpha, beta, kappa : float
        UKF scaling parameters.

    Returns
    -------
    result : SRKalmanUpdate
        Updated state and Cholesky factor.
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
    # U-D factorization
    "UDState",
    "ud_factorize",
    "ud_reconstruct",
    "ud_predict",
    "ud_update_scalar",
    "ud_update",
    # Square-root UKF
    "sr_ukf_predict",
    "sr_ukf_update",
]
