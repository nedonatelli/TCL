"""
Linear Kalman filter implementation.

This module provides the standard linear Kalman filter for systems with
linear dynamics and linear measurements with Gaussian noise.
"""

from typing import NamedTuple, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import cho_factor, cho_solve


class KalmanState(NamedTuple):
    """State of a Kalman filter.

    Attributes
    ----------
    x : ndarray
        State estimate.
    P : ndarray
        State covariance.
    """

    x: NDArray[np.floating]
    P: NDArray[np.floating]


class KalmanPrediction(NamedTuple):
    """Result of Kalman filter prediction step.

    Attributes
    ----------
    x : ndarray
        Predicted state estimate.
    P : ndarray
        Predicted state covariance.
    """

    x: NDArray[np.floating]
    P: NDArray[np.floating]


class KalmanUpdate(NamedTuple):
    """Result of Kalman filter update step.

    Attributes
    ----------
    x : ndarray
        Updated state estimate.
    P : ndarray
        Updated state covariance.
    y : ndarray
        Innovation (measurement residual).
    S : ndarray
        Innovation covariance.
    K : ndarray
        Kalman gain.
    likelihood : float
        Measurement likelihood (for association).
    """

    x: NDArray[np.floating]
    P: NDArray[np.floating]
    y: NDArray[np.floating]
    S: NDArray[np.floating]
    K: NDArray[np.floating]
    likelihood: float


def kf_predict(
    x: ArrayLike,
    P: ArrayLike,
    F: ArrayLike,
    Q: ArrayLike,
    B: Optional[ArrayLike] = None,
    u: Optional[ArrayLike] = None,
) -> KalmanPrediction:
    """
    Kalman filter prediction (time update) step.

    Computes the predicted state and covariance:
        x_pred = F @ x + B @ u
        P_pred = F @ P @ F' + Q

    Parameters
    ----------
    x : array_like
        Current state estimate, shape (n,).
    P : array_like
        Current state covariance, shape (n, n).
    F : array_like
        State transition matrix, shape (n, n).
    Q : array_like
        Process noise covariance, shape (n, n).
    B : array_like, optional
        Control input matrix, shape (n, m).
    u : array_like, optional
        Control input, shape (m,).

    Returns
    -------
    result : KalmanPrediction
        Named tuple with predicted state x and covariance P.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([0.0, 1.0])  # position=0, velocity=1
    >>> P = np.eye(2) * 0.1
    >>> F = np.array([[1, 1], [0, 1]])  # CV model, T=1
    >>> Q = np.array([[0.25, 0.5], [0.5, 1.0]])
    >>> pred = kf_predict(x, P, F, Q)
    >>> pred.x
    array([1., 1.])

    See Also
    --------
    kf_update : Measurement update step.
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    P = np.asarray(P, dtype=np.float64)
    F = np.asarray(F, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)

    # Predicted state
    x_pred = F @ x

    # Add control input if provided
    if B is not None and u is not None:
        B = np.asarray(B, dtype=np.float64)
        u = np.asarray(u, dtype=np.float64).flatten()
        x_pred = x_pred + B @ u

    # Predicted covariance
    P_pred = F @ P @ F.T + Q

    # Ensure symmetry
    P_pred = (P_pred + P_pred.T) / 2

    return KalmanPrediction(x=x_pred, P=P_pred)


def kf_update(
    x: ArrayLike,
    P: ArrayLike,
    z: ArrayLike,
    H: ArrayLike,
    R: ArrayLike,
) -> KalmanUpdate:
    """
    Kalman filter update (measurement update) step.

    Computes the updated state and covariance given a measurement:
        y = z - H @ x           (innovation)
        S = H @ P @ H' + R      (innovation covariance)
        K = P @ H' @ S^{-1}     (Kalman gain)
        x_upd = x + K @ y       (updated state)
        P_upd = (I - K @ H) @ P (updated covariance)

    Parameters
    ----------
    x : array_like
        Predicted state estimate, shape (n,).
    P : array_like
        Predicted state covariance, shape (n, n).
    z : array_like
        Measurement, shape (m,).
    H : array_like
        Measurement matrix, shape (m, n).
    R : array_like
        Measurement noise covariance, shape (m, m).

    Returns
    -------
    result : KalmanUpdate
        Named tuple with updated state, covariance, innovation, etc.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([1.0, 1.0])
    >>> P = np.array([[0.35, 0.5], [0.5, 1.1]])
    >>> z = np.array([1.2])  # position measurement
    >>> H = np.array([[1, 0]])
    >>> R = np.array([[0.1]])
    >>> upd = kf_update(x, P, z, H, R)
    >>> upd.x  # updated state
    array([1.15..., 1.22...])

    See Also
    --------
    kf_predict : Time prediction step.
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    P = np.asarray(P, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64).flatten()
    H = np.asarray(H, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)

    # Innovation (measurement residual)
    y = z - H @ x

    # Innovation covariance
    S = H @ P @ H.T + R

    # Use Cholesky decomposition for efficient solving (reused for gain and likelihood)
    # This is more numerically stable and efficient than repeated solve() calls
    try:
        S_cho = cho_factor(S)
        # Kalman gain: K = P @ H' @ S^{-1}
        K = cho_solve(S_cho, H @ P.T).T
        # Mahalanobis distance for likelihood
        mahal_sq = y @ cho_solve(S_cho, y)
        # Log determinant from Cholesky factor (more stable than det)
        log_det_S = 2 * np.sum(np.log(np.diag(S_cho[0])))
        m = len(z)
        likelihood = np.exp(-0.5 * (mahal_sq + log_det_S + m * np.log(2 * np.pi)))
    except np.linalg.LinAlgError:
        # Fallback if Cholesky fails (S not positive definite)
        K = np.linalg.solve(S.T, H @ P.T).T
        likelihood = 0.0

    # Updated state
    x_upd = x + K @ y

    # Updated covariance (Joseph form for numerical stability)
    I_KH = np.eye(len(x)) - K @ H
    P_upd = I_KH @ P @ I_KH.T + K @ R @ K.T

    # Ensure symmetry
    P_upd = (P_upd + P_upd.T) / 2

    return KalmanUpdate(
        x=x_upd,
        P=P_upd,
        y=y,
        S=S,
        K=K,
        likelihood=likelihood,
    )


def kf_predict_update(
    x: ArrayLike,
    P: ArrayLike,
    z: ArrayLike,
    F: ArrayLike,
    Q: ArrayLike,
    H: ArrayLike,
    R: ArrayLike,
    B: Optional[ArrayLike] = None,
    u: Optional[ArrayLike] = None,
) -> KalmanUpdate:
    """
    Combined Kalman filter prediction and update step.

    Parameters
    ----------
    x : array_like
        Current state estimate.
    P : array_like
        Current state covariance.
    z : array_like
        Measurement.
    F : array_like
        State transition matrix.
    Q : array_like
        Process noise covariance.
    H : array_like
        Measurement matrix.
    R : array_like
        Measurement noise covariance.
    B : array_like, optional
        Control input matrix.
    u : array_like, optional
        Control input.

    Returns
    -------
    result : KalmanUpdate
        Updated state and covariance with innovation statistics.

    See Also
    --------
    kf_predict : Prediction step only.
    kf_update : Update step only.
    """
    pred = kf_predict(x, P, F, Q, B, u)
    return kf_update(pred.x, pred.P, z, H, R)


def kf_smooth(
    x_filt: ArrayLike,
    P_filt: ArrayLike,
    x_pred: ArrayLike,
    P_pred: ArrayLike,
    x_smooth_next: ArrayLike,
    P_smooth_next: ArrayLike,
    F: ArrayLike,
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Rauch-Tung-Striebel (RTS) smoother backward step.

    Given forward filter results and next smoothed state, compute
    current smoothed state.

    Parameters
    ----------
    x_filt : array_like
        Filtered state at current time.
    P_filt : array_like
        Filtered covariance at current time.
    x_pred : array_like
        Predicted state at next time (from filter).
    P_pred : array_like
        Predicted covariance at next time.
    x_smooth_next : array_like
        Smoothed state at next time.
    P_smooth_next : array_like
        Smoothed covariance at next time.
    F : array_like
        State transition matrix.

    Returns
    -------
    x_smooth : ndarray
        Smoothed state at current time.
    P_smooth : ndarray
        Smoothed covariance at current time.

    Notes
    -----
    The RTS smoother runs backward through the filter results:
        G = P_filt @ F' @ P_pred^{-1}
        x_smooth = x_filt + G @ (x_smooth_next - x_pred)
        P_smooth = P_filt + G @ (P_smooth_next - P_pred) @ G'
    """
    x_filt = np.asarray(x_filt, dtype=np.float64).flatten()
    P_filt = np.asarray(P_filt, dtype=np.float64)
    x_pred = np.asarray(x_pred, dtype=np.float64).flatten()
    P_pred = np.asarray(P_pred, dtype=np.float64)
    x_smooth_next = np.asarray(x_smooth_next, dtype=np.float64).flatten()
    P_smooth_next = np.asarray(P_smooth_next, dtype=np.float64)
    F = np.asarray(F, dtype=np.float64)

    # Smoother gain
    G = np.linalg.solve(P_pred.T, F @ P_filt.T).T

    # Smoothed state
    x_smooth = x_filt + G @ (x_smooth_next - x_pred)

    # Smoothed covariance
    P_smooth = P_filt + G @ (P_smooth_next - P_pred) @ G.T

    # Ensure symmetry
    P_smooth = (P_smooth + P_smooth.T) / 2

    return x_smooth, P_smooth


def information_filter_predict(
    y: ArrayLike,
    Y: ArrayLike,
    F: ArrayLike,
    Q: ArrayLike,
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Information filter prediction step.

    The information filter uses the information matrix Y = P^{-1} and
    information vector y = P^{-1} @ x instead of P and x.

    Parameters
    ----------
    y : array_like
        Information vector (Y @ x).
    Y : array_like
        Information matrix (P^{-1}).
    F : array_like
        State transition matrix.
    Q : array_like
        Process noise covariance.

    Returns
    -------
    y_pred : ndarray
        Predicted information vector.
    Y_pred : ndarray
        Predicted information matrix.

    Notes
    -----
    The prediction requires converting to state space, predicting,
    and converting back (information form prediction is complex).
    """
    y = np.asarray(y, dtype=np.float64).flatten()
    Y = np.asarray(Y, dtype=np.float64)
    F = np.asarray(F, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)

    # Convert to state space
    P = np.linalg.inv(Y)
    x = P @ y

    # Predict in state space
    x_pred = F @ x
    P_pred = F @ P @ F.T + Q

    # Convert back to information form
    Y_pred = np.linalg.inv(P_pred)
    y_pred = Y_pred @ x_pred

    return y_pred, Y_pred


def information_filter_update(
    y: ArrayLike,
    Y: ArrayLike,
    z: ArrayLike,
    H: ArrayLike,
    R: ArrayLike,
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Information filter update step.

    Parameters
    ----------
    y : array_like
        Predicted information vector.
    Y : array_like
        Predicted information matrix.
    z : array_like
        Measurement.
    H : array_like
        Measurement matrix.
    R : array_like
        Measurement noise covariance.

    Returns
    -------
    y_upd : ndarray
        Updated information vector.
    Y_upd : ndarray
        Updated information matrix.

    Notes
    -----
    The update in information form is additive:
        Y_upd = Y + H' @ R^{-1} @ H
        y_upd = y + H' @ R^{-1} @ z
    """
    y = np.asarray(y, dtype=np.float64).flatten()
    Y = np.asarray(Y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64).flatten()
    H = np.asarray(H, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)

    # Information contribution from measurement
    R_inv = np.linalg.inv(R)
    I_z = H.T @ R_inv @ H  # Information matrix contribution
    i_z = H.T @ R_inv @ z  # Information vector contribution

    # Update (additive in information form)
    Y_upd = Y + I_z
    y_upd = y + i_z

    return y_upd, Y_upd


__all__ = [
    "KalmanState",
    "KalmanPrediction",
    "KalmanUpdate",
    "kf_predict",
    "kf_update",
    "kf_predict_update",
    "kf_smooth",
    "information_filter_predict",
    "information_filter_update",
]
