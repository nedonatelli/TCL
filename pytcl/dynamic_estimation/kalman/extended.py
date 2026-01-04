"""
Extended Kalman Filter (EKF) implementation.

The EKF handles nonlinear dynamics and/or measurements by linearizing
around the current state estimate using Jacobians.
"""

from typing import Any, Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pytcl.dynamic_estimation.kalman.linear import KalmanPrediction, KalmanUpdate


def ekf_predict(
    x: ArrayLike,
    P: ArrayLike,
    f: Callable[[NDArray[Any]], NDArray[Any]],
    F: ArrayLike,
    Q: ArrayLike,
) -> KalmanPrediction:
    """
    Extended Kalman filter prediction step.

    Uses nonlinear dynamics f(x) with Jacobian F for covariance propagation.

    Parameters
    ----------
    x : array_like
        Current state estimate, shape (n,).
    P : array_like
        Current state covariance, shape (n, n).
    f : callable
        Nonlinear state transition function f(x) -> x_next.
    F : array_like
        Jacobian of f evaluated at x, shape (n, n).
    Q : array_like
        Process noise covariance, shape (n, n).

    Returns
    -------
    result : KalmanPrediction
        Predicted state and covariance.

    Examples
    --------
    >>> import numpy as np
    >>> # Coordinated turn dynamics
    >>> def f_turn(x, omega=0.1, T=1.0):
    ...     px, vx, py, vy = x
    ...     return np.array([
    ...         px + np.sin(omega*T)/omega * vx - (1-np.cos(omega*T))/omega * vy,
    ...         np.cos(omega*T) * vx - np.sin(omega*T) * vy,
    ...         py + (1-np.cos(omega*T))/omega * vx + np.sin(omega*T)/omega * vy,
    ...         np.sin(omega*T) * vx + np.cos(omega*T) * vy,
    ...     ])
    >>> x = np.array([0, 10, 0, 0])
    >>> P = np.eye(4)
    >>> F = np.eye(4)  # Jacobian (simplified)
    >>> Q = np.eye(4) * 0.1
    >>> pred = ekf_predict(x, P, f_turn, F, Q)

    See Also
    --------
    ekf_update : EKF measurement update.
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    P = np.asarray(P, dtype=np.float64)
    F = np.asarray(F, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)

    # Predicted state using nonlinear dynamics
    x_pred = np.asarray(f(x), dtype=np.float64).flatten()

    # Predicted covariance using linearized dynamics
    P_pred = F @ P @ F.T + Q

    # Ensure symmetry
    P_pred = (P_pred + P_pred.T) / 2

    return KalmanPrediction(x=x_pred, P=P_pred)


def ekf_update(
    x: ArrayLike,
    P: ArrayLike,
    z: ArrayLike,
    h: Callable[[NDArray[Any]], NDArray[Any]],
    H: ArrayLike,
    R: ArrayLike,
) -> KalmanUpdate:
    """
    Extended Kalman filter update step.

    Uses nonlinear measurement function h(x) with Jacobian H.

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
    H : array_like
        Jacobian of h evaluated at x, shape (m, n).
    R : array_like
        Measurement noise covariance, shape (m, m).

    Returns
    -------
    result : KalmanUpdate
        Updated state, covariance, and innovation statistics.

    Examples
    --------
    >>> import numpy as np
    >>> # Range-bearing measurement
    >>> def h_rb(x):
    ...     px, vx, py, vy = x
    ...     r = np.sqrt(px**2 + py**2)
    ...     theta = np.arctan2(py, px)
    ...     return np.array([r, theta])
    >>> x = np.array([100, 10, 50, 5])
    >>> P = np.eye(4)
    >>> z = np.array([112, 0.46])
    >>> H = np.array([[0.89, 0, 0.45, 0], [-0.004, 0, 0.008, 0]])
    >>> R = np.diag([1.0, 0.01])
    >>> upd = ekf_update(x, P, z, h_rb, H, R)

    See Also
    --------
    ekf_predict : EKF prediction step.
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    P = np.asarray(P, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64).flatten()
    H = np.asarray(H, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)

    # Predicted measurement using nonlinear function
    z_pred = np.asarray(h(x), dtype=np.float64).flatten()

    # Innovation
    y = z - z_pred

    # Innovation covariance
    S = H @ P @ H.T + R

    # Kalman gain
    K = np.linalg.solve(S.T, H @ P.T).T

    # Updated state
    x_upd = x + K @ y

    # Updated covariance (Joseph form)
    I_KH = np.eye(len(x)) - K @ H
    P_upd = I_KH @ P @ I_KH.T + K @ R @ K.T

    # Ensure symmetry
    P_upd = (P_upd + P_upd.T) / 2

    # Likelihood
    m = len(z)
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


def numerical_jacobian(
    f: Callable[[NDArray[Any]], NDArray[Any]],
    x: ArrayLike,
    dx: float = 1e-7,
) -> NDArray[np.floating]:
    """
    Compute Jacobian numerically using central differences.

    Parameters
    ----------
    f : callable
        Function f(x) -> y.
    x : array_like
        Point at which to evaluate Jacobian.
    dx : float, optional
        Step size for finite differences.

    Returns
    -------
    J : ndarray
        Jacobian matrix of shape (m, n) where m = len(f(x)), n = len(x).

    Examples
    --------
    >>> import numpy as np
    >>> def f(x):
    ...     return np.array([x[0]**2, x[0]*x[1]])
    >>> x = np.array([2.0, 3.0])
    >>> J = numerical_jacobian(f, x)
    >>> J  # [[2*x[0], 0], [x[1], x[0]]] = [[4, 0], [3, 2]]
    array([[4., 0.],
           [3., 2.]])
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    f0 = np.asarray(f(x), dtype=np.float64).flatten()

    n = len(x)
    m = len(f0)
    J = np.zeros((m, n), dtype=np.float64)

    for i in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += dx
        x_minus[i] -= dx

        f_plus = np.asarray(f(x_plus), dtype=np.float64).flatten()
        f_minus = np.asarray(f(x_minus), dtype=np.float64).flatten()

        J[:, i] = (f_plus - f_minus) / (2 * dx)

    return J


def ekf_predict_auto(
    x: ArrayLike,
    P: ArrayLike,
    f: Callable[[NDArray[Any]], NDArray[Any]],
    Q: ArrayLike,
    dx: float = 1e-7,
) -> KalmanPrediction:
    """
    EKF prediction with automatic Jacobian computation.

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
    dx : float, optional
        Step size for numerical Jacobian.

    Returns
    -------
    result : KalmanPrediction
        Predicted state and covariance.
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    F = numerical_jacobian(f, x, dx)
    return ekf_predict(x, P, f, F, Q)


def ekf_update_auto(
    x: ArrayLike,
    P: ArrayLike,
    z: ArrayLike,
    h: Callable[[NDArray[Any]], NDArray[Any]],
    R: ArrayLike,
    dx: float = 1e-7,
) -> KalmanUpdate:
    """
    EKF update with automatic Jacobian computation.

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
    dx : float, optional
        Step size for numerical Jacobian.

    Returns
    -------
    result : KalmanUpdate
        Updated state and covariance.
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    H = numerical_jacobian(h, x, dx)
    return ekf_update(x, P, z, h, H, R)


def iterated_ekf_update(
    x: ArrayLike,
    P: ArrayLike,
    z: ArrayLike,
    h: Callable[[NDArray[Any]], NDArray[Any]],
    H_func: Callable[[NDArray[Any]], NDArray[Any]],
    R: ArrayLike,
    max_iter: int = 10,
    tol: float = 1e-6,
) -> KalmanUpdate:
    """
    Iterated Extended Kalman Filter (IEKF) update.

    The IEKF iteratively relinearizes around the updated estimate
    to improve accuracy for highly nonlinear measurements.

    Parameters
    ----------
    x : array_like
        Predicted state estimate.
    P : array_like
        Predicted state covariance.
    z : array_like
        Measurement.
    h : callable
        Nonlinear measurement function h(x) -> z.
    H_func : callable
        Function returning Jacobian H(x) at given state.
    R : array_like
        Measurement noise covariance.
    max_iter : int, optional
        Maximum iterations.
    tol : float, optional
        Convergence tolerance.

    Returns
    -------
    result : KalmanUpdate
        Updated state and covariance after convergence.
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    P = np.asarray(P, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64).flatten()
    R = np.asarray(R, dtype=np.float64)

    x_iter = x.copy()

    for _ in range(max_iter):
        H = np.asarray(H_func(x_iter), dtype=np.float64)
        z_pred = np.asarray(h(x_iter), dtype=np.float64).flatten()

        # Innovation at linearization point
        y = z - z_pred - H @ (x - x_iter)

        # Innovation covariance
        S = H @ P @ H.T + R

        # Kalman gain
        K = np.linalg.solve(S.T, H @ P.T).T

        # Updated state
        x_new = x + K @ y

        # Check convergence
        if np.linalg.norm(x_new - x_iter) < tol:
            x_iter = x_new
            break

        x_iter = x_new

    # Final update with converged linearization
    H = np.asarray(H_func(x_iter), dtype=np.float64)
    z_pred = np.asarray(h(x_iter), dtype=np.float64).flatten()
    y = z - z_pred

    S = H @ P @ H.T + R
    K = np.linalg.solve(S.T, H @ P.T).T

    I_KH = np.eye(len(x)) - K @ H
    P_upd = I_KH @ P @ I_KH.T + K @ R @ K.T
    P_upd = (P_upd + P_upd.T) / 2

    m = len(z)
    det_S = np.linalg.det(S)
    if det_S > 0:
        mahal_sq = y @ np.linalg.solve(S, y)
        likelihood = np.exp(-0.5 * mahal_sq) / np.sqrt((2 * np.pi) ** m * det_S)
    else:
        likelihood = 0.0

    return KalmanUpdate(
        x=x_iter,
        P=P_upd,
        y=y,
        S=S,
        K=K,
        likelihood=likelihood,
    )


__all__ = [
    "ekf_predict",
    "ekf_update",
    "numerical_jacobian",
    "ekf_predict_auto",
    "ekf_update_auto",
    "iterated_ekf_update",
]
