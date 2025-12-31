"""
Continuous-time dynamic models.

This module provides drift (a) and diffusion (D) functions for continuous-time
stochastic differential equations of the form:

    dx = a(x, t) dt + D(x, t) dW

where W is a Wiener process.
"""

from typing import Optional, Tuple

import numpy as np
import scipy.linalg
from numpy.typing import ArrayLike, NDArray


def drift_constant_velocity(
    x: ArrayLike,
    t: float = 0.0,
    num_dims: int = 3,
) -> NDArray[np.floating]:
    """
    Drift function for constant velocity model.

    Parameters
    ----------
    x : array_like
        State vector [pos_1, vel_1, pos_2, vel_2, ...].
    t : float, optional
        Time (not used, model is time-invariant).
    num_dims : int, optional
        Number of spatial dimensions.

    Returns
    -------
    a : ndarray
        Drift vector (time derivative of state).

    Examples
    --------
    >>> x = np.array([0, 1, 0, 2, 0, 3])  # 3D: pos=0, vel=[1,2,3]
    >>> a = drift_constant_velocity(x, num_dims=3)
    >>> a
    array([1., 0., 2., 0., 3., 0.])
    """
    x = np.asarray(x, dtype=np.float64)
    n = 2  # states per dimension (position, velocity)
    a = np.zeros_like(x)

    for d in range(num_dims):
        idx_pos = d * n
        idx_vel = d * n + 1
        a[idx_pos] = x[idx_vel]  # dx/dt = v
        a[idx_vel] = 0  # dv/dt = 0 (constant velocity)

    return a


def drift_constant_acceleration(
    x: ArrayLike,
    t: float = 0.0,
    num_dims: int = 3,
) -> NDArray[np.floating]:
    """
    Drift function for constant acceleration model.

    Parameters
    ----------
    x : array_like
        State vector [pos_1, vel_1, acc_1, pos_2, vel_2, acc_2, ...].
    t : float, optional
        Time (not used).
    num_dims : int, optional
        Number of spatial dimensions.

    Returns
    -------
    a : ndarray
        Drift vector.
    """
    x = np.asarray(x, dtype=np.float64)
    n = 3  # states per dimension
    a = np.zeros_like(x)

    for d in range(num_dims):
        idx_pos = d * n
        idx_vel = d * n + 1
        idx_acc = d * n + 2
        a[idx_pos] = x[idx_vel]  # dx/dt = v
        a[idx_vel] = x[idx_acc]  # dv/dt = a
        a[idx_acc] = 0  # da/dt = 0

    return a


def drift_singer(
    x: ArrayLike,
    t: float = 0.0,
    tau: float = 10.0,
    num_dims: int = 1,
) -> NDArray[np.floating]:
    """
    Drift function for Singer acceleration model.

    Parameters
    ----------
    x : array_like
        State vector [pos, vel, acc, ...].
    t : float, optional
        Time (not used).
    tau : float, optional
        Maneuver time constant in seconds.
    num_dims : int, optional
        Number of spatial dimensions.

    Returns
    -------
    a : ndarray
        Drift vector.

    Notes
    -----
    The Singer model has acceleration following a first-order Markov process:
        da/dt = -a/tau + w(t)
    """
    x = np.asarray(x, dtype=np.float64)
    n = 3  # states per dimension
    a = np.zeros_like(x)

    for d in range(num_dims):
        idx_pos = d * n
        idx_vel = d * n + 1
        idx_acc = d * n + 2
        a[idx_pos] = x[idx_vel]  # dx/dt = v
        a[idx_vel] = x[idx_acc]  # dv/dt = a
        a[idx_acc] = -x[idx_acc] / tau  # da/dt = -a/tau

    return a


def drift_coordinated_turn_2d(
    x: ArrayLike,
    t: float = 0.0,
) -> NDArray[np.floating]:
    """
    Drift function for 2D coordinated turn model.

    Parameters
    ----------
    x : array_like
        State vector [x, vx, y, vy, omega].
    t : float, optional
        Time (not used).

    Returns
    -------
    a : ndarray
        Drift vector.

    Notes
    -----
    The coordinated turn dynamics are:
        dx/dt = vx
        dvx/dt = -omega * vy
        dy/dt = vy
        dvy/dt = omega * vx
        domega/dt = 0
    """
    x = np.asarray(x, dtype=np.float64)
    pos_x, vel_x, pos_y, vel_y, omega = x

    a = np.array(
        [
            vel_x,  # dx/dt
            -omega * vel_y,  # dvx/dt
            vel_y,  # dy/dt
            omega * vel_x,  # dvy/dt
            0.0,  # domega/dt
        ],
        dtype=np.float64,
    )

    return a


def diffusion_constant_velocity(
    x: ArrayLike,
    t: float = 0.0,
    sigma_a: float = 1.0,
    num_dims: int = 3,
) -> NDArray[np.floating]:
    """
    Diffusion matrix for constant velocity model with acceleration noise.

    Parameters
    ----------
    x : array_like
        State vector (not used, included for interface consistency).
    t : float, optional
        Time (not used).
    sigma_a : float, optional
        Standard deviation of acceleration noise.
    num_dims : int, optional
        Number of spatial dimensions.

    Returns
    -------
    D : ndarray
        Diffusion matrix (noise enters through velocity derivative).
    """
    n = 2 * num_dims
    D = np.zeros((n, num_dims), dtype=np.float64)

    for d in range(num_dims):
        # Noise enters through velocity (acceleration noise)
        D[d * 2 + 1, d] = sigma_a

    return D


def diffusion_constant_acceleration(
    x: ArrayLike,
    t: float = 0.0,
    sigma_j: float = 1.0,
    num_dims: int = 3,
) -> NDArray[np.floating]:
    """
    Diffusion matrix for constant acceleration model with jerk noise.

    Parameters
    ----------
    x : array_like
        State vector (not used).
    t : float, optional
        Time (not used).
    sigma_j : float, optional
        Standard deviation of jerk noise.
    num_dims : int, optional
        Number of spatial dimensions.

    Returns
    -------
    D : ndarray
        Diffusion matrix.
    """
    n = 3 * num_dims
    D = np.zeros((n, num_dims), dtype=np.float64)

    for d in range(num_dims):
        # Noise enters through acceleration (jerk noise)
        D[d * 3 + 2, d] = sigma_j

    return D


def diffusion_singer(
    x: ArrayLike,
    t: float = 0.0,
    sigma_m: float = 1.0,
    tau: float = 10.0,
    num_dims: int = 1,
) -> NDArray[np.floating]:
    """
    Diffusion matrix for Singer model.

    Parameters
    ----------
    x : array_like
        State vector (not used).
    t : float, optional
        Time (not used).
    sigma_m : float, optional
        Standard deviation of target acceleration.
    tau : float, optional
        Maneuver time constant.
    num_dims : int, optional
        Number of spatial dimensions.

    Returns
    -------
    D : ndarray
        Diffusion matrix.

    Notes
    -----
    The diffusion coefficient for Singer is sqrt(2*sigma_m^2/tau).
    """
    n = 3 * num_dims
    D = np.zeros((n, num_dims), dtype=np.float64)

    # Noise intensity for Singer model
    sigma_w = np.sqrt(2 * sigma_m**2 / tau)

    for d in range(num_dims):
        D[d * 3 + 2, d] = sigma_w

    return D


def continuous_to_discrete(
    A: ArrayLike,
    G: ArrayLike,
    Q_c: ArrayLike,
    T: float,
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Convert continuous-time model to discrete-time using Van Loan method.

    Given continuous-time model:
        dx/dt = A*x + G*w,  E[w*w'] = Q_c

    Compute discrete-time model:
        x_{k+1} = F*x_k + v_k,  E[v_k*v_k'] = Q_d

    Parameters
    ----------
    A : array_like
        Continuous-time state matrix.
    G : array_like
        Noise input matrix.
    Q_c : array_like
        Continuous-time process noise covariance.
    T : float
        Time step in seconds.

    Returns
    -------
    F : ndarray
        Discrete-time state transition matrix.
    Q_d : ndarray
        Discrete-time process noise covariance.

    Examples
    --------
    >>> # 1D constant velocity model
    >>> A = np.array([[0, 1], [0, 0]])
    >>> G = np.array([[0], [1]])
    >>> Q_c = np.array([[1.0]])  # acceleration variance
    >>> F, Q_d = continuous_to_discrete(A, G, Q_c, T=0.1)

    Notes
    -----
    Uses the Van Loan method which computes F and Q_d by exponentiating
    the augmented matrix:

        M = [[-A, G*Q_c*G'],
             [ 0,       A']] * T

        exp(M) = [[  *,  F^{-1}*Q_d],
                  [  0,       F'   ]]

    References
    ----------
    .. [1] Van Loan, C.F., "Computing Integrals Involving the Matrix
           Exponential", IEEE Trans. Automatic Control, 1978.
    """
    A = np.asarray(A, dtype=np.float64)
    G = np.asarray(G, dtype=np.float64)
    Q_c = np.asarray(Q_c, dtype=np.float64)

    n = A.shape[0]

    # Build augmented matrix
    GQG = G @ Q_c @ G.T
    M = np.zeros((2 * n, 2 * n), dtype=np.float64)
    M[:n, :n] = -A
    M[:n, n:] = GQG
    M[n:, n:] = A.T

    # Matrix exponential
    M_exp = scipy.linalg.expm(M * T)

    # Extract F and Q_d
    F = M_exp[n:, n:].T
    Q_d = F @ M_exp[:n, n:]

    # Ensure symmetry
    Q_d = (Q_d + Q_d.T) / 2

    return F, Q_d


def discretize_lti(
    A: ArrayLike,
    B: Optional[ArrayLike] = None,
    T: float = 1.0,
) -> Tuple[NDArray[np.floating], Optional[NDArray[np.floating]]]:
    """
    Discretize a linear time-invariant system.

    Given continuous-time system:
        dx/dt = A*x + B*u

    Compute discrete-time system:
        x_{k+1} = F*x_k + G*u_k

    Parameters
    ----------
    A : array_like
        Continuous-time state matrix.
    B : array_like, optional
        Continuous-time input matrix.
    T : float, optional
        Time step in seconds.

    Returns
    -------
    F : ndarray
        Discrete-time state transition matrix.
    G : ndarray or None
        Discrete-time input matrix (None if B is None).
    """
    A = np.asarray(A, dtype=np.float64)
    n = A.shape[0]

    # F = exp(A*T)
    F = scipy.linalg.expm(A * T)

    if B is None:
        return F, None

    B = np.asarray(B, dtype=np.float64)
    m = B.shape[1] if B.ndim > 1 else 1

    # G = integral_0^T exp(A*s) ds * B
    # Use augmented matrix method
    M = np.zeros((n + m, n + m), dtype=np.float64)
    M[:n, :n] = A
    M[:n, n:] = B.reshape(n, -1)

    M_exp = scipy.linalg.expm(M * T)
    G = M_exp[:n, n:]

    return F, G


def state_jacobian_cv(
    x: ArrayLike,
    num_dims: int = 3,
) -> NDArray[np.floating]:
    """
    State Jacobian (A matrix) for constant velocity model.

    Parameters
    ----------
    x : array_like
        State vector (not used, included for interface consistency).
    num_dims : int, optional
        Number of spatial dimensions.

    Returns
    -------
    A : ndarray
        Continuous-time state matrix.
    """
    n = 2  # states per dimension
    A_1d = np.array(
        [
            [0, 1],
            [0, 0],
        ],
        dtype=np.float64,
    )

    if num_dims == 1:
        return A_1d

    A = np.zeros((n * num_dims, n * num_dims), dtype=np.float64)
    for d in range(num_dims):
        start = d * n
        end = start + n
        A[start:end, start:end] = A_1d

    return A


def state_jacobian_ca(
    x: ArrayLike,
    num_dims: int = 3,
) -> NDArray[np.floating]:
    """
    State Jacobian (A matrix) for constant acceleration model.

    Parameters
    ----------
    x : array_like
        State vector (not used).
    num_dims : int, optional
        Number of spatial dimensions.

    Returns
    -------
    A : ndarray
        Continuous-time state matrix.
    """
    n = 3  # states per dimension
    A_1d = np.array(
        [
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0],
        ],
        dtype=np.float64,
    )

    if num_dims == 1:
        return A_1d

    A = np.zeros((n * num_dims, n * num_dims), dtype=np.float64)
    for d in range(num_dims):
        start = d * n
        end = start + n
        A[start:end, start:end] = A_1d

    return A


def state_jacobian_singer(
    x: ArrayLike,
    tau: float = 10.0,
    num_dims: int = 1,
) -> NDArray[np.floating]:
    """
    State Jacobian (A matrix) for Singer acceleration model.

    Parameters
    ----------
    x : array_like
        State vector (not used).
    tau : float, optional
        Maneuver time constant.
    num_dims : int, optional
        Number of spatial dimensions.

    Returns
    -------
    A : ndarray
        Continuous-time state matrix.
    """
    n = 3  # states per dimension
    A_1d = np.array(
        [
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, -1 / tau],
        ],
        dtype=np.float64,
    )

    if num_dims == 1:
        return A_1d

    A = np.zeros((n * num_dims, n * num_dims), dtype=np.float64)
    for d in range(num_dims):
        start = d * n
        end = start + n
        A[start:end, start:end] = A_1d

    return A


__all__ = [
    "drift_constant_velocity",
    "drift_constant_acceleration",
    "drift_singer",
    "drift_coordinated_turn_2d",
    "diffusion_constant_velocity",
    "diffusion_constant_acceleration",
    "diffusion_singer",
    "continuous_to_discrete",
    "discretize_lti",
    "state_jacobian_cv",
    "state_jacobian_ca",
    "state_jacobian_singer",
]
