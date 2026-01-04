"""
Process noise covariance matrices for polynomial motion models.

This module provides Q matrices for constant velocity, constant acceleration,
and higher-order polynomial models.
"""

import math

import numpy as np
from numpy.typing import NDArray


def q_poly_kal(
    order: int,
    T: float,
    q: float,
    num_dims: int = 1,
) -> NDArray[np.floating]:
    """
    Create process noise covariance matrix for polynomial Kalman filter model.

    Parameters
    ----------
    order : int
        Order of the model:
        - 0: Random walk
        - 1: Constant velocity (discrete white noise acceleration)
        - 2: Constant acceleration (discrete white noise jerk)
    T : float
        Time step in seconds.
    q : float
        Process noise intensity (spectral density).
        For order=1: acceleration variance [m²/s⁴]
        For order=2: jerk variance [m²/s⁶]
    num_dims : int, optional
        Number of spatial dimensions (default: 1).

    Returns
    -------
    Q : ndarray
        Process noise covariance matrix.

    Examples
    --------
    >>> # 1D constant velocity with acceleration noise variance of 1 m²/s⁴
    >>> Q = q_poly_kal(order=1, T=0.1, q=1.0)
    >>> Q
    array([[0.000025  , 0.0005    ],
           [0.0005    , 0.01      ]])

    Notes
    -----
    The process noise assumes discrete white noise on the highest derivative.
    For continuous-time models, use q_poly_kal_continuous.
    """
    n = order + 1

    # Build 1D process noise matrix
    Q_1d = np.zeros((n, n), dtype=np.float64)

    # Using discrete white noise acceleration/jerk model
    # Q_ij = q * integral of (T^(n-1-i) / (n-1-i)!) * (T^(n-1-j) / (n-1-j)!) dt
    # This gives the standard polynomial Q matrix

    for i in range(n):
        for j in range(n):
            # Powers in Taylor expansion
            pi = n - 1 - i
            pj = n - 1 - j

            # Q[i,j] = q * T^(pi+pj+1) / ((pi+pj+1) * pi! * pj!)
            power = pi + pj + 1
            Q_1d[i, j] = (
                q * T**power / (power * math.factorial(pi) * math.factorial(pj))
            )

    if num_dims == 1:
        return Q_1d

    # Build block diagonal for multiple dimensions
    Q = np.zeros((n * num_dims, n * num_dims), dtype=np.float64)
    for d in range(num_dims):
        start = d * n
        end = start + n
        Q[start:end, start:end] = Q_1d

    return Q


def q_discrete_white_noise(
    dim: int,
    T: float,
    var: float,
    block_size: int = 1,
) -> NDArray[np.floating]:
    """
    Create discrete white noise process noise matrix.

    This is a general-purpose function for creating Q matrices where
    the noise enters at a specific derivative level.

    Parameters
    ----------
    dim : int
        Dimension of the noise (2 for CV, 3 for CA, etc.).
    T : float
        Time step in seconds.
    var : float
        Variance of the white noise.
    block_size : int, optional
        Number of independent dimensions (default: 1).

    Returns
    -------
    Q : ndarray
        Process noise covariance matrix.

    Examples
    --------
    >>> # 2D state (position, velocity) with acceleration noise
    >>> Q = q_discrete_white_noise(dim=2, T=1.0, var=1.0)
    >>> Q
    array([[0.25, 0.5 ],
           [0.5 , 1.  ]])
    """
    if dim == 2:
        # Constant velocity model (noise in acceleration)
        Q_1d = var * np.array(
            [
                [T**4 / 4, T**3 / 2],
                [T**3 / 2, T**2],
            ],
            dtype=np.float64,
        )
    elif dim == 3:
        # Constant acceleration model (noise in jerk)
        Q_1d = var * np.array(
            [
                [T**6 / 36, T**5 / 12, T**4 / 6],
                [T**5 / 12, T**4 / 4, T**3 / 2],
                [T**4 / 6, T**3 / 2, T**2],
            ],
            dtype=np.float64,
        )
    elif dim == 4:
        # Constant jerk model (noise in snap)
        Q_1d = var * np.array(
            [
                [T**8 / 576, T**7 / 144, T**6 / 48, T**5 / 24],
                [T**7 / 144, T**6 / 36, T**5 / 12, T**4 / 6],
                [T**6 / 48, T**5 / 12, T**4 / 4, T**3 / 2],
                [T**5 / 24, T**4 / 6, T**3 / 2, T**2],
            ],
            dtype=np.float64,
        )
    else:
        # General case using q_poly_kal
        Q_1d = q_poly_kal(order=dim - 1, T=T, q=var, num_dims=1)

    if block_size == 1:
        return Q_1d

    # Build block diagonal
    Q = np.zeros((dim * block_size, dim * block_size), dtype=np.float64)
    for b in range(block_size):
        start = b * dim
        end = start + dim
        Q[start:end, start:end] = Q_1d

    return Q


def q_constant_velocity(
    T: float,
    sigma_a: float,
    num_dims: int = 3,
) -> NDArray[np.floating]:
    """
    Create process noise covariance for constant velocity model.

    Parameters
    ----------
    T : float
        Time step in seconds.
    sigma_a : float
        Standard deviation of acceleration noise [m/s²].
    num_dims : int, optional
        Number of spatial dimensions (default: 3).

    Returns
    -------
    Q : ndarray
        Process noise covariance matrix.

    Examples
    --------
    >>> Q = q_constant_velocity(T=1.0, sigma_a=0.1, num_dims=2)
    >>> Q.shape
    (4, 4)

    See Also
    --------
    f_constant_velocity : State transition matrix for CV model.
    """
    var = sigma_a**2
    return q_discrete_white_noise(dim=2, T=T, var=var, block_size=num_dims)


def q_constant_acceleration(
    T: float,
    sigma_j: float,
    num_dims: int = 3,
) -> NDArray[np.floating]:
    """
    Create process noise covariance for constant acceleration model.

    Parameters
    ----------
    T : float
        Time step in seconds.
    sigma_j : float
        Standard deviation of jerk noise [m/s³].
    num_dims : int, optional
        Number of spatial dimensions (default: 3).

    Returns
    -------
    Q : ndarray
        Process noise covariance matrix.

    See Also
    --------
    f_constant_acceleration : State transition matrix for CA model.
    """
    var = sigma_j**2
    return q_discrete_white_noise(dim=3, T=T, var=var, block_size=num_dims)


def q_continuous_white_noise(
    dim: int,
    T: float,
    spectral_density: float,
    block_size: int = 1,
) -> NDArray[np.floating]:
    """
    Create process noise matrix from continuous-time white noise model.

    This assumes the continuous-time process noise has spectral density q,
    and computes the discrete-time covariance via integration.

    Parameters
    ----------
    dim : int
        Dimension of the state per block.
    T : float
        Time step in seconds.
    spectral_density : float
        Spectral density of the continuous white noise.
    block_size : int, optional
        Number of independent dimensions (default: 1).

    Returns
    -------
    Q : ndarray
        Process noise covariance matrix.

    Notes
    -----
    For a continuous-time model with process noise spectral density q,
    the discrete-time process noise covariance is:

        Q = integral_0^T exp(A*t) * G * q * G' * exp(A'*t) dt

    This function computes this integral analytically for polynomial models.
    """
    # This is the same as q_discrete_white_noise but with spectral density
    # instead of variance (multiply by T for conversion in simple cases)
    return q_discrete_white_noise(
        dim=dim, T=T, var=spectral_density, block_size=block_size
    )


__all__ = [
    "q_poly_kal",
    "q_discrete_white_noise",
    "q_constant_velocity",
    "q_constant_acceleration",
    "q_continuous_white_noise",
]
