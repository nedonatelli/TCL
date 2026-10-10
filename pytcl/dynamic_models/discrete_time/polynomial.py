"""
Polynomial (constant velocity/acceleration) state transition models.

This module provides F matrices for polynomial motion models where the state
consists of position, velocity, and optionally acceleration components.
"""

import math

import numpy as np
from numpy.typing import NDArray


def f_poly_kal(
    order: int,
    T: float,
    num_dims: int = 1,
) -> NDArray[np.floating]:
    """
    Create state transition matrix for polynomial Kalman filter model.

    The polynomial model assumes constant highest derivative. For order=1
    (constant velocity), the state is [position, velocity]. For order=2
    (constant acceleration), state is [position, velocity, acceleration].

    Parameters
    ----------
    order : int
        Order of the model:
        - 0: Constant position (random walk)
        - 1: Constant velocity (nearly constant velocity)
        - 2: Constant acceleration (nearly constant acceleration)
    T : float
        Time step in seconds.
    num_dims : int, optional
        Number of spatial dimensions (default: 1).
        For num_dims > 1, creates block diagonal matrix.

    Returns
    -------
    F : ndarray
        State transition matrix of shape ((order+1)*num_dims, (order+1)*num_dims).

    Examples
    --------
    >>> # 1D constant velocity model
    >>> F = f_poly_kal(order=1, T=0.1)
    >>> F
    array([[1. , 0.1],
           [0. , 1. ]])

    >>> # 2D constant velocity model
    >>> F = f_poly_kal(order=1, T=0.1, num_dims=2)
    >>> F.shape
    (4, 4)

    Notes
    -----
    The state vector is ordered as [x, vx, ax, y, vy, ay, ...] for each dimension.

    **State layout differs from MATLAB.** This is the same model as
    ``FPolyKal`` in the MATLAB Tracker Component Library, but under a
    permutation of the state. ``FPolyKal`` is ``kron(F1, eye(numDim))``,
    which is derivative-major: ``[x, y, vx, vy]``. This function is
    block-diagonal per spatial dimension, which is interleaved:
    ``[x, vx, y, vy]``. For ``order=1, T=1, num_dims=2`` the largest
    elementwise difference between the two matrices is 1.0. They agree
    after reordering the state (``F_matlab = F[p][:, p]`` with
    ``p = [0, 2, 1, 3]``). Neither is wrong, so do not change the numbers
    to make them match.

    :func:`pytcl.dynamic_estimation.batch_estimation.two_point_diff_init`
    returns the derivative-major layout. Applying this ``F`` to its output
    does not raise and silently produces a wrong state: for positions
    (0, 1) then (2, 0) at ``T=1`` the initializer returns
    ``[2, 0, 2, -1]``, this ``F`` maps it to ``[2, 0, 1, -1]``, and the
    correct propagated state is ``[4, -1, 2, -1]``.
    """
    n = order + 1

    # Build 1D transition matrix using Taylor expansion
    F_1d = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(i, n):
            # F[i,j] = T^(j-i) / (j-i)!
            F_1d[i, j] = T ** (j - i) / math.factorial(j - i)

    if num_dims == 1:
        return F_1d

    # Build block diagonal for multiple dimensions
    F = np.zeros((n * num_dims, n * num_dims), dtype=np.float64)
    for d in range(num_dims):
        start = d * n
        end = start + n
        F[start:end, start:end] = F_1d

    return F


def f_constant_velocity(
    T: float,
    num_dims: int = 3,
) -> NDArray[np.floating]:
    """
    Create state transition matrix for constant velocity (CV) model.

    This is a convenience wrapper for f_poly_kal with order=1.

    Parameters
    ----------
    T : float
        Time step in seconds.
    num_dims : int, optional
        Number of spatial dimensions (default: 3).

    Returns
    -------
    F : ndarray
        State transition matrix of shape (2*num_dims, 2*num_dims).

    Examples
    --------
    >>> F = f_constant_velocity(T=1.0, num_dims=2)
    >>> F
    array([[1., 1., 0., 0.],
           [0., 1., 0., 0.],
           [0., 0., 1., 1.],
           [0., 0., 0., 1.]])

    See Also
    --------
    f_poly_kal : General polynomial model.
    q_constant_velocity : Process noise for CV model.
    """
    return f_poly_kal(order=1, T=T, num_dims=num_dims)


def f_constant_acceleration(
    T: float,
    num_dims: int = 3,
) -> NDArray[np.floating]:
    """
    Create state transition matrix for constant acceleration (CA) model.

    This is a convenience wrapper for f_poly_kal with order=2.

    Parameters
    ----------
    T : float
        Time step in seconds.
    num_dims : int, optional
        Number of spatial dimensions (default: 3).

    Returns
    -------
    F : ndarray
        State transition matrix of shape (3*num_dims, 3*num_dims).

    Examples
    --------
    >>> F = f_constant_acceleration(T=1.0, num_dims=1)
    >>> F
    array([[1. , 1. , 0.5],
           [0. , 1. , 1. ],
           [0. , 0. , 1. ]])

    See Also
    --------
    f_poly_kal : General polynomial model.
    q_constant_acceleration : Process noise for CA model.
    """
    return f_poly_kal(order=2, T=T, num_dims=num_dims)


def f_discrete_white_noise_accel(
    T: float,
    num_dims: int = 3,
) -> NDArray[np.floating]:
    """
    Create state transition matrix for discrete white noise acceleration (DWNA).

    This is equivalent to the constant velocity model but emphasizes that
    acceleration is modeled as discrete white noise.

    Parameters
    ----------
    T : float
        Time step in seconds.
    num_dims : int, optional
        Number of spatial dimensions (default: 3).

    Returns
    -------
    F : ndarray
        State transition matrix.

    See Also
    --------
    f_constant_velocity : Same model, different naming convention.
    """
    return f_constant_velocity(T=T, num_dims=num_dims)


def f_piecewise_white_noise_jerk(
    T: float,
    num_dims: int = 3,
) -> NDArray[np.floating]:
    """
    Create state transition matrix for piecewise white noise jerk model.

    This is the constant acceleration model where jerk (derivative of
    acceleration) is modeled as piecewise constant white noise.

    Parameters
    ----------
    T : float
        Time step in seconds.
    num_dims : int, optional
        Number of spatial dimensions (default: 3).

    Returns
    -------
    F : ndarray
        State transition matrix.

    See Also
    --------
    f_constant_acceleration : Same model, different naming convention.
    """
    return f_constant_acceleration(T=T, num_dims=num_dims)


__all__ = [
    "f_poly_kal",
    "f_constant_velocity",
    "f_constant_acceleration",
    "f_discrete_white_noise_accel",
    "f_piecewise_white_noise_jerk",
]
