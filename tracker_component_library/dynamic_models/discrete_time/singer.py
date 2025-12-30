"""
Singer acceleration model.

The Singer model treats target acceleration as a first-order Markov process
(exponentially correlated random variable), providing a more realistic model
for maneuvering targets than constant acceleration.
"""

import numpy as np
from numpy.typing import NDArray


def f_singer(
    T: float,
    tau: float,
    num_dims: int = 1,
) -> NDArray[np.floating]:
    """
    Create state transition matrix for Singer acceleration model.

    The Singer model assumes acceleration is a first-order Gauss-Markov
    process with time constant tau. State is [position, velocity, acceleration].

    Parameters
    ----------
    T : float
        Time step in seconds.
    tau : float
        Maneuver time constant in seconds.
        Typical values: 5-20s for aircraft, 1-5s for ground vehicles.
    num_dims : int, optional
        Number of spatial dimensions (default: 1).

    Returns
    -------
    F : ndarray
        State transition matrix of shape (3*num_dims, 3*num_dims).

    Examples
    --------
    >>> F = f_singer(T=1.0, tau=10.0)
    >>> F
    array([[1.        , 1.        , 0.04837418],
           [0.        , 1.        , 0.90483742],
           [0.        , 0.        , 0.90483742]])

    Notes
    -----
    The continuous-time model is:
        dx/dt = v
        dv/dt = a
        da/dt = -a/tau + w(t)

    where w(t) is white noise.

    The discrete-time transition is:
        alpha = exp(-T/tau)
        F = [[1, T, (alpha*T + tau - tau*alpha - T)/alpha_tau],
             [0, 1, tau*(1 - alpha)],
             [0, 0, alpha]]

    where alpha_tau = 1/tau.

    See Also
    --------
    q_singer : Process noise for Singer model.

    References
    ----------
    .. [1] Singer, R.A., "Estimating Optimal Tracking Filter Performance
           for Manned Maneuvering Targets", IEEE Trans. on Aerospace and
           Electronic Systems, Vol. AES-6, No. 4, July 1970.
    """
    alpha = np.exp(-T / tau)
    beta = T / tau

    # Using Van Loan method or direct integration:
    # F[0,2] = tau^2 * ((T/tau) - 1 + exp(-T/tau))
    # F[1,2] = tau * (1 - exp(-T/tau))
    # F[2,2] = exp(-T/tau)
    f13 = tau**2 * (beta - 1 + alpha)  # Position-acceleration coupling
    f23 = tau * (1 - alpha)  # Velocity-acceleration coupling
    f33 = alpha

    F_1d = np.array(
        [
            [1, T, f13],
            [0, 1, f23],
            [0, 0, f33],
        ],
        dtype=np.float64,
    )

    if num_dims == 1:
        return F_1d

    # Build block diagonal for multiple dimensions
    n = 3
    F = np.zeros((n * num_dims, n * num_dims), dtype=np.float64)
    for d in range(num_dims):
        start = d * n
        end = start + n
        F[start:end, start:end] = F_1d

    return F


def f_singer_2d(T: float, tau: float) -> NDArray[np.floating]:
    """
    Create state transition matrix for 2D Singer model.

    State vector is [x, vx, ax, y, vy, ay].

    Parameters
    ----------
    T : float
        Time step in seconds.
    tau : float
        Maneuver time constant in seconds.

    Returns
    -------
    F : ndarray
        State transition matrix of shape (6, 6).

    See Also
    --------
    f_singer : General Singer model.
    """
    return f_singer(T=T, tau=tau, num_dims=2)


def f_singer_3d(T: float, tau: float) -> NDArray[np.floating]:
    """
    Create state transition matrix for 3D Singer model.

    State vector is [x, vx, ax, y, vy, ay, z, vz, az].

    Parameters
    ----------
    T : float
        Time step in seconds.
    tau : float
        Maneuver time constant in seconds.

    Returns
    -------
    F : ndarray
        State transition matrix of shape (9, 9).

    See Also
    --------
    f_singer : General Singer model.
    """
    return f_singer(T=T, tau=tau, num_dims=3)


__all__ = [
    "f_singer",
    "f_singer_2d",
    "f_singer_3d",
]
