"""
Process noise covariance matrices for Singer acceleration model.

This module provides functions to construct process noise covariance matrices
(Q matrices) for the Singer acceleration motion model. The Singer model
treats target acceleration as a first-order Gauss-Markov process, making it
well-suited for tracking maneuvering targets with random accelerations.

The Singer model is characterized by:
- A maneuver time constant (tau) that controls how quickly acceleration
  correlations decay
- An RMS maneuver level (sigma_m) that sets the expected acceleration magnitude
- State vector [position, velocity, acceleration] per dimension

The model dynamics are:
    da/dt = -a/tau + w(t)

where w(t) is white noise with spectral density 2*sigma_m²/tau.

Available functions:
- ``q_singer``: Generic N-dimensional Singer process noise
- ``q_singer_2d``: 2D Singer model (6x6 state)
- ``q_singer_3d``: 3D Singer model (9x9 state)

These Q matrices are designed to work with the corresponding state transition
matrices in :mod:`pytcl.dynamic_models.singer`.

Examples
--------
Create process noise for tracking a maneuvering aircraft:

>>> from pytcl.dynamic_models.process_noise import q_singer
>>> Q = q_singer(T=1.0, tau=20.0, sigma_m=3.0)  # tau=20s, 3g maneuvers
>>> Q.shape
(3, 3)

For 3D tracking:

>>> Q = q_singer_3d(T=0.1, tau=10.0, sigma_m=2.0)
>>> Q.shape
(9, 9)

See Also
--------
pytcl.dynamic_models.singer : State transition matrices
pytcl.dynamic_models.process_noise.constant_acceleration : CA process noise

References
----------
.. [1] Singer, R.A., "Estimating Optimal Tracking Filter Performance for
       Manned Maneuvering Targets", IEEE Trans. Aerospace and Electronic
       Systems, Vol. AES-6, No. 4, July 1970, pp. 473-483.
.. [2] Bar-Shalom, Y., Li, X.R., and Kirubarajan, T., "Estimation with
       Applications to Tracking and Navigation", Wiley, 2001, Chapter 6.
"""

import numpy as np
from numpy.typing import NDArray


def q_singer(
    T: float,
    tau: float,
    sigma_m: float,
    num_dims: int = 1,
) -> NDArray[np.floating]:
    """
    Create process noise covariance matrix for Singer acceleration model.

    Parameters
    ----------
    T : float
        Time step in seconds.
    tau : float
        Maneuver time constant in seconds.
    sigma_m : float
        Standard deviation of target acceleration [m/s²].
        This is the RMS maneuver level.
    num_dims : int, optional
        Number of spatial dimensions (default: 1).

    Returns
    -------
    Q : ndarray
        Process noise covariance matrix of shape (3*num_dims, 3*num_dims).

    Examples
    --------
    >>> Q = q_singer(T=1.0, tau=10.0, sigma_m=1.0)
    >>> Q.shape
    (3, 3)

    Notes
    -----
    The Singer model assumes acceleration is a first-order Gauss-Markov process:
        da/dt = -a/tau + w(t)

    where w(t) is white noise with spectral density 2*sigma_m²/tau.

    The discrete-time process noise covariance is computed by integrating
    the continuous-time dynamics.

    See Also
    --------
    f_singer : State transition matrix for Singer model.

    References
    ----------
    .. [1] Singer, R.A., "Estimating Optimal Tracking Filter Performance
           for Manned Maneuvering Targets", IEEE Trans. AES, 1970.
    """
    # Use the standard formulas from Bar-Shalom, Li, Kirubarajan
    # "Estimation with Applications to Tracking"
    alpha = np.exp(-T / tau)
    alpha2 = alpha * alpha
    tau2 = tau * tau
    tau3 = tau2 * tau
    tau4 = tau3 * tau
    tau5 = tau4 * tau
    T2 = T * T
    T3 = T2 * T

    q_c = 2 * sigma_m**2 / tau

    Q_1d = np.zeros((3, 3), dtype=np.float64)

    Q_1d[0, 0] = q_c * (
        tau5 / 2 * (1 - alpha2 + 2 * T / tau * alpha2)
        - tau4 * T * (1 - alpha) ** 2
        - tau3 * T2 * (1 - alpha)
        + tau2 * T3 / 3
    )

    Q_1d[0, 1] = q_c * (
        tau4 / 2 * (alpha2 - 1 + 2 * T / tau - 2 * T / tau * alpha)
        + tau3 * T * (1 - alpha)
        - tau2 * T2 / 2
    )
    Q_1d[1, 0] = Q_1d[0, 1]

    Q_1d[0, 2] = q_c * tau3 / 2 * (1 - alpha) ** 2
    Q_1d[2, 0] = Q_1d[0, 2]

    Q_1d[1, 1] = q_c * tau3 / 2 * (4 * alpha - alpha2 - 3 + 2 * T / tau)

    Q_1d[1, 2] = q_c * tau2 / 2 * (1 - 2 * alpha + alpha2)
    Q_1d[2, 1] = Q_1d[1, 2]

    Q_1d[2, 2] = q_c * tau / 2 * (1 - alpha2)

    if num_dims == 1:
        return Q_1d

    # Build block diagonal for multiple dimensions
    n = 3
    Q = np.zeros((n * num_dims, n * num_dims), dtype=np.float64)
    for d in range(num_dims):
        start = d * n
        end = start + n
        Q[start:end, start:end] = Q_1d

    return Q


def q_singer_2d(T: float, tau: float, sigma_m: float) -> NDArray[np.floating]:
    """
    Create process noise covariance for 2D Singer model.

    Parameters
    ----------
    T : float
        Time step in seconds.
    tau : float
        Maneuver time constant in seconds.
    sigma_m : float
        Standard deviation of target acceleration [m/s²].

    Returns
    -------
    Q : ndarray
        Process noise covariance matrix of shape (6, 6).
    """
    return q_singer(T=T, tau=tau, sigma_m=sigma_m, num_dims=2)


def q_singer_3d(T: float, tau: float, sigma_m: float) -> NDArray[np.floating]:
    """
    Create process noise covariance for 3D Singer model.

    Parameters
    ----------
    T : float
        Time step in seconds.
    tau : float
        Maneuver time constant in seconds.
    sigma_m : float
        Standard deviation of target acceleration [m/s²].

    Returns
    -------
    Q : ndarray
        Process noise covariance matrix of shape (9, 9).
    """
    return q_singer(T=T, tau=tau, sigma_m=sigma_m, num_dims=3)


__all__ = [
    "q_singer",
    "q_singer_2d",
    "q_singer_3d",
]
