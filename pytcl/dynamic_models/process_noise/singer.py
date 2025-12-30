"""
Process noise covariance matrices for Singer acceleration model.
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
