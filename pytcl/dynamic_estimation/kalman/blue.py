"""
BLUE measurement updates for polar and spherical measurements.

Best linear unbiased estimation (BLUE) updates that handle the
nonlinear polar/spherical measurement conversion in closed form with
multiplicative-bias (debiasing) corrections, instead of linearizing or
sampling. Ports of the MATLAB TCL ``BLUEPolarMeasUpdateApprox.m`` and
``BLUESpherMeasUpdateApprox.m``.

References
----------
.. [1] Z. Zhao, X. R. Li, and V. P. Jilkov, "Best linear unbiased
   filtering with nonlinear measurements for target tracking," IEEE
   Transactions on Aerospace and Electronic Systems, vol. 40, no. 4,
   pp. 1324-1336, Oct. 2004.
"""

from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray


class BLUEUpdateResult(NamedTuple):
    """Result of a BLUE measurement update.

    Attributes
    ----------
    x : ndarray
        Updated state, in the same ordering as the input state.
    P : ndarray
        Updated covariance.
    innov : ndarray
        Cartesian-converted innovation.
    pzz : ndarray
        Innovation covariance in converted-measurement space.
    gain : ndarray
        Filter gain (in the internal position/velocity-interleaved
        ordering used by the equations).
    """

    x: NDArray[np.floating]
    P: NDArray[np.floating]
    innov: NDArray[np.floating]
    pzz: NDArray[np.floating]
    gain: NDArray[np.floating]


def blue_polar_meas_update(
    x_state_pred: ArrayLike,
    p_pred: ArrayLike,
    z: ArrayLike,
    r: ArrayLike,
) -> BLUEUpdateResult:
    """
    BLUE measurement update with a 2D polar measurement.

    Parameters
    ----------
    x_state_pred : array_like
        (4,) predicted state ``[x, y, xdot, ydot]``.
    p_pred : array_like
        (4, 4) predicted covariance, same ordering.
    z : array_like
        (2,) one-way polar measurement ``[range, azimuth]``, azimuth
        counterclockwise from the x-axis in radians.
    r : array_like
        (2, 2) diagonal measurement covariance
        ``diag(sigma_r^2, sigma_azimuth^2)``.

    Returns
    -------
    result : BLUEUpdateResult

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([1000.0, 500.0, 10.0, -5.0])
    >>> P = np.diag([100.0, 100.0, 25.0, 25.0])
    >>> z = np.array([np.hypot(1010.0, 495.0), np.arctan2(495.0, 1010.0)])
    >>> R = np.diag([25.0, 1e-4])
    >>> res = blue_polar_meas_update(x, P, z, R)
    >>> bool(np.all(np.diag(res.P)[:2] < np.diag(P)[:2]))
    True

    Notes
    -----
    Port of ``BLUEPolarMeasUpdateApprox.m``, implementing the
    approximate BLUE update of [1]_. The equations run in an internal
    ``[x, xdot, y, ydot]`` ordering; inputs and outputs use
    ``[x, y, xdot, ydot]``.
    """
    x_state_pred = np.asarray(x_state_pred, dtype=np.float64).ravel()
    p_pred = np.asarray(p_pred, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64).ravel()
    r = np.asarray(r, dtype=np.float64)

    perm = [0, 2, 1, 3]
    undo = [0, 2, 1, 3]
    x_bar_state = x_state_pred[perm]
    p_bar = p_pred[np.ix_(perm, perm)]

    x_bar = x_bar_state[0]
    y_bar = x_bar_state[2]

    cov_x = p_bar[0, 0]
    cov_y = p_bar[2, 2]
    cov_xy = p_bar[0, 2]

    sigma_r2 = r[0, 0]
    sigma_theta2 = r[1, 1]

    denom = (x_bar**2 + y_bar**2) ** 3
    exy_rat1 = (
        (y_bar**2 - x_bar**2) / (x_bar**2 + y_bar**2)
        + 2.0 * y_bar**2 * (y_bar**2 - 3.0 * x_bar**2) * cov_x / denom
        + 4.0 * x_bar * y_bar * (x_bar**2 - y_bar**2) * cov_xy / denom
        - 2.0 * x_bar**2 * (x_bar**2 - 3.0 * y_bar**2) * cov_y / denom
    )
    exy_rat2 = x_bar * y_bar / (x_bar**2 + y_bar**2) + 0.5 * (
        2.0 * x_bar * y_bar * (x_bar**2 - 3.0 * y_bar**2) * cov_x / denom
        + (6.0 * x_bar**2 * y_bar**2 - x_bar**4 - y_bar**4) * cov_xy / denom
        + 2.0 * x_bar * y_bar * (y_bar**2 - 3.0 * x_bar**2) * cov_y / denom
    )

    lam1 = np.exp(-sigma_theta2 / 2.0)
    lam2 = 0.5 * (1.0 + np.exp(-2.0 * sigma_theta2))
    lam3 = 0.5 * (1.0 - np.exp(-2.0 * sigma_theta2))

    pzz = np.zeros((2, 2))
    pzz[0, 0] = (
        lam2 * cov_x
        + lam3 * cov_y
        + 0.5 * sigma_r2
        + lam3 * y_bar**2
        + (lam2 - lam1**2) * x_bar**2
        + 0.5 * sigma_r2 * np.exp(-2.0 * sigma_theta2) * (-exy_rat1)
    )
    pzz[1, 1] = (
        lam2 * cov_y
        + lam3 * cov_x
        + 0.5 * sigma_r2
        + lam3 * x_bar**2
        + (lam2 - lam1**2) * y_bar**2
        + 0.5 * sigma_r2 * np.exp(-2.0 * sigma_theta2) * exy_rat1
    )
    pzz[0, 1] = (
        np.exp(-2.0 * sigma_theta2) * cov_xy
        + (np.exp(-2.0 * sigma_theta2) - lam1**2) * x_bar * y_bar
        - sigma_r2 * np.exp(-2.0 * sigma_theta2) * exy_rat2
    )
    pzz[1, 0] = pzz[0, 1]

    cov_xz = lam1 * np.column_stack([p_bar[:, 0], p_bar[:, 2]])
    gain = np.linalg.solve(pzz.T, cov_xz.T).T
    z_cart_pred = lam1 * np.array([x_bar, y_bar])

    z_cart = np.array([z[0] * np.cos(z[1]), z[0] * np.sin(z[1])])
    innov = z_cart - z_cart_pred
    x_update = x_bar_state + gain @ innov
    p_update = p_bar - gain @ pzz @ gain.T

    return BLUEUpdateResult(
        x_update[undo], p_update[np.ix_(undo, undo)], innov, pzz, gain
    )


def blue_spher_meas_update(
    x_state_pred: ArrayLike,
    p_pred: ArrayLike,
    z: ArrayLike,
    r: ArrayLike,
) -> BLUEUpdateResult:
    """
    BLUE measurement update with a 3D spherical measurement.

    Parameters
    ----------
    x_state_pred : array_like
        (6,) predicted state ``[x, y, z, xdot, ydot, zdot]``.
    p_pred : array_like
        (6, 6) predicted covariance, same ordering.
    z : array_like
        (3,) one-way spherical measurement ``[range, azimuth,
        elevation]``, azimuth from the x-axis in the x-y plane,
        elevation up from the x-y plane, radians.
    r : array_like
        (3, 3) diagonal measurement covariance
        ``diag(sigma_r^2, sigma_azimuth^2, sigma_elevation^2)``.

    Returns
    -------
    result : BLUEUpdateResult

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([2e3, 1e3, 500.0, 10.0, -5.0, 2.0])
    >>> P = np.diag([100.0, 100.0, 100.0, 25.0, 25.0, 25.0])
    >>> true_pos = np.array([2010.0, 995.0, 505.0])
    >>> rng_ = np.linalg.norm(true_pos)
    >>> z = np.array([rng_, np.arctan2(995.0, 2010.0),
    ...               np.arcsin(505.0 / rng_)])
    >>> R = np.diag([25.0, 1e-4, 1e-4])
    >>> res = blue_spher_meas_update(x, P, z, R)
    >>> bool(np.all(np.diag(res.P)[:3] < 100.0))
    True

    Notes
    -----
    Port of ``BLUESpherMeasUpdateApprox.m``. The equations run in an
    internal ``[x, xdot, y, ydot, z, zdot]`` ordering; inputs and
    outputs use ``[x, y, z, xdot, ydot, zdot]``.
    """
    x_state_pred = np.asarray(x_state_pred, dtype=np.float64).ravel()
    p_pred = np.asarray(p_pred, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64).ravel()
    r = np.asarray(r, dtype=np.float64)

    perm = [0, 3, 1, 4, 2, 5]
    undo = [0, 2, 4, 1, 3, 5]
    x_bar_state = x_state_pred[perm]
    p_bar = p_pred[np.ix_(perm, perm)]

    x_bar = x_bar_state[0]
    y_bar = x_bar_state[2]
    z_bar = x_bar_state[4]

    sigma_r2 = r[0, 0]
    sigma_theta2 = r[1, 1]
    sigma_phi2 = r[2, 2]

    lam1 = np.exp(-sigma_theta2 / 2.0)
    lam2 = 0.5 * (1.0 + np.exp(-2.0 * sigma_theta2))
    lam3 = 0.5 * (1.0 - np.exp(-2.0 * sigma_theta2))
    mu1 = np.exp(-sigma_phi2 / 2.0)
    mu2 = 0.5 * (1.0 + np.exp(-2.0 * sigma_phi2))
    mu3 = 0.5 * (1.0 - np.exp(-2.0 * sigma_phi2))

    r_bar = np.sqrt(x_bar**2 + y_bar**2 + z_bar**2)
    r1_bar = np.sqrt(x_bar**2 + y_bar**2)
    alpha = (
        mu2 * sigma_r2 / r_bar**2
        + mu3 * z_bar**2 / r1_bar**2
        + mu3 * sigma_r2 * z_bar**2 / (r1_bar**2 * r_bar**2)
    )
    alpha1 = (lam2 * mu2 - lam1**2 * mu1**2) * x_bar**2 + lam3 * mu2 * y_bar**2
    alpha2 = (lam2 * mu2 - lam1**2 * mu1**2) * y_bar**2 + lam3 * mu2 * x_bar**2
    alpha3 = (mu2 - mu1**2) * z_bar**2 + mu3 * (x_bar**2 + y_bar**2)
    alpha4 = (mu2 * (lam2 - lam3) - lam1**2 * mu1**2) * x_bar * y_bar
    alpha5 = (lam1 * (mu2 - mu3) - lam1 * mu1**2) * z_bar

    s = np.zeros((3, 3))
    s[0, 0] = (
        lam2 * mu2 * p_bar[0, 0]
        + lam3 * mu2 * p_bar[2, 2]
        + alpha * (lam2 * x_bar**2 + lam3 * y_bar**2)
        + alpha1
    )
    s[1, 1] = (
        lam2 * mu2 * p_bar[2, 2]
        + lam3 * mu2 * p_bar[0, 0]
        + alpha * (lam3 * x_bar**2 + lam2 * y_bar**2)
        + alpha2
    )
    s[2, 2] = (
        mu2 * p_bar[4, 4]
        + mu3
        * (
            p_bar[0, 0]
            + p_bar[2, 2]
            + mu2 * sigma_r2 * z_bar**2 / r_bar**2
            + mu3 * sigma_r2 * r1_bar**2 / r_bar**2
        )
        + alpha3
    )
    s[0, 1] = (lam2 - lam3) * (mu2 * p_bar[0, 2] + alpha * x_bar * y_bar) + alpha4
    s[1, 0] = s[0, 1]
    s[0, 2] = (
        lam1 * (mu2 - mu3) * (p_bar[0, 4] + sigma_r2 * x_bar * z_bar / r_bar**2)
        + alpha5 * x_bar
    )
    s[2, 0] = s[0, 2]
    s[1, 2] = (
        lam1 * (mu2 - mu3) * (p_bar[2, 4] + sigma_r2 * y_bar * z_bar / r_bar**2)
        + alpha5 * y_bar
    )
    s[2, 1] = s[1, 2]

    cov_xz = mu1 * np.column_stack(
        [lam1 * p_bar[:, 0], lam1 * p_bar[:, 2], p_bar[:, 4]]
    )
    gain = np.linalg.solve(s.T, cov_xz.T).T

    z_cart = np.array(
        [
            z[0] * np.cos(z[1]) * np.cos(z[2]),
            z[0] * np.sin(z[1]) * np.cos(z[2]),
            z[0] * np.sin(z[2]),
        ]
    )
    innov = z_cart - mu1 * np.array([lam1 * x_bar, lam1 * y_bar, z_bar])
    x_update = x_bar_state + gain @ innov
    p_update = p_bar - gain @ s @ gain.T

    return BLUEUpdateResult(
        x_update[undo], p_update[np.ix_(undo, undo)], innov, s, gain
    )


__all__ = [
    "BLUEUpdateResult",
    "blue_polar_meas_update",
    "blue_spher_meas_update",
]
