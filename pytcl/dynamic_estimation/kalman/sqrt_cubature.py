"""
Square-root cubature Kalman filter steps.

Prediction and measurement update propagating a lower-triangular
square root of the covariance through arbitrary cubature points, via
triangularization instead of covariance arithmetic. Ports of the
MATLAB TCL ``sqrtDiscCubKalPred.m`` and ``sqrtCubKalUpdate.m``.
Cubature points use pytcl's (num_points, n) row convention.

References
----------
.. [1] I. Arasaratnam and S. Haykin, "Cubature Kalman filters," IEEE
   Transactions on Automatic Control, vol. 54, no. 6, pp. 1254-1269,
   Jun. 2009.
"""

from typing import Callable, NamedTuple, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pytcl.mathematical_functions.basic_matrix import tria_sqrt
from pytcl.mathematical_functions.numerical_integration.cubature_points import (
    spherical_radial_points,
    transform_cubature_points,
)


class SqrtCKFPrediction(NamedTuple):
    """Result of :func:`sqrt_ckf_predict`.

    Attributes
    ----------
    x : ndarray
        (x_dim,) predicted state.
    S : ndarray
        (x_dim, x_dim) lower-triangular root of the predicted
        covariance (P = S @ S.T).
    x_prop_cen_points : ndarray
        (x_dim, num_points) weighted, centered propagated points, as
        needed by the square-root cubature smoother.
    """

    x: NDArray[np.floating]
    S: NDArray[np.floating]
    x_prop_cen_points: NDArray[np.floating]


class SqrtCKFUpdate(NamedTuple):
    """Result of :func:`sqrt_ckf_update`.

    Attributes
    ----------
    x : ndarray
        (x_dim,) updated state.
    S : ndarray
        (x_dim, x_dim) lower-triangular root of the updated covariance.
    innov : ndarray
        (z_dim,) innovation.
    szz : ndarray
        (z_dim, z_dim) root innovation covariance.
    gain : ndarray
        (x_dim, z_dim) filter gain.
    """

    x: NDArray[np.floating]
    S: NDArray[np.floating]
    innov: NDArray[np.floating]
    szz: NDArray[np.floating]
    gain: NDArray[np.floating]


def _points(xi, w, x_dim) -> Tuple[NDArray, NDArray]:
    if xi is None:
        # Third-order spherical-radial (CKF) points: their weights are
        # all positive, which the sqrt(w) factorization below requires
        # (fifth-order weights go negative for x_dim > 4).
        return spherical_radial_points(x_dim, 3)
    return np.asarray(xi, dtype=np.float64), np.asarray(w, dtype=np.float64).ravel()


def sqrt_ckf_predict(
    x_prev: ArrayLike,
    s_prev: ArrayLike,
    f: Callable,
    s_q: ArrayLike,
    xi: Optional[ArrayLike] = None,
    w: Optional[ArrayLike] = None,
    state_diff_trans: Optional[Callable] = None,
    state_avg_fun: Optional[Callable] = None,
    state_trans: Optional[Callable] = None,
) -> SqrtCKFPrediction:
    """
    Square-root cubature Kalman filter prediction step.

    Parameters
    ----------
    x_prev : array_like
        (x_dim,) previous state estimate.
    s_prev : array_like
        (x_dim, x_dim) lower-triangular root of the previous
        covariance.
    f : callable
        State transition ``f(x)``.
    s_q : array_like
        (x_dim, x_dim) lower-triangular root of the process noise
        covariance.
    xi, w : array_like, optional
        Cubature points (num_points, x_dim) and weights for a unit
        Gaussian; the weights must all be positive for the sqrt(w)
        factorization to exist. Default: third-order spherical-radial
        (CKF) points.
    state_diff_trans, state_avg_fun, state_trans : callable, optional
        Hooks for circular state components. Defaults identity /
        weighted mean / identity.

    Returns
    -------
    result : SqrtCKFPrediction

    Examples
    --------
    >>> import numpy as np
    >>> F = np.array([[1.0, 1.0], [0.0, 1.0]])
    >>> pred = sqrt_ckf_predict(np.array([1.0, -0.5]), np.eye(2),
    ...                         lambda x: F @ x, 0.1 * np.eye(2))
    >>> np.allclose(pred.S @ pred.S.T,
    ...             F @ F.T + 0.01 * np.eye(2), atol=1e-8)
    True

    Notes
    -----
    Port of ``sqrtDiscCubKalPred.m``.
    """
    x_prev = np.asarray(x_prev, dtype=np.float64).ravel()
    s_prev = np.asarray(s_prev, dtype=np.float64)
    s_q = np.asarray(s_q, dtype=np.float64)
    x_dim = len(x_prev)
    xi_arr, w_arr = _points(xi, w, x_dim)
    if state_diff_trans is None:
        state_diff_trans = lambda x: x  # noqa: E731
    if state_avg_fun is None:
        state_avg_fun = lambda pts, wt: pts @ wt  # noqa: E731
    if state_trans is None:
        state_trans = lambda x: x  # noqa: E731

    n_pts = len(w_arr)
    pts, _ = transform_cubature_points(xi_arr, w_arr, x_prev, s_prev)
    x_points = np.asarray(state_trans(pts.T), dtype=np.float64)
    x_prop = np.zeros((x_dim, n_pts))
    for k in range(n_pts):
        x_prop[:, k] = f(x_points[:, k])
    x_pred = np.asarray(state_avg_fun(x_prop, w_arr)).ravel()
    x_diff = state_diff_trans(x_prop - x_pred[:, np.newaxis])
    x_prop_cen = x_diff * np.sqrt(w_arr)[np.newaxis, :]
    s_pred = tria_sqrt(np.hstack([x_prop_cen, s_q]))
    return SqrtCKFPrediction(x_pred, s_pred, x_prop_cen)


def sqrt_ckf_update(
    x_pred: ArrayLike,
    s_pred: ArrayLike,
    z: ArrayLike,
    s_r: ArrayLike,
    h: Callable,
    xi: Optional[ArrayLike] = None,
    w: Optional[ArrayLike] = None,
    innov_trans: Optional[Callable] = None,
    meas_avg_fun: Optional[Callable] = None,
    state_diff_trans: Optional[Callable] = None,
    state_trans: Optional[Callable] = None,
) -> SqrtCKFUpdate:
    """
    Square-root cubature Kalman filter measurement update.

    Parameters
    ----------
    x_pred : array_like
        (x_dim,) predicted state.
    s_pred : array_like
        (x_dim, x_dim) lower-triangular root of the predicted
        covariance.
    z : array_like
        (z_dim,) measurement.
    s_r : array_like
        (z_dim, z_dim) lower-triangular root of the measurement noise
        covariance.
    h : callable
        Measurement function ``h(x)``.
    xi, w : array_like, optional
        Cubature points and weights (all weights positive). Default:
        third-order spherical-radial (CKF) points.
    innov_trans, meas_avg_fun, state_diff_trans, state_trans : callable, optional
        Hooks for circular components.

    Returns
    -------
    result : SqrtCKFUpdate

    Examples
    --------
    >>> import numpy as np
    >>> H = np.array([[1.0, 0.0]])
    >>> up = sqrt_ckf_update(np.array([1.0, -0.5]), np.eye(2),
    ...                      np.array([1.3]), 0.5 * np.eye(1),
    ...                      lambda x: H @ x)
    >>> bool(1.0 < up.x[0] < 1.3)
    True

    Notes
    -----
    Port of ``sqrtCubKalUpdate.m``.
    """
    x_pred = np.asarray(x_pred, dtype=np.float64).ravel()
    s_pred = np.asarray(s_pred, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64).ravel()
    s_r = np.asarray(s_r, dtype=np.float64)
    x_dim = len(x_pred)
    z_dim = len(z)
    xi_arr, w_arr = _points(xi, w, x_dim)
    if innov_trans is None:
        innov_trans = lambda a, b: a - b  # noqa: E731
    if meas_avg_fun is None:
        meas_avg_fun = lambda pts, wt: pts @ wt  # noqa: E731
    if state_diff_trans is None:
        state_diff_trans = lambda x: x  # noqa: E731
    if state_trans is None:
        state_trans = lambda x: x  # noqa: E731

    n_pts = len(w_arr)
    sqrt_w = np.sqrt(w_arr)
    pts, _ = transform_cubature_points(xi_arr, w_arr, x_pred, s_pred)
    x_points = np.asarray(state_trans(pts.T), dtype=np.float64)
    x_cen = state_diff_trans(x_points - x_pred[:, np.newaxis]) * sqrt_w[np.newaxis, :]
    z_points = np.zeros((z_dim, n_pts))
    for k in range(n_pts):
        z_points[:, k] = np.atleast_1d(h(x_points[:, k]))
    z_pred = np.asarray(meas_avg_fun(z_points, w_arr)).ravel()
    z_cen = innov_trans(z_points, z_pred[:, np.newaxis]) * sqrt_w[np.newaxis, :]
    szz = tria_sqrt(np.hstack([z_cen, s_r]))
    pxz = x_cen @ z_cen.T
    # MATLAB's W = (Pxz/Szz')/Szz.
    tmp = np.linalg.solve(szz, pxz.T).T
    gain = np.linalg.solve(szz.T, tmp.T).T
    innov = innov_trans(z, z_pred)
    x_up = np.asarray(state_trans(x_pred + gain @ innov)).ravel()
    s_up = tria_sqrt(np.hstack([state_diff_trans(x_cen - gain @ z_cen), gain @ s_r]))
    return SqrtCKFUpdate(x_up, s_up, innov, szz, gain)


__all__ = [
    "SqrtCKFPrediction",
    "SqrtCKFUpdate",
    "sqrt_ckf_predict",
    "sqrt_ckf_update",
]
