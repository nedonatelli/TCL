"""
Ensemble Kalman filter (EnKF).

The EnKF propagates an ensemble of state samples instead of a mean and
covariance, making it suited to high-dimensional and strongly nonlinear
problems. Ports of the MATLAB TCL ``EnKFDiscPred.m`` and
``EnKFUpdate.m``.

References
----------
.. [1] S. Gillijns, O. Barrero Mendoza, J. Chandrasekar, B. L. R. De
   Moor, D. S. Bernstein, and A. Ridley, "What is the ensemble Kalman
   filter and how well does it work?" in Proceedings of the American
   Control Conference, Minneapolis, MN, Jun. 2006.
.. [2] G. Evensen, "The ensemble Kalman filter: Theoretical formulation
   and practical implementation," Ocean Dynamics, vol. 53, no. 4, pp.
   343-367, Nov. 2003.
"""

from typing import Callable, NamedTuple, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray


class EnKFPrediction(NamedTuple):
    """Result of :func:`enkf_predict`.

    Attributes
    ----------
    x_ensemble : ndarray
        (x_dim, num_samples) propagated ensemble.
    x_pred : ndarray
        (x_dim,) ensemble mean.
    p_pred : ndarray
        (x_dim, x_dim) ensemble covariance.
    v_samp : ndarray
        The process-noise samples used, for reproducibility.
    """

    x_ensemble: NDArray[np.floating]
    x_pred: NDArray[np.floating]
    p_pred: NDArray[np.floating]
    v_samp: NDArray[np.floating]


class EnKFUpdate(NamedTuple):
    """Result of :func:`enkf_update`.

    Attributes
    ----------
    x_ensemble : ndarray
        (x_dim, num_samples) updated ensemble.
    x_update : ndarray
        (x_dim,) ensemble mean.
    p_update : ndarray
        (x_dim, x_dim) ensemble covariance.
    w_samp : ndarray
        The measurement-noise samples used.
    pzz : ndarray
        (z_dim, z_dim) innovation covariance estimate.
    gain : ndarray
        (x_dim, z_dim) ensemble Kalman gain.
    innov_points : ndarray
        (z_dim, num_samples) per-sample innovations.
    """

    x_ensemble: NDArray[np.floating]
    x_update: NDArray[np.floating]
    p_update: NDArray[np.floating]
    w_samp: NDArray[np.floating]
    pzz: NDArray[np.floating]
    gain: NDArray[np.floating]
    innov_points: NDArray[np.floating]


def _draw_centered(
    sqrt_mat: NDArray[np.floating],
    dim: int,
    num_samples: int,
    rng: Optional[np.random.Generator],
) -> NDArray[np.floating]:
    """Zero-mean noise samples, as in [2]_ (the mean is subtracted)."""
    if rng is None:
        rng = np.random.default_rng()
    samp = sqrt_mat @ rng.standard_normal((dim, num_samples))
    return samp - np.mean(samp, axis=1, keepdims=True)


def enkf_predict(
    x_ensemble: ArrayLike,
    f: Callable,
    sq: ArrayLike,
    filter_type: int = 0,
    state_diff_trans: Optional[Callable] = None,
    v_samp: Optional[ArrayLike] = None,
    rng: Optional[np.random.Generator] = None,
) -> EnKFPrediction:
    """
    Ensemble Kalman filter prediction (time-update) step.

    Parameters
    ----------
    x_ensemble : array_like
        (x_dim, num_samples) ensemble of state samples.
    f : callable
        State transition. With ``filter_type`` 0 it maps ``f(x)`` and
        the noise is additive; with 1 it maps ``f(x, v)``.
    sq : array_like
        (x_dim, x_dim) lower-triangular square root of the process
        noise covariance (used only when ``v_samp`` is not given).
    filter_type : int, optional
        0 (default) additive noise; 1 non-additive noise.
    state_diff_trans : callable, optional
        Transform applied to state differences before averaging
        (wraps circular components). Default identity.
    v_samp : array_like, optional
        (x_dim, num_samples) explicit process-noise samples. When
        given, the step is fully deterministic; when omitted, samples
        are drawn zero-mean from ``rng``.
    rng : numpy.random.Generator, optional
        Source for the noise draws. Default: a fresh generator.

    Returns
    -------
    result : EnKFPrediction

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(7)
    >>> ens = rng.standard_normal((2, 500))
    >>> F = np.array([[1.0, 0.5], [0.0, 1.0]])
    >>> pred = enkf_predict(ens, lambda x: F @ x, 0.1 * np.eye(2), rng=rng)
    >>> pred.x_ensemble.shape
    (2, 500)
    >>> bool(np.all(np.linalg.eigvalsh(pred.p_pred) > 0))
    True

    Notes
    -----
    Port of ``EnKFDiscPred.m`` with one upstream defect fixed and
    documented: the original's multi-output form calls
    ``stateAvgFun``, which is not among its parameters, so requesting
    the mean and covariance crashes; the ensemble mean is used here,
    matching ``EnKFUpdate.m``'s default. The original also silently
    accepts an invalid ``filterType``; here it raises.
    """
    x_ens = np.array(x_ensemble, dtype=np.float64, copy=True)
    if state_diff_trans is None:
        state_diff_trans = lambda x: x  # noqa: E731
    x_dim, num_samples = x_ens.shape

    if v_samp is None:
        v_arr = _draw_centered(
            np.asarray(sq, dtype=np.float64), x_dim, num_samples, rng
        )
    else:
        v_arr = np.asarray(v_samp, dtype=np.float64)

    if filter_type == 0:
        for k in range(num_samples):
            x_ens[:, k] = f(x_ens[:, k]) + v_arr[:, k]
    elif filter_type == 1:
        for k in range(num_samples):
            x_ens[:, k] = f(x_ens[:, k], v_arr[:, k])
    else:
        raise ValueError("Invalid filter type given")

    x_pred = np.mean(x_ens, axis=1)
    cen = state_diff_trans(x_ens - x_pred[:, np.newaxis])
    p_pred = (cen @ cen.T) / (num_samples - 1)
    return EnKFPrediction(x_ens, x_pred, p_pred, v_arr)


def enkf_update(
    x_ensemble: ArrayLike,
    z: ArrayLike,
    sr: ArrayLike,
    h: Callable,
    filter_type: int = 0,
    innov_trans: Optional[Callable] = None,
    meas_avg_fun: Optional[Callable] = None,
    state_diff_trans: Optional[Callable] = None,
    state_avg_fun: Optional[Callable] = None,
    state_trans: Optional[Callable] = None,
    w_samp: Optional[ArrayLike] = None,
    rng: Optional[np.random.Generator] = None,
) -> EnKFUpdate:
    """
    Ensemble Kalman filter measurement-update step.

    Parameters
    ----------
    x_ensemble : array_like
        (x_dim, num_samples) predicted ensemble.
    z : array_like
        (z_dim,) measurement.
    sr : array_like
        (z_dim, z_dim) lower-triangular square root of the measurement
        noise covariance (used only when ``w_samp`` is not given).
    h : callable
        Measurement function; ``h(x)`` for filter types 0 and 2,
        ``h(x, w)`` for type 1.
    filter_type : int, optional
        0 (default): noise perturbs the predicted measurements; 1: the
        same with non-additive noise; 2: the formulation of [1]_ and
        [2]_, where noise instead perturbs the measurement in each
        innovation.
    innov_trans : callable, optional
        Difference function for innovations (wraps circular
        components). Default plain subtraction.
    meas_avg_fun, state_avg_fun : callable, optional
        Averaging functions over measurement / state sample sets.
        Default: the sample mean.
    state_diff_trans, state_trans : callable, optional
        Transforms for state differences and updated states. Default
        identity.
    w_samp : array_like, optional
        (z_dim, num_samples) explicit measurement-noise samples; makes
        the step fully deterministic.
    rng : numpy.random.Generator, optional
        Source for the noise draws. Default: a fresh generator.

    Returns
    -------
    result : EnKFUpdate

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(3)
    >>> ens = np.array([1.0, -0.5])[:, None] + 0.5 * rng.standard_normal((2, 800))
    >>> H = np.array([[1.0, 0.0]])
    >>> upd = enkf_update(ens, np.array([1.2]), 0.3 * np.eye(1),
    ...                   lambda x: H @ x, rng=rng)
    >>> upd.x_update.shape, upd.p_update.shape
    ((2,), (2, 2))

    Notes
    -----
    Port of ``EnKFUpdate.m``. The gain uses the pseudoinverse of the
    innovation covariance, as in the original.
    """
    x_ens = np.array(x_ensemble, dtype=np.float64, copy=True)
    z = np.asarray(z, dtype=np.float64).ravel()
    z_dim = len(z)
    x_dim, num_samples = x_ens.shape

    if innov_trans is None:
        innov_trans = lambda a, b: a - b  # noqa: E731
    if meas_avg_fun is None:
        meas_avg_fun = lambda pts: np.mean(pts, axis=1)  # noqa: E731
    if state_diff_trans is None:
        state_diff_trans = lambda x: x  # noqa: E731
    if state_avg_fun is None:
        state_avg_fun = lambda pts: np.mean(pts, axis=1)  # noqa: E731
    if state_trans is None:
        state_trans = lambda x: x  # noqa: E731

    if w_samp is None:
        w_arr = _draw_centered(
            np.asarray(sr, dtype=np.float64), z_dim, num_samples, rng
        )
    else:
        w_arr = np.asarray(w_samp, dtype=np.float64)

    z_pert = np.zeros((z_dim, num_samples))
    if filter_type == 0:
        for k in range(num_samples):
            z_pert[:, k] = np.atleast_1d(h(x_ens[:, k])) + w_arr[:, k]
    elif filter_type == 1:
        for k in range(num_samples):
            z_pert[:, k] = np.atleast_1d(h(x_ens[:, k], w_arr[:, k]))
    elif filter_type == 2:
        for k in range(num_samples):
            z_pert[:, k] = np.atleast_1d(h(x_ens[:, k]))
    else:
        raise ValueError("Invalid filter type given")

    z_pred = meas_avg_fun(z_pert)
    z_cen = innov_trans(z_pert, np.asarray(z_pred).reshape(z_dim, 1))
    pzz = (z_cen @ z_cen.T) / (num_samples - 1)

    x_pred = state_avg_fun(x_ens)
    x_cen = state_diff_trans(x_ens - np.asarray(x_pred).reshape(x_dim, 1))
    pxz = (x_cen @ z_cen.T) / (num_samples - 1)

    gain = pxz @ np.linalg.pinv(pzz)

    innov_points = np.zeros((z_dim, num_samples))
    if filter_type in (0, 1):
        for k in range(num_samples):
            innov_points[:, k] = innov_trans(z, z_pert[:, k])
            x_ens[:, k] = state_trans(x_ens[:, k] + gain @ innov_points[:, k])
    else:
        for k in range(num_samples):
            innov_points[:, k] = innov_trans(z + w_arr[:, k], z_pert[:, k])
            x_ens[:, k] = state_trans(x_ens[:, k] + gain @ innov_points[:, k])

    x_update = np.asarray(state_avg_fun(x_ens)).ravel()
    x_up_cen = state_diff_trans(x_ens - x_update[:, np.newaxis])
    p_update = (x_up_cen @ x_up_cen.T) / (num_samples - 1)

    return EnKFUpdate(x_ens, x_update, p_update, w_arr, pzz, gain, innov_points)


__all__ = [
    "EnKFPrediction",
    "EnKFUpdate",
    "enkf_predict",
    "enkf_update",
]
