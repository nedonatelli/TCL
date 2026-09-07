"""
Monte-Carlo Kalman filter (the MATLAB TCL "QMC" family).

Prediction and update steps that approximate the moment integrals by
sampling the prior, plus the decomposed measurement-prediction /
gain / update-with-prediction parts that let one measurement
prediction serve several measurement updates. Ports of
``discQMCKalPred.m``, ``QMCKalUpdate.m``, ``QMCKalMeasPred.m``,
``QMCKalUpdateWithPred.m`` and ``calcQMCKalmanGain.m``. Despite the
family name, the originals draw plain pseudo-random normal samples,
not quasi-Monte-Carlo sequences; the port is faithful to that.

References
----------
.. [1] D. F. Crouse, "Basic tracking using nonlinear 3D monostatic and
   bistatic measurements," IEEE Aerospace and Electronic Systems
   Magazine, vol. 29, no. 8, Part II, pp. 4-53, Aug. 2014.
"""

from typing import Callable, NamedTuple, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pytcl.mathematical_functions.basic_matrix import chol_semi_def


class QMCKFPrediction(NamedTuple):
    """Result of :func:`qmc_kf_predict`.

    Attributes
    ----------
    x : ndarray
        (x_dim,) predicted state.
    P : ndarray
        (x_dim, x_dim) predicted covariance.
    """

    x: NDArray[np.floating]
    P: NDArray[np.floating]


class QMCKFUpdateResult(NamedTuple):
    """Result of :func:`qmc_kf_update` and
    :func:`qmc_kf_update_with_pred`.

    Attributes
    ----------
    x : ndarray
        Updated state, (x_dim,) or (x_dim, num_comp).
    P : ndarray
        Updated covariance, (x_dim, x_dim) or (x_dim, x_dim, num_comp).
    innov : ndarray
        Innovation(s).
    pzz : ndarray
        Innovation covariance(s).
    gain : ndarray
        Kalman gain(s).
    """

    x: NDArray[np.floating]
    P: NDArray[np.floating]
    innov: NDArray[np.floating]
    pzz: NDArray[np.floating]
    gain: NDArray[np.floating]


class QMCMeasPredInfo(NamedTuple):
    """Measurement-prediction information from
    :func:`qmc_kf_meas_pred`, consumed by
    :func:`qmc_kf_update_with_pred` and :func:`calc_qmc_kalman_gain`.

    Attributes
    ----------
    z_pred : ndarray
        (z_dim, num_comp) predicted measurement per component.
    pz_pred : ndarray
        (z_dim, z_dim, num_comp) measurement-prediction covariances
        (without R).
    pxz : ndarray
        (x_dim, z_dim, num_comp) cross covariances.
    x_pred : ndarray
        The (x_dim, num_comp) predicted states passed in.
    p_pred : ndarray
        The (x_dim, x_dim, num_comp) predicted covariances passed in.
    innov_trans : callable
        The innovation difference function used.
    state_trans : callable
        The state transform used.
    """

    z_pred: NDArray[np.floating]
    pz_pred: NDArray[np.floating]
    pxz: NDArray[np.floating]
    x_pred: NDArray[np.floating]
    p_pred: NDArray[np.floating]
    innov_trans: Callable
    state_trans: Callable


def _weighted_mean(points, w):
    return points @ np.asarray(w).ravel()


def qmc_kf_predict(
    x_prev: ArrayLike,
    p_prev: ArrayLike,
    f: Callable,
    q: ArrayLike,
    num_samples: int = 100,
    state_diff_trans: Optional[Callable] = None,
    state_avg_fun: Optional[Callable] = None,
    state_trans: Optional[Callable] = None,
    rng: Optional[np.random.Generator] = None,
) -> QMCKFPrediction:
    """
    Monte-Carlo Kalman filter prediction step.

    Samples the prior, propagates each sample through the possibly
    nonlinear dynamics, and re-estimates the moments.

    Parameters
    ----------
    x_prev : array_like
        (x_dim,) previous state estimate.
    p_prev : array_like
        (x_dim, x_dim) previous covariance (may be semidefinite).
    f : callable
        State transition ``f(x)``.
    q : array_like
        (x_dim, x_dim) process noise covariance (added, not sampled).
    num_samples : int, optional
        Number of Monte-Carlo samples. Default 100.
    state_diff_trans, state_avg_fun, state_trans : callable, optional
        Hooks for circular state components: difference transform,
        weighted average (``fun(points, weights)``), and sample
        transform. Defaults: identity / weighted mean / identity.
    rng : numpy.random.Generator, optional
        Sample source (the MATLAB original uses the global RNG; a
        Generator argument replaces that for reproducibility).

    Returns
    -------
    result : QMCKFPrediction

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(5)
    >>> F = np.array([[1.0, 1.0], [0.0, 1.0]])
    >>> pred = qmc_kf_predict(np.zeros(2), np.eye(2), lambda x: F @ x,
    ...                       0.1 * np.eye(2), 4000, rng=rng)
    >>> bool(np.allclose(pred.P, F @ F.T + 0.1 * np.eye(2), atol=0.2))
    True

    Notes
    -----
    Port of ``discQMCKalPred.m``.
    """
    if rng is None:
        rng = np.random.default_rng()
    if state_diff_trans is None:
        state_diff_trans = lambda x: x  # noqa: E731
    if state_avg_fun is None:
        state_avg_fun = _weighted_mean
    if state_trans is None:
        state_trans = lambda x: x  # noqa: E731

    x_prev = np.asarray(x_prev, dtype=np.float64).ravel()
    p_prev = np.asarray(p_prev, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    x_dim = len(x_prev)

    s = chol_semi_def(p_prev)
    x_samples = state_trans(
        x_prev[:, np.newaxis] + s @ rng.standard_normal((x_dim, num_samples))
    )
    x_samples = np.asarray(x_samples, dtype=np.float64)
    for k in range(num_samples):
        x_samples[:, k] = f(x_samples[:, k])

    w = np.full(num_samples, 1.0 / num_samples)
    x_pred = np.asarray(state_avg_fun(x_samples, w)).ravel()

    cen = state_diff_trans(x_samples - x_pred[:, np.newaxis])
    p_pred = q + (cen @ cen.T) / num_samples
    return QMCKFPrediction(x_pred, p_pred)


def qmc_kf_update(
    x_pred: ArrayLike,
    p_pred: ArrayLike,
    z: ArrayLike,
    r: ArrayLike,
    h: Callable,
    num_samples: int = 100,
    innov_trans: Optional[Callable] = None,
    meas_avg_fun: Optional[Callable] = None,
    state_diff_trans: Optional[Callable] = None,
    state_trans: Optional[Callable] = None,
    rng: Optional[np.random.Generator] = None,
) -> QMCKFUpdateResult:
    """
    Monte-Carlo Kalman filter measurement update.

    Parameters
    ----------
    x_pred : array_like
        (x_dim,) predicted state.
    p_pred : array_like
        (x_dim, x_dim) predicted covariance.
    z : array_like
        (z_dim,) measurement.
    r : array_like
        (z_dim, z_dim) measurement noise covariance.
    h : callable
        Measurement function ``h(x)``.
    num_samples : int, optional
        Number of Monte-Carlo samples. Default 100.
    innov_trans, meas_avg_fun, state_diff_trans, state_trans : callable, optional
        Hooks for circular components, as in :func:`qmc_kf_predict`.
    rng : numpy.random.Generator, optional
        Sample source.

    Returns
    -------
    result : QMCKFUpdateResult

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(11)
    >>> H = np.array([[1.0, 0.0]])
    >>> up = qmc_kf_update(np.array([1.0, 0.5]), np.eye(2), np.array([1.4]),
    ...                    0.5 * np.eye(1), lambda x: H @ x, 4000, rng=rng)
    >>> bool(1.0 < up.x[0] < 1.4)
    True

    Notes
    -----
    Port of ``QMCKalUpdate.m``. The updated covariance is symmetrized
    as in the original.
    """
    if rng is None:
        rng = np.random.default_rng()
    if innov_trans is None:
        innov_trans = lambda a, b: a - b  # noqa: E731
    if meas_avg_fun is None:
        meas_avg_fun = _weighted_mean
    if state_diff_trans is None:
        state_diff_trans = lambda x: x  # noqa: E731
    if state_trans is None:
        state_trans = lambda x: x  # noqa: E731

    x_pred = np.asarray(x_pred, dtype=np.float64).ravel()
    p_pred = np.asarray(p_pred, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64).ravel()
    r = np.asarray(r, dtype=np.float64)
    x_dim = len(x_pred)
    z_dim = len(z)

    s = chol_semi_def(p_pred)
    x_samples = x_pred[:, np.newaxis] + s @ rng.standard_normal((x_dim, num_samples))
    w = 1.0 / num_samples

    z_points = np.zeros((z_dim, num_samples))
    for k in range(num_samples):
        z_points[:, k] = np.atleast_1d(h(x_samples[:, k]))

    z_pred = np.asarray(meas_avg_fun(z_points, np.full(num_samples, w))).ravel()
    innov = innov_trans(z, z_pred)

    pzz = r.astype(np.float64, copy=True)
    pxz = np.zeros((x_dim, z_dim))
    for k in range(num_samples):
        diff = innov_trans(z_points[:, k], z_pred)
        pzz = pzz + w * np.outer(diff, diff)
        pxz = pxz + w * np.outer(state_diff_trans(x_samples[:, k] - x_pred), diff)

    gain = np.linalg.solve(pzz.T, pxz.T).T
    x_update = np.asarray(state_trans(x_pred + gain @ innov)).ravel()
    p_update = p_pred - gain @ pzz @ gain.T
    p_update = (p_update + p_update.T) / 2.0
    return QMCKFUpdateResult(x_update, p_update, innov, pzz, gain)


def qmc_kf_meas_pred(
    x_pred: ArrayLike,
    p_pred: ArrayLike,
    z_dim: int,
    h: Callable,
    num_samples: int = 100,
    innov_trans: Optional[Callable] = None,
    meas_avg_fun: Optional[Callable] = None,
    state_diff_trans: Optional[Callable] = None,
    state_trans: Optional[Callable] = None,
    rng: Optional[np.random.Generator] = None,
) -> QMCMeasPredInfo:
    """
    Monte-Carlo measurement prediction, separated from the update.

    Computing the measurement prediction once lets
    :func:`qmc_kf_update_with_pred` apply several measurements without
    resampling. Supports a bank of components: ``x_pred`` may be
    (x_dim, num_comp) with matching stacked covariances.

    Parameters
    ----------
    x_pred : array_like
        (x_dim,) or (x_dim, num_comp) predicted state(s).
    p_pred : array_like
        (x_dim, x_dim) or (x_dim, x_dim, num_comp) covariance(s).
    z_dim : int
        Measurement dimensionality.
    h : callable
        Measurement function.
    num_samples : int, optional
        Samples per component. Default 100.
    innov_trans, meas_avg_fun, state_diff_trans, state_trans : callable, optional
        Hooks as in :func:`qmc_kf_update`.
    rng : numpy.random.Generator, optional
        Sample source.

    Returns
    -------
    info : QMCMeasPredInfo

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(2)
    >>> H = np.array([[1.0, 0.0]])
    >>> info = qmc_kf_meas_pred(np.array([1.0, -0.5]), np.eye(2), 1,
    ...                         lambda x: H @ x, 4000, rng=rng)
    >>> bool(abs(info.z_pred[0, 0] - 1.0) < 0.1)
    True

    Notes
    -----
    Port of ``QMCKalMeasPred.m``.
    """
    if rng is None:
        rng = np.random.default_rng()
    if innov_trans is None:
        innov_trans = lambda a, b: a - b  # noqa: E731
    if meas_avg_fun is None:
        meas_avg_fun = _weighted_mean
    if state_diff_trans is None:
        state_diff_trans = lambda x: x  # noqa: E731
    if state_trans is None:
        state_trans = lambda x: x  # noqa: E731

    x_pred = np.asarray(x_pred, dtype=np.float64)
    if x_pred.ndim == 1:
        x_pred = x_pred[:, np.newaxis]
    p_pred = np.asarray(p_pred, dtype=np.float64)
    if p_pred.ndim == 2:
        p_pred = p_pred[:, :, np.newaxis]
    x_dim, num_comp = x_pred.shape

    z_pred = np.zeros((z_dim, num_comp))
    pz_pred = np.zeros((z_dim, z_dim, num_comp))
    pxz = np.zeros((x_dim, z_dim, num_comp))
    w = 1.0 / num_samples

    for c in range(num_comp):
        s = chol_semi_def(p_pred[:, :, c])
        x_samples = x_pred[:, c : c + 1] + s @ rng.standard_normal((x_dim, num_samples))
        z_points = np.zeros((z_dim, num_samples))
        for k in range(num_samples):
            z_points[:, k] = np.atleast_1d(h(x_samples[:, k]))
        z_pred[:, c] = np.asarray(
            meas_avg_fun(z_points, np.full(num_samples, w))
        ).ravel()
        z_cen = innov_trans(z_points, z_pred[:, c : c + 1])
        for k in range(num_samples):
            diff = z_cen[:, k]
            pz_pred[:, :, c] += w * np.outer(diff, diff)
            pxz[:, :, c] += w * np.outer(
                state_diff_trans(x_samples[:, k] - x_pred[:, c]), diff
            )

    return QMCMeasPredInfo(
        z_pred, pz_pred, pxz, x_pred, p_pred, innov_trans, state_trans
    )


def qmc_kf_update_with_pred(
    z: ArrayLike,
    r: ArrayLike,
    other_info: QMCMeasPredInfo,
) -> QMCKFUpdateResult:
    """
    Measurement update from a precomputed measurement prediction.

    Parameters
    ----------
    z : array_like
        (z_dim,) measurement.
    r : array_like
        (z_dim, z_dim) measurement noise covariance.
    other_info : QMCMeasPredInfo
        Output of :func:`qmc_kf_meas_pred`.

    Returns
    -------
    result : QMCKFUpdateResult
        Component-stacked outputs: ``x`` is (x_dim, num_comp), ``P`` is
        (x_dim, x_dim, num_comp), and so on. This part is fully
        deterministic given ``other_info``.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(2)
    >>> H = np.array([[1.0, 0.0]])
    >>> info = qmc_kf_meas_pred(np.array([1.0, -0.5]), np.eye(2), 1,
    ...                         lambda x: H @ x, 4000, rng=rng)
    >>> up = qmc_kf_update_with_pred(np.array([1.3]), 0.5 * np.eye(1), info)
    >>> up.x.shape
    (2, 1)

    Notes
    -----
    Port of ``QMCKalUpdateWithPred.m``.
    """
    z = np.asarray(z, dtype=np.float64).ravel()
    r = np.asarray(r, dtype=np.float64)
    x_pred = other_info.x_pred
    p_pred = other_info.p_pred
    x_dim, num_comp = x_pred.shape
    z_dim = len(z)

    x_update = np.zeros((x_dim, num_comp))
    p_update = np.zeros((x_dim, x_dim, num_comp))
    innov = np.zeros((z_dim, num_comp))
    pzz = np.zeros((z_dim, z_dim, num_comp))
    gain = np.zeros((x_dim, z_dim, num_comp))

    for c in range(num_comp):
        pzz[:, :, c] = other_info.pz_pred[:, :, c] + r
        innov[:, c] = other_info.innov_trans(z, other_info.z_pred[:, c])
        gain[:, :, c] = np.linalg.solve(pzz[:, :, c].T, other_info.pxz[:, :, c].T).T
        x_update[:, c] = np.asarray(
            other_info.state_trans(x_pred[:, c] + gain[:, :, c] @ innov[:, c])
        ).ravel()
        p_update[:, :, c] = (
            p_pred[:, :, c] - gain[:, :, c] @ pzz[:, :, c] @ gain[:, :, c].T
        )
    return QMCKFUpdateResult(x_update, p_update, innov, pzz, gain)


def calc_qmc_kalman_gain(
    r: ArrayLike,
    pz_pred: ArrayLike,
    other_info: QMCMeasPredInfo,
) -> NDArray[np.floating]:
    """
    Kalman gain from a precomputed measurement prediction.

    Parameters
    ----------
    r : array_like
        (z_dim, z_dim) measurement noise covariance.
    pz_pred : array_like
        (z_dim, z_dim) measurement-prediction covariance (without R),
        e.g. one component slice of :attr:`QMCMeasPredInfo.pz_pred`.
    other_info : QMCMeasPredInfo
        Output of :func:`qmc_kf_meas_pred` (single component).

    Returns
    -------
    gain : ndarray
        (x_dim, z_dim) filter gain.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(2)
    >>> H = np.array([[1.0, 0.0]])
    >>> info = qmc_kf_meas_pred(np.array([1.0, -0.5]), np.eye(2), 1,
    ...                         lambda x: H @ x, 4000, rng=rng)
    >>> g = calc_qmc_kalman_gain(0.5 * np.eye(1), info.pz_pred[:, :, 0], info)
    >>> g.shape
    (2, 1)

    Notes
    -----
    Port of ``calcQMCKalmanGain.m``, which delegates to
    ``calcCubKalGain``: the gain is ``Pxz @ inv(PzPred + R)``.
    """
    r = np.asarray(r, dtype=np.float64)
    pz_pred = np.asarray(pz_pred, dtype=np.float64)
    pxz = other_info.pxz[:, :, 0]
    pzz = pz_pred + r
    return np.linalg.solve(pzz.T, pxz.T).T


__all__ = [
    "QMCKFPrediction",
    "QMCKFUpdateResult",
    "QMCMeasPredInfo",
    "calc_qmc_kalman_gain",
    "qmc_kf_meas_pred",
    "qmc_kf_predict",
    "qmc_kf_update",
    "qmc_kf_update_with_pred",
]
