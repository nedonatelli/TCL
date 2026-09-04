"""
Batch and interval Kalman smoothers.

Ports of the MATLAB TCL ``Dynamic_Estimation/Batch_and_Smoothing``
smoothers: the forward-backward Kalman/RTS batch smoother, the
Fraser-Potter two-filter information smoother (batch and sliding
interval), the extended (iterated) Kalman batch smoother, the
square-root cubature Kalman batch smoother, the square-root
information smoother, the sliding/growing-interval Kalman smoother,
and the FIR (finite impulse response) Kalman smoother with its
coefficient generator.

Array layout follows the MATLAB originals: measurement batches are
(z_dim, N), state batches (x_dim, N), and covariance batches
(x_dim, x_dim, N) stacked along the last axis. The requested step
``k_d`` is zero-based (MATLAB's 1-based ``kD`` maps to
``k_d = kD - 1``).

References
----------
.. [1] Y. Bar-Shalom, X. R. Li, and T. Kirubarajan, Estimation with
   Applications to Tracking and Navigation. New York: John Wiley and
   Sons, Inc, 2001, ch. 8.6.
.. [2] D. C. Fraser and J. E. Potter, "The optimal linear smoother as
   a combination of two optimum linear filters," IEEE Transactions on
   Automatic Control, pp. 387-390, Aug. 1969.
.. [3] G. J. Bierman, Factorization Methods for Discrete Sequential
   Estimation. New York: Academic Press, 1977.
.. [4] I. Arasaratnam and S. Haykin, "Cubature Kalman smoothers,"
   Automatica, vol. 47, no. 10, pp. 2245-2250, Oct. 2011.
.. [5] D. F. Crouse, P. Willett, and Y. Bar-Shalom, "A low-complexity
   sliding-window Kalman FIR smoother for discrete-time models," IEEE
   Signal Processing Letters, vol. 17, no. 2, pp. 177-180, Feb. 2010.
"""

from typing import Callable, List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pytcl.dynamic_estimation.kalman.extended import numerical_jacobian
from pytcl.dynamic_estimation.kalman.linear import kf_predict, kf_update
from pytcl.dynamic_estimation.kalman.sqrt_cubature import (
    sqrt_ckf_predict,
    sqrt_ckf_update,
)
from pytcl.mathematical_functions.basic_matrix import tria_sqrt
from pytcl.mathematical_functions.numerical_integration.cubature_points import (
    spherical_radial_points,
    transform_cubature_points,
)


class BatchSmootherResult(NamedTuple):
    """Smoothed batch estimate.

    Attributes
    ----------
    x : ndarray
        Smoothed states, (x_dim, N), or (x_dim,) when a single step
        ``k_d`` was requested.
    P : ndarray
        Smoothed covariances, (x_dim, x_dim, N) or (x_dim, x_dim).
    """

    x: NDArray[np.floating]
    P: NDArray[np.floating]


class FPInfoBatchSmootherResult(NamedTuple):
    """Result of :func:`fp_info_batch_smoother`.

    Attributes
    ----------
    y : ndarray
        Smoothed information states, (x_dim, N) or (x_dim,).
    p_inv : ndarray
        Smoothed inverse covariances, (x_dim, x_dim, N) or
        (x_dim, x_dim).
    x : ndarray
        The state estimates corresponding to ``y``.
    P : ndarray
        The covariances corresponding to ``p_inv``.
    """

    y: NDArray[np.floating]
    p_inv: NDArray[np.floating]
    x: NDArray[np.floating]
    P: NDArray[np.floating]


class EKalmanBatchSmootherResult(NamedTuple):
    """Result of :func:`ekalman_batch_smoother`.

    Attributes
    ----------
    x : ndarray
        Smoothed states, (x_dim, N) or (x_dim,).
    P : ndarray
        Smoothed covariances, (x_dim, x_dim, N) or (x_dim, x_dim).
    x_upd : ndarray
        (x_dim, N) forward-filter (unsmoothed) state estimates.
    p_upd : ndarray
        (x_dim, x_dim, N) forward-filter covariances.
    """

    x: NDArray[np.floating]
    P: NDArray[np.floating]
    x_upd: NDArray[np.floating]
    p_upd: NDArray[np.floating]


class SqrtCubKalBatchSmootherResult(NamedTuple):
    """Result of :func:`sqrt_cub_kal_batch_smoother`.

    Attributes
    ----------
    x : ndarray
        Smoothed states, (x_dim, N) or (x_dim,).
    S : ndarray
        Lower-triangular roots of the smoothed covariances,
        (x_dim, x_dim, N) or (x_dim, x_dim).
    x_upd : ndarray
        (x_dim, N) forward-filter (unsmoothed) state estimates.
    s_upd : ndarray
        (x_dim, x_dim, N) forward-filter covariance roots.
    """

    x: NDArray[np.floating]
    S: NDArray[np.floating]
    x_upd: NDArray[np.floating]
    s_upd: NDArray[np.floating]


class SqrtInfoBatchSmootherResult(NamedTuple):
    """Result of :func:`sqrt_info_batch_smoother`.

    Attributes
    ----------
    y_sqrt : ndarray
        Smoothed square-root information states, (x_dim, N) or
        (x_dim,). The state is ``solve(p_inv_sqrt, y_sqrt)``.
    p_inv_sqrt : ndarray
        Inverse square-root covariances, (x_dim, x_dim, N) or
        (x_dim, x_dim); ``P^-1 = p_inv_sqrt' @ p_inv_sqrt``.
    """

    y_sqrt: NDArray[np.floating]
    p_inv_sqrt: NDArray[np.floating]


class KalmanIntervalSmootherResult(NamedTuple):
    """Result of :func:`kalman_interval_smoother`.

    Attributes
    ----------
    x : ndarray
        (x_dim, N) smoothed states over the interval, oldest first.
    P : ndarray
        (x_dim, x_dim, N) covariances for ``x``.
    x_fwd_pred : ndarray
        (x_dim, N-1) forward predicted states to feed the next call.
        Column k is the prediction to the time of column k+1 of
        ``x_fwd_post``.
    p_fwd_pred : ndarray
        (x_dim, x_dim, N-1) covariances for ``x_fwd_pred``.
    x_fwd_post : ndarray
        (x_dim, N) forward posterior states to feed the next call.
    p_fwd_post : ndarray
        (x_dim, x_dim, N) covariances for ``x_fwd_post``.
    """

    x: NDArray[np.floating]
    P: NDArray[np.floating]
    x_fwd_pred: NDArray[np.floating]
    p_fwd_pred: NDArray[np.floating]
    x_fwd_post: NDArray[np.floating]
    p_fwd_post: NDArray[np.floating]


class FPInfoIntervalSmootherResult(NamedTuple):
    """Result of :func:`fp_info_interval_smoother`.

    Attributes
    ----------
    y : ndarray
        (x_dim, N) smoothed information states over the interval.
    p_inv : ndarray
        (x_dim, x_dim, N) inverse covariances for ``y``.
    y_fwd_pred : ndarray
        (x_dim, N) forward predicted information states to feed the
        next call.
    p_inv_fwd_pred : ndarray
        (x_dim, x_dim, N) inverse covariances for ``y_fwd_pred``.
    y_fwd_end : ndarray
        (x_dim,) forward-updated information state at the newest step
        (the smoothed estimate there), to feed the next call's
        ``y_fwd_prev``.
    p_inv_fwd_end : ndarray
        (x_dim, x_dim) inverse covariance for ``y_fwd_end``.
    """

    y: NDArray[np.floating]
    p_inv: NDArray[np.floating]
    y_fwd_pred: NDArray[np.floating]
    p_inv_fwd_pred: NDArray[np.floating]
    y_fwd_end: NDArray[np.floating]
    p_inv_fwd_end: NDArray[np.floating]


class FIRSmootherCoeffs(NamedTuple):
    """Result of :func:`kalman_fir_smoother_coeffs`.

    Attributes
    ----------
    a : ndarray
        (x_dim, z_dim, N) measurement coefficients.
    b : ndarray
        (x_dim, x_dim, N-1) control-input coefficients.
    p_kn : ndarray
        (x_dim, x_dim) covariance of the smoothed estimate at ``k_d``.
    """

    a: NDArray[np.floating]
    b: NDArray[np.floating]
    p_kn: NDArray[np.floating]


def _rdiv(a: NDArray, b: NDArray) -> NDArray:
    """MATLAB A/B: a @ inv(b), via a linear solve."""
    return np.linalg.solve(b.T, a.T).T


def _sym(a: NDArray) -> NDArray:
    return (a + a.T) / 2.0


def _rep3(m: ArrayLike, n: int) -> NDArray[np.floating]:
    """Broadcast a single matrix to an (r, c, n) stack; pass stacks through."""
    m = np.asarray(m, dtype=np.float64)
    if m.ndim == 2:
        return np.repeat(m[:, :, np.newaxis], n, axis=2)
    return m


def _as_funs(f: Union[Callable, Sequence[Callable]], n: int) -> List[Callable]:
    if callable(f):
        return [f] * n
    return list(f)


def _info_filter_update(
    y: NDArray, p_inv: NDArray, z: NDArray, r: NDArray, h: NDArray
) -> Tuple[NDArray, NDArray]:
    """Port of ``infoFilterUpdate.m``."""
    ht_rinv = np.linalg.solve(r.T, h).T
    return y + ht_rinv @ z, _sym(p_inv + ht_rinv @ h)


def _info_filter_disc_pred(
    y: NDArray,
    p_inv: NDArray,
    f: NDArray,
    q: NDArray,
    u: Optional[NDArray] = None,
) -> Tuple[NDArray, NDArray]:
    """Port of ``infoFilterDiscPred.m``."""
    p_inv_finv = _rdiv(p_inv, f)
    d_inv = f.T + p_inv_finv @ q
    p_inv_pred = _sym(np.linalg.solve(d_inv, p_inv_finv))
    y_pred = np.linalg.solve(d_inv, y)
    if u is not None:
        y_pred = y_pred + p_inv_pred @ u
    return y_pred, p_inv_pred


def _info_filter_disc_pred_rev(
    y: NDArray,
    p_inv: NDArray,
    f: NDArray,
    q: NDArray,
    u: Optional[NDArray] = None,
) -> Tuple[NDArray, NDArray]:
    """Port of ``infoFilterDiscPredRev.m``."""
    d_inv = np.linalg.inv(f.T) + _rdiv(p_inv @ q, f.T)
    p_inv_pred = _sym(np.linalg.solve(d_inv, p_inv) @ f)
    y_pred = np.linalg.solve(d_inv, y)
    if u is not None:
        y_pred = y_pred - _rdiv(p_inv_pred, f) @ u
    return y_pred, p_inv_pred


def _sqrt_info_filter_update(
    y_sqrt: NDArray, p_inv_sqrt: NDArray, z: NDArray, s_r: NDArray, h: NDArray
) -> Tuple[NDArray, NDArray]:
    """Port of ``sqrtInfoFilterUpdate.m``."""
    x_dim = len(y_sqrt)
    a = np.block(
        [
            [p_inv_sqrt, y_sqrt[:, np.newaxis]],
            [np.linalg.solve(s_r, h), np.linalg.solve(s_r, z)[:, np.newaxis]],
        ]
    )
    t = np.linalg.qr(a, mode="r")
    return t[:x_dim, x_dim], t[:x_dim, :x_dim]


def _sqrt_info_filter_disc_pred(
    y_sqrt: NDArray,
    p_inv_sqrt: NDArray,
    f: NDArray,
    s_q: NDArray,
    u: Optional[NDArray],
    gamma: NDArray,
) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    """Port of ``sqrtInfoFilterDiscPred.m`` (with the Rw/Rwx extras)."""
    x_dim = len(y_sqrt)
    q_dim = s_q.shape[0]
    if u is None:
        u = np.zeros(q_dim)
    p_inv_sqrt_tilde = _rdiv(p_inv_sqrt, f)
    a = np.block(
        [
            [
                np.linalg.inv(s_q),
                np.zeros((q_dim, x_dim)),
                np.linalg.solve(s_q, u)[:, np.newaxis],
            ],
            [
                -p_inv_sqrt_tilde @ gamma,
                p_inv_sqrt_tilde,
                y_sqrt[:, np.newaxis],
            ],
        ]
    )
    t = np.linalg.qr(a, mode="r")
    y_sqrt_pred = t[q_dim:, -1]
    p_inv_sqrt_pred = t[q_dim:, q_dim : q_dim + x_dim]
    rw = t[:q_dim, :q_dim]
    rwx = t[:q_dim, q_dim : q_dim + x_dim]
    return y_sqrt_pred, p_inv_sqrt_pred, rw, rwx


def _sqrt_info_smooth_step(
    z_star: NDArray,
    p_inv_sqrt_prev: NDArray,
    f: NDArray,
    ru: NDArray,
    rux: NDArray,
    gamma: NDArray,
) -> Tuple[NDArray, NDArray]:
    """The Bierman smoothing step nested in ``sqrtInfoBatchSmoother.m``.

    The original places the control input u in the data column of the
    top block; the recursion runs on state residuals, in which the
    control (already present in both the smoothed and predicted
    trajectories) cancels identically, so that entry is zero. With u
    there, the control is injected a second time and the smoothed
    states with nonzero control drift off the optimum.
    """
    x_dim = len(z_star)
    q_dim = ru.shape[0]
    a = np.block(
        [
            [ru + rux @ gamma, rux @ f, np.zeros((q_dim, 1))],
            [p_inv_sqrt_prev @ gamma, p_inv_sqrt_prev @ f, z_star[:, np.newaxis]],
        ]
    )
    t = np.linalg.qr(a, mode="r")
    z_smooth = t[x_dim:, -1]
    p_inv_sqrt = t[x_dim:, -1 - x_dim : -1]
    return z_smooth, p_inv_sqrt


def _ekf_update(
    x_pred: NDArray,
    p_pred: NDArray,
    z: NDArray,
    r: NDArray,
    h: Callable,
    h_jacob: Callable,
    num_iter: int,
) -> Tuple[NDArray, NDArray]:
    """The (optionally iterated) EKF update of ``EKFUpdate.m``."""
    h_mat = np.atleast_2d(h_jacob(x_pred))
    z_pred = np.atleast_1d(h(x_pred))
    innov = z - z_pred
    pzz = _sym(r + h_mat @ p_pred @ h_mat.T)
    gain = _rdiv(p_pred @ h_mat.T, pzz)
    x_up = x_pred + gain @ innov
    temp = np.eye(len(x_pred)) - gain @ h_mat
    p_up = _sym(temp @ p_pred @ temp.T + gain @ r @ gain.T)
    for _ in range(num_iter):
        h_mat = np.atleast_2d(h_jacob(x_up))
        pzz = _sym(r + h_mat @ p_pred @ h_mat.T)
        gain = _rdiv(p_pred @ h_mat.T, pzz)
        temp = np.eye(len(x_pred)) - gain @ h_mat
        p_up = _sym(temp @ p_pred @ temp.T + gain @ r @ gain.T)
        z_pred = np.atleast_1d(h(x_up))
        x_up = (
            x_up
            + p_up @ h_mat.T @ np.linalg.lstsq(r, z - z_pred, rcond=None)[0]
            - p_up @ np.linalg.lstsq(p_pred, x_up - x_pred, rcond=None)[0]
        )
    return x_up, p_up


def _disc_ekf_pred(
    x: NDArray, p: NDArray, f: Callable, f_jacob: Callable, q: NDArray
) -> Tuple[NDArray, NDArray]:
    """Port of ``discEKFPred.m`` (first-order form)."""
    f_mat = np.atleast_2d(f_jacob(x))
    return np.atleast_1d(f(x)), _sym(f_mat @ p @ f_mat.T + q)


def _pick_step(x: NDArray, p: NDArray, k_d: Optional[int]):
    if k_d is None:
        return x, p
    return x[:, k_d], p[:, :, k_d]


def kalman_batch_smoother(
    x_init: ArrayLike,
    p_init: ArrayLike,
    z: ArrayLike,
    u: Optional[ArrayLike],
    h: ArrayLike,
    f: ArrayLike,
    r: ArrayLike,
    q: ArrayLike,
    k_d: Optional[int] = None,
    use_fp: bool = True,
) -> BatchSmootherResult:
    """
    Forward-backward Kalman smoother for a batch of linear measurements.

    Parameters
    ----------
    x_init : array_like
        (x_dim,) predicted state at the time of the first measurement.
    p_init : array_like
        (x_dim, x_dim) covariance of ``x_init``.
    z : array_like
        (z_dim, N) batch of measurements.
    u : array_like or None
        (x_dim, N-1) control inputs, or None for no control.
    h : array_like
        (z_dim, x_dim) measurement matrix, or (z_dim, x_dim, N) stack.
    f : array_like
        (x_dim, x_dim) state transition matrix, or a (x_dim, x_dim,
        N-1) stack.
    r : array_like
        (z_dim, z_dim) measurement covariance, or (z_dim, z_dim, N).
    q : array_like
        (x_dim, x_dim) process noise covariance, or a stack of N-1.
    k_d : int, optional
        Zero-based step whose smoothed estimate is desired; None
        (default) returns the whole batch.
    use_fp : bool, optional
        If True (default), delegate to the Fraser-Potter information
        smoother; otherwise run the RTS forward-backward recursion.

    Returns
    -------
    result : BatchSmootherResult

    Examples
    --------
    >>> import numpy as np
    >>> F = np.array([[1.0, 1.0], [0.0, 1.0]])
    >>> H = np.array([[1.0, 0.0]])
    >>> z = np.array([[0.05, 1.1, 1.95, 3.02]])
    >>> res = kalman_batch_smoother([0.0, 1.0], np.eye(2), z, None, H, F,
    ...                             np.array([[0.01]]), 0.001 * np.eye(2))
    >>> rts = kalman_batch_smoother([0.0, 1.0], np.eye(2), z, None, H, F,
    ...                             np.array([[0.01]]), 0.001 * np.eye(2),
    ...                             use_fp=False)
    >>> np.allclose(res.x, rts.x, atol=1e-9)
    True

    Notes
    -----
    Port of ``KalmanBatchSmoother.m``.
    """
    x_init = np.asarray(x_init, dtype=np.float64).ravel()
    p_init = np.asarray(p_init, dtype=np.float64)
    z = np.atleast_2d(np.asarray(z, dtype=np.float64))
    if use_fp:
        y_init = np.linalg.solve(p_init, x_init)
        p_inv_init = np.linalg.inv(p_init)
        fp = fp_info_batch_smoother(y_init, p_inv_init, z, u, h, f, r, q, k_d)
        return BatchSmootherResult(fp.x, fp.P)

    h = _rep3(h, z.shape[1])
    x_dim = h.shape[1]
    n = z.shape[1]
    u_arr = (
        np.zeros((x_dim, n - 1))
        if u is None
        else np.atleast_2d(np.asarray(u, dtype=np.float64))
    )
    f = _rep3(f, n - 1)
    r = _rep3(r, n)
    q = _rep3(q, n - 1)
    eye = np.eye(x_dim)

    x_pred = np.zeros((x_dim, n))
    p_pred = np.zeros((x_dim, x_dim, n))
    x_upd = np.zeros((x_dim, n))
    p_upd = np.zeros((x_dim, x_dim, n))
    x_pred[:, 0] = x_init
    p_pred[:, :, 0] = p_init
    for k in range(n - 1):
        up = kf_update(x_pred[:, k], p_pred[:, :, k], z[:, k], h[:, :, k], r[:, :, k])
        x_upd[:, k], p_upd[:, :, k] = up.x, up.P
        pr = kf_predict(
            x_upd[:, k], p_upd[:, :, k], f[:, :, k], q[:, :, k], eye, u_arr[:, k]
        )
        x_pred[:, k + 1], p_pred[:, :, k + 1] = pr.x, pr.P

    x_smooth = np.zeros((x_dim, n))
    p_smooth = np.zeros((x_dim, x_dim, n))
    up = kf_update(
        x_pred[:, n - 1],
        p_pred[:, :, n - 1],
        z[:, n - 1],
        h[:, :, n - 1],
        r[:, :, n - 1],
    )
    x_smooth[:, n - 1], p_smooth[:, :, n - 1] = up.x, up.P
    for k in range(n - 2, -1, -1):
        c = _rdiv(p_upd[:, :, k] @ f[:, :, k].T, p_pred[:, :, k + 1])
        x_smooth[:, k] = x_upd[:, k] + c @ (x_smooth[:, k + 1] - x_pred[:, k + 1])
        p_smooth[:, :, k] = (
            p_upd[:, :, k] + c @ (p_smooth[:, :, k + 1] - p_pred[:, :, k + 1]) @ c.T
        )
    xs, ps = _pick_step(x_smooth, p_smooth, k_d)
    return BatchSmootherResult(xs, ps)


def fp_info_batch_smoother(
    y_pred: Optional[ArrayLike],
    p_inv_pred: Optional[ArrayLike],
    z: ArrayLike,
    u: Optional[ArrayLike],
    h: ArrayLike,
    f: ArrayLike,
    r: ArrayLike,
    q: ArrayLike,
    k_d: Optional[int] = None,
) -> FPInfoBatchSmootherResult:
    """
    Fraser-Potter two-filter information smoother on a batch.

    Unlike :func:`kalman_batch_smoother`, the prior may be
    uninformative: pass None for ``y_pred``/``p_inv_pred``.

    Parameters
    ----------
    y_pred : array_like or None
        (x_dim,) predicted information state at the first measurement,
        or None for no prior information.
    p_inv_pred : array_like or None
        (x_dim, x_dim) inverse covariance of ``y_pred``, or None.
    z, u, h, f, r, q, k_d
        As in :func:`kalman_batch_smoother`.

    Returns
    -------
    result : FPInfoBatchSmootherResult

    Examples
    --------
    >>> import numpy as np
    >>> F = np.array([[1.0, 1.0], [0.0, 1.0]])
    >>> H = np.array([[1.0, 0.0]])
    >>> z = np.array([[0.0, 1.0, 2.0, 3.0]])
    >>> res = fp_info_batch_smoother(None, None, z, None, H, F,
    ...                              np.array([[0.01]]), 1e-8 * np.eye(2))
    >>> np.allclose(res.x[:, 2], [2.0, 1.0], atol=1e-3)
    True

    Notes
    -----
    Port of ``FPInfoBatchSmoother.m``. The original's whole-batch
    state-recovery block allocates ``zeros(xDim, kD)`` with ``kD``
    empty and relies on MATLAB's implicit array growth; the port
    allocates the (x_dim, N) result directly.
    """
    z = np.atleast_2d(np.asarray(z, dtype=np.float64))
    n = z.shape[1]
    h = _rep3(h, n)
    x_dim = h.shape[1]
    y0 = (
        np.zeros(x_dim)
        if y_pred is None
        else np.asarray(y_pred, dtype=np.float64).ravel()
    )
    p_inv0 = (
        np.zeros((x_dim, x_dim))
        if p_inv_pred is None
        else np.asarray(p_inv_pred, dtype=np.float64)
    )
    u_arr = (
        np.zeros((x_dim, n - 1))
        if u is None
        else np.atleast_2d(np.asarray(u, dtype=np.float64))
    )
    f = _rep3(f, n - 1)
    r = _rep3(r, n)
    q = _rep3(q, n - 1)

    y_fwd_pred = np.zeros((x_dim, n))
    p_inv_fwd_pred = np.zeros((x_dim, x_dim, n))
    y_fwd_pred[:, 0] = y0
    p_inv_fwd_pred[:, :, 0] = p_inv0
    for k in range(n - 1):
        y_fwd, p_inv_fwd = _info_filter_update(
            y_fwd_pred[:, k], p_inv_fwd_pred[:, :, k], z[:, k], r[:, :, k], h[:, :, k]
        )
        y_fwd_pred[:, k + 1], p_inv_fwd_pred[:, :, k + 1] = _info_filter_disc_pred(
            y_fwd, p_inv_fwd, f[:, :, k], q[:, :, k], u_arr[:, k]
        )

    y_rev = np.zeros((x_dim, n))
    p_inv_rev = np.zeros((x_dim, x_dim, n))
    y_rev[:, n - 1], p_inv_rev[:, :, n - 1] = _info_filter_update(
        np.zeros(x_dim),
        np.zeros((x_dim, x_dim)),
        z[:, n - 1],
        r[:, :, n - 1],
        h[:, :, n - 1],
    )
    for k in range(n - 2, -1, -1):
        y_rev_pred, p_inv_rev_pred = _info_filter_disc_pred_rev(
            y_rev[:, k + 1], p_inv_rev[:, :, k + 1], f[:, :, k], q[:, :, k], u_arr[:, k]
        )
        y_rev[:, k], p_inv_rev[:, :, k] = _info_filter_update(
            y_rev_pred, p_inv_rev_pred, z[:, k], r[:, :, k], h[:, :, k]
        )

    y_est = y_fwd_pred + y_rev
    p_inv_est = p_inv_fwd_pred + p_inv_rev
    x_est = np.zeros((x_dim, n))
    p_est = np.zeros((x_dim, x_dim, n))
    for k in range(n):
        x_est[:, k] = np.linalg.solve(p_inv_est[:, :, k], y_est[:, k])
        p_est[:, :, k] = np.linalg.inv(p_inv_est[:, :, k])
    if k_d is not None:
        return FPInfoBatchSmootherResult(
            y_est[:, k_d], p_inv_est[:, :, k_d], x_est[:, k_d], p_est[:, :, k_d]
        )
    return FPInfoBatchSmootherResult(y_est, p_inv_est, x_est, p_est)


def ekalman_batch_smoother(
    x_init: ArrayLike,
    p_init: ArrayLike,
    z: ArrayLike,
    h: Union[Callable, Sequence[Callable]],
    h_jacob: Union[Callable, Sequence[Callable], None],
    f: Union[Callable, Sequence[Callable]],
    f_jacob: Union[Callable, Sequence[Callable], None],
    r: ArrayLike,
    q: ArrayLike,
    k_d: Optional[int] = None,
    num_iter: int = 0,
) -> EKalmanBatchSmootherResult:
    """
    Extended (optionally iterated) Kalman batch smoother.

    Parameters
    ----------
    x_init : array_like
        (x_dim,) predicted state at the time of the first measurement.
    p_init : array_like
        (x_dim, x_dim) covariance of ``x_init``.
    z : array_like
        (z_dim, N) batch of measurements.
    h : callable or sequence of callables
        Measurement function(s) ``h(x)``; a single callable is used
        for every step.
    h_jacob : callable, sequence of callables, or None
        Measurement Jacobian(s); None uses numerical differentiation.
    f : callable or sequence of callables
        State transition function(s) ``f(x)``.
    f_jacob : callable, sequence of callables, or None
        Transition Jacobian(s); None uses numerical differentiation.
    r : array_like
        (z_dim, z_dim) measurement covariance, or (z_dim, z_dim, N).
    q : array_like
        (x_dim, x_dim) process noise covariance, or a stack of N-1.
    k_d : int, optional
        Zero-based step whose smoothed estimate is desired; None
        (default) returns the whole batch.
    num_iter : int, optional
        Iterations for the iterated EKF update and the sawtooth
        smoothing iteration of Johnston and Krishnamurthy. Default 0.

    Returns
    -------
    result : EKalmanBatchSmootherResult

    Examples
    --------
    >>> import numpy as np
    >>> F = np.array([[1.0, 1.0], [0.0, 1.0]])
    >>> z = np.array([[1.02, 4.1, 8.9, 16.3]])
    >>> res = ekalman_batch_smoother(
    ...     [1.0, 1.0], np.eye(2), z,
    ...     lambda x: np.array([x[0] ** 2]),
    ...     lambda x: np.array([[2.0 * x[0], 0.0]]),
    ...     lambda x: F @ x, lambda x: F,
    ...     np.array([[0.04]]), 0.001 * np.eye(2))
    >>> res.x.shape
    (2, 4)

    Notes
    -----
    Port of ``EKalmanBatchSmoother.m``. The smoothing iteration
    (``num_iter > 0``) iterates the smoothed state but, as in the
    original, not the smoothed covariance. The original cannot run as
    shipped -- its forward pass calls ``DiscEKFPred`` while the
    library ships ``discEKFPred.m`` (a case mismatch MATLAB does not
    resolve); reference values were captured with a case shim.
    """
    x_init = np.asarray(x_init, dtype=np.float64).ravel()
    p_init = np.asarray(p_init, dtype=np.float64)
    z = np.atleast_2d(np.asarray(z, dtype=np.float64))
    x_dim = len(x_init)
    n = z.shape[1]

    f_list = _as_funs(f, n)
    h_list = _as_funs(h, n)
    if f_jacob is None:
        f_jacob_list = [
            (lambda fk: lambda x: numerical_jacobian(fk, x))(fk) for fk in f_list
        ]
    else:
        f_jacob_list = _as_funs(f_jacob, n)
    if h_jacob is None:
        h_jacob_list = [
            (lambda hk: lambda x: numerical_jacobian(hk, x))(hk) for hk in h_list
        ]
    else:
        h_jacob_list = _as_funs(h_jacob, n)
    r = _rep3(r, n)
    q = _rep3(q, n - 1)

    x_pred = np.zeros((x_dim, n))
    p_pred = np.zeros((x_dim, x_dim, n))
    x_upd = np.zeros((x_dim, n))
    p_upd = np.zeros((x_dim, x_dim, n))
    x_pred[:, 0] = x_init
    p_pred[:, :, 0] = p_init
    for k in range(n - 1):
        x_upd[:, k], p_upd[:, :, k] = _ekf_update(
            x_pred[:, k],
            p_pred[:, :, k],
            z[:, k],
            r[:, :, k],
            h_list[k],
            h_jacob_list[k],
            num_iter,
        )
        x_pred[:, k + 1], p_pred[:, :, k + 1] = _disc_ekf_pred(
            x_upd[:, k], p_upd[:, :, k], f_list[k], f_jacob_list[k], q[:, :, k]
        )
    x_upd[:, n - 1], p_upd[:, :, n - 1] = _ekf_update(
        x_pred[:, n - 1],
        p_pred[:, :, n - 1],
        z[:, n - 1],
        r[:, :, n - 1],
        h_list[n - 1],
        h_jacob_list[n - 1],
        num_iter,
    )

    x_smooth = np.zeros((x_dim, n))
    p_smooth = np.zeros((x_dim, x_dim, n))
    x_smooth[:, n - 1] = x_upd[:, n - 1]
    p_smooth[:, :, n - 1] = p_upd[:, :, n - 1]
    for k in range(n - 2, -1, -1):
        f_mat = np.atleast_2d(f_jacob_list[k](x_upd[:, k]))
        c = _rdiv(p_upd[:, :, k] @ f_mat.T, p_pred[:, :, k + 1])
        x_smooth[:, k] = x_upd[:, k] + c @ (x_smooth[:, k + 1] - x_pred[:, k + 1])
        p_smooth[:, :, k] = (
            p_upd[:, :, k] + c @ (p_smooth[:, :, k + 1] - p_pred[:, :, k + 1]) @ c.T
        )
        for _ in range(num_iter):
            f_mat = np.atleast_2d(f_jacob_list[k](x_smooth[:, k]))
            h_mat = np.atleast_2d(h_jacob_list[k](x_smooth[:, k]))
            b = np.linalg.pinv(
                np.linalg.pinv(p_upd[:, :, k])
                + h_mat.T @ np.linalg.solve(r[:, :, k], h_mat)
                + f_mat.T @ np.linalg.solve(q[:, :, k], f_mat)
            )
            meas_term = h_mat.T @ np.linalg.solve(
                r[:, :, k],
                z[:, k]
                - np.atleast_1d(h_list[k](x_pred[:, k]))
                - h_mat @ (x_pred[:, k] - x_smooth[:, k]),
            )
            dyn_term = f_mat.T @ np.linalg.solve(
                q[:, :, k], x_smooth[:, k + 1] - x_pred[:, k + 1]
            )
            x_smooth[:, k] = x_upd[:, k] + b @ (meas_term + dyn_term)

    xs, ps = _pick_step(x_smooth, p_smooth, k_d)
    if k_d is not None:
        return EKalmanBatchSmootherResult(xs, ps, x_upd, p_upd)
    return EKalmanBatchSmootherResult(x_smooth, p_smooth, x_upd, p_upd)


def sqrt_cub_kal_batch_smoother(
    x_init: ArrayLike,
    s_init: ArrayLike,
    z: ArrayLike,
    h: Union[Callable, Sequence[Callable]],
    f: Union[Callable, Sequence[Callable]],
    s_r: ArrayLike,
    s_q: ArrayLike,
    xi: Optional[ArrayLike] = None,
    w: Optional[ArrayLike] = None,
    k_d: Optional[int] = None,
    innov_trans: Optional[Callable] = None,
    meas_avg_fun: Optional[Callable] = None,
    state_diff_trans: Optional[Callable] = None,
    state_trans: Optional[Callable] = None,
    state_avg_fun: Optional[Callable] = None,
) -> SqrtCubKalBatchSmootherResult:
    """
    Forward-backward square-root cubature Kalman batch smoother.

    Parameters
    ----------
    x_init : array_like
        (x_dim,) predicted state at the time of the first measurement.
    s_init : array_like
        (x_dim, x_dim) lower-triangular root of its covariance.
    z : array_like
        (z_dim, N) batch of measurements.
    h : callable or sequence of callables
        Measurement function(s).
    f : callable or sequence of callables
        State transition function(s).
    s_r : array_like
        (z_dim, z_dim) root measurement covariance, or a stack of N.
    s_q : array_like
        (x_dim, x_dim) root process noise covariance, or a stack of N
        (note: N, not N-1 -- see Notes).
    xi, w : array_like, optional
        Cubature points (num_points, x_dim) and positive weights.
        Default: third-order spherical-radial points.
    k_d : int, optional
        Zero-based step whose smoothed estimate is desired; None
        (default) returns the whole batch.
    innov_trans, meas_avg_fun, state_diff_trans, state_trans, state_avg_fun : callable, optional
        Hooks for circular measurement/state components.

    Returns
    -------
    result : SqrtCubKalBatchSmootherResult

    Examples
    --------
    >>> import numpy as np
    >>> F = np.array([[1.0, 1.0], [0.0, 1.0]])
    >>> z = np.array([[0.05, 1.1, 1.95, 3.02]])
    >>> res = sqrt_cub_kal_batch_smoother(
    ...     [0.0, 1.0], np.eye(2), z,
    ...     lambda x: np.array([x[0]]), lambda x: F @ x,
    ...     0.1 * np.eye(1), 0.05 * np.eye(2))
    >>> res.x.shape
    (2, 4)

    Notes
    -----
    Port of ``sqrtCubKalBatchSmoother.m``. As in the original, a
    single ``s_q`` is tiled to N slices and the backward pass at step
    k reads slice k+1 -- one later than the slice the forward pass
    used for the same transition. For a time-invariant ``s_q`` the two
    coincide; a time-varying stack must carry N slices and inherits
    the original's indexing.
    """
    x_init = np.asarray(x_init, dtype=np.float64).ravel()
    s_init = np.asarray(s_init, dtype=np.float64)
    z = np.atleast_2d(np.asarray(z, dtype=np.float64))
    x_dim = len(x_init)
    n = z.shape[1]
    if state_diff_trans is None:
        state_diff_trans = lambda x: x  # noqa: E731
    if state_trans is None:
        state_trans = lambda x: x  # noqa: E731

    h_list = _as_funs(h, n)
    f_list = _as_funs(f, n)
    s_r = _rep3(s_r, n)
    s_q = _rep3(s_q, n)

    if xi is None:
        xi_arr, w_arr = spherical_radial_points(x_dim, 3)
    else:
        xi_arr = np.asarray(xi, dtype=np.float64)
        w_arr = np.asarray(w, dtype=np.float64).ravel()
    n_cub = len(w_arr)

    x_pred = np.zeros((x_dim, n))
    s_pred = np.zeros((x_dim, x_dim, n))
    x_prop_cen = np.zeros((x_dim, n_cub, n))
    x_upd = np.zeros((x_dim, n))
    s_upd = np.zeros((x_dim, x_dim, n))
    x_pred[:, 0] = x_init
    s_pred[:, :, 0] = s_init
    for k in range(n - 1):
        up = sqrt_ckf_update(
            x_pred[:, k],
            s_pred[:, :, k],
            z[:, k],
            s_r[:, :, k],
            h_list[k],
            xi_arr,
            w_arr,
            innov_trans,
            meas_avg_fun,
            state_diff_trans,
            state_trans,
        )
        x_upd[:, k], s_upd[:, :, k] = up.x, up.S
        pr = sqrt_ckf_predict(
            x_upd[:, k],
            s_upd[:, :, k],
            f_list[k],
            s_q[:, :, k],
            xi_arr,
            w_arr,
            state_diff_trans,
            state_avg_fun,
            state_trans,
        )
        x_pred[:, k + 1], s_pred[:, :, k + 1] = pr.x, pr.S
        x_prop_cen[:, :, k + 1] = pr.x_prop_cen_points
    up = sqrt_ckf_update(
        x_pred[:, n - 1],
        s_pred[:, :, n - 1],
        z[:, n - 1],
        s_r[:, :, n - 1],
        h_list[n - 1],
        xi_arr,
        w_arr,
        innov_trans,
        meas_avg_fun,
        state_diff_trans,
        state_trans,
    )
    x_upd[:, n - 1], s_upd[:, :, n - 1] = up.x, up.S

    sqrt_w = np.sqrt(w_arr)
    x_smooth = np.zeros((x_dim, n))
    s_smooth = np.zeros((x_dim, x_dim, n))
    x_smooth[:, n - 1] = x_upd[:, n - 1]
    s_smooth[:, :, n - 1] = s_upd[:, :, n - 1]
    for k in range(n - 2, -1, -1):
        pts, _ = transform_cubature_points(xi_arr, w_arr, x_upd[:, k], s_upd[:, :, k])
        x_fwd_points = np.asarray(state_trans(pts.T), dtype=np.float64)
        x_fwd_cen = (
            state_diff_trans(x_fwd_points - x_upd[:, k][:, np.newaxis])
            * sqrt_w[np.newaxis, :]
        )
        spp = tria_sqrt(np.hstack([x_prop_cen[:, :, k + 1], s_q[:, :, k + 1]]))
        pfp = x_fwd_cen @ x_prop_cen[:, :, k + 1].T
        tmp = np.linalg.solve(spp, pfp.T).T
        g = np.linalg.solve(spp.T, tmp.T).T
        x_diff = state_diff_trans(x_smooth[:, k + 1] - x_pred[:, k + 1])
        x_smooth[:, k] = np.asarray(state_trans(x_upd[:, k] + g @ x_diff)).ravel()
        s_smooth[:, :, k] = tria_sqrt(
            np.hstack(
                [
                    state_diff_trans(x_fwd_cen - g @ x_prop_cen[:, :, k + 1]),
                    g @ s_q[:, :, k + 1],
                    g @ s_smooth[:, :, k + 1],
                ]
            )
        )

    xs, ss = _pick_step(x_smooth, s_smooth, k_d)
    return SqrtCubKalBatchSmootherResult(xs, ss, x_upd, s_upd)


def sqrt_info_batch_smoother(
    y_sqrt_pred: Optional[ArrayLike],
    p_inv_sqrt_pred: Optional[ArrayLike],
    z: ArrayLike,
    u: Optional[ArrayLike],
    h: ArrayLike,
    f: ArrayLike,
    s_r: ArrayLike,
    s_q: ArrayLike,
    gamma: Optional[ArrayLike] = None,
    k_d: Optional[int] = None,
) -> SqrtInfoBatchSmootherResult:
    """
    Square-root information smoother (SRIF/SRIS) on a batch.

    The prior must be informative enough that the forward filter's
    information matrix is invertible at every step of the backward
    pass (as in the original, whose empty-prior option shares the
    same restriction).

    Parameters
    ----------
    y_sqrt_pred : array_like or None
        (x_dim,) predicted square-root information state at the first
        measurement (``p_inv_sqrt_pred`` times the state), or None.
    p_inv_sqrt_pred : array_like or None
        (x_dim, x_dim) inverse square-root covariance, or None.
    z : array_like
        (z_dim, N) batch of measurements.
    u : array_like or None
        (x_dim, N-1) control inputs, or None.
    h : array_like
        (z_dim, x_dim) measurement matrix, or a stack of N.
    f : array_like
        (x_dim, x_dim) transition matrix, or a stack of N-1.
    s_r : array_like
        Invertible lower-triangular roots of the measurement
        covariances, (z_dim, z_dim) or a stack of N.
    s_q : array_like
        Invertible lower-triangular roots of the process noise
        covariances, (x_dim, x_dim) or a stack of N-1.
    gamma : array_like, optional
        (x_dim, x_dim) process-noise-to-state transform (or a stack of
        N-1); identity if omitted.
    k_d : int, optional
        Zero-based step whose smoothed estimate is desired; None
        (default) returns the whole batch.

    Returns
    -------
    result : SqrtInfoBatchSmootherResult

    Examples
    --------
    >>> import numpy as np
    >>> F = np.array([[1.0, 1.0], [0.0, 1.0]])
    >>> H = np.array([[1.0, 0.0]])
    >>> z = np.array([[0.05, 1.1, 1.95, 3.02]])
    >>> res = sqrt_info_batch_smoother([0.0, 1.0], np.eye(2), z, None, H, F,
    ...                                0.1 * np.eye(1), 0.05 * np.eye(2))
    >>> x2 = np.linalg.solve(res.p_inv_sqrt[:, :, 2], res.y_sqrt[:, 2])
    >>> bool(abs(x2[0] - 2.0) < 0.2)
    True

    Notes
    -----
    Port of ``sqrtInfoBatchSmoother.m`` (the ODTBX-derived backward
    recursion of Bierman's smoother), with two deliberate fixes of
    upstream defects. First, the original's backward pass reads the
    stored ``Rw``/``Rwx`` prediction factors one index below where its
    forward pass stores them, so its first smoothed step consumes
    never-written zeros and later steps pair each transition with the
    previous transition's factors. Second, the original places the
    control input in the data column of the smoothing step's QR
    stack, but that recursion runs on state residuals in which the
    control cancels identically, so the entry is zero; keeping u
    there injects the control a second time. With both corrected, the
    result agrees with :func:`kalman_batch_smoother` to machine
    precision (with and without control inputs), which the original
    does not.
    """
    z = np.atleast_2d(np.asarray(z, dtype=np.float64))
    n = z.shape[1]
    h = _rep3(h, n)
    x_dim = h.shape[1]
    y0 = (
        np.zeros(x_dim)
        if y_sqrt_pred is None
        else np.asarray(y_sqrt_pred, dtype=np.float64).ravel()
    )
    p_inv_sqrt0 = (
        np.zeros((x_dim, x_dim))
        if p_inv_sqrt_pred is None
        else np.asarray(p_inv_sqrt_pred, dtype=np.float64)
    )
    u_arr = (
        np.zeros((x_dim, n - 1))
        if u is None
        else np.atleast_2d(np.asarray(u, dtype=np.float64))
    )
    f = _rep3(f, n - 1)
    s_r = _rep3(s_r, n)
    s_q = _rep3(s_q, n - 1)
    gamma_arr = _rep3(np.eye(x_dim) if gamma is None else gamma, n - 1)

    y_fwd_pred = np.zeros((x_dim, n))
    p_inv_sqrt_fwd_pred = np.zeros((x_dim, x_dim, n))
    y_fwd_upd = np.zeros((x_dim, n))
    p_inv_sqrt_fwd_upd = np.zeros((x_dim, x_dim, n))
    rw = np.zeros((x_dim, x_dim, n))
    rwx = np.zeros((x_dim, x_dim, n))
    y_fwd_pred[:, 0] = y0
    p_inv_sqrt_fwd_pred[:, :, 0] = p_inv_sqrt0
    for k in range(n - 1):
        y_fwd_upd[:, k], p_inv_sqrt_fwd_upd[:, :, k] = _sqrt_info_filter_update(
            y_fwd_pred[:, k],
            p_inv_sqrt_fwd_pred[:, :, k],
            z[:, k],
            s_r[:, :, k],
            h[:, :, k],
        )
        (
            y_fwd_pred[:, k + 1],
            p_inv_sqrt_fwd_pred[:, :, k + 1],
            rw[:, :, k + 1],
            rwx[:, :, k + 1],
        ) = _sqrt_info_filter_disc_pred(
            y_fwd_upd[:, k],
            p_inv_sqrt_fwd_upd[:, :, k],
            f[:, :, k],
            s_q[:, :, k],
            u_arr[:, k],
            gamma_arr[:, :, k],
        )

    y_est = np.zeros((x_dim, n))
    p_inv_sqrt_est = np.zeros((x_dim, x_dim, n))
    y_est[:, n - 1], p_inv_sqrt_est[:, :, n - 1] = _sqrt_info_filter_update(
        y_fwd_pred[:, n - 1],
        p_inv_sqrt_fwd_pred[:, :, n - 1],
        z[:, n - 1],
        s_r[:, :, n - 1],
        h[:, :, n - 1],
    )
    for k in range(n - 2, -1, -1):
        x_pred = np.linalg.solve(p_inv_sqrt_fwd_pred[:, :, k + 1], y_fwd_pred[:, k + 1])
        x_star = np.linalg.solve(p_inv_sqrt_est[:, :, k + 1], y_est[:, k + 1])
        z_star = p_inv_sqrt_est[:, :, k + 1] @ (x_star - x_pred)
        # The original reads Rw(:,:,curStep)/Rwx(:,:,curStep) here, but the
        # forward pass stores the factors of the transition curStep ->
        # curStep+1 at index curStep+1, so that read is off by one: its
        # first smoothed step consumes never-written zeros (degenerating
        # to a zero information root) and the rest use the factors of the
        # wrong transition. With the index corrected, the recursion
        # reproduces the RTS/Fraser-Potter optimum to machine precision.
        z_smooth, p_inv_sqrt_est[:, :, k] = _sqrt_info_smooth_step(
            z_star,
            p_inv_sqrt_est[:, :, k + 1],
            f[:, :, k],
            rw[:, :, k + 1],
            rwx[:, :, k + 1],
            gamma_arr[:, :, k],
        )
        x_upd = np.linalg.solve(p_inv_sqrt_fwd_upd[:, :, k], y_fwd_upd[:, k])
        y_est[:, k] = p_inv_sqrt_est[:, :, k] @ (
            x_upd + np.linalg.solve(p_inv_sqrt_est[:, :, k], z_smooth)
        )

    if k_d is not None:
        return SqrtInfoBatchSmootherResult(y_est[:, k_d], p_inv_sqrt_est[:, :, k_d])
    return SqrtInfoBatchSmootherResult(y_est, p_inv_sqrt_est)


def kalman_interval_smoother(
    x_fwd_pred: Optional[ArrayLike],
    p_fwd_pred: Optional[ArrayLike],
    x_fwd_post: ArrayLike,
    p_fwd_post: ArrayLike,
    n_interval: Optional[int],
    z_cur: Optional[ArrayLike],
    r_cur: Optional[ArrayLike],
    h_cur: Optional[ArrayLike],
    f_interval: ArrayLike,
    q_prev: Optional[ArrayLike],
    u_prev: Optional[ArrayLike] = None,
    has_last_pred: bool = False,
    has_last_update: bool = False,
) -> KalmanIntervalSmootherResult:
    """
    Sliding- or growing-interval forward-backward Kalman smoother.

    Feed each new measurement together with the forward quantities
    returned by the previous call; the interval grows until it reaches
    ``n_interval`` steps and slides thereafter.

    Parameters
    ----------
    x_fwd_pred : array_like or None
        (x_dim, N-1) forward predicted states from the previous call
        (None on the first call). Column k is the prediction to the
        time of posterior column k+1; unless ``has_last_pred``, the
        last column predicts to the step before ``z_cur``.
    p_fwd_pred : array_like or None
        Covariances for ``x_fwd_pred``.
    x_fwd_post : array_like
        (x_dim, N) forward posterior states (the first call passes the
        single initial estimate).
    p_fwd_post : array_like
        Covariances for ``x_fwd_post``.
    n_interval : int or None
        Desired interval length (>= 2); None keeps the current length.
    z_cur, r_cur, h_cur : array_like or None
        The new measurement, its covariance, and measurement matrix
        (each may be None when ``has_last_update``).
    f_interval : array_like
        (x_dim, x_dim) transition matrix, or a stack over the
        interval; the last slice predicts the previous posterior to
        the time of ``z_cur``.
    q_prev : array_like or None
        Process noise covariance for that last transition (None
        allowed when ``has_last_pred``).
    u_prev : array_like, optional
        (x_dim,) control input for that transition.
    has_last_pred : bool, optional
        Whether ``x_fwd_pred`` already contains the prediction to the
        time of ``z_cur``. Default False.
    has_last_update : bool, optional
        Whether ``x_fwd_post`` already contains the posterior at the
        time of ``z_cur``. Default False.

    Returns
    -------
    result : KalmanIntervalSmootherResult

    Examples
    --------
    >>> import numpy as np
    >>> F = np.array([[1.0, 1.0], [0.0, 1.0]])
    >>> H = np.array([[1.0, 0.0]])
    >>> R = np.array([[0.01]])
    >>> Q = 0.001 * np.eye(2)
    >>> res = kalman_interval_smoother(
    ...     None, None, np.array([[0.0], [1.0]]),
    ...     np.eye(2)[:, :, np.newaxis], 3,
    ...     np.array([1.05]), R, H, F, Q)
    >>> res.x.shape
    (2, 2)
    >>> res2 = kalman_interval_smoother(
    ...     res.x_fwd_pred, res.p_fwd_pred, res.x_fwd_post,
    ...     res.p_fwd_post, 3, np.array([2.02]), R, H, F, Q)
    >>> res2.x.shape
    (2, 3)

    Notes
    -----
    Port of ``KalmanIntervalSmoother.m``.
    """
    x_fwd_post = np.atleast_2d(np.asarray(x_fwd_post, dtype=np.float64))
    p_fwd_post = np.asarray(p_fwd_post, dtype=np.float64)
    if p_fwd_post.ndim == 2:
        p_fwd_post = p_fwd_post[:, :, np.newaxis]
    x_dim = x_fwd_post.shape[0]
    n_cur = x_fwd_post.shape[1]
    if x_fwd_pred is None:
        x_fwd_pred = np.zeros((x_dim, 0))
        p_fwd_pred = np.zeros((x_dim, x_dim, 0))
    else:
        x_fwd_pred = np.atleast_2d(np.asarray(x_fwd_pred, dtype=np.float64))
        p_fwd_pred = np.asarray(p_fwd_pred, dtype=np.float64)
        if p_fwd_pred.ndim == 2:
            p_fwd_pred = p_fwd_pred[:, :, np.newaxis]
    n = n_cur if n_interval is None else n_interval

    nf_end = n_cur if n > n_cur else n - 1
    f_interval = _rep3(f_interval, nf_end)

    if not has_last_pred:
        pr = kf_predict(
            x_fwd_post[:, n_cur - 1],
            p_fwd_post[:, :, n_cur - 1],
            f_interval[:, :, nf_end - 1],
            np.asarray(q_prev, dtype=np.float64),
            np.eye(x_dim),
            None if u_prev is None else np.asarray(u_prev, dtype=np.float64).ravel(),
        )
        if n_cur == n:
            # The interval is sliding.
            x_fwd_pred = np.hstack([x_fwd_pred[:, 1 : n - 1], pr.x[:, np.newaxis]])
            p_fwd_pred = np.concatenate(
                [p_fwd_pred[:, :, 1 : n - 1], pr.P[:, :, np.newaxis]], axis=2
            )
        else:
            # The interval is growing.
            x_fwd_pred = np.hstack([x_fwd_pred, pr.x[:, np.newaxis]])
            p_fwd_pred = np.concatenate([p_fwd_pred, pr.P[:, :, np.newaxis]], axis=2)
        has_last_update = False

    if not has_last_update:
        up = kf_update(
            x_fwd_pred[:, nf_end - 1],
            p_fwd_pred[:, :, nf_end - 1],
            np.asarray(z_cur, dtype=np.float64).ravel(),
            np.atleast_2d(np.asarray(h_cur, dtype=np.float64)),
            np.atleast_2d(np.asarray(r_cur, dtype=np.float64)),
        )
        if n_cur == n:
            # The interval is sliding.
            x_fwd_post = np.hstack([x_fwd_post[:, 1:n], up.x[:, np.newaxis]])
            p_fwd_post = np.concatenate(
                [p_fwd_post[:, :, 1:n], up.P[:, :, np.newaxis]], axis=2
            )
        else:
            # The interval is growing.
            x_fwd_post = np.hstack([x_fwd_post, up.x[:, np.newaxis]])
            p_fwd_post = np.concatenate([p_fwd_post, up.P[:, :, np.newaxis]], axis=2)
            n_cur = n_cur + 1

    x_est = np.zeros((x_dim, n_cur))
    p_est = np.zeros((x_dim, x_dim, n_cur))
    x_est[:, n_cur - 1] = x_fwd_post[:, n_cur - 1]
    p_est[:, :, n_cur - 1] = p_fwd_post[:, :, n_cur - 1]
    for k in range(n_cur - 2, -1, -1):
        c = _rdiv(p_fwd_post[:, :, k] @ f_interval[:, :, k].T, p_fwd_pred[:, :, k])
        x_est[:, k] = x_fwd_post[:, k] + c @ (x_est[:, k + 1] - x_fwd_pred[:, k])
        p_est[:, :, k] = (
            p_fwd_post[:, :, k] + c @ (p_est[:, :, k + 1] - p_fwd_pred[:, :, k]) @ c.T
        )

    return KalmanIntervalSmootherResult(
        x_est, p_est, x_fwd_pred, p_fwd_pred, x_fwd_post, p_fwd_post
    )


def fp_info_interval_smoother(
    y_fwd_pred: ArrayLike,
    p_inv_fwd_pred: ArrayLike,
    y_fwd_prev: ArrayLike,
    p_inv_fwd_prev: ArrayLike,
    n_interval: Optional[int],
    z: ArrayLike,
    r: ArrayLike,
    h: ArrayLike,
    f: ArrayLike,
    q: ArrayLike,
    u: Optional[ArrayLike] = None,
    has_last_pred: bool = False,
) -> FPInfoIntervalSmootherResult:
    """
    Sliding- or growing-interval Fraser-Potter information smoother.

    Parameters
    ----------
    y_fwd_pred : array_like
        (x_dim, N) forward predicted information states over the
        interval (from the previous call; the first call passes the
        single initial predicted information state as a column).
    p_inv_fwd_pred : array_like
        Inverse covariances for ``y_fwd_pred``.
    y_fwd_prev : array_like
        (x_dim,) forward information estimate at the previous step
        (``y_fwd_end`` from the previous call).
    p_inv_fwd_prev : array_like
        (x_dim, x_dim) inverse covariance of ``y_fwd_prev``.
    n_interval : int or None
        Desired interval length; None keeps the current length.
    z : array_like
        (z_dim, N_cur) measurements over the whole current window
        (newest last).
    r, h : array_like
        Measurement covariance(s) and matrix(es), single or stacks of
        N_cur.
    f, q : array_like
        Transition matrix(es) and process noise covariance(s), single
        or stacks of N_cur - 1.
    u : array_like, optional
        (x_dim,) or (x_dim, N_cur - 1) control inputs.
    has_last_pred : bool, optional
        Whether ``y_fwd_pred`` already includes the prediction to the
        newest step. Default False.

    Returns
    -------
    result : FPInfoIntervalSmootherResult

    Examples
    --------
    >>> import numpy as np
    >>> F = np.array([[1.0, 1.0], [0.0, 1.0]])
    >>> H = np.array([[1.0, 0.0]])
    >>> R = np.array([[0.01]])
    >>> Q = 0.001 * np.eye(2)
    >>> y0 = np.zeros((2, 1))
    >>> pinv0 = np.zeros((2, 2, 1))
    >>> res = fp_info_interval_smoother(
    ...     y0, pinv0, np.zeros(2), np.zeros((2, 2)), 3,
    ...     np.array([[0.05, 1.1]]), R, H, F, Q)
    >>> res.y.shape
    (2, 2)

    Notes
    -----
    Port of ``FPInfoIntervalSmoother.m``. The returned ``y_fwd_end``
    equals the smoothed information state at the newest step, which is
    the forward-updated estimate there.
    """
    y_fwd_pred = np.atleast_2d(np.asarray(y_fwd_pred, dtype=np.float64))
    p_inv_fwd_pred = np.asarray(p_inv_fwd_pred, dtype=np.float64)
    if p_inv_fwd_pred.ndim == 2:
        p_inv_fwd_pred = p_inv_fwd_pred[:, :, np.newaxis]
    y_fwd_prev = np.asarray(y_fwd_prev, dtype=np.float64).ravel()
    p_inv_fwd_prev = np.asarray(p_inv_fwd_prev, dtype=np.float64)
    z = np.atleast_2d(np.asarray(z, dtype=np.float64))
    x_dim = y_fwd_pred.shape[0]
    n_cur = z.shape[1]
    n = n_cur if n_interval is None else n_interval
    nf_end = n_cur - 1

    f = _rep3(f, nf_end)
    q = _rep3(q, nf_end)
    if u is None:
        u_arr = np.zeros((x_dim, nf_end))
    else:
        u_arr = np.atleast_2d(np.asarray(u, dtype=np.float64))
        if u_arr.shape[1] == 1:
            u_arr = np.repeat(u_arr, nf_end, axis=1)
    h = _rep3(h, n_cur)
    r = _rep3(r, n_cur)

    if not has_last_pred:
        y_pred_end, p_inv_pred_end = _info_filter_disc_pred(
            y_fwd_prev,
            p_inv_fwd_prev,
            f[:, :, nf_end - 1],
            q[:, :, nf_end - 1],
            u_arr[:, nf_end - 1],
        )
        if n_cur == n and y_fwd_pred.shape[1] == n:
            # The interval is sliding.
            y_fwd_pred = np.hstack([y_fwd_pred[:, 1:n], y_pred_end[:, np.newaxis]])
            p_inv_fwd_pred = np.concatenate(
                [p_inv_fwd_pred[:, :, 1:n], p_inv_pred_end[:, :, np.newaxis]],
                axis=2,
            )
        else:
            # The interval is growing.
            y_fwd_pred = np.hstack([y_fwd_pred, y_pred_end[:, np.newaxis]])
            p_inv_fwd_pred = np.concatenate(
                [p_inv_fwd_pred, p_inv_pred_end[:, :, np.newaxis]], axis=2
            )

    y_rev = np.zeros((x_dim, n_cur))
    p_inv_rev = np.zeros((x_dim, x_dim, n_cur))
    y_rev[:, n_cur - 1], p_inv_rev[:, :, n_cur - 1] = _info_filter_update(
        np.zeros(x_dim),
        np.zeros((x_dim, x_dim)),
        z[:, n_cur - 1],
        r[:, :, n_cur - 1],
        h[:, :, n_cur - 1],
    )
    for k in range(n_cur - 2, -1, -1):
        y_rev_pred, p_inv_rev_pred = _info_filter_disc_pred_rev(
            y_rev[:, k + 1], p_inv_rev[:, :, k + 1], f[:, :, k], q[:, :, k], u_arr[:, k]
        )
        y_rev[:, k], p_inv_rev[:, :, k] = _info_filter_update(
            y_rev_pred, p_inv_rev_pred, z[:, k], r[:, :, k], h[:, :, k]
        )

    y_est = y_fwd_pred + y_rev
    p_inv_est = p_inv_fwd_pred + p_inv_rev
    return FPInfoIntervalSmootherResult(
        y_est,
        p_inv_est,
        y_fwd_pred,
        p_inv_fwd_pred,
        y_est[:, -1],
        p_inv_est[:, :, -1],
    )


def kalman_fir_smoother_coeffs(
    h: ArrayLike,
    f: ArrayLike,
    r: ArrayLike,
    q: ArrayLike,
    k_d: int,
) -> FIRSmootherCoeffs:
    """
    Coefficients for the linear Kalman FIR smoother.

    Produces matrices A and B such that summing ``A[:, :, k] @ z[:, k]
    + B[:, :, k] @ u[:, k]`` over the window yields the smoothed state
    at step ``k_d`` with no prior information.

    Parameters
    ----------
    h : array_like
        (z_dim, x_dim, N) stack of measurement matrices (or a single
        matrix).
    f : array_like
        (x_dim, x_dim, N-1) stack of invertible transition matrices
        (or a single matrix).
    r : array_like
        (z_dim, z_dim, N) stack of measurement covariances (or one).
    q : array_like
        (x_dim, x_dim, N-1) stack of invertible process noise
        covariances (or one).
    k_d : int
        Zero-based step at which the smoothed estimate is desired.

    Returns
    -------
    result : FIRSmootherCoeffs

    Examples
    --------
    >>> import numpy as np
    >>> F = np.array([[1.0, 1.0], [0.0, 1.0]])
    >>> H = np.array([[1.0, 0.0]])
    >>> coeffs = kalman_fir_smoother_coeffs(
    ...     np.repeat(H[:, :, np.newaxis], 4, axis=2),
    ...     np.repeat(F[:, :, np.newaxis], 3, axis=2),
    ...     np.full((1, 1, 4), 0.01), np.repeat(
    ...         (0.1 * np.eye(2))[:, :, np.newaxis], 3, axis=2), 1)
    >>> coeffs.a.shape
    (2, 1, 4)

    Notes
    -----
    Port of ``KalmanFIRSmootherCoeffs.m``; the strict orderings of the
    D-matrix products (backward for the forward coefficients, forward
    for the backward coefficients) follow the original.
    """
    h = np.asarray(h, dtype=np.float64)
    if h.ndim == 2:
        raise ValueError("h must be a (z_dim, x_dim, N) stack to define N")
    z_dim, x_dim, n = h.shape
    f = _rep3(f, n - 1)
    r = _rep3(r, n)
    q = _rep3(q, n - 1)
    kd = k_d + 1  # The 1-based index the MATLAB algebra is written in.
    eye = np.eye(x_dim)

    def _ht_rinv_h(k):
        return np.linalg.solve(r[:, :, k].T, h[:, :, k]).T @ h[:, :, k]

    def _ht_rinv(k):
        return np.linalg.solve(r[:, :, k].T, h[:, :, k]).T

    p_inv_1 = np.zeros((x_dim, x_dim, kd))
    p_inv_pred_1 = np.zeros((x_dim, x_dim, kd))
    d_1 = np.zeros((x_dim, x_dim, kd))
    p_inv_1[:, :, 0] = _ht_rinv_h(0)
    for k in range(1, kd):
        d_inv_1 = (
            f[:, :, k - 1].T
            + _rdiv(p_inv_1[:, :, k - 1], f[:, :, k - 1]) @ q[:, :, k - 1]
        )
        p_inv_pred_1[:, :, k - 1] = np.linalg.solve(
            d_inv_1, _rdiv(p_inv_1[:, :, k - 1], f[:, :, k - 1])
        )
        p_inv_1[:, :, k] = p_inv_pred_1[:, :, k - 1] + _ht_rinv_h(k)
        d_1[:, :, k - 1] = np.linalg.inv(d_inv_1)

    p_inv_n = np.zeros((x_dim, x_dim, n))
    p_inv_pred_n = np.zeros((x_dim, x_dim, n))
    d_n = np.zeros((x_dim, x_dim, n))
    p_inv_n[:, :, n - 1] = _ht_rinv_h(n - 1)
    for k in range(n - 2, kd - 2, -1):
        d_inv_n = np.linalg.inv(f[:, :, k]).T + _rdiv(
            p_inv_n[:, :, k + 1] @ q[:, :, k], f[:, :, k].T
        )
        p_inv_pred_n[:, :, k + 1] = (
            np.linalg.solve(d_inv_n, p_inv_n[:, :, k + 1]) @ f[:, :, k]
        )
        p_inv_n[:, :, k] = p_inv_pred_n[:, :, k + 1] + _ht_rinv_h(k)
        d_n[:, :, k + 1] = np.linalg.inv(d_inv_n)

    a = np.zeros((x_dim, z_dim, n))
    b = np.zeros((x_dim, x_dim, n - 1))
    if kd > 1:
        p_inv_kn = p_inv_pred_1[:, :, kd - 2] + p_inv_n[:, :, kd - 1]
    else:
        p_inv_kn = p_inv_n[:, :, kd - 1]
    p_kn = np.linalg.inv(p_inv_kn)

    # The forward coefficients. The order of the D products MUST be
    # backwards, as in the original.
    for j in range(max(kd - 1, 1)):
        d_prod = eye.copy()
        for m in range(kd - 2, j - 1, -1):
            d_prod = d_prod @ d_1[:, :, m]
        a[:, :, j] = np.linalg.solve(p_inv_kn, d_prod @ _ht_rinv(j))

        d_prod = eye.copy()
        for m in range(kd - 2, j, -1):
            d_prod = d_prod @ d_1[:, :, m]
        b[:, :, j] = np.linalg.solve(p_inv_kn, d_prod @ p_inv_pred_1[:, :, j])

    # The backward coefficients. The order of the D products MUST be
    # forwards, as in the original.
    for j in range(kd - 1, n):
        d_prod = eye.copy()
        for m in range(kd, j + 1):
            d_prod = d_prod @ d_n[:, :, m]
        a[:, :, j] = np.linalg.solve(p_inv_kn, d_prod @ _ht_rinv(j))

        if j < n - 1:
            b[:, :, j] = -np.linalg.solve(
                p_inv_kn, _rdiv(d_prod @ p_inv_pred_n[:, :, j + 1], f[:, :, j])
            )

    return FIRSmootherCoeffs(a, b, p_kn)


def kalman_fir_smoother(
    z: ArrayLike,
    u: Optional[ArrayLike],
    h: ArrayLike,
    f: ArrayLike,
    r: ArrayLike,
    q: ArrayLike,
    k_d: int,
) -> BatchSmootherResult:
    """
    Linear Kalman FIR smoother over a batch with no prior information.

    Parameters
    ----------
    z : array_like
        (z_dim, N) batch of measurements.
    u : array_like or None
        (x_dim, N-1) control inputs, or None.
    h : array_like
        (z_dim, x_dim) measurement matrix, or a stack of N.
    f : array_like
        (x_dim, x_dim) invertible transition matrix, or a stack of
        N-1.
    r : array_like
        (z_dim, z_dim) measurement covariance, or a stack of N.
    q : array_like
        (x_dim, x_dim) invertible process noise covariance, or a
        stack of N-1.
    k_d : int
        Zero-based step at which the smoothed estimate is desired.

    Returns
    -------
    result : BatchSmootherResult
        ``x`` is the (x_dim,) smoothed state at ``k_d`` and ``P`` its
        covariance.

    Examples
    --------
    >>> import numpy as np
    >>> F = np.array([[1.0, 1.0], [0.0, 1.0]])
    >>> H = np.array([[1.0, 0.0]])
    >>> z = np.array([[0.0, 1.0, 2.0, 3.0]])
    >>> res = kalman_fir_smoother(z, None, H, F, np.array([[0.01]]),
    ...                           1e-6 * np.eye(2), 1)
    >>> np.allclose(res.x, [1.0, 1.0], atol=1e-2)
    True

    Notes
    -----
    Port of ``KalmanFIRSmoother.m``.
    """
    z = np.atleast_2d(np.asarray(z, dtype=np.float64))
    n = z.shape[1]
    h = _rep3(h, n)
    x_dim = h.shape[1]
    u_arr = (
        np.zeros((x_dim, n - 1))
        if u is None
        else np.atleast_2d(np.asarray(u, dtype=np.float64))
    )
    f = _rep3(f, n - 1)
    r = _rep3(r, n)
    q = _rep3(q, n - 1)

    coeffs = kalman_fir_smoother_coeffs(h, f, r, q, k_d)
    x_est = np.zeros(x_dim)
    for k in range(n - 1):
        x_est = x_est + coeffs.a[:, :, k] @ z[:, k] + coeffs.b[:, :, k] @ u_arr[:, k]
    # No control input is used at the final step.
    x_est = x_est + coeffs.a[:, :, n - 1] @ z[:, n - 1]
    return BatchSmootherResult(x_est, coeffs.p_kn)


__all__ = [
    "BatchSmootherResult",
    "EKalmanBatchSmootherResult",
    "FIRSmootherCoeffs",
    "FPInfoBatchSmootherResult",
    "FPInfoIntervalSmootherResult",
    "KalmanIntervalSmootherResult",
    "SqrtCubKalBatchSmootherResult",
    "SqrtInfoBatchSmootherResult",
    "ekalman_batch_smoother",
    "fp_info_batch_smoother",
    "fp_info_interval_smoother",
    "kalman_batch_smoother",
    "kalman_fir_smoother",
    "kalman_fir_smoother_coeffs",
    "kalman_interval_smoother",
    "sqrt_cub_kal_batch_smoother",
    "sqrt_info_batch_smoother",
]
