"""
Batch least-squares state estimation.

Estimators that fit a state (or a whole trajectory) to a batch of
measurements at once: the closed-form linear batch estimator, iterated
Gauss-Newton estimators for nonlinear measurements with linear or
folded-in dynamics, Levenberg-Marquardt variants, and two-point
differencing initialization. Ports of the MATLAB TCL
``batchLSLinMeasLinDyn.m``, ``batchLSNonlinMeasLinDyn.m``,
``batchLSNonlinMeasNonlinDyn.m``, ``batchLSNonlinMeasLinDynLM.m``,
``batchLSNonlinMeasNonlinDynLM.m`` and ``twoPointDiffInit.m``.

Time indices are zero-based here: ``k_d`` selects which step the
returned state/covariance refers to, where the MATLAB originals use
1-based ``kD``.

References
----------
.. [1] Y. Bar-Shalom, X. R. Li, and T. Kirubarajan, Estimation with
   Applications to Tracking and Navigation. New York: John Wiley and
   Sons, 2001.
"""

from typing import Callable, NamedTuple, Optional, Sequence, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray


class BatchLSResult(NamedTuple):
    """Result of a batch least-squares estimate.

    Attributes
    ----------
    x : ndarray or None
        (x_dim,) state estimate at step ``k_d`` (None when only the
        covariance was requested).
    P : ndarray
        (x_dim, x_dim) covariance estimate at step ``k_d``.
    """

    x: Optional[NDArray[np.floating]]
    P: NDArray[np.floating]


class BatchLSLMResult(NamedTuple):
    """Result of a Levenberg-Marquardt batch estimate.

    Attributes
    ----------
    x : ndarray
        (x_dim,) state estimate at step ``k_d``.
    P : ndarray
        (x_dim, x_dim) linearized covariance estimate.
    x_batch : ndarray
        (x_dim, num_steps) trajectory estimate (predicted from ``x``
        when no process noise was modeled).
    success : bool
        Whether the optimizer reported convergence.
    """

    x: NDArray[np.floating]
    P: NDArray[np.floating]
    x_batch: NDArray[np.floating]
    success: bool


def _get_trans_mats(F: NDArray[np.floating]) -> NDArray[np.floating]:
    """All state-transition products (``getTransMats``, full form).

    ``FMats[:, :, p, i]`` maps the state at step ``i`` to step ``p``
    (zero-based); backward maps invert the transition matrices.
    """
    x_dim = F.shape[0]
    n = F.shape[2] + 1
    mats = np.zeros((x_dim, x_dim, n, n))
    for k in range(n):
        mats[:, :, k, k] = np.eye(x_dim)
    for i in range(n - 1):
        f_recur = F[:, :, i]
        for p in range(i + 1, n):
            mats[:, :, p, i] = f_recur
            if p < n - 1:
                f_recur = F[:, :, p] @ f_recur
    for p in range(n - 1):
        f_recur = F[:, :, p]
        for i in range(p + 1, n):
            mats[:, :, p, i] = np.linalg.inv(f_recur)
            if i < n - 1:
                f_recur = F[:, :, i] @ f_recur
    return mats


def _get_trans_mats_base(
    F: NDArray[np.floating], base_idx: int
) -> NDArray[np.floating]:
    """Transitions from ``base_idx`` to every step
    (``getTransMats`` with a base index)."""
    x_dim = F.shape[0]
    n = F.shape[2] + 1
    mats = np.zeros((x_dim, x_dim, n))
    mats[:, :, base_idx] = np.eye(x_dim)
    if base_idx < n - 1:
        f_recur = F[:, :, base_idx]
        for p in range(base_idx + 1, n):
            mats[:, :, p] = f_recur
            if p < n - 1:
                f_recur = F[:, :, p] @ f_recur
    for p in range(base_idx - 1, -1, -1):
        mats[:, :, p] = np.linalg.inv(F[:, :, p]) @ mats[:, :, p + 1]
    return mats


def _tile3(a: ArrayLike, n: int) -> NDArray[np.floating]:
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]
    if arr.shape[2] == 1:
        arr = np.repeat(arr, n, axis=2)
    return arr


def _per_step_callables(fun, num_steps):
    if callable(fun):
        return [fun] * num_steps
    return list(fun)


def _default_jacobians(h_jacob, h_list, z_dim, num_steps):
    from pytcl.dynamic_estimation.kalman.extended import numerical_jacobian

    if h_jacob is None:
        return [
            (lambda hk: lambda x: numerical_jacobian(hk, x))(h_list[k])
            for k in range(num_steps)
        ]
    return _per_step_callables(h_jacob, num_steps)


def batch_ls_lin_meas_lin_dyn(
    z: Optional[ArrayLike],
    H: ArrayLike,
    F: ArrayLike,
    R: ArrayLike,
    k_d: int,
    Q: Optional[ArrayLike] = None,
    num_meas: Optional[int] = None,
) -> BatchLSResult:
    """
    Closed-form batch estimate with linear measurements and dynamics.

    Estimates the state at step ``k_d`` from the whole measurement
    batch, accounting for process noise accumulated between each
    measurement and the estimation step.

    Parameters
    ----------
    z : array_like or None
        (z_dim, num_meas) measurements, or None to compute only the
        covariance (then ``num_meas`` is required).
    H : array_like
        (z_dim, x_dim) measurement matrix, or (z_dim, x_dim, num_meas)
        per-step matrices.
    F : array_like
        (x_dim, x_dim) state transition, or (x_dim, x_dim,
        num_meas - 1) per-step. Must be invertible when ``k_d`` is not
        the first step.
    R : array_like
        (z_dim, z_dim[, num_meas]) measurement covariance(s).
    k_d : int
        Zero-based step at which the state is estimated (the MATLAB
        original's ``kD`` is 1-based).
    Q : array_like, optional
        (x_dim, x_dim[, num_meas - 1]) process noise covariance(s).
        Default zero.
    num_meas : int, optional
        Number of measurements when ``z`` is None.

    Returns
    -------
    result : BatchLSResult

    Examples
    --------
    >>> import numpy as np
    >>> F = np.array([[1.0, 1.0], [0.0, 1.0]])
    >>> H = np.array([[1.0, 0.0]])
    >>> R = np.eye(1) * 0.01
    >>> z = np.array([[0.0, 1.0, 2.0]])  # constant velocity 1
    >>> res = batch_ls_lin_meas_lin_dyn(z, H, F, R, 0)
    >>> np.round(res.x, 6)
    array([0., 1.])

    Notes
    -----
    Port of ``batchLSLinMeasLinDyn.m``.
    """
    H = np.asarray(H, dtype=np.float64)
    F = np.asarray(F, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)

    if z is not None:
        z = np.asarray(z, dtype=np.float64)
        if z.ndim == 1:
            z = z[np.newaxis, :]
        num_meas = z.shape[1]
    if num_meas is None:
        raise ValueError("num_meas is required when z is None.")

    x_dim = F.shape[0]
    z_dim = R.shape[0]
    kd = k_d + 1  # 1-based index used by the original's index algebra.

    h_mats = _tile3(H, num_meas)
    F = _tile3(F, num_meas - 1)
    R = _tile3(R, num_meas)
    if Q is None:
        q_mats = np.zeros((x_dim, x_dim, num_meas))
    else:
        q_mats = _tile3(Q, num_meas - 1)

    trans = _get_trans_mats(F)

    w_mat = np.zeros((z_dim * num_meas, z_dim * num_meas))
    for p in range(1, num_meas + 1):
        p_span = slice((p - 1) * z_dim, p * z_dim)
        for q in range(1, num_meas + 1):
            q_span = slice((q - 1) * z_dim, q * z_dim)
            if p < kd and q < kd:
                cum = np.zeros((x_dim, x_dim))
                for i in range(max(p, q) + 1, kd + 1):
                    cum += (
                        trans[:, :, p - 1, i - 1]
                        @ q_mats[:, :, i - 2]
                        @ trans[:, :, q - 1, i - 1].T
                    )
                w_mat[p_span, q_span] = (
                    h_mats[:, :, p - 1] @ cum @ h_mats[:, :, q - 1].T
                    + (p == q) * R[:, :, p - 1]
                )
            elif p > kd and q > kd:
                cum = np.zeros((x_dim, x_dim))
                for i in range(kd + 1, min(p, q) + 1):
                    cum += (
                        trans[:, :, p - 1, i - 1]
                        @ q_mats[:, :, i - 2]
                        @ trans[:, :, q - 1, i - 1].T
                    )
                w_mat[p_span, q_span] = (p == q) * R[:, :, p - 1] + h_mats[
                    :, :, p - 1
                ] @ cum @ h_mats[:, :, q - 1].T
            elif p == kd and q == kd:
                w_mat[p_span, q_span] = R[:, :, p - 1]

    big_h = np.zeros((z_dim * num_meas, x_dim))
    for m in range(num_meas):
        big_h[m * z_dim : (m + 1) * z_dim, :] = h_mats[:, :, m] @ trans[:, :, m, kd - 1]

    w_inv = np.linalg.inv(w_mat)
    p_est_inv = big_h.T @ w_inv @ big_h
    p_est = np.linalg.inv(p_est_inv)
    if z is None:
        return BatchLSResult(None, p_est)
    x_est = np.linalg.solve(
        p_est_inv, big_h.T @ np.linalg.solve(w_mat, z.ravel(order="F"))
    )
    return BatchLSResult(x_est, p_est)


def batch_ls_nonlin_meas_lin_dyn(
    x_init: ArrayLike,
    z: ArrayLike,
    h: Union[Callable, Sequence[Callable]],
    F: ArrayLike,
    R: ArrayLike,
    k_d: int,
    h_jacob: Union[Callable, Sequence[Callable], None] = None,
    num_iter: int = 10,
) -> BatchLSResult:
    """
    Iterated Gauss-Newton batch estimate, nonlinear measurements.

    Parameters
    ----------
    x_init : array_like
        (x_dim,) initial estimate of the state at step ``k_d``.
    z : array_like
        (z_dim, num_steps) measurements.
    h : callable or sequence of callables
        Measurement function(s) ``h(x)``, shared or per step.
    F : array_like
        (x_dim, x_dim[, num_steps - 1]) state transition(s); must be
        invertible when ``k_d`` is not the first step.
    R : array_like
        (z_dim, z_dim[, num_steps]) measurement covariance(s).
    k_d : int
        Zero-based step of the estimated state.
    h_jacob : callable or sequence, optional
        Measurement Jacobian(s); None uses numerical Jacobians.
    num_iter : int, optional
        Gauss-Newton iterations. Default 10.

    Returns
    -------
    result : BatchLSResult

    Examples
    --------
    >>> import numpy as np
    >>> F = np.array([[1.0, 1.0], [0.0, 1.0]])
    >>> h = lambda x: np.array([x[0] ** 2])
    >>> hj = lambda x: np.array([[2.0 * x[0], 0.0]])
    >>> true0 = np.array([2.0, 0.5])
    >>> zs = []
    >>> x = true0.copy()
    >>> for _ in range(4):
    ...     zs.append(h(x))
    ...     x = F @ x
    >>> z = np.array(zs).T
    >>> res = batch_ls_nonlin_meas_lin_dyn(
    ...     np.array([1.8, 0.4]), z, h, F, 0.01 * np.eye(1), 0, hj)
    >>> np.round(res.x, 5)
    array([2. , 0.5])

    Notes
    -----
    Port of ``batchLSNonlinMeasLinDyn.m``.
    """
    x_est = np.asarray(x_init, dtype=np.float64).ravel().copy()
    z = np.asarray(z, dtype=np.float64)
    if z.ndim == 1:
        z = z[np.newaxis, :]
    F = np.asarray(F, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)

    z_dim, num_steps = z.shape
    x_dim = len(x_est)

    h_list = _per_step_callables(h, num_steps)
    hj_list = _default_jacobians(h_jacob, h_list, z_dim, num_steps)
    R = _tile3(R, num_steps)
    F = _tile3(F, num_steps - 1)

    r_stacked_inv = np.linalg.inv(_block_diag([R[:, :, k] for k in range(num_steps)]))
    trans = _get_trans_mats_base(F, k_d)

    j = np.zeros((z_dim * num_steps, x_dim))
    z_pred = np.zeros(z_dim * num_steps)
    for _ in range(num_iter):
        for k in range(num_steps):
            span = slice(k * z_dim, (k + 1) * z_dim)
            x_cur = trans[:, :, k] @ x_est
            j[span, :] = np.atleast_2d(hj_list[k](x_cur)) @ trans[:, :, k]
            z_pred[span] = np.atleast_1d(h_list[k](x_cur))
        lhs = j.T @ r_stacked_inv @ j
        rhs = j.T @ r_stacked_inv @ (z.ravel(order="F") - z_pred)
        x_est = x_est + np.linalg.lstsq(lhs, rhs, rcond=None)[0]

    for k in range(num_steps):
        span = slice(k * z_dim, (k + 1) * z_dim)
        x_cur = trans[:, :, k] @ x_est
        j[span, :] = np.atleast_2d(hj_list[k](x_cur)) @ trans[:, :, k]
    p_est = np.linalg.pinv(j.T @ r_stacked_inv @ j)
    return BatchLSResult(x_est, p_est)


def batch_ls_nonlin_meas_nonlin_dyn(
    x_init: ArrayLike,
    z: ArrayLike,
    h: Union[Callable, Sequence[Callable]],
    R: ArrayLike,
    h_jacob: Union[Callable, Sequence[Callable], None] = None,
    num_iter: int = 10,
) -> BatchLSResult:
    """
    Iterated Gauss-Newton batch estimate with dynamics folded into h.

    Each per-step function maps the state being estimated directly to
    that step's measurement, so arbitrary (nonlinear) dynamics are
    expressed inside the ``h`` sequence.

    Parameters
    ----------
    x_init : array_like
        (x_dim,) initial estimate.
    z : array_like
        (z_dim, num_steps) measurements.
    h : callable or sequence of callables
        Per-step maps from the estimated state to each measurement.
    R : array_like
        (z_dim, z_dim[, num_steps]) measurement covariance(s).
    h_jacob : callable or sequence, optional
        Jacobian(s); None uses numerical Jacobians.
    num_iter : int, optional
        Gauss-Newton iterations. Default 10.

    Returns
    -------
    result : BatchLSResult

    Examples
    --------
    >>> import numpy as np
    >>> F = np.array([[1.0, 1.0], [0.0, 1.0]])
    >>> hs = [(lambda k: (lambda x: (np.linalg.matrix_power(F, k) @ x)[:1]))(k)
    ...       for k in range(4)]
    >>> true0 = np.array([2.0, 0.5])
    >>> z = np.column_stack([hs[k](true0) for k in range(4)])
    >>> res = batch_ls_nonlin_meas_nonlin_dyn(
    ...     np.array([1.5, 0.2]), z, hs, 0.01 * np.eye(1))
    >>> np.round(res.x, 5)
    array([2. , 0.5])

    Notes
    -----
    Port of ``batchLSNonlinMeasNonlinDyn.m``.
    """
    x_est = np.asarray(x_init, dtype=np.float64).ravel().copy()
    z = np.asarray(z, dtype=np.float64)
    if z.ndim == 1:
        z = z[np.newaxis, :]
    R = np.asarray(R, dtype=np.float64)

    z_dim, num_steps = z.shape
    x_dim = len(x_est)

    h_list = _per_step_callables(h, num_steps)
    hj_list = _default_jacobians(h_jacob, h_list, z_dim, num_steps)
    R = _tile3(R, num_steps)

    r_stacked_inv = np.linalg.inv(_block_diag([R[:, :, k] for k in range(num_steps)]))

    j = np.zeros((z_dim * num_steps, x_dim))
    z_pred = np.zeros(z_dim * num_steps)
    for _ in range(num_iter):
        for k in range(num_steps):
            span = slice(k * z_dim, (k + 1) * z_dim)
            j[span, :] = np.atleast_2d(hj_list[k](x_est))
            z_pred[span] = np.atleast_1d(h_list[k](x_est))
        lhs = j.T @ r_stacked_inv @ j
        rhs = j.T @ r_stacked_inv @ (z.ravel(order="F") - z_pred)
        x_est = x_est + np.linalg.lstsq(lhs, rhs, rcond=None)[0]

    for k in range(num_steps):
        span = slice(k * z_dim, (k + 1) * z_dim)
        j[span, :] = np.atleast_2d(hj_list[k](x_est))
    p_est = np.linalg.pinv(j.T @ r_stacked_inv @ j)
    return BatchLSResult(x_est, p_est)


def _block_diag(mats):
    from scipy.linalg import block_diag

    return block_diag(*mats)


def batch_ls_nonlin_meas_lin_dyn_lm(
    x_init: ArrayLike,
    z: ArrayLike,
    h: Union[Callable, Sequence[Callable]],
    F: ArrayLike,
    R: ArrayLike,
    k_d: int = 0,
    Q: Optional[ArrayLike] = None,
    h_jacob: Union[Callable, Sequence[Callable], None] = None,
    max_iter: Optional[int] = None,
) -> BatchLSLMResult:
    """
    Levenberg-Marquardt batch estimate, nonlinear measurements.

    Without process noise the single state at step ``k_d`` is
    estimated; with ``Q`` given, the whole trajectory is estimated
    with process-noise-whitened dynamics residuals.

    Parameters
    ----------
    x_init : array_like
        (x_dim,) initial state at ``k_d``, or (x_dim, num_steps)
        initial trajectory in the process-noise mode.
    z, h, F, R, k_d, h_jacob
        As in :func:`batch_ls_nonlin_meas_lin_dyn`.
    Q : array_like, optional
        (x_dim, x_dim[, num_steps - 1]) process noise; enables the
        trajectory mode.
    max_iter : int, optional
        Maximum optimizer iterations.

    Returns
    -------
    result : BatchLSLMResult

    Examples
    --------
    >>> import numpy as np
    >>> F = np.array([[1.0, 1.0], [0.0, 1.0]])
    >>> h = lambda x: np.array([x[0] ** 2])
    >>> true0 = np.array([2.0, 0.5])
    >>> zs, x = [], true0.copy()
    >>> for _ in range(4):
    ...     zs.append(h(x)); x = F @ x
    >>> res = batch_ls_nonlin_meas_lin_dyn_lm(
    ...     np.array([1.8, 0.4]), np.array(zs).T, h, F, 0.01 * np.eye(1), 0)
    >>> bool(res.success), tuple(np.round(res.x, 5))
    (True, (2.0, 0.5))

    Notes
    -----
    Port of ``batchLSNonlinMeasLinDynLM.m``. SciPy's
    ``least_squares(method="lm")`` substitutes the original's
    ``LSEstLMarquardt`` (same whitened residuals, same optimum,
    different damping schedule). The covariance comes from
    :func:`batch_ls_lin_meas_lin_dyn` with the linearized measurement
    matrices, as in the original.
    """
    from scipy.optimize import least_squares

    x_init = np.asarray(x_init, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    if z.ndim == 1:
        z = z[np.newaxis, :]
    F_in = np.asarray(F, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)

    z_dim, num_steps = z.shape
    x_dim = x_init.shape[0]

    h_list = _per_step_callables(h, num_steps)
    F = _tile3(F_in, num_steps - 1)
    R3 = _tile3(R, num_steps)
    sr = np.zeros_like(R3)
    for k in range(num_steps):
        sr[:, :, k] = np.linalg.cholesky(R3[:, :, k])

    kd = k_d  # zero-based throughout

    if Q is not None:
        q3 = _tile3(np.asarray(Q, dtype=np.float64), num_steps - 1)
        sq = np.zeros_like(q3)
        for k in range(num_steps - 1):
            if np.any(q3[:, :, k]):
                sq[:, :, k] = np.linalg.cholesky(q3[:, :, k])

        if x_init.ndim > 1 and x_init.shape[1] > 1:
            x_batch0 = x_init.copy()
        else:
            x_batch0 = np.zeros((x_dim, num_steps))
            x_batch0[:, kd] = x_init.ravel()
            for k in range(kd + 1, num_steps):
                x_batch0[:, k] = F[:, :, k - 1] @ x_batch0[:, k - 1]
            for k in range(kd - 1, -1, -1):
                x_batch0[:, k] = np.linalg.solve(F[:, :, k], x_batch0[:, k + 1])

        def _resid(x_flat):
            xb = x_flat.reshape(x_dim, num_steps, order="F")
            out = np.zeros(z_dim * num_steps + x_dim * (num_steps - 1))
            out[:z_dim] = np.linalg.solve(
                sr[:, :, 0], np.atleast_1d(h_list[0](xb[:, 0])) - z[:, 0]
            )
            idx = z_dim
            for k in range(1, num_steps):
                out[idx : idx + z_dim] = np.linalg.solve(
                    sr[:, :, k], np.atleast_1d(h_list[k](xb[:, k])) - z[:, k]
                )
                idx += z_dim
                d = xb[:, k] - F[:, :, k - 1] @ xb[:, k - 1]
                out[idx : idx + x_dim] = np.linalg.solve(sq[:, :, k - 1], d)
                idx += x_dim
            return out

        res = least_squares(
            _resid, x_batch0.ravel(order="F"), method="lm", max_nfev=max_iter
        )
        x_batch = res.x.reshape(x_dim, num_steps, order="F")
        x_est = x_batch[:, kd]
    else:

        def _resid(x0):
            out = np.zeros(z_dim * num_steps)
            out[:z_dim] = np.linalg.solve(
                sr[:, :, kd], np.atleast_1d(h_list[kd](x0)) - z[:, kd]
            )
            idx = z_dim
            x_cur = x0
            for k in range(kd + 1, num_steps):
                x_cur = F[:, :, k - 1] @ x_cur
                out[idx : idx + z_dim] = np.linalg.solve(
                    sr[:, :, k], np.atleast_1d(h_list[k](x_cur)) - z[:, k]
                )
                idx += z_dim
            x_cur = x0
            for k in range(kd - 1, -1, -1):
                x_cur = np.linalg.solve(F[:, :, k], x_cur)
                out[idx : idx + z_dim] = np.linalg.solve(
                    sr[:, :, k], np.atleast_1d(h_list[k](x_cur)) - z[:, k]
                )
                idx += z_dim
            return out

        res = least_squares(_resid, x_init.ravel(), method="lm", max_nfev=max_iter)
        x_est = res.x
        x_batch = np.zeros((x_dim, num_steps))
        x_batch[:, kd] = x_est
        for k in range(kd + 1, num_steps):
            x_batch[:, k] = F[:, :, k - 1] @ x_batch[:, k - 1]
        for k in range(kd - 1, -1, -1):
            x_batch[:, k] = np.linalg.solve(F[:, :, k], x_batch[:, k + 1])

    hj_list = _default_jacobians(h_jacob, h_list, z_dim, num_steps)
    h_mats = np.zeros((z_dim, x_dim, num_steps))
    for k in range(num_steps):
        h_mats[:, :, k] = np.atleast_2d(hj_list[k](x_batch[:, k]))
    p_est = batch_ls_lin_meas_lin_dyn(None, h_mats, F, R3, k_d, Q, num_steps).P
    return BatchLSLMResult(x_est, p_est, x_batch, bool(res.success))


def batch_ls_nonlin_meas_nonlin_dyn_lm(
    x_init: ArrayLike,
    z: ArrayLike,
    h: Union[Callable, Sequence[Callable]],
    R: ArrayLike,
    k_d: int = 0,
    h_jacob: Union[Callable, Sequence[Callable], None] = None,
    max_iter: Optional[int] = None,
) -> BatchLSLMResult:
    """
    Levenberg-Marquardt batch estimate with dynamics folded into h.

    Parameters
    ----------
    x_init : array_like
        (x_dim,) initial estimate.
    z, h, R, h_jacob
        As in :func:`batch_ls_nonlin_meas_nonlin_dyn`.
    k_d : int, optional
        Zero-based index of the step whose residual is ordered first
        (does not change the optimum). Default 0.
    max_iter : int, optional
        Maximum optimizer iterations.

    Returns
    -------
    result : BatchLSLMResult
        ``x_batch`` equals ``x`` tiled, since the dynamics live inside
        ``h``.

    Examples
    --------
    >>> import numpy as np
    >>> F = np.array([[1.0, 1.0], [0.0, 1.0]])
    >>> hs = [(lambda k: (lambda x: (np.linalg.matrix_power(F, k) @ x)[:1]))(k)
    ...       for k in range(4)]
    >>> true0 = np.array([2.0, 0.5])
    >>> z = np.column_stack([hs[k](true0) for k in range(4)])
    >>> res = batch_ls_nonlin_meas_nonlin_dyn_lm(
    ...     np.array([1.5, 0.2]), z, hs, 0.01 * np.eye(1))
    >>> bool(res.success), tuple(np.round(res.x, 5))
    (True, (2.0, 0.5))

    Notes
    -----
    Port of ``batchLSNonlinMeasNonlinDynLM.m``, with one upstream
    defect fixed and documented: the original's covariance inverts the
    stacked Cholesky factors of R instead of R itself, disagreeing
    with its own non-LM sibling; here the covariance uses R inverse,
    matching :func:`batch_ls_nonlin_meas_nonlin_dyn`. SciPy's
    ``least_squares(method="lm")`` substitutes ``LSEstLMarquardt``.
    """
    from scipy.optimize import least_squares

    x_init = np.asarray(x_init, dtype=np.float64).ravel()
    z = np.asarray(z, dtype=np.float64)
    if z.ndim == 1:
        z = z[np.newaxis, :]
    R = np.asarray(R, dtype=np.float64)

    z_dim, num_steps = z.shape

    h_list = _per_step_callables(h, num_steps)
    R3 = _tile3(R, num_steps)
    sr = np.zeros_like(R3)
    for k in range(num_steps):
        sr[:, :, k] = np.linalg.cholesky(R3[:, :, k])

    order = [k_d] + [k for k in range(num_steps) if k != k_d]

    def _resid(x0):
        out = np.zeros(z_dim * num_steps)
        for i, k in enumerate(order):
            out[i * z_dim : (i + 1) * z_dim] = np.linalg.solve(
                sr[:, :, k], np.atleast_1d(h_list[k](x0)) - z[:, k]
            )
        return out

    res = least_squares(_resid, x_init, method="lm", max_nfev=max_iter)
    x_est = res.x

    hj_list = _default_jacobians(h_jacob, h_list, z_dim, num_steps)
    j = np.zeros((z_dim * num_steps, len(x_est)))
    for k in range(num_steps):
        j[k * z_dim : (k + 1) * z_dim, :] = np.atleast_2d(hj_list[k](x_est))
    r_stacked_inv = np.linalg.inv(_block_diag([R3[:, :, k] for k in range(num_steps)]))
    p_est = np.linalg.pinv(j.T @ r_stacked_inv @ j)
    x_batch = np.tile(x_est[:, np.newaxis], (1, num_steps))
    return BatchLSLMResult(x_est, p_est, x_batch, bool(res.success))


def two_point_diff_init(
    T: float,
    z: ArrayLike,
    R: ArrayLike,
    q: float = 0.0,
) -> BatchLSResult:
    """
    Two-point differencing track initialization.

    Builds position/velocity states and covariances from consecutive
    position-measurement pairs.

    Parameters
    ----------
    T : float
        Time between the two measurements.
    z : array_like
        (z_dim, 2) one measurement pair, or (z_dim, 2, N) a batch of
        pairs (each column pair ordered [earlier, later]).
    R : array_like
        (z_dim, z_dim) shared covariance, or per-measurement /
        per-pair stacked covariances (z_dim, z_dim, 2[, N]).
    q : float, optional
        Process noise power spectral density, adding the standard
        (1/3) q T bias term to the velocity covariance. Default 0.

    Returns
    -------
    result : BatchLSResult
        ``x`` is (2*z_dim,) or (2*z_dim, N); ``P`` is stacked
        accordingly.

    Examples
    --------
    >>> import numpy as np
    >>> z = np.array([[0.0, 2.0], [1.0, 0.0]])  # two 2D positions
    >>> res = two_point_diff_init(2.0, z, np.eye(2))
    >>> np.round(res.x, 6)
    array([ 2. ,  0. ,  1. , -0.5])

    Notes
    -----
    Port of ``twoPointDiffInit.m`` (Equations 39, 40 and 56 of the
    initialization survey it cites).
    """
    z = np.asarray(z, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    single = z.ndim == 2
    if single:
        z = z[:, :, np.newaxis]
    z_dim = z.shape[0]
    n = z.shape[2]

    if R.ndim == 2:
        R = R[:, :, np.newaxis]
    if R.shape[2] == 1:
        R = np.repeat(R, 2, axis=2)
    if R.ndim == 3:
        R = R[:, :, :, np.newaxis]
    if R.shape[3] == 1:
        R = np.repeat(R, n, axis=3)

    x_dim = 2 * z_dim
    x = np.zeros((x_dim, n))
    P = np.zeros((x_dim, x_dim, n))
    pos = slice(0, z_dim)
    vel = slice(z_dim, 2 * z_dim)
    for idx in range(n):
        x[pos, idx] = z[:, 1, idx]
        x[vel, idx] = (z[:, 1, idx] - z[:, 0, idx]) / T
        P[pos, pos, idx] = R[:, :, 1, idx]
        P[pos, vel, idx] = R[:, :, 1, idx] / T
        P[vel, pos, idx] = R[:, :, 1, idx] / T
        P[vel, vel, idx] = (R[:, :, 0, idx] + R[:, :, 1, idx]) / T**2 + (
            1.0 / 3.0
        ) * q * T * np.eye(z_dim)

    if single:
        return BatchLSResult(x[:, 0], P[:, :, 0])
    return BatchLSResult(x, P)


__all__ = [
    "BatchLSLMResult",
    "BatchLSResult",
    "batch_ls_lin_meas_lin_dyn",
    "batch_ls_nonlin_meas_lin_dyn",
    "batch_ls_nonlin_meas_lin_dyn_lm",
    "batch_ls_nonlin_meas_nonlin_dyn",
    "batch_ls_nonlin_meas_nonlin_dyn_lm",
    "two_point_diff_init",
]
