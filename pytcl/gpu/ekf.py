"""
GPU-accelerated Extended Kalman Filter.

This module provides GPU-accelerated implementations of the Extended Kalman
Filter (EKF) for batch processing of multiple tracks with nonlinear dynamics.

The EKF handles nonlinear systems by linearizing around the current estimate:
    x_k = f(x_{k-1}) + w       (nonlinear dynamics)
    z_k = h(x_k) + v           (nonlinear measurement)

Key Features
------------
- Batch processing of multiple tracks with same or different dynamics
- Support for user-provided Jacobian functions
- Numerical Jacobian computation when analytic unavailable
- Runs on either GPU backend (CuPy on CUDA, MLX on Apple Silicon) through
  :func:`pytcl.gpu._backend.get_compute_backend`

Backends and Precision
----------------------
The linear-algebra work is written against the backend-neutral operation
surface in :mod:`pytcl.gpu._backend`. CuPy computes in float64; MLX computes
in float32 (float64 is unsupported on the MLX GPU stream), so results on
Apple Silicon are precision-limited to roughly 1e-5 relative error against
the CPU reference in :mod:`pytcl.dynamic_estimation.kalman.extended`.

The user-supplied ``f``, ``h``, and Jacobian callables receive the whole batch
as a single device array of the active backend and are called once, not once
per track. This is the contract shared by every filter in :mod:`pytcl.gpu`:

- ``f(x)`` and ``h(x)`` take ``(N, state_dim)`` and return ``(N, out_dim)``;
- ``F_jacobian(x)`` and ``H_jacobian(x)`` take ``(N, state_dim)`` and return
  ``(N, out_dim, state_dim)``.

Write them against :func:`pytcl.gpu.utils.get_array_module` rather than NumPy
directly, so the same callable runs on either backend. A callable that mixes a
host NumPy array into the expression raises ``TypeError`` on CuPy.

Examples
--------
>>> from pytcl.gpu.ekf import batch_ekf_predict, batch_ekf_update
>>> import numpy as np
>>>
>>> from pytcl.gpu.utils import get_array_module
>>>
>>> # Batched: x is (N, 2), and the result is (N, 2)
>>> def f_dynamics(x):
...     xp = get_array_module(x)
...     return xp.stack([x[:, 0] + x[:, 1], x[:, 1] * 0.99], axis=1)
>>>
>>> # Batched Jacobian: (N, 2, 2), constant here so broadcast it
>>> def F_jacobian(x):
...     xp = get_array_module(x)
...     return xp.broadcast_to(
...         xp.array([[1.0, 1.0], [0.0, 0.99]]), (x.shape[0], 2, 2)
...     )
>>>
>>> # Batch prediction over three tracks with a 2-D state
>>> x = np.zeros((3, 2))
>>> P = np.stack([np.eye(2)] * 3)
>>> Q = np.stack([np.eye(2) * 0.01] * 3)
>>> x_pred, P_pred = batch_ekf_predict(x, P, f_dynamics, F_jacobian, Q)
>>> x_pred.shape
(3, 2)

See Also
--------
pytcl.gpu._backend : Backend dispatch layer (CuPy / MLX).
pytcl.dynamic_estimation.kalman.extended : CPU reference implementation.
"""

from typing import Any, Callable, NamedTuple, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pytcl.gpu._backend import get_compute_backend


class BatchEKFPrediction(NamedTuple):
    """Result of batch EKF prediction.

    Attributes
    ----------
    x : ndarray
        Predicted state estimates, shape (n_tracks, state_dim).
    P : ndarray
        Predicted covariances, shape (n_tracks, state_dim, state_dim).
    """

    x: NDArray[np.floating]
    P: NDArray[np.floating]


class BatchEKFUpdate(NamedTuple):
    """Result of batch EKF update.

    Attributes
    ----------
    x : ndarray
        Updated state estimates.
    P : ndarray
        Updated covariances.
    y : ndarray
        Innovations.
    S : ndarray
        Innovation covariances.
    K : ndarray
        Kalman gains.
    likelihood : ndarray
        Measurement likelihoods.
    """

    x: NDArray[np.floating]
    P: NDArray[np.floating]
    y: NDArray[np.floating]
    S: NDArray[np.floating]
    K: NDArray[np.floating]
    likelihood: NDArray[np.floating]


def _compute_numerical_jacobian(
    f: Callable[[Any], Any],
    x: Any,
    eps: Optional[float] = None,
) -> Any:
    """
    Central-difference Jacobian of a batched callback.

    Parameters
    ----------
    f : callable
        Maps ``(N, n)`` to ``(N, m)`` on the active backend.
    x : array
        Evaluation points, shape ``(N, n)``, on the active backend.
    eps : float, optional
        Finite-difference step. Defaults to a value matched to the backend's
        precision: a float32 backend cannot resolve the 1e-7 step that is
        right for float64, and using it there returns noise rather than a
        derivative.

    Returns
    -------
    J : array
        Jacobians, shape ``(N, m, n)``, on the active backend.

    Notes
    -----
    One pair of evaluations per input dimension for the whole batch, rather
    than per item: ``2 * n`` calls instead of ``2 * N * n``.
    """
    b = get_compute_backend()
    if eps is None:
        eps = 1e-7 if b.supports_float64 else 1e-3

    x = b.asarray(x)
    n = x.shape[1]
    basis = b.eye(n)

    columns = []
    for i in range(n):
        step = basis[i] * eps
        f_plus = b.asarray(f(x + step))
        f_minus = b.asarray(f(x - step))
        columns.append((f_plus - f_minus) / (2 * eps))

    return b.stack(columns, axis=-1)


def batch_ekf_predict(
    x: ArrayLike,
    P: ArrayLike,
    f: Callable[[NDArray[np.floating[Any]]], NDArray[np.floating[Any]]],
    F_jacobian: Optional[
        Callable[[NDArray[np.floating[Any]]], NDArray[np.floating[Any]]]
    ],
    Q: ArrayLike,
) -> BatchEKFPrediction:
    """
    Batch EKF prediction for multiple tracks.

    Parameters
    ----------
    x : array_like
        Current state estimates, shape (n_tracks, state_dim).
    P : array_like
        Current covariances, shape (n_tracks, state_dim, state_dim).
    f : callable
        Batched dynamics. Takes the whole ``(n_tracks, state_dim)`` device
        array and returns ``(n_tracks, state_dim)``. Called once.
    F_jacobian : callable or None
        Batched Jacobian df/dx. Takes ``(n_tracks, state_dim)`` and returns
        ``(n_tracks, state_dim, state_dim)``. If None, computed numerically
        with ``2 * state_dim`` evaluations of ``f`` over the whole batch.
    Q : array_like
        Process noise covariance, shape (state_dim, state_dim)
        or (n_tracks, state_dim, state_dim).

    Returns
    -------
    result : BatchEKFPrediction
        Predicted states and covariances, as device arrays of the active
        backend. Use :func:`pytcl.gpu.utils.to_cpu` to bring them back.

    Raises
    ------
    DependencyError
        If neither CuPy nor MLX is installed.

    Examples
    --------
    >>> import numpy as np
    >>> from pytcl.gpu.ekf import batch_ekf_predict
    >>> from pytcl.gpu.utils import get_array_module
    >>> # Coordinated turn, evaluated for the whole batch at once
    >>> def f_turn(x):
    ...     xp = get_array_module(x)
    ...     w = 0.01
    ...     return xp.stack([x[:, 0] + xp.cos(w) * x[:, 2],
    ...                      x[:, 1] + xp.sin(w) * x[:, 3],
    ...                      x[:, 2], x[:, 3]], axis=1)
    >>> def F_jacobian(x):
    ...     xp = get_array_module(x)
    ...     w = 0.01
    ...     J = xp.array([[1.0, 0.0, xp.cos(w).item(), 0.0],
    ...                   [0.0, 1.0, xp.sin(w).item(), 0.0],
    ...                   [0.0, 0.0, 1.0, 0.0],
    ...                   [0.0, 0.0, 0.0, 1.0]])
    ...     return xp.broadcast_to(J, (x.shape[0], 4, 4))
    >>> n_tracks = 30
    >>> x = np.random.randn(n_tracks, 4) * 0.1
    >>> P = np.tile(np.eye(4) * 0.01, (n_tracks, 1, 1))
    >>> Q = np.eye(4) * 0.001
    >>> result = batch_ekf_predict(x, P, f_turn, F_jacobian, Q)
    >>> result.x.shape
    (30, 4)

    Notes
    -----
    Both the dynamics and the covariance propagation stay on the device. The
    callback is invoked once for the batch rather than once per track.
    """
    b = get_compute_backend()

    x_gpu = b.asarray(x)
    P_gpu = b.asarray(P)
    Q_gpu = b.asarray(Q)

    n_tracks = x_gpu.shape[0]
    state_dim = x_gpu.shape[1]

    # One call for the whole batch. This used to convert to numpy and loop,
    # invoking the callback once per track and once more per dimension for the
    # numerical Jacobian.
    x_pred_gpu = b.asarray(f(x_gpu))
    if F_jacobian is not None:
        F_gpu = b.asarray(F_jacobian(x_gpu))
    else:
        F_gpu = _compute_numerical_jacobian(f, x_gpu)

    # Handle Q dimensions
    if Q_gpu.ndim == 2:
        Q_batch = b.broadcast_to(Q_gpu, (n_tracks, state_dim, state_dim))
    else:
        Q_batch = Q_gpu

    # Covariance prediction on GPU: P_pred = F @ P @ F' + Q
    FP = b.einsum("nij,njk->nik", F_gpu, P_gpu)
    P_pred = b.einsum("nij,nkj->nik", FP, F_gpu) + Q_batch

    # Ensure symmetry
    P_pred = (P_pred + b.swapaxes(P_pred, -2, -1)) / 2

    return BatchEKFPrediction(x=x_pred_gpu, P=P_pred)


def batch_ekf_update(
    x: ArrayLike,
    P: ArrayLike,
    z: ArrayLike,
    h: Callable[[NDArray[np.floating[Any]]], NDArray[np.floating[Any]]],
    H_jacobian: Optional[
        Callable[[NDArray[np.floating[Any]]], NDArray[np.floating[Any]]]
    ],
    R: ArrayLike,
) -> BatchEKFUpdate:
    """
    Batch EKF update for multiple tracks.

    Parameters
    ----------
    x : array_like
        Predicted state estimates, shape (n_tracks, state_dim).
    P : array_like
        Predicted covariances, shape (n_tracks, state_dim, state_dim).
    z : array_like
        Measurements, shape (n_tracks, meas_dim).
    h : callable
        Batched measurement function. Takes ``(n_tracks, state_dim)`` and
        returns ``(n_tracks, meas_dim)``. Called once.
    H_jacobian : callable or None
        Batched Jacobian dh/dx. Takes ``(n_tracks, state_dim)`` and returns
        ``(n_tracks, meas_dim, state_dim)``. If None, computed numerically.
    R : array_like
        Measurement noise covariance.

    Returns
    -------
    result : BatchEKFUpdate
        Update results including states, covariances, and statistics, as
        device arrays of the active backend.

    Raises
    ------
    DependencyError
        If neither CuPy nor MLX is installed.

    Examples
    --------
    >>> import numpy as np
    >>> from pytcl.gpu.ekf import batch_ekf_update
    >>> from pytcl.gpu.utils import get_array_module
    >>> # Polar measurement from a Cartesian state, batched
    >>> def h_polar(x):
    ...     xp = get_array_module(x)
    ...     r = xp.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2)
    ...     theta = xp.arctan2(x[:, 1], x[:, 0])
    ...     return xp.stack([r, theta], axis=1)
    >>> def H_jacobian(x):
    ...     xp = get_array_module(x)
    ...     r = xp.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2)
    ...     row0 = xp.stack([x[:, 0] / r, x[:, 1] / r], axis=1)
    ...     row1 = xp.stack([-x[:, 1] / r**2, x[:, 0] / r**2], axis=1)
    ...     return xp.stack([row0, row1], axis=1)
    >>> n_tracks = 20
    >>> x = np.random.randn(n_tracks, 2)
    >>> P = np.tile(np.eye(2), (n_tracks, 1, 1))
    >>> z = np.random.randn(n_tracks, 2) * [100, 0.1]  # r, theta
    >>> R = np.diag([10.0, 0.01])
    >>> result = batch_ekf_update(x, P, z, h_polar, H_jacobian, R)
    >>> result.x.shape
    (20, 2)
    """
    b = get_compute_backend()

    x_gpu = b.asarray(x)
    P_gpu = b.asarray(P)
    z_gpu = b.asarray(z)
    R_gpu = b.asarray(R)

    n_tracks = x_gpu.shape[0]
    state_dim = x_gpu.shape[1]
    meas_dim = z_gpu.shape[1]

    z_pred_gpu = b.asarray(h(x_gpu))
    if H_jacobian is not None:
        H_gpu = b.asarray(H_jacobian(x_gpu))
    else:
        H_gpu = _compute_numerical_jacobian(h, x_gpu)

    # Handle R dimensions
    if R_gpu.ndim == 2:
        R_batch = b.broadcast_to(R_gpu, (n_tracks, meas_dim, meas_dim))
    else:
        R_batch = R_gpu

    # Innovation
    y = z_gpu - z_pred_gpu

    # Innovation covariance: S = H @ P @ H' + R
    HP = b.einsum("nij,njk->nik", H_gpu, P_gpu)
    S = b.einsum("nij,nkj->nik", HP, H_gpu) + R_batch

    # Kalman gain: K = P @ H' @ S^{-1}
    PHT = b.einsum("nij,nkj->nik", P_gpu, H_gpu)
    S_inv = b.inv(S)
    K = b.einsum("nij,njk->nik", PHT, S_inv)

    # Updated state
    x_upd = x_gpu + b.einsum("nij,nj->ni", K, y)

    # Updated covariance (Joseph form)
    eye = b.eye(state_dim)
    I_KH = eye - b.einsum("nij,njk->nik", K, H_gpu)
    P_upd = b.einsum("nij,njk->nik", I_KH, P_gpu)
    P_upd = b.einsum("nij,nkj->nik", P_upd, I_KH)
    KRK = b.einsum("nij,njk,nlk->nil", K, R_batch, K)
    P_upd = P_upd + KRK

    # Ensure symmetry
    P_upd = (P_upd + b.swapaxes(P_upd, -2, -1)) / 2

    # Likelihoods
    mahal_sq = b.einsum("ni,nij,nj->n", y, S_inv, y)
    _sign, logdet = b.slogdet(S)
    log_likelihood = -0.5 * (mahal_sq + logdet + meas_dim * np.log(2 * np.pi))
    likelihood = b.exp(log_likelihood)

    return BatchEKFUpdate(
        x=x_upd,
        P=P_upd,
        y=y,
        S=S,
        K=K,
        likelihood=likelihood,
    )


class CuPyExtendedKalmanFilter:
    """
    GPU-accelerated Extended Kalman Filter for batch processing.

    Despite the historical name, this class runs on whichever GPU backend is
    available: CuPy on CUDA devices, MLX on Apple Silicon.

    Parameters
    ----------
    state_dim : int
        Dimension of state vector.
    meas_dim : int
        Dimension of measurement vector.
    f : callable
        Nonlinear dynamics function f(x) -> x_next.
    h : callable
        Nonlinear measurement function h(x) -> z.
    F_jacobian : callable, optional
        Jacobian of dynamics. If None, computed numerically.
    H_jacobian : callable, optional
        Jacobian of measurement. If None, computed numerically.
    Q : array_like, optional
        Process noise covariance.
    R : array_like, optional
        Measurement noise covariance.

    Raises
    ------
    DependencyError
        If neither CuPy nor MLX is installed.

    Examples
    --------
    >>> import numpy as np
    >>> from pytcl.gpu.ekf import CuPyExtendedKalmanFilter
    >>>
    >>> # Nonlinear dynamics
    >>> def f(x):
    ...     return np.array([x[0] + x[1], x[1] * 0.99])
    >>>
    >>> def h(x):
    ...     return np.array([np.sqrt(x[0]**2 + x[1]**2)])
    >>>
    >>> ekf = CuPyExtendedKalmanFilter(
    ...     state_dim=2, meas_dim=1,
    ...     f=f, h=h,
    ...     Q=np.eye(2) * 0.01,
    ...     R=np.array([[0.1]]),
    ... )
    """

    def __init__(
        self,
        state_dim: int,
        meas_dim: int,
        f: Callable[[NDArray[np.floating[Any]]], NDArray[np.floating[Any]]],
        h: Callable[[NDArray[np.floating[Any]]], NDArray[np.floating[Any]]],
        F_jacobian: Optional[
            Callable[[NDArray[np.floating[Any]]], NDArray[np.floating[Any]]]
        ] = None,
        H_jacobian: Optional[
            Callable[[NDArray[np.floating[Any]]], NDArray[np.floating[Any]]]
        ] = None,
        Q: Optional[ArrayLike] = None,
        R: Optional[ArrayLike] = None,
    ):
        b = get_compute_backend()

        self.state_dim = state_dim
        self.meas_dim = meas_dim
        self.f = f
        self.h = h
        self.F_jacobian = F_jacobian
        self.H_jacobian = H_jacobian

        if Q is None:
            self.Q = b.eye(state_dim) * 0.01
        else:
            self.Q = b.asarray(Q)

        if R is None:
            self.R = b.eye(meas_dim)
        else:
            self.R = b.asarray(R)

    def predict(
        self,
        x: ArrayLike,
        P: ArrayLike,
    ) -> BatchEKFPrediction:
        """Perform batch EKF prediction."""
        return batch_ekf_predict(x, P, self.f, self.F_jacobian, self.Q)

    def update(
        self,
        x: ArrayLike,
        P: ArrayLike,
        z: ArrayLike,
    ) -> BatchEKFUpdate:
        """Perform batch EKF update."""
        return batch_ekf_update(x, P, z, self.h, self.H_jacobian, self.R)

    def predict_update(
        self,
        x: ArrayLike,
        P: ArrayLike,
        z: ArrayLike,
    ) -> BatchEKFUpdate:
        """Combined prediction and update."""
        pred = self.predict(x, P)
        return self.update(pred.x, pred.P, z)


__all__ = [
    "BatchEKFPrediction",
    "BatchEKFUpdate",
    "batch_ekf_predict",
    "batch_ekf_update",
    "CuPyExtendedKalmanFilter",
]
