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

The user-supplied ``f``, ``h``, and Jacobian callables are always evaluated on
the CPU in NumPy float64: they are arbitrary Python functions and cannot be
assumed to accept device arrays.

Examples
--------
>>> from pytcl.gpu.ekf import batch_ekf_predict, batch_ekf_update
>>> import numpy as np
>>>
>>> # Define nonlinear dynamics (on CPU, applied per-track)
>>> def f_dynamics(x):
...     return np.array([x[0] + x[1], x[1] * 0.99])
>>>
>>> def F_jacobian(x):
...     return np.array([[1, 1], [0, 0.99]])
>>>
>>> # Batch prediction
>>> x_pred, P_pred = batch_ekf_predict(x, P, f_dynamics, F_jacobian, Q)

See Also
--------
pytcl.gpu._backend : Backend dispatch layer (CuPy / MLX).
pytcl.dynamic_estimation.kalman.extended : CPU reference implementation.
"""

from typing import Any, Callable, NamedTuple, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pytcl.gpu._backend import get_compute_backend
from pytcl.gpu.utils import to_cpu


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
    f: Callable[[NDArray[np.floating[Any]]], NDArray[np.floating[Any]]],
    x: NDArray[np.floating[Any]],
    eps: float = 1e-7,
) -> NDArray[np.floating[Any]]:
    """
    Compute numerical Jacobian using central differences.

    Parameters
    ----------
    f : callable
        Function to differentiate.
    x : ndarray
        Point at which to evaluate Jacobian.
    eps : float
        Finite difference step size.

    Returns
    -------
    J : ndarray
        Jacobian matrix, shape (output_dim, input_dim).
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    n = len(x)
    f0 = np.asarray(f(x), dtype=np.float64).flatten()
    m = len(f0)

    J = np.zeros((m, n))
    for i in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        f_plus = np.asarray(f(x_plus), dtype=np.float64).flatten()
        f_minus = np.asarray(f(x_minus), dtype=np.float64).flatten()
        J[:, i] = (f_plus - f_minus) / (2 * eps)

    return J


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
        Nonlinear dynamics function f(x) -> x_next.
        Applied to each track's state vector.
    F_jacobian : callable or None
        Jacobian of dynamics df/dx. If None, computed numerically.
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
    >>> # Nonlinear dynamics: coordinated turn
    >>> def f_turn(x):
    ...     w = 0.01  # Turn rate
    ...     return np.array([x[0] + np.cos(w)*x[2],
    ...                      x[1] + np.sin(w)*x[3],
    ...                      x[2], x[3]])
    >>> def F_jacobian(x):
    ...     w = 0.01
    ...     return np.array([[1, 0, np.cos(w), 0],
    ...                      [0, 1, np.sin(w), 0],
    ...                      [0, 0, 1, 0],
    ...                      [0, 0, 0, 1]])
    >>> n_tracks = 30
    >>> x = np.random.randn(n_tracks, 4) * 0.1
    >>> P = np.tile(np.eye(4) * 0.01, (n_tracks, 1, 1))
    >>> Q = np.eye(4) * 0.001
    >>> result = batch_ekf_predict(x, P, f_turn, F_jacobian, Q)
    >>> result.x.shape
    (30, 4)

    Notes
    -----
    The nonlinear dynamics are applied on CPU (Python function), then
    covariance propagation is performed on GPU. This is efficient when
    the number of tracks is large relative to the cost of the dynamics.
    """
    b = get_compute_backend()

    # Convert to numpy for dynamics evaluation. ``to_cpu`` is required because
    # device arrays refuse implicit conversion via ``np.asarray``.
    x_np = np.asarray(to_cpu(x), dtype=np.float64)
    P_gpu = b.asarray(P)
    Q_gpu = b.asarray(Q)

    n_tracks = x_np.shape[0]
    state_dim = x_np.shape[1]

    # Apply nonlinear dynamics to each track (on CPU)
    x_pred_np = np.zeros((n_tracks, state_dim), dtype=np.float64)
    F_matrices = np.zeros((n_tracks, state_dim, state_dim))

    for i in range(n_tracks):
        x_i = x_np[i]
        x_pred_np[i] = f(x_i)

        # Compute Jacobian
        if F_jacobian is not None:
            F_matrices[i] = F_jacobian(x_i)
        else:
            F_matrices[i] = _compute_numerical_jacobian(f, x_i)

    # Move to GPU
    x_pred_gpu = b.asarray(x_pred_np)
    F_gpu = b.asarray(F_matrices)

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
        Nonlinear measurement function h(x) -> z_predicted.
    H_jacobian : callable or None
        Jacobian of measurement function dh/dx. If None, computed numerically.
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
    >>> # Polar measurement from Cartesian state
    >>> def h_polar(x):
    ...     r = np.sqrt(x[0]**2 + x[1]**2)
    ...     theta = np.arctan2(x[1], x[0])
    ...     return np.array([r, theta])
    >>> def H_jacobian(x):
    ...     r = np.sqrt(x[0]**2 + x[1]**2)
    ...     return np.array([[x[0]/r, x[1]/r],
    ...                      [-x[1]/r**2, x[0]/r**2]])
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

    # Convert to numpy for measurement evaluation. ``to_cpu`` is required
    # because device arrays refuse implicit conversion via ``np.asarray``.
    x_np = np.asarray(to_cpu(x), dtype=np.float64)
    z_np = np.asarray(to_cpu(z), dtype=np.float64)
    P_gpu = b.asarray(P)
    z_gpu = b.asarray(z_np)
    R_gpu = b.asarray(R)

    n_tracks = x_np.shape[0]
    state_dim = x_np.shape[1]
    meas_dim = z_np.shape[1]

    # Evaluate measurement function and Jacobian for each track
    z_pred_np = np.zeros((n_tracks, meas_dim))
    H_matrices = np.zeros((n_tracks, meas_dim, state_dim))

    for i in range(n_tracks):
        x_i = x_np[i]
        z_pred_np[i] = h(x_i)

        if H_jacobian is not None:
            H_matrices[i] = H_jacobian(x_i)
        else:
            H_matrices[i] = _compute_numerical_jacobian(h, x_i)

    # Move to GPU
    x_gpu = b.asarray(x_np)
    z_pred_gpu = b.asarray(z_pred_np)
    H_gpu = b.asarray(H_matrices)

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
