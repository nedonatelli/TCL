"""
GPU-accelerated Extended Kalman Filter using CuPy.

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
- Memory-efficient operations using CuPy

Examples
--------
>>> from pytcl.gpu.ekf import batch_ekf_predict, batch_ekf_update
>>> import numpy as np
>>>
>>> # Define nonlinear dynamics (on CPU, applied per-particle)
>>> def f_dynamics(x):
...     return np.array([x[0] + x[1], x[1] * 0.99])
>>>
>>> def F_jacobian(x):
...     return np.array([[1, 1], [0, 0.99]])
>>>
>>> # Batch prediction
>>> x_pred, P_pred = batch_ekf_predict(x, P, f_dynamics, F_jacobian, Q)
"""

from typing import Callable, NamedTuple, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pytcl.core.optional_deps import import_optional, requires
from pytcl.gpu.utils import ensure_gpu_array, to_cpu


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
    f: Callable[[NDArray], NDArray],
    x: NDArray,
    eps: float = 1e-7,
) -> NDArray:
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
    x = np.asarray(x).flatten()
    n = len(x)
    f0 = np.asarray(f(x)).flatten()
    m = len(f0)

    J = np.zeros((m, n))
    for i in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        f_plus = np.asarray(f(x_plus)).flatten()
        f_minus = np.asarray(f(x_minus)).flatten()
        J[:, i] = (f_plus - f_minus) / (2 * eps)

    return J


@requires("cupy", extra="gpu", feature="GPU Extended Kalman filter")
def batch_ekf_predict(
    x: ArrayLike,
    P: ArrayLike,
    f: Callable[[NDArray], NDArray],
    F_jacobian: Optional[Callable[[NDArray], NDArray]],
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
        Predicted states and covariances.

    Notes
    -----
    The nonlinear dynamics are applied on CPU (Python function), then
    covariance propagation is performed on GPU. This is efficient when
    the number of tracks is large relative to the cost of the dynamics.
    """
    cp = import_optional("cupy", extra="gpu", feature="GPU Extended Kalman filter")

    # Convert to numpy for dynamics evaluation
    x_np = np.asarray(x)
    P_gpu = ensure_gpu_array(P, dtype=cp.float64)
    Q_gpu = ensure_gpu_array(Q, dtype=cp.float64)

    n_tracks = x_np.shape[0]
    state_dim = x_np.shape[1]

    # Apply nonlinear dynamics to each track (on CPU)
    x_pred_np = np.zeros_like(x_np)
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
    x_pred_gpu = ensure_gpu_array(x_pred_np, dtype=cp.float64)
    F_gpu = ensure_gpu_array(F_matrices, dtype=cp.float64)

    # Handle Q dimensions
    if Q_gpu.ndim == 2:
        Q_batch = cp.broadcast_to(Q_gpu, (n_tracks, state_dim, state_dim))
    else:
        Q_batch = Q_gpu

    # Covariance prediction on GPU: P_pred = F @ P @ F' + Q
    FP = cp.einsum("nij,njk->nik", F_gpu, P_gpu)
    P_pred = cp.einsum("nij,nkj->nik", FP, F_gpu) + Q_batch

    # Ensure symmetry
    P_pred = (P_pred + cp.swapaxes(P_pred, -2, -1)) / 2

    return BatchEKFPrediction(x=x_pred_gpu, P=P_pred)


@requires("cupy", extra="gpu", feature="GPU Extended Kalman filter")
def batch_ekf_update(
    x: ArrayLike,
    P: ArrayLike,
    z: ArrayLike,
    h: Callable[[NDArray], NDArray],
    H_jacobian: Optional[Callable[[NDArray], NDArray]],
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
        Update results including states, covariances, and statistics.
    """
    cp = import_optional("cupy", extra="gpu", feature="GPU Extended Kalman filter")

    # Convert to numpy for measurement evaluation
    x_np = np.asarray(to_cpu(x))
    z_np = np.asarray(z)
    P_gpu = ensure_gpu_array(P, dtype=cp.float64)
    z_gpu = ensure_gpu_array(z, dtype=cp.float64)
    R_gpu = ensure_gpu_array(R, dtype=cp.float64)

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
    x_gpu = ensure_gpu_array(x_np, dtype=cp.float64)
    z_pred_gpu = ensure_gpu_array(z_pred_np, dtype=cp.float64)
    H_gpu = ensure_gpu_array(H_matrices, dtype=cp.float64)

    # Handle R dimensions
    if R_gpu.ndim == 2:
        R_batch = cp.broadcast_to(R_gpu, (n_tracks, meas_dim, meas_dim))
    else:
        R_batch = R_gpu

    # Innovation
    y = z_gpu - z_pred_gpu

    # Innovation covariance: S = H @ P @ H' + R
    HP = cp.einsum("nij,njk->nik", H_gpu, P_gpu)
    S = cp.einsum("nij,nkj->nik", HP, H_gpu) + R_batch

    # Kalman gain: K = P @ H' @ S^{-1}
    PHT = cp.einsum("nij,nkj->nik", P_gpu, H_gpu)
    S_inv = cp.linalg.inv(S)
    K = cp.einsum("nij,njk->nik", PHT, S_inv)

    # Updated state
    x_upd = x_gpu + cp.einsum("nij,nj->ni", K, y)

    # Updated covariance (Joseph form)
    eye = cp.eye(state_dim, dtype=cp.float64)
    I_KH = eye - cp.einsum("nij,njk->nik", K, H_gpu)
    P_upd = cp.einsum("nij,njk->nik", I_KH, P_gpu)
    P_upd = cp.einsum("nij,nkj->nik", P_upd, I_KH)
    KRK = cp.einsum("nij,njk,nlk->nil", K, R_batch, K)
    P_upd = P_upd + KRK

    # Ensure symmetry
    P_upd = (P_upd + cp.swapaxes(P_upd, -2, -1)) / 2

    # Likelihoods
    mahal_sq = cp.einsum("ni,nij,nj->n", y, S_inv, y)
    sign, logdet = cp.linalg.slogdet(S)
    log_likelihood = -0.5 * (mahal_sq + logdet + meas_dim * np.log(2 * np.pi))
    likelihood = cp.exp(log_likelihood)

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

    @requires("cupy", extra="gpu", feature="GPU Extended Kalman filter")
    def __init__(
        self,
        state_dim: int,
        meas_dim: int,
        f: Callable[[NDArray], NDArray],
        h: Callable[[NDArray], NDArray],
        F_jacobian: Optional[Callable[[NDArray], NDArray]] = None,
        H_jacobian: Optional[Callable[[NDArray], NDArray]] = None,
        Q: Optional[ArrayLike] = None,
        R: Optional[ArrayLike] = None,
    ):
        cp = import_optional("cupy", extra="gpu", feature="GPU Extended Kalman filter")

        self.state_dim = state_dim
        self.meas_dim = meas_dim
        self.f = f
        self.h = h
        self.F_jacobian = F_jacobian
        self.H_jacobian = H_jacobian

        if Q is None:
            self.Q = cp.eye(state_dim, dtype=cp.float64) * 0.01
        else:
            self.Q = ensure_gpu_array(Q, dtype=cp.float64)

        if R is None:
            self.R = cp.eye(meas_dim, dtype=cp.float64)
        else:
            self.R = ensure_gpu_array(R, dtype=cp.float64)

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
