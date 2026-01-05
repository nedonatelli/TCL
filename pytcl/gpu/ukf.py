"""
GPU-accelerated Unscented Kalman Filter using CuPy.

This module provides GPU-accelerated implementations of the Unscented Kalman
Filter (UKF) for batch processing of multiple tracks with nonlinear dynamics.

The UKF uses sigma points to propagate uncertainty through nonlinear functions
without requiring Jacobian computation.

Key Features
------------
- Batch processing of multiple tracks
- Configurable sigma point parameters (alpha, beta, kappa)
- GPU-accelerated sigma point generation and transformation
- Support for nonlinear dynamics and measurements

Examples
--------
>>> from pytcl.gpu.ukf import batch_ukf_predict
>>> import numpy as np
>>>
>>> def f_dynamics(x):
...     return np.array([x[0] + x[1], x[1] * 0.99])
>>>
>>> x_pred, P_pred = batch_ukf_predict(x, P, f_dynamics, Q)
"""

from typing import Callable, NamedTuple, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pytcl.core.optional_deps import import_optional, requires
from pytcl.gpu.utils import ensure_gpu_array, to_cpu


class BatchUKFPrediction(NamedTuple):
    """Result of batch UKF prediction.

    Attributes
    ----------
    x : ndarray
        Predicted state estimates, shape (n_tracks, state_dim).
    P : ndarray
        Predicted covariances, shape (n_tracks, state_dim, state_dim).
    """

    x: NDArray[np.floating]
    P: NDArray[np.floating]


class BatchUKFUpdate(NamedTuple):
    """Result of batch UKF update.

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
    likelihood : ndarray
        Measurement likelihoods.
    """

    x: NDArray[np.floating]
    P: NDArray[np.floating]
    y: NDArray[np.floating]
    S: NDArray[np.floating]
    likelihood: NDArray[np.floating]


def _compute_sigma_weights(
    n: int,
    alpha: float = 1e-3,
    beta: float = 2.0,
    kappa: float = 0.0,
) -> Tuple[NDArray, NDArray]:
    """
    Compute UKF sigma point weights (Merwe scaled sigma points).

    Parameters
    ----------
    n : int
        State dimension.
    alpha : float
        Spread of sigma points (1e-4 to 1).
    beta : float
        Prior knowledge (2 is optimal for Gaussian).
    kappa : float
        Secondary scaling parameter (0 or 3-n).

    Returns
    -------
    Wm : ndarray
        Mean weights, shape (2n+1,).
    Wc : ndarray
        Covariance weights, shape (2n+1,).
    """
    lambda_ = alpha**2 * (n + kappa) - n

    # Weight for mean: first point
    Wm = np.full(2 * n + 1, 1 / (2 * (n + lambda_)))
    Wm[0] = lambda_ / (n + lambda_)

    # Weight for covariance
    Wc = Wm.copy()
    Wc[0] = Wm[0] + (1 - alpha**2 + beta)

    return Wm, Wc


@requires("cupy", extra="gpu", feature="GPU Unscented Kalman filter")
def _generate_sigma_points(
    x: ArrayLike,
    P: ArrayLike,
    alpha: float = 1e-3,
    kappa: float = 0.0,
) -> NDArray:
    """
    Generate sigma points for batch of tracks.

    Parameters
    ----------
    x : array_like
        State estimates, shape (n_tracks, state_dim).
    P : array_like
        Covariances, shape (n_tracks, state_dim, state_dim).
    alpha : float
        Spread parameter.
    kappa : float
        Secondary scaling.

    Returns
    -------
    sigma_points : ndarray
        Sigma points, shape (n_tracks, 2*state_dim+1, state_dim).
    """
    cp = import_optional("cupy", extra="gpu", feature="GPU Unscented Kalman filter")

    x_gpu = ensure_gpu_array(x, dtype=cp.float64)
    P_gpu = ensure_gpu_array(P, dtype=cp.float64)

    n_tracks = x_gpu.shape[0]
    n = x_gpu.shape[1]  # state dim
    n_sigma = 2 * n + 1

    lambda_ = alpha**2 * (n + kappa) - n
    gamma = cp.sqrt(n + lambda_)

    # Cholesky decomposition of P
    # CuPy's cholesky returns lower triangular
    try:
        L = cp.linalg.cholesky(P_gpu)
    except cp.linalg.LinAlgError:
        # Fallback: eigendecomposition for non-positive-definite
        eigvals, eigvecs = cp.linalg.eigh(P_gpu)
        eigvals = cp.maximum(eigvals, 1e-10)
        L = eigvecs @ cp.diag(cp.sqrt(eigvals)).T
        L = cp.swapaxes(L, -2, -1)  # Make it "lower triangular-like"

    # Scale by gamma
    scaled_L = gamma * L  # shape: (n_tracks, n, n)

    # Generate sigma points
    sigma_points = cp.zeros((n_tracks, n_sigma, n), dtype=cp.float64)

    # First point is the mean
    sigma_points[:, 0, :] = x_gpu

    # Remaining points: x ± scaled_L columns
    for i in range(n):
        sigma_points[:, i + 1, :] = x_gpu + scaled_L[:, :, i]
        sigma_points[:, n + i + 1, :] = x_gpu - scaled_L[:, :, i]

    return sigma_points


@requires("cupy", extra="gpu", feature="GPU Unscented Kalman filter")
def batch_ukf_predict(
    x: ArrayLike,
    P: ArrayLike,
    f: Callable[[NDArray], NDArray],
    Q: ArrayLike,
    alpha: float = 1e-3,
    beta: float = 2.0,
    kappa: float = 0.0,
) -> BatchUKFPrediction:
    """
    Batch UKF prediction for multiple tracks.

    Parameters
    ----------
    x : array_like
        Current state estimates, shape (n_tracks, state_dim).
    P : array_like
        Current covariances, shape (n_tracks, state_dim, state_dim).
    f : callable
        Nonlinear dynamics function f(x) -> x_next.
    Q : array_like
        Process noise covariance.
    alpha, beta, kappa : float
        Sigma point parameters.

    Returns
    -------
    result : BatchUKFPrediction
        Predicted states and covariances.
    """
    cp = import_optional("cupy", extra="gpu", feature="GPU Unscented Kalman filter")

    x_gpu = ensure_gpu_array(x, dtype=cp.float64)
    P_gpu = ensure_gpu_array(P, dtype=cp.float64)
    Q_gpu = ensure_gpu_array(Q, dtype=cp.float64)

    n_tracks = x_gpu.shape[0]
    n = x_gpu.shape[1]
    n_sigma = 2 * n + 1

    # Generate sigma points
    sigma_points = _generate_sigma_points(x_gpu, P_gpu, alpha, kappa)

    # Compute weights
    Wm, Wc = _compute_sigma_weights(n, alpha, beta, kappa)
    Wm_gpu = ensure_gpu_array(Wm, dtype=cp.float64)
    Wc_gpu = ensure_gpu_array(Wc, dtype=cp.float64)

    # Propagate sigma points through dynamics (on CPU)
    sigma_np = to_cpu(sigma_points)
    sigma_pred_np = np.zeros_like(sigma_np)

    for i in range(n_tracks):
        for j in range(n_sigma):
            sigma_pred_np[i, j] = f(sigma_np[i, j])

    sigma_pred = ensure_gpu_array(sigma_pred_np, dtype=cp.float64)

    # Predicted mean: sum of weighted sigma points
    x_pred = cp.einsum("j,nj...->n...", Wm_gpu, sigma_pred)

    # Predicted covariance
    diff = sigma_pred - x_pred[:, None, :]  # (n_tracks, n_sigma, n)
    P_pred = cp.einsum("j,nji,njk->nik", Wc_gpu, diff, diff)

    # Add process noise
    if Q_gpu.ndim == 2:
        P_pred = P_pred + Q_gpu
    else:
        P_pred = P_pred + Q_gpu

    # Ensure symmetry
    P_pred = (P_pred + cp.swapaxes(P_pred, -2, -1)) / 2

    return BatchUKFPrediction(x=x_pred, P=P_pred)


@requires("cupy", extra="gpu", feature="GPU Unscented Kalman filter")
def batch_ukf_update(
    x: ArrayLike,
    P: ArrayLike,
    z: ArrayLike,
    h: Callable[[NDArray], NDArray],
    R: ArrayLike,
    alpha: float = 1e-3,
    beta: float = 2.0,
    kappa: float = 0.0,
) -> BatchUKFUpdate:
    """
    Batch UKF update for multiple tracks.

    Parameters
    ----------
    x : array_like
        Predicted state estimates, shape (n_tracks, state_dim).
    P : array_like
        Predicted covariances, shape (n_tracks, state_dim, state_dim).
    z : array_like
        Measurements, shape (n_tracks, meas_dim).
    h : callable
        Nonlinear measurement function h(x) -> z.
    R : array_like
        Measurement noise covariance.
    alpha, beta, kappa : float
        Sigma point parameters.

    Returns
    -------
    result : BatchUKFUpdate
        Update results.
    """
    cp = import_optional("cupy", extra="gpu", feature="GPU Unscented Kalman filter")

    x_gpu = ensure_gpu_array(x, dtype=cp.float64)
    P_gpu = ensure_gpu_array(P, dtype=cp.float64)
    z_gpu = ensure_gpu_array(z, dtype=cp.float64)
    R_gpu = ensure_gpu_array(R, dtype=cp.float64)

    n_tracks = x_gpu.shape[0]
    n = x_gpu.shape[1]
    m = z_gpu.shape[1]
    n_sigma = 2 * n + 1

    # Generate sigma points
    sigma_points = _generate_sigma_points(x_gpu, P_gpu, alpha, kappa)

    # Compute weights
    Wm, Wc = _compute_sigma_weights(n, alpha, beta, kappa)
    Wm_gpu = ensure_gpu_array(Wm, dtype=cp.float64)
    Wc_gpu = ensure_gpu_array(Wc, dtype=cp.float64)

    # Transform sigma points through measurement function (on CPU)
    sigma_np = to_cpu(sigma_points)
    gamma_np = np.zeros((n_tracks, n_sigma, m))

    for i in range(n_tracks):
        for j in range(n_sigma):
            gamma_np[i, j] = h(sigma_np[i, j])

    gamma = ensure_gpu_array(gamma_np, dtype=cp.float64)

    # Predicted measurement: weighted sum
    z_pred = cp.einsum("j,njk->nk", Wm_gpu, gamma)

    # Innovation
    y = z_gpu - z_pred

    # Innovation covariance
    z_diff = gamma - z_pred[:, None, :]  # (n_tracks, n_sigma, m)
    S = cp.einsum("j,nji,njk->nik", Wc_gpu, z_diff, z_diff)

    # Add measurement noise
    if R_gpu.ndim == 2:
        S = S + R_gpu
    else:
        S = S + R_gpu

    # Cross covariance
    x_np = to_cpu(x_gpu)
    x_diff = sigma_np - x_np[:, None, :]  # On CPU
    x_diff_gpu = ensure_gpu_array(x_diff, dtype=cp.float64)

    Pxz = cp.einsum("j,nji,njk->nik", Wc_gpu, x_diff_gpu, z_diff)

    # Kalman gain
    S_inv = cp.linalg.inv(S)
    K = cp.einsum("nij,njk->nik", Pxz, S_inv)

    # Updated state
    x_upd = x_gpu + cp.einsum("nij,nj->ni", K, y)

    # Updated covariance
    P_upd = P_gpu - cp.einsum("nij,njk,nlk->nil", K, S, K)

    # Ensure symmetry
    P_upd = (P_upd + cp.swapaxes(P_upd, -2, -1)) / 2

    # Likelihoods
    mahal_sq = cp.einsum("ni,nij,nj->n", y, S_inv, y)
    sign, logdet = cp.linalg.slogdet(S)
    log_likelihood = -0.5 * (mahal_sq + logdet + m * np.log(2 * np.pi))
    likelihood = cp.exp(log_likelihood)

    return BatchUKFUpdate(
        x=x_upd,
        P=P_upd,
        y=y,
        S=S,
        likelihood=likelihood,
    )


class CuPyUnscentedKalmanFilter:
    """
    GPU-accelerated Unscented Kalman Filter for batch processing.

    Parameters
    ----------
    state_dim : int
        Dimension of state vector.
    meas_dim : int
        Dimension of measurement vector.
    f : callable
        Nonlinear dynamics function.
    h : callable
        Nonlinear measurement function.
    Q : array_like, optional
        Process noise covariance.
    R : array_like, optional
        Measurement noise covariance.
    alpha : float
        Spread of sigma points (default 1e-3).
    beta : float
        Prior knowledge parameter (default 2.0).
    kappa : float
        Secondary scaling (default 0.0).

    Examples
    --------
    >>> import numpy as np
    >>> from pytcl.gpu.ukf import CuPyUnscentedKalmanFilter
    >>>
    >>> def f(x):
    ...     return np.array([x[0] + x[1], x[1]])
    >>>
    >>> def h(x):
    ...     return np.array([np.sqrt(x[0]**2 + x[1]**2)])
    >>>
    >>> ukf = CuPyUnscentedKalmanFilter(
    ...     state_dim=2, meas_dim=1,
    ...     f=f, h=h,
    ... )
    """

    @requires("cupy", extra="gpu", feature="GPU Unscented Kalman filter")
    def __init__(
        self,
        state_dim: int,
        meas_dim: int,
        f: Callable[[NDArray], NDArray],
        h: Callable[[NDArray], NDArray],
        Q: Optional[ArrayLike] = None,
        R: Optional[ArrayLike] = None,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0,
    ):
        cp = import_optional("cupy", extra="gpu", feature="GPU Unscented Kalman filter")

        self.state_dim = state_dim
        self.meas_dim = meas_dim
        self.f = f
        self.h = h
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

        if Q is None:
            self.Q = cp.eye(state_dim, dtype=cp.float64) * 0.01
        else:
            self.Q = ensure_gpu_array(Q, dtype=cp.float64)

        if R is None:
            self.R = cp.eye(meas_dim, dtype=cp.float64)
        else:
            self.R = ensure_gpu_array(R, dtype=cp.float64)

    def predict(self, x: ArrayLike, P: ArrayLike) -> BatchUKFPrediction:
        """Perform batch UKF prediction."""
        return batch_ukf_predict(
            x, P, self.f, self.Q, self.alpha, self.beta, self.kappa
        )

    def update(self, x: ArrayLike, P: ArrayLike, z: ArrayLike) -> BatchUKFUpdate:
        """Perform batch UKF update."""
        return batch_ukf_update(
            x, P, z, self.h, self.R, self.alpha, self.beta, self.kappa
        )

    def predict_update(
        self, x: ArrayLike, P: ArrayLike, z: ArrayLike
    ) -> BatchUKFUpdate:
        """Combined prediction and update."""
        pred = self.predict(x, P)
        return self.update(pred.x, pred.P, z)


__all__ = [
    "BatchUKFPrediction",
    "BatchUKFUpdate",
    "batch_ukf_predict",
    "batch_ukf_update",
    "CuPyUnscentedKalmanFilter",
]
