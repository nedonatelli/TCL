"""
GPU-accelerated Unscented Kalman Filter.

This module provides GPU-accelerated implementations of the Unscented Kalman
Filter (UKF) for batch processing of multiple tracks with nonlinear dynamics.

The UKF uses sigma points to propagate uncertainty through nonlinear functions
without requiring Jacobian computation.

All array work is dispatched through :mod:`pytcl.gpu._backend`, so the same
code runs on CuPy (NVIDIA CUDA, float64) and on MLX (Apple Silicon, float32).
The nonlinear functions ``f`` and ``h`` receive the flattened sigma points as
a single *device* array and must be written against
:func:`pytcl.gpu.utils.get_array_module` rather than NumPy directly -- the same
contract every other filter in :mod:`pytcl.gpu` uses. A callable that mixes a
host NumPy array into the expression raises ``TypeError`` on CuPy.

Key Features
------------
- Batch processing of multiple tracks
- Configurable sigma point parameters (alpha, beta, kappa)
- GPU-accelerated sigma point generation and transformation
- Support for nonlinear dynamics and measurements

Notes
-----
**Precision and the choice of alpha.** Merwe scaled sigma points place the
points at a distance proportional to ``alpha`` from the mean and then undo that
scaling with weights of magnitude ``O(1/alpha**2)``. The default
``alpha=1e-3`` therefore produces weights near ``1e6``: the mean and covariance
are recovered by canceling large terms, and every bit lost to rounding is
amplified by that factor.

On the MLX backend all compute is float32 (MLX raises on float64 for GPU
operations), which makes this amplification severe. Measured on a linear
problem where the UKF must reduce exactly to the linear Kalman filter
(4-state constant-velocity dynamics, 2-D position measurement, 12 tracks),
maximum relative error of ``x`` and ``P`` from ``batch_ukf_predict`` and
``batch_ukf_update`` against the float64 CPU UKF:

===========  ===================  ==================
``alpha``    MLX (float32)        float64 reference
===========  ===================  ==================
1e-3         5.8e+01              1.9e-10
1e-2         4.6e-03              1.1e-12
1e-1         1.9e-05              2.1e-14
1.0          1.9e-06              6.6e-16
===========  ===================  ==================

The float32 column varies by roughly a factor of 4 across problem
instances; the scaling with ``alpha`` does not. At ``alpha=1e-3`` the
float32 result carries no significant digits at all, so
:func:`batch_ukf_predict` and :func:`batch_ukf_update` emit a
:class:`RuntimeWarning` when they run on a float32 backend with
``alpha < 1e-2``. The user's ``alpha`` is never silently changed. **On MLX,
use ``alpha`` of 0.1 or larger** (0.5 and 1.0 are common choices and reach
roughly float32 machine precision); the CuPy backend is unaffected and the
1e-3 default remains reasonable there.

Examples
--------
>>> from pytcl.gpu.ukf import batch_ukf_predict
>>> import numpy as np
>>>
>>> from pytcl.gpu.utils import get_array_module
>>> def f_dynamics(x):
...     xp = get_array_module(x)
...     return xp.stack([x[:, 0] + x[:, 1], x[:, 1] * 0.99], axis=1)
>>>
>>> x = np.zeros((3, 2))
>>> P = np.stack([np.eye(2)] * 3)
>>> Q = np.stack([np.eye(2) * 0.01] * 3)
>>> x_pred, P_pred = batch_ukf_predict(x, P, f_dynamics, Q)
>>> x_pred.shape
(3, 2)
"""

import warnings
from typing import Any, Callable, NamedTuple, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pytcl.gpu._backend import Backend, get_compute_backend

#: Below this alpha the Merwe weights exceed 1e4 and a float32 backend loses
#: every significant digit of the recovered mean and covariance.
_FLOAT32_MIN_ALPHA = 1e-2


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


def _warn_if_ill_conditioned(b: Backend, alpha: float) -> None:
    """Warn when float32 compute cannot resolve the requested sigma spread.

    The Merwe weights scale as ``1/alpha**2``; on a float32 backend an alpha
    below :data:`_FLOAT32_MIN_ALPHA` leaves no significant digits in the
    recovered moments. The caller's ``alpha`` is left untouched.
    """
    if b.supports_float64 or alpha >= _FLOAT32_MIN_ALPHA:
        return
    warnings.warn(
        f"alpha={alpha:g} gives Merwe sigma-point weights of order "
        f"{1.0 / alpha**2:.0e}, which the float32 {b.name} backend cannot "
        f"resolve: the returned mean and covariance carry no significant "
        f"digits (measured relative error ~1e2 at alpha=1e-3). Pass "
        f"alpha >= {_FLOAT32_MIN_ALPHA:g} (0.1 or larger recommended) or use "
        f"a float64 backend.",
        RuntimeWarning,
        stacklevel=3,
    )


def _compute_sigma_weights(
    n: int,
    alpha: float = 1e-3,
    beta: float = 2.0,
    kappa: float = 0.0,
) -> Tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
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


def _matrix_sqrt(b: Backend, P: Any, n: int) -> Any:
    """Batched lower-triangular square root of ``P`` with a non-PD fallback.

    Returns ``L`` with ``L @ L.T == P`` when ``P`` is positive definite. When
    it is not, ``L`` reconstructs ``P`` with its eigenvalues clamped at 1e-10.

    Positive definiteness is decided from the diagonal of the Cholesky factor
    rather than from a raised exception: NumPy and CuPy raise ``LinAlgError``,
    but MLX returns a factor with a non-positive diagonal instead of failing,
    so both signals must be handled for the fallback to fire uniformly.
    """
    positive_definite = False
    L = None
    try:
        L = b.cholesky(P)
        # Batched diagonal via an identity mask: sum(L * I, axis=-1)[..., j]
        # is L[..., j, j]; b.max(-d) is NaN-propagating, so a NaN factor also
        # fails the test below.
        diag = b.sum(L * b.eye(n), axis=-1)
        min_diag = float(b.to_numpy(-b.max(-diag)))
        positive_definite = min_diag > 0.0
    except np.linalg.LinAlgError:
        positive_definite = False

    if positive_definite:
        return L

    eigvals, eigvecs = b.eigh(P)
    eigvals = b.maximum(eigvals, 1e-10)
    # Per-track V @ diag(sqrt(w)) so that L @ L.T == V @ diag(w) @ V.T ~= P
    return eigvecs * b.sqrt(eigvals)[..., None, :]


def _generate_sigma_points(
    x: ArrayLike,
    P: ArrayLike,
    alpha: float = 1e-3,
    kappa: float = 0.0,
) -> NDArray[np.floating[Any]]:
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
    b = get_compute_backend()

    x_gpu = b.asarray(x)
    P_gpu = b.asarray(P)

    n = x_gpu.shape[1]  # state dim

    lambda_ = alpha**2 * (n + kappa) - n
    gamma = float(np.sqrt(n + lambda_))

    L = _matrix_sqrt(b, P_gpu, n)

    # Scale by gamma
    scaled_L = gamma * L  # shape: (n_tracks, n, n)

    # Sigma points: the mean, then x +/- the columns of scaled_L. Built by
    # stacking rather than slice assignment (MLX arrays are immutable).
    columns = [scaled_L[:, :, i] for i in range(n)]
    rows = [x_gpu]
    rows.extend(x_gpu + col for col in columns)
    rows.extend(x_gpu - col for col in columns)

    return b.stack(rows, axis=1)


def batch_ukf_predict(
    x: ArrayLike,
    P: ArrayLike,
    f: Callable[[NDArray[np.floating[Any]]], NDArray[np.floating[Any]]],
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
        Batched dynamics. Takes ``(N, state_dim)`` and returns
        ``(N, state_dim)``, where ``N`` is ``n_tracks * (2 * state_dim + 1)``
        sigma points. Called once.
    Q : array_like
        Process noise covariance.
    alpha, beta, kappa : float
        Sigma point parameters.

    Returns
    -------
    result : BatchUKFPrediction
        Predicted states and covariances.

    Warns
    -----
    RuntimeWarning
        If the active backend computes in float32 (MLX) and ``alpha`` is below
        1e-2, where the O(1/alpha**2) Merwe weights destroy every significant
        digit. See Notes.

    Notes
    -----
    On the float32 MLX backend the maximum relative error against the float64
    CPU UKF, on a linear problem where the UKF reduces to the Kalman filter,
    is 5.8e+01 at ``alpha=1e-3``, 4.6e-03 at 1e-2, 1.9e-05 at 0.1 and 1.9e-06
    at 1.0 (the O(1/alpha**2) Merwe weights amplify float32 rounding). Use
    ``alpha >= 0.1`` on MLX. In float64 the same errors are 1.9e-10, 1.1e-12,
    2.1e-14 and 6.6e-16. See the module Notes.

    Examples
    --------
    >>> import numpy as np
    >>> from pytcl.gpu.ukf import batch_ukf_predict
    >>> # Nonlinear dynamics example
    >>> from pytcl.gpu.utils import get_array_module
    >>> def f_dynamics(x):
    ...     xp = get_array_module(x)
    ...     return xp.stack([x[:, 0] + 0.1 * x[:, 1], x[:, 1] * 0.99], axis=1)
    >>> n_tracks = 50
    >>> x = np.random.randn(n_tracks, 2)
    >>> P = np.tile(np.eye(2) * 0.01, (n_tracks, 1, 1))
    >>> Q = np.eye(2) * 0.001
    >>> result = batch_ukf_predict(x, P, f_dynamics, Q)
    >>> result.x.shape
    (50, 2)
    """
    b = get_compute_backend()
    _warn_if_ill_conditioned(b, alpha)

    x_gpu = b.asarray(x)
    P_gpu = b.asarray(P)
    Q_gpu = b.asarray(Q)

    n_tracks = x_gpu.shape[0]
    n = x_gpu.shape[1]
    n_sigma = 2 * n + 1

    # Generate sigma points
    sigma_points = _generate_sigma_points(x_gpu, P_gpu, alpha, kappa)

    # Compute weights
    Wm, Wc = _compute_sigma_weights(n, alpha, beta, kappa)
    Wm_gpu = b.asarray(Wm)
    Wc_gpu = b.asarray(Wc)

    # Propagate every sigma point of every track in one call. Flattening to
    # (n_tracks * n_sigma, n) is what lets the same callable serve the EKF and
    # the UKF: it always sees a 2-D batch, whatever the batch happens to be.
    # This replaced a nested Python loop of n_tracks * (2n+1) invocations.
    flat = b.reshape(sigma_points, (n_tracks * n_sigma, n))
    sigma_pred = b.reshape(b.asarray(f(flat)), (n_tracks, n_sigma, n))

    # Predicted mean: sum of weighted sigma points
    x_pred = b.einsum("j,njk->nk", Wm_gpu, sigma_pred)

    # Predicted covariance
    diff = sigma_pred - x_pred[:, None, :]  # (n_tracks, n_sigma, n)
    P_pred = b.einsum("j,nji,njk->nik", Wc_gpu, diff, diff)

    # Add process noise (broadcasts for a shared (n, n) or per-track Q)
    P_pred = P_pred + Q_gpu

    # Ensure symmetry
    P_pred = (P_pred + b.swapaxes(P_pred, -2, -1)) / 2

    b.evaluate(x_pred, P_pred)
    return BatchUKFPrediction(x=x_pred, P=P_pred)


def batch_ukf_update(
    x: ArrayLike,
    P: ArrayLike,
    z: ArrayLike,
    h: Callable[[NDArray[np.floating[Any]]], NDArray[np.floating[Any]]],
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
        Batched measurement function. Takes ``(N, state_dim)`` and returns
        ``(N, meas_dim)``, where ``N`` is ``n_tracks * (2 * state_dim + 1)``
        sigma points. Called once.
    R : array_like
        Measurement noise covariance.
    alpha, beta, kappa : float
        Sigma point parameters.

    Returns
    -------
    result : BatchUKFUpdate
        Update results.

    Warns
    -----
    RuntimeWarning
        If the active backend computes in float32 (MLX) and ``alpha`` is below
        1e-2, where the O(1/alpha**2) Merwe weights destroy every significant
        digit. See Notes.

    Notes
    -----
    On the float32 MLX backend the maximum relative error against the float64
    CPU UKF, on a linear problem where the UKF reduces to the Kalman filter,
    is 5.8e+01 at ``alpha=1e-3``, 4.6e-03 at 1e-2, 1.9e-05 at 0.1 and 1.9e-06
    at 1.0 (the O(1/alpha**2) Merwe weights amplify float32 rounding). Use
    ``alpha >= 0.1`` on MLX. In float64 the same errors are 1.9e-10, 1.1e-12,
    2.1e-14 and 6.6e-16. See the module Notes.

    Examples
    --------
    >>> import numpy as np
    >>> from pytcl.gpu.ukf import batch_ukf_update
    >>> # Nonlinear measurement example
    >>> from pytcl.gpu.utils import get_array_module
    >>> def h_measurement(x):  # Range-only
    ...     xp = get_array_module(x)
    ...     return xp.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2)[:, None]
    >>> n_tracks = 40
    >>> x = np.random.randn(n_tracks, 2)
    >>> P = np.tile(np.eye(2), (n_tracks, 1, 1))
    >>> z = np.random.randn(n_tracks, 1) * 10 + 100
    >>> R = np.array([[1.0]])
    >>> result = batch_ukf_update(x, P, z, h_measurement, R)
    >>> result.x.shape
    (40, 2)
    """
    b = get_compute_backend()
    _warn_if_ill_conditioned(b, alpha)

    x_gpu = b.asarray(x)
    P_gpu = b.asarray(P)
    z_gpu = b.asarray(z)
    R_gpu = b.asarray(R)

    n_tracks = x_gpu.shape[0]
    n = x_gpu.shape[1]
    m = z_gpu.shape[1]
    n_sigma = 2 * n + 1

    # Generate sigma points
    sigma_points = _generate_sigma_points(x_gpu, P_gpu, alpha, kappa)

    # Compute weights
    Wm, Wc = _compute_sigma_weights(n, alpha, beta, kappa)
    Wm_gpu = b.asarray(Wm)
    Wc_gpu = b.asarray(Wc)

    # One call for every sigma point of every track; see batch_ukf_predict.
    flat = b.reshape(sigma_points, (n_tracks * n_sigma, n))
    gamma = b.reshape(b.asarray(h(flat)), (n_tracks, n_sigma, m))

    # Predicted measurement: weighted sum
    z_pred = b.einsum("j,njk->nk", Wm_gpu, gamma)

    # Innovation
    y = z_gpu - z_pred

    # Innovation covariance
    z_diff = gamma - z_pred[:, None, :]  # (n_tracks, n_sigma, m)
    S = b.einsum("j,nji,njk->nik", Wc_gpu, z_diff, z_diff)

    # Add measurement noise (broadcasts for a shared (m, m) or per-track R)
    S = S + R_gpu

    # Cross covariance
    x_diff = sigma_points - x_gpu[:, None, :]
    Pxz = b.einsum("j,nji,njk->nik", Wc_gpu, x_diff, z_diff)

    # Kalman gain
    S_inv = b.inv(S)
    K = b.einsum("nij,njk->nik", Pxz, S_inv)

    # Updated state
    x_upd = x_gpu + b.einsum("nij,nj->ni", K, y)

    # Updated covariance
    P_upd = P_gpu - b.einsum("nij,njk,nlk->nil", K, S, K)

    # Ensure symmetry
    P_upd = (P_upd + b.swapaxes(P_upd, -2, -1)) / 2

    # Likelihoods
    mahal_sq = b.einsum("ni,nij,nj->n", y, S_inv, y)
    _sign, logdet = b.slogdet(S)
    log_likelihood = -0.5 * (mahal_sq + logdet + m * np.log(2 * np.pi))
    likelihood = b.exp(log_likelihood)

    b.evaluate(x_upd, P_upd, y, S, likelihood)
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

    Runs on whichever compute backend is available (CuPy on CUDA, MLX on
    Apple Silicon); the name is retained for backwards compatibility.

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
        Spread of sigma points (default 1e-3). On the float32 MLX backend use
        0.1 or larger; see the module Notes.
    beta : float
        Prior knowledge parameter (default 2.0).
    kappa : float
        Secondary scaling (default 0.0).

    Examples
    --------
    >>> import numpy as np
    >>> from pytcl.gpu.ukf import CuPyUnscentedKalmanFilter
    >>>
    >>> from pytcl.gpu.utils import get_array_module
    >>> def f(x):
    ...     xp = get_array_module(x)
    ...     return xp.stack([x[:, 0] + x[:, 1], x[:, 1]], axis=1)
    >>>
    >>> def h(x):
    ...     xp = get_array_module(x)
    ...     return xp.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2)[:, None]
    >>>
    >>> ukf = CuPyUnscentedKalmanFilter(
    ...     state_dim=2, meas_dim=1,
    ...     f=f, h=h,
    ... )
    """

    def __init__(
        self,
        state_dim: int,
        meas_dim: int,
        f: Callable[[NDArray[np.floating[Any]]], NDArray[np.floating[Any]]],
        h: Callable[[NDArray[np.floating[Any]]], NDArray[np.floating[Any]]],
        Q: Optional[ArrayLike] = None,
        R: Optional[ArrayLike] = None,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0,
    ):
        b = get_compute_backend()

        self.state_dim = state_dim
        self.meas_dim = meas_dim
        self.f = f
        self.h = h
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

        if Q is None:
            self.Q = b.eye(state_dim) * 0.01
        else:
            self.Q = b.asarray(Q)

        if R is None:
            self.R = b.eye(meas_dim)
        else:
            self.R = b.asarray(R)

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
