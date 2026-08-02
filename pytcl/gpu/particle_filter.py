"""
GPU-accelerated Particle Filter.

This module provides GPU-accelerated implementations of particle filtering
algorithms for highly nonlinear and non-Gaussian state estimation. The
algorithms are written against the backend-dispatch layer in
:mod:`pytcl.gpu._backend`, so they run on CuPy (NVIDIA CUDA, float64) or MLX
(Apple Silicon, float32) without change.

Key Features
------------
- GPU-accelerated resampling (systematic, multinomial, stratified)
- Parallel weight computation
- Batch processing of multiple particle filters
- Efficient memory management

Performance
-----------
The GPU implementation achieves 8-15x speedup compared to CPU for:
- Large particle counts (N > 1000)
- Parallel processing of multiple targets

Notes
-----
On the MLX backend all computation is single precision, so weights and
estimates are precision-limited relative to CuPy (relative error ~1e-7 rather
than ~1e-15). The resampling *properties* (systematic low-variance bound,
multinomial distribution, uniform post-resample weights) hold exactly on both.

Examples
--------
>>> from pytcl.gpu.particle_filter import CuPyParticleFilter
>>> import numpy as np
>>>
>>> def dynamics(particles, t):
...     # Propagate particles through nonlinear dynamics
...     return particles + np.random.randn(*particles.shape) * 0.1
>>>
>>> def likelihood(particles, measurement):
...     # Compute likelihood for each particle
...     diff = particles[:, 0] - measurement
...     return np.exp(-0.5 * diff**2)
>>>
>>> pf = CuPyParticleFilter(n_particles=1000, state_dim=2)
>>> pf.initialize(np.zeros(2), np.eye(2))
>>> pf.predict(lambda particles: dynamics(particles, 0.0))
>>> _ = pf.update(0.5, likelihood)  # returns the log-likelihood
>>> pf.get_estimate().shape
(2,)
"""

from typing import Any, Callable, NamedTuple, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pytcl.gpu._backend import Backend, get_compute_backend
from pytcl.gpu.utils import to_cpu


class ParticleFilterState(NamedTuple):
    """State of a particle filter.

    Attributes
    ----------
    particles : ndarray
        Particle states, shape (n_particles, state_dim).
    weights : ndarray
        Normalized particle weights, shape (n_particles,).
    ess : float
        Effective sample size.
    """

    particles: NDArray[np.floating]
    weights: NDArray[np.floating]
    ess: float


def _likelihood_floor(b: Backend) -> float:
    """Positive floor added to likelihoods before taking their log.

    Guards ``log(0)`` without perturbing representable likelihoods. The value
    must stay in the *normal* range of the backend's floating dtype: MLX
    evaluates ``log`` of a float32 subnormal as ``-inf`` on the GPU stream,
    which would propagate NaN through the log-sum-exp when every likelihood
    underflows.
    """
    return 1e-300 if b.supports_float64 else 1e-30


def gpu_effective_sample_size(weights: ArrayLike) -> float:
    """
    Compute effective sample size on GPU.

    ESS = 1 / sum(w_i^2)

    Parameters
    ----------
    weights : array_like
        Normalized particle weights.

    Returns
    -------
    ess : float
        Effective sample size.

    Examples
    --------
    >>> import numpy as np
    >>> from pytcl.gpu.particle_filter import gpu_effective_sample_size
    >>> weights = np.array([0.1, 0.2, 0.3, 0.4])
    >>> ess = gpu_effective_sample_size(weights)
    >>> ess > 0
    True
    >>> ess <= len(weights)
    True
    """
    b = get_compute_backend()
    w = b.asarray(weights)
    ess = 1.0 / float(b.sum(w**2))
    return ess


def gpu_resample_systematic(
    weights: ArrayLike, seed: Optional[int] = None
) -> NDArray[np.intp]:
    """
    GPU-accelerated systematic resampling.

    Systematic resampling uses a single random number to select particles,
    resulting in low variance and O(N) complexity.

    Parameters
    ----------
    weights : array_like
        Normalized particle weights, shape (n_particles,).
    seed : int, optional
        Seed for the single uniform draw. If None, the backend's global
        random state is used.

    Returns
    -------
    indices : ndarray
        Resampled particle indices, shape (n_particles,).

    Examples
    --------
    >>> import numpy as np
    >>> from pytcl.gpu.particle_filter import gpu_resample_systematic
    >>> weights = np.array([0.1, 0.3, 0.4, 0.2])
    >>> indices = gpu_resample_systematic(weights)
    >>> # Particles 1 and 2 will be selected more often

    Notes
    -----
    Every particle is selected either ``floor(n * w_i)`` or
    ``ceil(n * w_i)`` times, i.e. ``|count_i - n * w_i| < 1``.
    """
    b = get_compute_backend()

    w = b.asarray(weights)
    n = w.shape[0]

    # Cumulative sum of weights
    cumsum = b.cumsum(w)

    # Systematic sampling positions: a single offset u0 ~ U[0, 1/n) shared by
    # all n equally spaced strata.
    u0 = b.uniform((1,), key=seed)
    positions = (b.arange(n) + u0) / n

    # Find indices using searchsorted
    indices = b.searchsorted(cumsum, positions)

    # Clip to valid range
    indices = b.clip(indices, 0, n - 1)

    return indices


def gpu_resample_multinomial(
    weights: ArrayLike, seed: Optional[int] = None
) -> NDArray[np.intp]:
    """
    GPU-accelerated multinomial resampling.

    Multinomial resampling samples particles independently according
    to their weights.

    Parameters
    ----------
    weights : array_like
        Normalized particle weights, shape (n_particles,).
    seed : int, optional
        Seed for the uniform draws. If None, the backend's global random
        state is used.

    Returns
    -------
    indices : ndarray
        Resampled particle indices, shape (n_particles,).

    Examples
    --------
    >>> import numpy as np
    >>> from pytcl.gpu.particle_filter import gpu_resample_multinomial
    >>> from pytcl.gpu.utils import to_cpu
    >>> weights = np.array([0.1, 0.4, 0.5])
    >>> indices = np.asarray(to_cpu(gpu_resample_multinomial(weights)))
    >>> indices.shape
    (3,)
    >>> bool(np.all(indices < 3))
    True

    Notes
    -----
    Multinomial resampling has higher variance than systematic resampling
    but is simpler and can be more efficient on GPU for certain sizes.
    """
    b = get_compute_backend()

    w = b.asarray(weights)
    n = w.shape[0]

    # Cumulative sum
    cumsum = b.cumsum(w)

    # Generate random samples
    u = b.uniform((n,), key=seed)

    # Find indices
    indices = b.searchsorted(cumsum, u)
    indices = b.clip(indices, 0, n - 1)

    return indices


def gpu_resample_stratified(
    weights: ArrayLike, seed: Optional[int] = None
) -> NDArray[np.intp]:
    """
    GPU-accelerated stratified resampling.

    Stratified resampling divides the CDF into N equal strata and samples
    one particle from each stratum.

    Parameters
    ----------
    weights : array_like
        Normalized particle weights, shape (n_particles,).
    seed : int, optional
        Seed for the uniform draws. If None, the backend's global random
        state is used.

    Returns
    -------
    indices : ndarray
        Resampled particle indices, shape (n_particles,).
    """
    b = get_compute_backend()

    w = b.asarray(weights)
    n = w.shape[0]

    # Cumulative sum
    cumsum = b.cumsum(w)

    # Stratified sampling: one random number per stratum
    u = (b.arange(n) + b.uniform((n,), key=seed)) / n

    # Find indices
    indices = b.searchsorted(cumsum, u)
    indices = b.clip(indices, 0, n - 1)

    return indices


def gpu_normalize_weights(
    log_weights: ArrayLike,
) -> Tuple[NDArray[np.floating[Any]], float]:
    """
    Normalize log weights to proper weights on GPU.

    Uses log-sum-exp trick for numerical stability.

    Parameters
    ----------
    log_weights : array_like
        Unnormalized log weights, shape (n_particles,).

    Returns
    -------
    weights : ndarray
        Normalized weights, shape (n_particles,).
    log_likelihood : float
        Log of the normalization constant.

    Examples
    --------
    >>> import numpy as np
    >>> from pytcl.gpu.particle_filter import gpu_normalize_weights
    >>> log_w = np.array([-1.0, -0.5, -2.0])
    >>> from pytcl.gpu.utils import to_cpu
    >>> weights, log_likelihood = gpu_normalize_weights(log_w)
    >>> bool(np.allclose(np.asarray(to_cpu(weights)).sum(), 1.0))
    True
    >>> bool(np.isclose(float(np.asarray(to_cpu(log_likelihood))),
    ...                 np.log(np.exp(log_w).sum())))
    True
    """
    b = get_compute_backend()

    log_w = b.asarray(log_weights)

    # Log-sum-exp for numerical stability
    max_log_w = b.max(log_w)
    log_sum = max_log_w + b.log(b.sum(b.exp(log_w - max_log_w)))

    # Normalized weights
    weights = b.exp(log_w - log_sum)

    return weights, float(log_sum)


class CuPyParticleFilter:
    """
    GPU-accelerated Bootstrap Particle Filter.

    This class implements the Sequential Importance Resampling (SIR)
    particle filter with GPU acceleration.

    Parameters
    ----------
    n_particles : int
        Number of particles.
    state_dim : int
        Dimension of state vector.
    resample_method : str
        Resampling method: 'systematic', 'multinomial', or 'stratified'.
    resample_threshold : float
        ESS threshold for resampling (as fraction of n_particles).

    Attributes
    ----------
    particles : GPUArray
        Current particle states, shape (n_particles, state_dim).
    weights : GPUArray
        Current particle weights, shape (n_particles,).

    Examples
    --------
    >>> import numpy as np
    >>> from pytcl.gpu.particle_filter import CuPyParticleFilter
    >>>
    >>> pf = CuPyParticleFilter(n_particles=1000, state_dim=4)
    >>> pf.initialize(np.zeros(4), np.eye(4))
    >>> dynamics_fn = lambda particles: particles * 0.99
    >>> likelihood_fn = lambda particles, z: np.exp(-0.5 * (particles[:, 0] - z) ** 2)
    >>> for measurement in (0.1, 0.2, 0.3):
    ...     pf.predict(dynamics_fn)
    ...     _ = pf.update(measurement, likelihood_fn)
    >>> pf.get_estimate().shape
    (4,)
    """

    def __init__(
        self,
        n_particles: int,
        state_dim: int,
        resample_method: str = "systematic",
        resample_threshold: float = 0.5,
    ):
        b = get_compute_backend()
        self._b = b

        self.n_particles = n_particles
        self.state_dim = state_dim
        self.resample_threshold = resample_threshold

        # Select resampling function
        if resample_method == "systematic":
            self._resample_fn = gpu_resample_systematic
        elif resample_method == "multinomial":
            self._resample_fn = gpu_resample_multinomial
        elif resample_method == "stratified":
            self._resample_fn = gpu_resample_stratified
        else:
            raise ValueError(f"Unknown resample method: {resample_method}")

        # Initialize particles and weights
        self.particles = b.zeros((n_particles, state_dim))
        self.weights = b.ones(n_particles) / n_particles

    def initialize(
        self,
        mean: ArrayLike,
        cov: ArrayLike,
    ) -> None:
        """
        Initialize particles from Gaussian distribution.

        Parameters
        ----------
        mean : array_like
            Mean state, shape (state_dim,).
        cov : array_like
            Covariance matrix, shape (state_dim, state_dim).
        """
        b = self._b

        mean = np.asarray(mean).flatten()
        cov = np.asarray(cov)

        # Sample from multivariate normal on CPU (no GPU backend provides it)
        samples = np.random.multivariate_normal(mean, cov, self.n_particles)
        self.particles = b.asarray(samples)
        self.weights = b.ones(self.n_particles) / self.n_particles

    def initialize_uniform(
        self,
        low: ArrayLike,
        high: ArrayLike,
    ) -> None:
        """
        Initialize particles from uniform distribution.

        Parameters
        ----------
        low : array_like
            Lower bounds, shape (state_dim,).
        high : array_like
            Upper bounds, shape (state_dim,).
        """
        b = self._b

        low = b.asarray(low)
        high = b.asarray(high)

        # Sample uniformly
        u = b.uniform((self.n_particles, self.state_dim))
        self.particles = low + u * (high - low)
        self.weights = b.ones(self.n_particles) / self.n_particles

    def predict(
        self,
        dynamics_fn: Callable[[NDArray[np.floating[Any]]], NDArray[np.floating[Any]]],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Propagate particles through dynamics.

        Parameters
        ----------
        dynamics_fn : callable
            Function that takes particles (N, state_dim) and returns
            propagated particles (N, state_dim).
        *args, **kwargs
            Additional arguments passed to dynamics_fn.

        Notes
        -----
        The dynamics function receives backend arrays (CuPy or MLX). It should
        return arrays of the same type.
        """
        # Apply dynamics (may be on CPU or GPU depending on function)
        self.particles = dynamics_fn(self.particles, *args, **kwargs)

    def update(
        self,
        measurement: ArrayLike,
        likelihood_fn: Callable[
            [NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
            NDArray[np.floating[Any]],
        ],
    ) -> float:
        """
        Update weights based on measurement likelihood.

        Parameters
        ----------
        measurement : array_like
            Measurement vector.
        likelihood_fn : callable
            Function that computes likelihood for each particle.
            Takes (particles, measurement) and returns likelihoods (n_particles,).

        Returns
        -------
        log_likelihood : float
            Log of the marginal likelihood (normalization constant).
        """
        b = self._b

        z = b.asarray(measurement)

        # Compute likelihoods
        likelihoods = likelihood_fn(self.particles, z)
        likelihoods = b.asarray(likelihoods)

        # Update weights
        log_weights = b.log(self.weights) + b.log(likelihoods + _likelihood_floor(b))

        # Normalize
        self.weights, log_likelihood = gpu_normalize_weights(log_weights)

        # Resample if ESS drops below threshold
        ess = gpu_effective_sample_size(self.weights)
        if ess < self.resample_threshold * self.n_particles:
            self._resample()

        return log_likelihood

    def _resample(self) -> None:
        """Perform resampling."""
        b = self._b

        indices = self._resample_fn(self.weights)
        self.particles = self.particles[indices]
        self.weights = b.ones(self.n_particles) / self.n_particles

    def get_estimate(self) -> NDArray[np.floating]:
        """
        Compute weighted mean estimate.

        Returns
        -------
        estimate : ndarray
            Weighted mean state, shape (state_dim,).
        """
        b = self._b
        estimate = b.sum(self.particles * self.weights[:, None], axis=0)
        return estimate

    def get_covariance(self) -> NDArray[np.floating]:
        """
        Compute weighted covariance estimate.

        Returns
        -------
        cov : ndarray
            Weighted covariance, shape (state_dim, state_dim).
        """
        b = self._b

        mean = self.get_estimate()
        diff = self.particles - mean
        cov = b.einsum("n,ni,nj->ij", self.weights, diff, diff)
        return cov

    def get_ess(self) -> float:
        """Get current effective sample size."""
        return gpu_effective_sample_size(self.weights)

    def get_state(self) -> ParticleFilterState:
        """
        Get current filter state.

        Returns
        -------
        state : ParticleFilterState
            Named tuple with particles, weights, and ESS.
        """
        return ParticleFilterState(
            particles=self.particles,
            weights=self.weights,
            ess=self.get_ess(),
        )

    def get_particles_cpu(self) -> NDArray[np.floating]:
        """Get particles on CPU."""
        return to_cpu(self.particles)

    def get_weights_cpu(self) -> NDArray[np.floating]:
        """Get weights on CPU."""
        return to_cpu(self.weights)


def batch_particle_filter_update(
    particles: ArrayLike,
    weights: ArrayLike,
    measurements: ArrayLike,
    likelihood_fn: Callable[
        [NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
        NDArray[np.floating[Any]],
    ],
) -> Tuple[
    NDArray[np.floating[Any]], NDArray[np.floating[Any]], NDArray[np.floating[Any]]
]:
    """
    Batch update for multiple particle filters.

    Parameters
    ----------
    particles : array_like
        Particle states, shape (n_filters, n_particles, state_dim).
    weights : array_like
        Particle weights, shape (n_filters, n_particles).
    measurements : array_like
        Measurements, shape (n_filters, meas_dim).
    likelihood_fn : callable
        Function that computes likelihood for each particle.

    Returns
    -------
    weights_updated : ndarray
        Updated weights.
    log_likelihoods : ndarray
        Log likelihoods for each filter.
    ess : ndarray
        Effective sample sizes.
    """
    b = get_compute_backend()

    particles_gpu = b.asarray(particles)
    weights_gpu = b.asarray(weights)
    measurements_gpu = b.asarray(measurements)

    n_filters = particles_gpu.shape[0]
    floor = _likelihood_floor(b)

    # Rows are accumulated and stacked: backend arrays are immutable on MLX,
    # so no in-place row assignment is possible.
    weights_rows = []
    log_likelihood_rows = []
    ess_rows = []

    for i in range(n_filters):
        # Compute likelihoods
        likelihoods = likelihood_fn(particles_gpu[i], measurements_gpu[i])
        likelihoods = b.asarray(likelihoods)

        # Update weights
        log_weights = b.log(weights_gpu[i]) + b.log(likelihoods + floor)

        # Normalize
        max_log_w = b.max(log_weights)
        log_sum = max_log_w + b.log(b.sum(b.exp(log_weights - max_log_w)))
        weights_i = b.exp(log_weights - log_sum)

        weights_rows.append(weights_i)
        log_likelihood_rows.append(log_sum)

        # ESS
        ess_rows.append(1.0 / b.sum(weights_i**2))

    weights_updated = b.stack(weights_rows)
    log_likelihoods = b.stack(log_likelihood_rows)
    ess = b.stack(ess_rows)

    return weights_updated, log_likelihoods, ess


__all__ = [
    "ParticleFilterState",
    "gpu_effective_sample_size",
    "gpu_resample_systematic",
    "gpu_resample_multinomial",
    "gpu_resample_stratified",
    "gpu_normalize_weights",
    "CuPyParticleFilter",
    "batch_particle_filter_update",
]
