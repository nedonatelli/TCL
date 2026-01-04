"""
Particle filter (Sequential Monte Carlo) implementations.

This module provides particle filtering algorithms for nonlinear/non-Gaussian
state estimation.
"""

from typing import Any, Callable, NamedTuple, Optional, Tuple

import numpy as np
from numba import njit
from numpy.typing import ArrayLike, NDArray


class ParticleState(NamedTuple):
    """State of a particle filter.

    Attributes
    ----------
    particles : ndarray
        Particle states, shape (N, n).
    weights : ndarray
        Normalized particle weights, shape (N,).
    """

    particles: NDArray[np.floating]
    weights: NDArray[np.floating]


def resample_multinomial(
    particles: NDArray[np.floating],
    weights: NDArray[np.floating],
    rng: Optional[np.random.Generator] = None,
) -> NDArray[np.floating]:
    """
    Multinomial resampling.

    Parameters
    ----------
    particles : ndarray
        Particle states, shape (N, n).
    weights : ndarray
        Normalized weights, shape (N,).
    rng : Generator, optional
        Random number generator.

    Returns
    -------
    resampled : ndarray
        Resampled particles with uniform weights.
    """
    if rng is None:
        rng = np.random.default_rng()

    N = len(weights)
    indices = rng.choice(N, size=N, replace=True, p=weights)
    return particles[indices].copy()


def resample_systematic(
    particles: NDArray[np.floating],
    weights: NDArray[np.floating],
    rng: Optional[np.random.Generator] = None,
) -> NDArray[np.floating]:
    """
    Systematic (stratified) resampling.

    More efficient than multinomial resampling with lower variance.

    Parameters
    ----------
    particles : ndarray
        Particle states, shape (N, n).
    weights : ndarray
        Normalized weights, shape (N,).
    rng : Generator, optional
        Random number generator.

    Returns
    -------
    resampled : ndarray
        Resampled particles.
    """
    if rng is None:
        rng = np.random.default_rng()

    N = len(weights)

    # Cumulative sum of weights
    cumsum = np.cumsum(weights)

    # Systematic sampling positions
    u0 = rng.uniform(0, 1 / N)
    u = u0 + np.arange(N) / N

    # Find indices
    indices = np.searchsorted(cumsum, u)
    indices = np.clip(indices, 0, N - 1)

    return particles[indices].copy()


@njit(cache=True)
def _resample_residual_deterministic(
    particles: np.ndarray[Any, Any],
    floor_Nw: np.ndarray[Any, Any],
) -> Tuple[np.ndarray[Any, Any], int]:
    """JIT-compiled deterministic copy portion of residual resampling."""
    N = particles.shape[0]
    n = particles.shape[1]
    resampled = np.zeros((N, n), dtype=np.float64)

    idx = 0
    for i in range(N):
        count = floor_Nw[i]
        for _ in range(count):
            for k in range(n):
                resampled[idx, k] = particles[i, k]
            idx += 1

    return resampled, idx


def resample_residual(
    particles: NDArray[np.floating],
    weights: NDArray[np.floating],
    rng: Optional[np.random.Generator] = None,
) -> NDArray[np.floating]:
    """
    Residual resampling.

    Deterministically copies particles with weight > 1/N, then uses
    multinomial resampling for the remainder.

    Parameters
    ----------
    particles : ndarray
        Particle states, shape (N, n).
    weights : ndarray
        Normalized weights, shape (N,).
    rng : Generator, optional
        Random number generator.

    Returns
    -------
    resampled : ndarray
        Resampled particles.
    """
    if rng is None:
        rng = np.random.default_rng()

    N = len(weights)

    # Integer and fractional parts
    Nw = N * weights
    floor_Nw = np.floor(Nw).astype(np.int64)
    residual = Nw - floor_Nw

    # Deterministic copies (JIT-compiled)
    resampled, idx = _resample_residual_deterministic(
        particles.astype(np.float64), floor_Nw
    )

    # Multinomial resampling of residuals
    if idx < N:
        residual_weights = residual / np.sum(residual)
        residual_indices = rng.choice(N, size=N - idx, replace=True, p=residual_weights)
        resampled[idx:] = particles[residual_indices]

    return resampled


def effective_sample_size(weights: NDArray[np.floating]) -> float:
    """
    Compute effective sample size (ESS) of particle weights.

    Parameters
    ----------
    weights : ndarray
        Normalized particle weights.

    Returns
    -------
    ess : float
        Effective sample size, in range [1, N].

    Notes
    -----
    ESS = 1 / sum(w_i^2)

    ESS = N means all weights are equal (no degeneracy).
    ESS = 1 means one particle has all the weight.
    """
    return 1.0 / np.sum(weights**2)


def bootstrap_pf_predict(
    particles: NDArray[np.floating],
    f: Callable[[NDArray[Any]], NDArray[Any]],
    Q_sample: Callable[[int, Optional[np.random.Generator]], NDArray[Any]],
    rng: Optional[np.random.Generator] = None,
) -> NDArray[np.floating]:
    """
    Bootstrap particle filter prediction step.

    Propagates particles through dynamics with process noise.

    Parameters
    ----------
    particles : ndarray
        Current particles, shape (N, n).
    f : callable
        Dynamics function f(x) -> x_next (operates on single particle).
    Q_sample : callable
        Function to sample process noise: Q_sample(N, rng) -> noise (N, n).
    rng : Generator, optional
        Random number generator.

    Returns
    -------
    particles_pred : ndarray
        Predicted particles.
    """
    if rng is None:
        rng = np.random.default_rng()

    N = len(particles)
    n = particles.shape[1]

    # Sample process noise
    noise = Q_sample(N, rng)

    # Propagate each particle
    particles_pred = np.zeros((N, n), dtype=np.float64)
    for i in range(N):
        particles_pred[i] = f(particles[i]) + noise[i]

    return particles_pred


def bootstrap_pf_update(
    particles: NDArray[np.floating],
    weights: NDArray[np.floating],
    z: ArrayLike,
    likelihood_func: Callable[[NDArray[Any], NDArray[Any]], float],
) -> Tuple[NDArray[np.floating], float]:
    """
    Bootstrap particle filter update step.

    Updates weights based on measurement likelihood.

    Parameters
    ----------
    particles : ndarray
        Predicted particles, shape (N, n).
    weights : ndarray
        Current weights, shape (N,).
    z : array_like
        Measurement.
    likelihood_func : callable
        Function computing p(z|x) for a particle: likelihood_func(z, x) -> float.

    Returns
    -------
    weights_new : ndarray
        Updated normalized weights.
    log_likelihood : float
        Log marginal likelihood of measurement.
    """
    z = np.asarray(z, dtype=np.float64)
    N = len(particles)

    # Compute likelihoods
    likelihoods = np.array(
        [likelihood_func(z, particles[i]) for i in range(N)], dtype=np.float64
    )

    # Update weights
    weights_unnorm = weights * likelihoods

    # Normalize
    sum_weights = np.sum(weights_unnorm)
    if sum_weights > 0:
        weights_new = weights_unnorm / sum_weights
        log_likelihood = np.log(sum_weights / N)
    else:
        # All weights zero - degenerate case
        weights_new = np.ones(N) / N
        log_likelihood = -np.inf

    return weights_new, log_likelihood


def gaussian_likelihood(
    z: NDArray[np.floating],
    z_pred: NDArray[np.floating],
    R: NDArray[np.floating],
) -> float:
    """
    Gaussian measurement likelihood.

    Parameters
    ----------
    z : ndarray
        Measurement.
    z_pred : ndarray
        Predicted measurement for a particle.
    R : ndarray
        Measurement noise covariance.

    Returns
    -------
    likelihood : float
        p(z|x) = N(z; z_pred, R).
    """
    m = len(z)
    innovation = z - z_pred
    det_R = np.linalg.det(R)

    if det_R <= 0:
        return 0.0

    mahal_sq = innovation @ np.linalg.solve(R, innovation)
    return np.exp(-0.5 * mahal_sq) / np.sqrt((2 * np.pi) ** m * det_R)


def bootstrap_pf_step(
    particles: NDArray[np.floating],
    weights: NDArray[np.floating],
    z: ArrayLike,
    f: Callable[[NDArray[Any]], NDArray[Any]],
    h: Callable[[NDArray[Any]], NDArray[Any]],
    Q_sample: Callable[[int, Optional[np.random.Generator]], NDArray[Any]],
    R: ArrayLike,
    resample_threshold: float = 0.5,
    resample_method: str = "systematic",
    rng: Optional[np.random.Generator] = None,
) -> ParticleState:
    """
    Complete bootstrap particle filter step (predict + update + resample).

    Parameters
    ----------
    particles : ndarray
        Current particles, shape (N, n).
    weights : ndarray
        Current weights, shape (N,).
    z : array_like
        Measurement.
    f : callable
        Dynamics function.
    h : callable
        Measurement function h(x) -> z_pred.
    Q_sample : callable
        Process noise sampler.
    R : array_like
        Measurement noise covariance.
    resample_threshold : float, optional
        Resample when ESS/N < threshold (default: 0.5).
    resample_method : str, optional
        'multinomial', 'systematic', or 'residual'.
    rng : Generator, optional
        Random number generator.

    Returns
    -------
    state : ParticleState
        Updated particles and weights.
    """
    if rng is None:
        rng = np.random.default_rng()

    z = np.asarray(z, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    N = len(particles)

    # Predict
    particles_pred = bootstrap_pf_predict(particles, f, Q_sample, rng)

    # Update
    def likelihood_func(z: NDArray[Any], x: NDArray[Any]) -> Any:
        z_pred = h(x)
        return gaussian_likelihood(z, z_pred, R)

    weights_new, _ = bootstrap_pf_update(particles_pred, weights, z, likelihood_func)

    # Resample if needed
    ess = effective_sample_size(weights_new)
    if ess / N < resample_threshold:
        if resample_method == "multinomial":
            particles_new = resample_multinomial(particles_pred, weights_new, rng)
        elif resample_method == "systematic":
            particles_new = resample_systematic(particles_pred, weights_new, rng)
        elif resample_method == "residual":
            particles_new = resample_residual(particles_pred, weights_new, rng)
        else:
            raise ValueError(f"Unknown resample method: {resample_method}")
        weights_new = np.ones(N) / N
    else:
        particles_new = particles_pred

    return ParticleState(particles=particles_new, weights=weights_new)


def particle_mean(
    particles: NDArray[np.floating],
    weights: NDArray[np.floating],
) -> NDArray[np.floating]:
    """
    Compute weighted mean of particles.

    Parameters
    ----------
    particles : ndarray
        Particles, shape (N, n).
    weights : ndarray
        Weights, shape (N,).

    Returns
    -------
    mean : ndarray
        Weighted mean estimate.
    """
    return np.sum(weights[:, np.newaxis] * particles, axis=0)


@njit(cache=True)
def _particle_covariance_core(
    particles: np.ndarray[Any, Any],
    weights: np.ndarray[Any, Any],
    mean: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    """JIT-compiled core for particle covariance computation."""
    N = particles.shape[0]
    n = particles.shape[1]
    cov = np.zeros((n, n), dtype=np.float64)

    for i in range(N):
        w = weights[i]
        for j in range(n):
            diff_j = particles[i, j] - mean[j]
            for k in range(j, n):
                diff_k = particles[i, k] - mean[k]
                val = w * diff_j * diff_k
                cov[j, k] += val
                if j != k:
                    cov[k, j] += val

    return cov


def particle_covariance(
    particles: NDArray[np.floating],
    weights: NDArray[np.floating],
    mean: Optional[NDArray[np.floating]] = None,
) -> NDArray[np.floating]:
    """
    Compute weighted covariance of particles.

    Parameters
    ----------
    particles : ndarray
        Particles, shape (N, n).
    weights : ndarray
        Weights, shape (N,).
    mean : ndarray, optional
        Precomputed mean.

    Returns
    -------
    cov : ndarray
        Weighted covariance.
    """
    if mean is None:
        mean = particle_mean(particles, weights)

    return _particle_covariance_core(
        particles.astype(np.float64),
        weights.astype(np.float64),
        mean.astype(np.float64),
    )


def initialize_particles(
    x0: ArrayLike,
    P0: ArrayLike,
    N: int,
    rng: Optional[np.random.Generator] = None,
) -> ParticleState:
    """
    Initialize particles from Gaussian prior.

    Parameters
    ----------
    x0 : array_like
        Prior mean.
    P0 : array_like
        Prior covariance.
    N : int
        Number of particles.
    rng : Generator, optional
        Random number generator.

    Returns
    -------
    state : ParticleState
        Initial particles with uniform weights.
    """
    if rng is None:
        rng = np.random.default_rng()

    x0 = np.asarray(x0, dtype=np.float64).flatten()
    P0 = np.asarray(P0, dtype=np.float64)

    particles = rng.multivariate_normal(x0, P0, size=N)
    weights = np.ones(N) / N

    return ParticleState(particles=particles, weights=weights)


__all__ = [
    "ParticleState",
    "resample_multinomial",
    "resample_systematic",
    "resample_residual",
    "effective_sample_size",
    "bootstrap_pf_predict",
    "bootstrap_pf_update",
    "gaussian_likelihood",
    "bootstrap_pf_step",
    "particle_mean",
    "particle_covariance",
    "initialize_particles",
]
