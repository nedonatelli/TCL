"""
Particle filter (Sequential Monte Carlo) implementations.

This module provides:
- Bootstrap particle filter
- Resampling methods (multinomial, systematic, residual)
- Particle statistics (mean, covariance, ESS)
"""

from pytcl.dynamic_estimation.particle_filters.bootstrap import (
    ParticleState,
    bootstrap_pf_predict,
    bootstrap_pf_step,
    bootstrap_pf_update,
    effective_sample_size,
    gaussian_likelihood,
    initialize_particles,
    particle_covariance,
    particle_mean,
    resample_multinomial,
    resample_residual,
    resample_systematic,
)

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
