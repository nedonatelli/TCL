"""
Particle filter (Sequential Monte Carlo) implementations.

This module provides:
- Bootstrap particle filter
- Resampling methods (multinomial, systematic, residual)
- Particle statistics (mean, covariance, ESS)
"""

from tcl.dynamic_estimation.particle_filters.bootstrap import (
    ParticleState,
    resample_multinomial,
    resample_systematic,
    resample_residual,
    effective_sample_size,
    bootstrap_pf_predict,
    bootstrap_pf_update,
    gaussian_likelihood,
    bootstrap_pf_step,
    particle_mean,
    particle_covariance,
    initialize_particles,
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
