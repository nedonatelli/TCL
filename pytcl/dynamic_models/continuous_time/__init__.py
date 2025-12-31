"""
Continuous-time dynamic models.

This module provides drift and diffusion functions for continuous-time
stochastic differential equations, as well as utilities for discretization.
"""

from pytcl.dynamic_models.continuous_time.dynamics import (
    continuous_to_discrete,
    diffusion_constant_acceleration,
    diffusion_constant_velocity,
    diffusion_singer,
    discretize_lti,
    drift_constant_acceleration,
    drift_constant_velocity,
    drift_coordinated_turn_2d,
    drift_singer,
    state_jacobian_ca,
    state_jacobian_cv,
    state_jacobian_singer,
)

__all__ = [
    # Drift functions
    "drift_constant_velocity",
    "drift_constant_acceleration",
    "drift_singer",
    "drift_coordinated_turn_2d",
    # Diffusion functions
    "diffusion_constant_velocity",
    "diffusion_constant_acceleration",
    "diffusion_singer",
    # Discretization
    "continuous_to_discrete",
    "discretize_lti",
    # Jacobians
    "state_jacobian_cv",
    "state_jacobian_ca",
    "state_jacobian_singer",
]
