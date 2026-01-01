"""
Continuous-time dynamic models.

This module provides drift and diffusion functions for continuous-time
stochastic differential equations, as well as utilities for discretization.
"""

from pytcl.dynamic_models.continuous_time.dynamics import continuous_to_discrete
from pytcl.dynamic_models.continuous_time.dynamics import (
    diffusion_constant_acceleration,
)
from pytcl.dynamic_models.continuous_time.dynamics import diffusion_constant_velocity
from pytcl.dynamic_models.continuous_time.dynamics import diffusion_singer
from pytcl.dynamic_models.continuous_time.dynamics import discretize_lti
from pytcl.dynamic_models.continuous_time.dynamics import drift_constant_acceleration
from pytcl.dynamic_models.continuous_time.dynamics import drift_constant_velocity
from pytcl.dynamic_models.continuous_time.dynamics import drift_coordinated_turn_2d
from pytcl.dynamic_models.continuous_time.dynamics import drift_singer
from pytcl.dynamic_models.continuous_time.dynamics import state_jacobian_ca
from pytcl.dynamic_models.continuous_time.dynamics import state_jacobian_cv
from pytcl.dynamic_models.continuous_time.dynamics import state_jacobian_singer

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
