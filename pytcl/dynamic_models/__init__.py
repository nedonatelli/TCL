"""
Dynamic motion models for target tracking.

This module provides state transition matrices (F) and process noise
covariance matrices (Q) for various motion models:
- Constant velocity (CV)
- Constant acceleration (CA)
- Coordinated turn (2D and 3D)
- Singer acceleration model
- Nearly constant velocity/acceleration

It also provides continuous-time dynamics (drift and diffusion functions)
and utilities for discretizing continuous-time models.
"""

# Import submodules for easy access
from pytcl.dynamic_models import continuous_time, discrete_time, process_noise

# Continuous-time dynamics
from pytcl.dynamic_models.continuous_time import (
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

# Discrete-time state transition matrices
from pytcl.dynamic_models.discrete_time import (
    f_constant_acceleration,
    f_constant_velocity,
    f_coord_turn_2d,
    f_coord_turn_3d,
    f_coord_turn_polar,
    f_discrete_white_noise_accel,
    f_piecewise_white_noise_jerk,
    f_poly_kal,
    f_singer,
    f_singer_2d,
    f_singer_3d,
)

# Process noise covariance matrices
from pytcl.dynamic_models.process_noise import (
    q_constant_acceleration,
    q_constant_velocity,
    q_continuous_white_noise,
    q_coord_turn_2d,
    q_coord_turn_3d,
    q_coord_turn_polar,
    q_discrete_white_noise,
    q_poly_kal,
    q_singer,
    q_singer_2d,
    q_singer_3d,
)

# Re-export commonly used functions at the top level


__all__ = [
    # Submodules
    "discrete_time",
    "process_noise",
    "continuous_time",
    # Discrete-time F matrices
    "f_poly_kal",
    "f_constant_velocity",
    "f_constant_acceleration",
    "f_discrete_white_noise_accel",
    "f_piecewise_white_noise_jerk",
    "f_coord_turn_2d",
    "f_coord_turn_3d",
    "f_coord_turn_polar",
    "f_singer",
    "f_singer_2d",
    "f_singer_3d",
    # Process noise Q matrices
    "q_poly_kal",
    "q_discrete_white_noise",
    "q_constant_velocity",
    "q_constant_acceleration",
    "q_continuous_white_noise",
    "q_singer",
    "q_singer_2d",
    "q_singer_3d",
    "q_coord_turn_2d",
    "q_coord_turn_3d",
    "q_coord_turn_polar",
    # Continuous-time dynamics
    "drift_constant_velocity",
    "drift_constant_acceleration",
    "drift_singer",
    "drift_coordinated_turn_2d",
    "diffusion_constant_velocity",
    "diffusion_constant_acceleration",
    "diffusion_singer",
    "continuous_to_discrete",
    "discretize_lti",
    "state_jacobian_cv",
    "state_jacobian_ca",
    "state_jacobian_singer",
]
