"""
Process noise covariance matrices.

This module provides process noise covariance matrices (Q) for various
motion models used in target tracking applications.
"""

from pytcl.dynamic_models.process_noise.coordinated_turn import (
    q_coord_turn_2d,
    q_coord_turn_3d,
    q_coord_turn_polar,
)
from pytcl.dynamic_models.process_noise.polynomial import (
    q_constant_acceleration,
    q_constant_velocity,
    q_continuous_white_noise,
    q_discrete_white_noise,
    q_poly_kal,
)
from pytcl.dynamic_models.process_noise.singer import q_singer, q_singer_2d, q_singer_3d

__all__ = [
    # Polynomial models
    "q_poly_kal",
    "q_discrete_white_noise",
    "q_constant_velocity",
    "q_constant_acceleration",
    "q_continuous_white_noise",
    # Singer model
    "q_singer",
    "q_singer_2d",
    "q_singer_3d",
    # Coordinated turn models
    "q_coord_turn_2d",
    "q_coord_turn_3d",
    "q_coord_turn_polar",
]
