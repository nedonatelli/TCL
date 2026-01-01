"""
Discrete-time state transition models.

This module provides state transition matrices (F) for various motion models
used in target tracking applications.
"""

from pytcl.dynamic_models.discrete_time.coordinated_turn import (
    f_coord_turn_2d,
    f_coord_turn_3d,
    f_coord_turn_polar,
)
from pytcl.dynamic_models.discrete_time.polynomial import (
    f_constant_acceleration,
    f_constant_velocity,
    f_discrete_white_noise_accel,
    f_piecewise_white_noise_jerk,
    f_poly_kal,
)
from pytcl.dynamic_models.discrete_time.singer import f_singer, f_singer_2d, f_singer_3d

__all__ = [
    # Polynomial models
    "f_poly_kal",
    "f_constant_velocity",
    "f_constant_acceleration",
    "f_discrete_white_noise_accel",
    "f_piecewise_white_noise_jerk",
    # Coordinated turn models
    "f_coord_turn_2d",
    "f_coord_turn_3d",
    "f_coord_turn_polar",
    # Singer model
    "f_singer",
    "f_singer_2d",
    "f_singer_3d",
]
