"""
Gravity models module.

This module provides functions for computing gravitational acceleration
and potential using various models including WGS84 normal gravity and
spherical harmonic expansions.

Examples
--------
>>> from pytcl.gravity import normal_gravity, gravity_wgs84
>>> import numpy as np

>>> # Normal gravity at 45° latitude, sea level
>>> g = normal_gravity(np.radians(45), 0)
>>> print(f"Gravity: {g:.4f} m/s^2")

>>> # Full gravity vector
>>> result = gravity_wgs84(np.radians(45), 0, 1000)
>>> print(f"Gravity magnitude: {result.magnitude:.4f} m/s^2")
"""

from pytcl.gravity.spherical_harmonics import (
    associated_legendre,
    associated_legendre_derivative,
    spherical_harmonic_sum,
    gravity_acceleration,
)

from pytcl.gravity.models import (
    GravityConstants,
    GravityResult,
    WGS84,
    GRS80,
    normal_gravity_somigliana,
    normal_gravity,
    gravity_wgs84,
    gravity_j2,
    geoid_height_j2,
    gravitational_potential,
    free_air_anomaly,
    bouguer_anomaly,
)

__all__ = [
    # Spherical harmonics
    "associated_legendre",
    "associated_legendre_derivative",
    "spherical_harmonic_sum",
    "gravity_acceleration",
    # Constants and types
    "GravityConstants",
    "GravityResult",
    "WGS84",
    "GRS80",
    # Gravity functions
    "normal_gravity_somigliana",
    "normal_gravity",
    "gravity_wgs84",
    "gravity_j2",
    "geoid_height_j2",
    "gravitational_potential",
    "free_air_anomaly",
    "bouguer_anomaly",
]
