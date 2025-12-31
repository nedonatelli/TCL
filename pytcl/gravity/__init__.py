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

from pytcl.gravity.clenshaw import (
    clenshaw_gravity,
    clenshaw_potential,
    clenshaw_sum_order,
    clenshaw_sum_order_derivative,
)
from pytcl.gravity.egm import (
    EGMCoefficients,
    GeoidResult,
    GravityDisturbance,
    create_test_coefficients,
    deflection_of_vertical,
    geoid_height,
    geoid_heights,
    get_data_dir,
    gravity_anomaly,
    gravity_disturbance,
    load_egm_coefficients,
)
from pytcl.gravity.models import (
    GRS80,
    WGS84,
    GravityConstants,
    GravityResult,
    bouguer_anomaly,
    free_air_anomaly,
    geoid_height_j2,
    gravitational_potential,
    gravity_j2,
    gravity_wgs84,
    normal_gravity,
    normal_gravity_somigliana,
)
from pytcl.gravity.spherical_harmonics import (
    associated_legendre,
    associated_legendre_derivative,
    associated_legendre_scaled,
    gravity_acceleration,
    legendre_scaling_factors,
    spherical_harmonic_sum,
)
from pytcl.gravity.tides import (
    GRAVIMETRIC_FACTOR,
    LOVE_H2,
    LOVE_H3,
    LOVE_K2,
    LOVE_K3,
    SHIDA_L2,
    SHIDA_L3,
    TIDAL_CONSTITUENTS,
    OceanTideLoading,
    TidalDisplacement,
    TidalGravity,
    atmospheric_pressure_loading,
    fundamental_arguments,
    julian_centuries_j2000,
    moon_position_approximate,
    ocean_tide_loading_displacement,
    pole_tide_displacement,
    solid_earth_tide_displacement,
    solid_earth_tide_gravity,
    sun_position_approximate,
    tidal_gravity_correction,
    total_tidal_displacement,
)

__all__ = [
    # Spherical harmonics
    "associated_legendre",
    "associated_legendre_derivative",
    "spherical_harmonic_sum",
    "gravity_acceleration",
    "legendre_scaling_factors",
    "associated_legendre_scaled",
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
    # Clenshaw summation (high-degree spherical harmonics)
    "clenshaw_sum_order",
    "clenshaw_sum_order_derivative",
    "clenshaw_potential",
    "clenshaw_gravity",
    # EGM models (EGM96/EGM2008)
    "EGMCoefficients",
    "GeoidResult",
    "GravityDisturbance",
    "get_data_dir",
    "load_egm_coefficients",
    "geoid_height",
    "geoid_heights",
    "gravity_disturbance",
    "gravity_anomaly",
    "deflection_of_vertical",
    "create_test_coefficients",
    # Tidal effects
    "TidalDisplacement",
    "TidalGravity",
    "OceanTideLoading",
    "LOVE_H2",
    "LOVE_K2",
    "SHIDA_L2",
    "LOVE_H3",
    "LOVE_K3",
    "SHIDA_L3",
    "GRAVIMETRIC_FACTOR",
    "TIDAL_CONSTITUENTS",
    "julian_centuries_j2000",
    "fundamental_arguments",
    "moon_position_approximate",
    "sun_position_approximate",
    "solid_earth_tide_displacement",
    "solid_earth_tide_gravity",
    "ocean_tide_loading_displacement",
    "atmospheric_pressure_loading",
    "pole_tide_displacement",
    "total_tidal_displacement",
    "tidal_gravity_correction",
]
