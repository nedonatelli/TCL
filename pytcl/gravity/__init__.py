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

from pytcl.gravity.clenshaw import clenshaw_gravity
from pytcl.gravity.clenshaw import clenshaw_potential
from pytcl.gravity.clenshaw import clenshaw_sum_order
from pytcl.gravity.clenshaw import clenshaw_sum_order_derivative
from pytcl.gravity.egm import EGMCoefficients
from pytcl.gravity.egm import GeoidResult
from pytcl.gravity.egm import GravityDisturbance
from pytcl.gravity.egm import create_test_coefficients
from pytcl.gravity.egm import deflection_of_vertical
from pytcl.gravity.egm import geoid_height
from pytcl.gravity.egm import geoid_heights
from pytcl.gravity.egm import get_data_dir
from pytcl.gravity.egm import gravity_anomaly
from pytcl.gravity.egm import gravity_disturbance
from pytcl.gravity.egm import load_egm_coefficients
from pytcl.gravity.models import GRS80
from pytcl.gravity.models import WGS84
from pytcl.gravity.models import GravityConstants
from pytcl.gravity.models import GravityResult
from pytcl.gravity.models import bouguer_anomaly
from pytcl.gravity.models import free_air_anomaly
from pytcl.gravity.models import geoid_height_j2
from pytcl.gravity.models import gravitational_potential
from pytcl.gravity.models import gravity_j2
from pytcl.gravity.models import gravity_wgs84
from pytcl.gravity.models import normal_gravity
from pytcl.gravity.models import normal_gravity_somigliana
from pytcl.gravity.spherical_harmonics import associated_legendre
from pytcl.gravity.spherical_harmonics import associated_legendre_derivative
from pytcl.gravity.spherical_harmonics import associated_legendre_scaled
from pytcl.gravity.spherical_harmonics import gravity_acceleration
from pytcl.gravity.spherical_harmonics import legendre_scaling_factors
from pytcl.gravity.spherical_harmonics import spherical_harmonic_sum
from pytcl.gravity.tides import GRAVIMETRIC_FACTOR
from pytcl.gravity.tides import LOVE_H2
from pytcl.gravity.tides import LOVE_H3
from pytcl.gravity.tides import LOVE_K2
from pytcl.gravity.tides import LOVE_K3
from pytcl.gravity.tides import SHIDA_L2
from pytcl.gravity.tides import SHIDA_L3
from pytcl.gravity.tides import TIDAL_CONSTITUENTS
from pytcl.gravity.tides import OceanTideLoading
from pytcl.gravity.tides import TidalDisplacement
from pytcl.gravity.tides import TidalGravity
from pytcl.gravity.tides import atmospheric_pressure_loading
from pytcl.gravity.tides import fundamental_arguments
from pytcl.gravity.tides import julian_centuries_j2000
from pytcl.gravity.tides import moon_position_approximate
from pytcl.gravity.tides import ocean_tide_loading_displacement
from pytcl.gravity.tides import pole_tide_displacement
from pytcl.gravity.tides import solid_earth_tide_displacement
from pytcl.gravity.tides import solid_earth_tide_gravity
from pytcl.gravity.tides import sun_position_approximate
from pytcl.gravity.tides import tidal_gravity_correction
from pytcl.gravity.tides import total_tidal_displacement

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
