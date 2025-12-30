"""
Magnetism models module.

This module provides implementations of geomagnetic field models including
the World Magnetic Model (WMM) and International Geomagnetic Reference
Field (IGRF).

Examples
--------
>>> from pytcl.magnetism import wmm, magnetic_declination
>>> import numpy as np

>>> # Magnetic field at a location
>>> result = wmm(np.radians(40), np.radians(-105), 1.0, 2023.0)
>>> print(f"Declination: {np.degrees(result.D):.2f}°")
>>> print(f"Total intensity: {result.F:.0f} nT")

>>> # Just the declination
>>> D = magnetic_declination(np.radians(40), np.radians(-105))
>>> print(f"Declination: {np.degrees(D):.2f}°")
"""

from pytcl.magnetism.wmm import (
    MagneticResult,
    MagneticCoefficients,
    WMM2020,
    create_wmm2020_coefficients,
    magnetic_field_spherical,
    wmm,
    magnetic_declination,
    magnetic_inclination,
    magnetic_field_intensity,
)

from pytcl.magnetism.igrf import (
    IGRFModel,
    IGRF13,
    create_igrf13_coefficients,
    igrf,
    igrf_declination,
    igrf_inclination,
    dipole_moment,
    dipole_axis,
    magnetic_north_pole,
)

__all__ = [
    # Types and constants
    "MagneticResult",
    "MagneticCoefficients",
    "IGRFModel",
    # WMM
    "WMM2020",
    "create_wmm2020_coefficients",
    "magnetic_field_spherical",
    "wmm",
    "magnetic_declination",
    "magnetic_inclination",
    "magnetic_field_intensity",
    # IGRF
    "IGRF13",
    "create_igrf13_coefficients",
    "igrf",
    "igrf_declination",
    "igrf_inclination",
    "dipole_moment",
    "dipole_axis",
    "magnetic_north_pole",
]
