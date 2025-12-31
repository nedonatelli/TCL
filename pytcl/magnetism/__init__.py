"""
Magnetism models module.

This module provides implementations of geomagnetic field models including
the World Magnetic Model (WMM), International Geomagnetic Reference
Field (IGRF), and high-resolution models (EMM, WMMHR).

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

>>> # High-resolution models (require external coefficient files)
>>> from pytcl.magnetism import emm, wmmhr, create_emm_test_coefficients
>>> # Create test coefficients for demonstration
>>> coef = create_emm_test_coefficients(n_max=36)
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

from pytcl.magnetism.emm import (
    HighResCoefficients,
    EMM_PARAMETERS,
    get_data_dir as get_emm_data_dir,
    load_emm_coefficients,
    create_test_coefficients as create_emm_test_coefficients,
    emm,
    wmmhr,
    emm_declination,
    emm_inclination,
    emm_intensity,
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
    # EMM / WMMHR (high-resolution models)
    "HighResCoefficients",
    "EMM_PARAMETERS",
    "get_emm_data_dir",
    "load_emm_coefficients",
    "create_emm_test_coefficients",
    "emm",
    "wmmhr",
    "emm_declination",
    "emm_inclination",
    "emm_intensity",
]
