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
Declination: 7.83°
>>> print(f"Total intensity: {result.F:.0f} nT")
Total intensity: 51573 nT

>>> # Just the declination
>>> D = magnetic_declination(np.radians(40), np.radians(-105))
>>> print(f"Declination: {np.degrees(D):.2f}°")
Declination: 7.67°

>>> # High-resolution models (require external coefficient files)
>>> from pytcl.magnetism import emm, wmmhr, create_emm_test_coefficients
>>> # Create test coefficients for demonstration
>>> coef = create_emm_test_coefficients(n_max=36)
"""

from pytcl.magnetism.emm import (
    EMM_PARAMETERS,
    HighResCoefficients,
    emm,
    emm_declination,
    emm_inclination,
    emm_intensity,
    load_emm_coefficients,
    parse_emm_file,
    wmmhr,
)
from pytcl.magnetism.emm import create_test_coefficients as create_emm_test_coefficients
from pytcl.magnetism.emm import get_data_dir as get_emm_data_dir
from pytcl.magnetism.igrf import (
    IGRF13,
    IGRF14,
    IGRFModel,
    create_igrf13_coefficients,
    create_igrf14_coefficients,
    dipole_axis,
    dipole_moment,
    igrf,
    igrf_declination,
    igrf_inclination,
    magnetic_north_pole,
)
from pytcl.magnetism.wmm import (
    WMM2020,
    WMM2025,
    MagneticCoefficients,
    MagneticResult,
    clear_magnetic_cache,
    configure_magnetic_cache,
    create_wmm2020_coefficients,
    create_wmm2025_coefficients,
    get_magnetic_cache_info,
    magnetic_declination,
    magnetic_field_intensity,
    magnetic_field_spherical,
    magnetic_inclination,
    wmm,
)

__all__ = [
    # Data-file parsing, for callers supplying their own coefficient set.
    "parse_emm_file",
    # Types and constants
    "MagneticResult",
    "MagneticCoefficients",
    "IGRFModel",
    # WMM
    "WMM2020",
    "WMM2025",
    "create_wmm2020_coefficients",
    "create_wmm2025_coefficients",
    "magnetic_field_spherical",
    "wmm",
    "magnetic_declination",
    "magnetic_inclination",
    "magnetic_field_intensity",
    # Cache management
    "get_magnetic_cache_info",
    "clear_magnetic_cache",
    "configure_magnetic_cache",
    # IGRF
    "IGRF13",
    "IGRF14",
    "create_igrf13_coefficients",
    "create_igrf14_coefficients",
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
