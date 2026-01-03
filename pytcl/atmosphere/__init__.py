"""
Atmospheric models for tracking applications.

This module provides standard atmosphere models used for computing
temperature, pressure, density, and other properties at various altitudes.

Submodules
----------
models : Standard atmosphere models (US76, ISA)
ionosphere : Ionospheric models for GPS/GNSS corrections
"""

from pytcl.atmosphere.ionosphere import (
    DEFAULT_KLOBUCHAR,
    F_L1,
    F_L2,
    IonosphereState,
    KlobucharCoefficients,
    dual_frequency_tec,
    ionospheric_delay_from_tec,
    klobuchar_delay,
    magnetic_latitude,
    scintillation_index,
    simple_iri,
)
from pytcl.atmosphere.models import G0  # Constants
from pytcl.atmosphere.models import (
    GAMMA,
    P0,
    RHO0,
    T0,
    AtmosphereState,
    R,
    altitude_from_pressure,
    isa_atmosphere,
    mach_number,
    true_airspeed_from_mach,
    us_standard_atmosphere_1976,
)
from pytcl.atmosphere.nrlmsise00 import (
    NRLMSISE00,
    F107Index,
    NRLMSISE00Output,
    nrlmsise00,
)

__all__ = [
    # Atmosphere state and models
    "AtmosphereState",
    "us_standard_atmosphere_1976",
    "isa_atmosphere",
    "altitude_from_pressure",
    "mach_number",
    "true_airspeed_from_mach",
    # NRLMSISE-00 High-Fidelity Model
    "NRLMSISE00",
    "NRLMSISE00Output",
    "F107Index",
    "nrlmsise00",
    # Atmosphere constants
    "T0",
    "P0",
    "RHO0",
    "G0",
    "R",
    "GAMMA",
    # Ionosphere
    "IonosphereState",
    "KlobucharCoefficients",
    "DEFAULT_KLOBUCHAR",
    "klobuchar_delay",
    "dual_frequency_tec",
    "ionospheric_delay_from_tec",
    "simple_iri",
    "magnetic_latitude",
    "scintillation_index",
    "F_L1",
    "F_L2",
]
