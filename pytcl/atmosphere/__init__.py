"""
Atmospheric models for tracking applications.

This module provides standard atmosphere models used for computing
temperature, pressure, density, and other properties at various altitudes.
"""

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

__all__ = [
    "AtmosphereState",
    "us_standard_atmosphere_1976",
    "isa_atmosphere",
    "altitude_from_pressure",
    "mach_number",
    "true_airspeed_from_mach",
    # Constants
    "T0",
    "P0",
    "RHO0",
    "G0",
    "R",
    "GAMMA",
]
