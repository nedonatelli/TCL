"""
Atmospheric models for tracking applications.

This module provides standard atmosphere models used for computing
temperature, pressure, density, and other properties at various altitudes.
"""

from tracker_component_library.atmosphere.models import (
    AtmosphereState,
    us_standard_atmosphere_1976,
    isa_atmosphere,
    altitude_from_pressure,
    mach_number,
    true_airspeed_from_mach,
    # Constants
    T0,
    P0,
    RHO0,
    G0,
    R,
    GAMMA,
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
