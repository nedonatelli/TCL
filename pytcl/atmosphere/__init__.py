"""
Atmospheric models for tracking applications.

This module provides standard atmosphere models used for computing
temperature, pressure, density, and other properties at various altitudes.
"""

from pytcl.atmosphere.models import G0  # Constants
from pytcl.atmosphere.models import GAMMA
from pytcl.atmosphere.models import P0
from pytcl.atmosphere.models import RHO0
from pytcl.atmosphere.models import T0
from pytcl.atmosphere.models import AtmosphereState
from pytcl.atmosphere.models import R
from pytcl.atmosphere.models import altitude_from_pressure
from pytcl.atmosphere.models import isa_atmosphere
from pytcl.atmosphere.models import mach_number
from pytcl.atmosphere.models import true_airspeed_from_mach
from pytcl.atmosphere.models import us_standard_atmosphere_1976

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
