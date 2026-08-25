"""
Atmospheric models for tracking applications.

This module provides standard atmosphere models used for computing
temperature, pressure, density, and other properties at various altitudes.

Submodules
----------
models : Standard atmosphere models (US76, ISA)
ionosphere : Ionospheric models for GPS/GNSS corrections
humidity : Humidity conversions and dew-point calculations
refraction : Atmospheric refractivity helpers
"""

from pytcl.atmosphere.humidity import (
    H2O_MOLAR_MASS,
    abs_humid_to_number_density,
    abs_humid_to_rel_humid,
    abs_humid_to_spec_humid,
    dew_point_pressure,
    dew_point_temperature,
    number_density_to_abs_humid,
    rel_humid_to_abs_humid,
    rel_humid_to_spec_humid,
    spec_humid_to_abs_humid,
    spec_humid_to_rel_humid,
)
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
from pytcl.atmosphere.models import (
    G0,  # Constants
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
from pytcl.atmosphere.refraction import (
    AstroRefParams,
    AstroRefractionResult,
    ExpDecayConstResult,
    SinclairAtmosResult,
    add_astro_refraction,
    approx_refractivity,
    atmos_exp_decay_const,
    remove_astro_refraction,
    simple_astro_ref_params,
    sinclair_atmosphere,
)
from pytcl.atmosphere.thermosphere import (
    F107Index,
    SimplifiedThermosphere,
    ThermosphereState,
    simplified_thermosphere,
)

__all__ = [
    # Atmosphere state and models
    "AtmosphereState",
    "us_standard_atmosphere_1976",
    "isa_atmosphere",
    "altitude_from_pressure",
    "mach_number",
    "true_airspeed_from_mach",
    # Simplified thermosphere model (see gh-79 for its accuracy envelope)
    "SimplifiedThermosphere",
    "ThermosphereState",
    "F107Index",
    "simplified_thermosphere",
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
    # Humidity and dew point
    "H2O_MOLAR_MASS",
    "abs_humid_to_number_density",
    "abs_humid_to_rel_humid",
    "abs_humid_to_spec_humid",
    "dew_point_pressure",
    "dew_point_temperature",
    "number_density_to_abs_humid",
    "rel_humid_to_abs_humid",
    "rel_humid_to_spec_humid",
    "spec_humid_to_abs_humid",
    "spec_humid_to_rel_humid",
    # Refractivity and astronomical refraction
    "ExpDecayConstResult",
    "approx_refractivity",
    "atmos_exp_decay_const",
    "AstroRefParams",
    "AstroRefractionResult",
    "SinclairAtmosResult",
    "add_astro_refraction",
    "remove_astro_refraction",
    "simple_astro_ref_params",
    "sinclair_atmosphere",
]
