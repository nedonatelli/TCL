"""
Atmospheric models for tracking applications.

This module provides standard atmosphere models used for computing
temperature, pressure, and density at various altitudes.
"""

import warnings
from typing import NamedTuple, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pytcl.core.constants import UNIVERSAL_GAS_CONSTANT


class AtmosphereState(NamedTuple):
    """
    Atmospheric state at a given altitude.

    Attributes
    ----------
    temperature : float or ndarray
        Temperature in Kelvin.
    pressure : float or ndarray
        Pressure in Pascals.
    density : float or ndarray
        Density in kg/m³.
    speed_of_sound : float or ndarray
        Speed of sound in m/s.
    """

    temperature: float | NDArray[np.float64]
    pressure: float | NDArray[np.float64]
    density: float | NDArray[np.float64]
    speed_of_sound: float | NDArray[np.float64]


# US Standard Atmosphere 1976 constants
# Sea level conditions
T0 = 288.15  # Temperature at sea level (K)
P0 = 101325.0  # Pressure at sea level (Pa)
RHO0 = 1.225  # Density at sea level (kg/m³)
G0 = 9.80665  # Standard gravity (m/s²)
R = 287.05287  # Specific gas constant for air (J/(kg·K))
GAMMA = 1.4  # Ratio of specific heats for air
R_EARTH_US76 = 6356766.0  # Earth radius adopted by US76 for geopotential height (m)

# Layer boundaries and lapse rates (altitude in m, lapse rate in K/m)
# Layer: (base altitude, base temperature, lapse rate)
US76_LAYERS = [
    (0, 288.15, -0.0065),  # Troposphere
    (11000, 216.65, 0.0),  # Tropopause
    (20000, 216.65, 0.001),  # Stratosphere 1
    (32000, 228.65, 0.0028),  # Stratosphere 2
    (47000, 270.65, 0.0),  # Stratopause
    (51000, 270.65, -0.0028),  # Mesosphere 1
    (71000, 214.65, -0.002),  # Mesosphere 2
    (84852, 186.95, 0.0),  # Mesopause (end of model)
]


def _get_layer(altitude: float) -> Tuple[int, float, float, float]:
    """Get layer parameters for given altitude."""
    for i, (h, T, L) in enumerate(US76_LAYERS):
        if i == len(US76_LAYERS) - 1:
            return i, h, T, L
        if altitude < US76_LAYERS[i + 1][0]:
            return i, h, T, L
    return len(US76_LAYERS) - 1, *US76_LAYERS[-1]


def us_standard_atmosphere_1976(
    altitude: ArrayLike,
) -> AtmosphereState:
    """
    Compute atmospheric properties using US Standard Atmosphere 1976.

    Parameters
    ----------
    altitude : array_like
        Geometric altitude in meters. Valid from 0 to ~86 km.

    Returns
    -------
    state : AtmosphereState
        Atmospheric state containing temperature, pressure, density,
        and speed of sound.

    Examples
    --------
    >>> state = us_standard_atmosphere_1976(10000)
    >>> round(state.temperature, 3)
    223.252
    >>> round(state.pressure, 1)
    26499.9

    Notes
    -----
    The US Standard Atmosphere 1976 is a model of the Earth's atmosphere
    that defines temperature, pressure, and density as functions of altitude.
    It is valid from sea level to approximately 86 km altitude.

    References
    ----------
    - U.S. Standard Atmosphere, 1976, U.S. Government Printing Office,
      Washington, D.C., 1976.
    """
    altitude = np.asarray(altitude, dtype=np.float64)
    scalar_input = altitude.ndim == 0
    altitude = np.atleast_1d(altitude)

    temperature = np.zeros_like(altitude)
    pressure = np.zeros_like(altitude)

    # Process each altitude point
    for i, z in enumerate(altitude):
        # The US76 layer table is defined in geopotential height
        h = R_EARTH_US76 * z / (R_EARTH_US76 + z)
        # Clamp altitude to valid range
        h = np.clip(h, 0, 84852)

        # Find which layer we're in
        layer_idx, h_base, T_base, L = _get_layer(h)

        # Calculate pressure at base of current layer
        P_base = P0
        for j in range(layer_idx):
            h_j, T_j, L_j = US76_LAYERS[j]
            h_next = US76_LAYERS[j + 1][0]
            dh = h_next - h_j

            if L_j != 0:
                # Gradient layer
                P_base *= (T_j / (T_j + L_j * dh)) ** (G0 / (R * L_j))
            else:
                # Isothermal layer
                P_base *= np.exp(-G0 * dh / (R * T_j))

        # Calculate temperature and pressure at altitude h
        dh = h - h_base

        if L != 0:
            # Gradient layer
            temperature[i] = T_base + L * dh
            pressure[i] = P_base * (T_base / temperature[i]) ** (G0 / (R * L))
        else:
            # Isothermal layer
            temperature[i] = T_base
            pressure[i] = P_base * np.exp(-G0 * dh / (R * T_base))

    # Calculate derived quantities
    density = pressure / (R * temperature)
    speed_of_sound = np.sqrt(GAMMA * R * temperature)

    if scalar_input:
        return AtmosphereState(
            temperature=float(temperature[0]),
            pressure=float(pressure[0]),
            density=float(density[0]),
            speed_of_sound=float(speed_of_sound[0]),
        )

    return AtmosphereState(
        temperature=temperature,
        pressure=pressure,
        density=density,
        speed_of_sound=speed_of_sound,
    )


def isa_atmosphere(
    altitude: ArrayLike,
    temperature_offset: float = 0.0,
) -> AtmosphereState:
    """
    Compute atmospheric properties using International Standard Atmosphere (ISA).

    This is essentially the troposphere portion of US Standard Atmosphere 1976
    with an optional temperature offset for non-standard days.

    Parameters
    ----------
    altitude : array_like
        Geometric altitude in meters.
    temperature_offset : float, optional
        Temperature offset from ISA conditions in Kelvin (default: 0).
        Positive values indicate warmer than standard day.

    Returns
    -------
    state : AtmosphereState
        Atmospheric state.

    Examples
    --------
    >>> # Standard day at 5000m
    >>> state = isa_atmosphere(5000)
    >>> # Hot day (+15K) at 5000m
    >>> state = isa_atmosphere(5000, temperature_offset=15)
    """
    altitude = np.asarray(altitude, dtype=np.float64)
    scalar_input = altitude.ndim == 0
    altitude = np.atleast_1d(altitude)
    # ISA lapse-rate formulas are defined in geopotential height
    altitude = R_EARTH_US76 * altitude / (R_EARTH_US76 + altitude)

    # Simple ISA model (troposphere + stratosphere)
    L = -0.0065  # Lapse rate in troposphere (K/m)
    h_trop = 11000  # Tropopause altitude (m)
    T_trop = T0 + L * h_trop  # Temperature at tropopause

    temperature = np.zeros_like(altitude)
    pressure = np.zeros_like(altitude)

    # Troposphere
    trop_mask = altitude <= h_trop
    temperature[trop_mask] = T0 + L * altitude[trop_mask] + temperature_offset
    # Barometric formula for gradient layer: P = P0 * (T0/T)^(g0/(R*L))
    # Since L is negative, g0/(R*L) is negative, so (T0/T)^negative = (T/T0)^positive
    pressure[trop_mask] = P0 * ((T0 + temperature_offset) / temperature[trop_mask]) ** (
        G0 / (R * L)
    )

    # Stratosphere (isothermal)
    strat_mask = altitude > h_trop
    temperature[strat_mask] = T_trop + temperature_offset
    # Pressure at tropopause
    P_trop = P0 * ((T0 + temperature_offset) / (T_trop + temperature_offset)) ** (
        G0 / (R * L)
    )
    pressure[strat_mask] = P_trop * np.exp(
        -G0 * (altitude[strat_mask] - h_trop) / (R * (T_trop + temperature_offset))
    )

    density = pressure / (R * temperature)
    speed_of_sound = np.sqrt(GAMMA * R * temperature)

    if scalar_input:
        return AtmosphereState(
            temperature=float(temperature[0]),
            pressure=float(pressure[0]),
            density=float(density[0]),
            speed_of_sound=float(speed_of_sound[0]),
        )

    return AtmosphereState(
        temperature=temperature,
        pressure=pressure,
        density=density,
        speed_of_sound=speed_of_sound,
    )


def altitude_from_pressure(
    pressure: ArrayLike,
) -> NDArray[np.float64]:
    """
    Compute geometric altitude from pressure (pressure altitude).

    Parameters
    ----------
    pressure : array_like
        Atmospheric pressure in Pascals.

    Returns
    -------
    altitude : ndarray
        Geometric altitude in meters.

    Examples
    --------
    >>> # Sea level pressure
    >>> bool(abs(altitude_from_pressure(101325)) < 1e-6)
    True
    >>> # Pressure at approximately 5000m
    >>> alt = altitude_from_pressure(54000)
    >>> 4800 < alt < 5200
    True

    Notes
    -----
    This is an approximate inversion of the ISA model, valid primarily
    in the troposphere.
    """
    pressure = np.asarray(pressure, dtype=np.float64)

    L = -0.0065  # Lapse rate
    exponent = -R * L / G0

    # Invert P = P0 * (T/T0)^(-g0/(R*L)) with T = T0 + L*h (geopotential),
    # then convert geopotential height back to geometric altitude
    h = (T0 / L) * ((pressure / P0) ** exponent - 1.0)
    altitude = R_EARTH_US76 * h / (R_EARTH_US76 - h)
    return altitude


def mach_number(
    velocity: ArrayLike,
    altitude: ArrayLike,
) -> NDArray[np.float64]:
    """
    Compute Mach number from velocity and altitude.

    Parameters
    ----------
    velocity : array_like
        True airspeed in m/s.
    altitude : array_like
        Geometric altitude in meters.

    Returns
    -------
    mach : ndarray
        Mach number.

    Examples
    --------
    >>> # Aircraft at 300 m/s at sea level
    >>> mach_number(300, 0)  # doctest: +ELLIPSIS
    0.88...
    >>> # Same speed at 10 km altitude (lower speed of sound)
    >>> mach_number(300, 10000)  # doctest: +ELLIPSIS
    1.00...
    """
    velocity = np.asarray(velocity, dtype=np.float64)
    altitude = np.asarray(altitude, dtype=np.float64)

    state = us_standard_atmosphere_1976(altitude)
    return velocity / np.asarray(state.speed_of_sound)


def true_airspeed_from_mach(
    mach: ArrayLike,
    altitude: ArrayLike,
) -> NDArray[np.float64]:
    """
    Compute true airspeed from Mach number and altitude.

    Parameters
    ----------
    mach : array_like
        Mach number.
    altitude : array_like
        Geometric altitude in meters.

    Returns
    -------
    velocity : ndarray
        True airspeed in m/s.

    Examples
    --------
    >>> # Mach 0.8 at cruise altitude (10 km)
    >>> tas = true_airspeed_from_mach(0.8, 10000)
    >>> 230 < tas < 250  # approximately 240 m/s
    True
    >>> # Supersonic at sea level
    >>> true_airspeed_from_mach(1.0, 0)  # doctest: +ELLIPSIS
    340.2...
    """
    mach = np.asarray(mach, dtype=np.float64)
    altitude = np.asarray(altitude, dtype=np.float64)

    state = us_standard_atmosphere_1976(altitude)
    return mach * np.asarray(state.speed_of_sound)


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


#: Speed of sound at standard temperature and pressure [m/s], the
#: reference value of Smith & Harlow (1963) returned by MATLAB's
#: ``speedOfSoundInAir`` when called with no arguments.
STANDARD_SPEED_OF_SOUND = 331.45


def speed_of_sound_ideal_gas(
    temperature: float,
    rel_humid: float = 0.0,
) -> float:
    """
    Speed of sound in air from the ideal-gas approximation.

    The simple approximation of Wong & Embleton, assuming standard
    pressure (101325 Pa). Derived for temperatures of 0-30 degrees
    Celsius; a warning is emitted outside that range.

    Port of ``speedOfSoundInAir.m`` algorithm 1. (Algorithm 0, which
    needs a gas-constituent table from NRLMSISE-00, is not ported; see
    the parity inventory.)

    Parameters
    ----------
    temperature : float
        Temperature in Kelvin.
    rel_humid : float, optional
        Relative humidity as a fraction in [0, 1]. Default 0.

    Returns
    -------
    c : float
        Speed of sound in meters per second.

    References
    ----------
    - G. S. K. Wong and T. F. W. Embleton, "Variation of the speed of
      sound in air with humidity and temperature," Journal of the
      Acoustical Society of America, vol. 77, no. 5, May 1985.

    Examples
    --------
    >>> round(speed_of_sound_ideal_gas(293.15, 0.5), 4)
    343.8478
    """
    t = temperature - 273.15
    if t < 0 or t > 30:
        warnings.warn(
            "The temperature supplied is outside of the range used "
            "(0-30 degrees C) in the paper deriving the ideal gas "
            "approximation. The results might have reduced accuracy.",
            stacklevel=2,
        )
    # Wong & Embleton Equation 4.
    a_t = 9.2e-5 + 5.5e-6 * t + 4.25e-7 * t**2
    # Equation 3 in moles per gram, converted to moles per kilogram.
    gamma_over_m = (0.04833 + (rel_humid - 0.023) * a_t) * 1000.0
    # Equation 1.
    return float(np.sqrt(gamma_over_m * UNIVERSAL_GAS_CONSTANT * temperature))


def speed_of_sound_cramer(
    temperature: float,
    pressure: float,
    h2o_fraction: float = 0.0,
    co2_fraction: float = 0.0,
) -> float:
    """
    Speed of sound in air from Cramer's polynomial approximation.

    Valid for 0-30 degrees Celsius, 75-102 kPa, water-vapor mole
    fractions up to 0.06 and CO2 mole fractions up to 0.01; warnings are
    emitted outside those ranges.

    Port of ``speedOfSoundInAir.m`` algorithm 2, transcribed exactly:
    note that the MATLAB source evaluates the ``a12`` CO2 term with the
    temperature in Kelvin squared where Cramer's Equation 15 uses
    Celsius, so CO2-laden results follow MATLAB, not the paper.

    Parameters
    ----------
    temperature : float
        Temperature in Kelvin.
    pressure : float
        Pressure in Pascals.
    h2o_fraction : float, optional
        Water-vapor mole fraction. Default 0 (dry air).
    co2_fraction : float, optional
        Carbon-dioxide mole fraction. Default 0.

    Returns
    -------
    c : float
        Speed of sound in meters per second.

    References
    ----------
    - O. Cramer, "The variation of the specific heat ratio and the speed
      of sound in air with temperature, pressure, humidity, and CO2
      concentration," Journal of the Acoustical Society of America,
      vol. 93, no. 5, pp. 2510-2516, May 1993.

    Examples
    --------
    >>> round(speed_of_sound_cramer(293.15, 101325.0, 0.01), 4)
    343.9366
    """
    t = temperature - 273.15
    if t < 0 or t > 30:
        warnings.warn(
            "The temperature supplied is outside of the range used "
            "(0-30 degrees C) in the paper deriving the speed "
            "approximation. The results might have reduced accuracy.",
            stacklevel=2,
        )
    if pressure < 75000 or pressure > 102000:
        warnings.warn(
            "The pressure supplied is outside of the range used "
            "(75000-102000 Pa) in the paper deriving the speed "
            "approximation. The results might have reduced accuracy.",
            stacklevel=2,
        )
    if h2o_fraction < 0:
        raise ValueError("Invalid mole fraction of water provided.")
    if h2o_fraction > 0.06:
        warnings.warn(
            "The water-vapor mole fraction is outside of the range used "
            "(0-0.06) in the paper deriving the speed approximation. "
            "The results might have reduced accuracy.",
            stacklevel=2,
        )
    if co2_fraction < 0:
        raise ValueError("Invalid mole fraction of carbon dioxide provided.")
    if co2_fraction > 0.01:
        warnings.warn(
            "The CO2 mole fraction is outside of the range used (0-0.01) "
            "in the paper deriving the speed approximation. The results "
            "might have reduced accuracy.",
            stacklevel=2,
        )

    # Cramer Table III coefficients for Equation 15.
    a = [
        331.5024,
        0.603055,
        -0.000528,
        51.471935,
        0.1495874,
        -0.000782,
        -1.82e-7,
        3.73e-8,
        -2.93e-10,
        -85.20931,
        -0.228525,
        5.91e-5,
        -2.835149,
        -2.15e-13,
        29.179762,
        0.000486,
    ]
    p = pressure
    xw = h2o_fraction
    xc = co2_fraction
    # Equation 15. The a[11] term uses Kelvin squared, matching the
    # MATLAB source (Cramer's paper uses Celsius there).
    return float(
        a[0]
        + a[1] * t
        + a[2] * t**2
        + (a[3] + a[4] * t + a[5] * t**2) * xw
        + (a[6] + a[7] * t + a[8] * t**2) * p
        + (a[9] + a[10] * t + a[11] * temperature**2) * xc
        + a[12] * xw**2
        + a[13] * p**2
        + a[14] * xc**2
        + a[15] * xw * p * xc
    )
