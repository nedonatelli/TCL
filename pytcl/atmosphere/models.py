"""
Atmospheric models for tracking applications.

This module provides standard atmosphere models used for computing
temperature, pressure, and density at various altitudes.
"""

from typing import NamedTuple, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray


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
    >>> state.temperature
    223.25...
    >>> state.pressure
    26499.9...

    Notes
    -----
    The US Standard Atmosphere 1976 is a model of the Earth's atmosphere
    that defines temperature, pressure, and density as functions of altitude.
    It is valid from sea level to approximately 86 km altitude.

    References
    ----------
    .. [1] U.S. Standard Atmosphere, 1976, U.S. Government Printing Office,
           Washington, D.C., 1976.
    """
    altitude = np.asarray(altitude, dtype=np.float64)
    scalar_input = altitude.ndim == 0
    altitude = np.atleast_1d(altitude)

    temperature = np.zeros_like(altitude)
    pressure = np.zeros_like(altitude)

    # Process each altitude point
    for i, h in enumerate(altitude):
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
    >>> altitude_from_pressure(101325)
    0.0
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
    exponent = R * L / G0

    altitude = (T0 / L) * (1 - (pressure / P0) ** exponent)
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
