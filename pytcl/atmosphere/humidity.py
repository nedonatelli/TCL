"""
Humidity conversions and dew-point calculations.

Ports of the humidity functions from the MATLAB Tracker Component Library's
``Atmosphere_and_Refraction`` directory: pairwise conversions between
absolute, relative and specific humidity, water number density, and the
saturation (dew-point) pressure/temperature of water.

Conventions
-----------
- Temperatures in Kelvin, pressures in Pascals.
- Relative humidity is a fraction in [0, 1], not a percent.
- Absolute humidity is kilograms of water per cubic meter of air.
- Specific humidity is dimensionless; see the ``definition`` parameter.

The dew-point algorithms are shared by every function that touches relative
humidity:

- ``0`` — corrected Clausius-Clapeyron equation (Koutsoyiannis 2012), for
  use over land or in the upper air.
- ``1`` — Magnus-type equation over water (Alduchov & Eskridge 1996),
  valid for -40 C to +50 C.
- ``2`` — Magnus-type equation over ice (Alduchov & Eskridge 1996),
  valid for -80 C to 0 C.

References
----------
- D. Koutsoyiannis, "Clausius-Clapeyron equation and saturation vapour
  pressure: simple theory reconciled with practice," European Journal of
  Physics, vol. 33, no. 2, pp. 295-305, Mar. 2012.
- O. A. Alduchov and R. E. Eskridge, "Improved Magnus form approximation
  of saturation vapor pressure," Journal of Applied Meteorology, vol. 35,
  no. 4, pp. 601-609, Apr. 1996.
"""

from typing import Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pytcl.core.constants import (
    ATOMIC_MASS_UNIT,
    UNIVERSAL_GAS_CONSTANT,
)

__all__ = [
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
]

#: Molar mass of water [g/mol], 2*H + O from the 2013 CIAAW standard atomic
#: weights (interval midpoints), matching the MATLAB TCL ``Constants`` class.
H2O_MOLAR_MASS: float = 2 * 1.007975 + 15.9994

_ABSOLUTE_ZERO_C = -273.15


def dew_point_pressure(
    temperature: ArrayLike,
    algorithm: int = 0,
) -> Union[float, NDArray[np.floating]]:
    """
    Saturation partial pressure of water for a given temperature.

    This is the partial vapor pressure of water that (in equilibrium)
    cannot be exceeded — the dew-point pressure. Above it, water begins to
    condense out of gaseous form at this temperature.

    Port of ``dewPointPres4Temp.m``.

    Parameters
    ----------
    temperature : array_like
        Temperature(s) in Kelvin.
    algorithm : int, optional
        ``0`` (default) corrected Clausius-Clapeyron equation; ``1``
        Magnus-type over water (-40 C to +50 C); ``2`` Magnus-type over
        ice (-80 C to 0 C). See the module docstring.

    Returns
    -------
    pressure : float or ndarray
        Saturation pressure(s) of water in Pascals.

    Examples
    --------
    >>> round(float(dew_point_pressure(288.15)), 4)
    1706.632
    >>> round(float(dew_point_pressure(288.15, algorithm=1)), 4)
    1701.9828
    >>> round(float(dew_point_pressure(263.15, algorithm=2)), 4)
    259.6718
    """
    T = np.asarray(temperature, dtype=np.float64)
    if algorithm == 1:
        # Alduchov & Eskridge Eq. 21 (over water); hectopascals.
        TC = T + _ABSOLUTE_ZERO_C
        p = 6.1094 * np.exp(17.625 * TC / (243.04 + TC))
    elif algorithm == 2:
        # Alduchov & Eskridge Eq. 23 (over ice); hectopascals.
        TC = T + _ABSOLUTE_ZERO_C
        p = 6.1121 * np.exp(22.587 * TC / (273.86 + TC))
    elif algorithm == 0:
        # Koutsoyiannis Eq. 23 with the paper's tweaked constants for water
        # vapor; p0, T0 are the triple point of water (hPa, K).
        p0 = 6.11657
        T0 = 273.16
        p = p0 * np.exp(24.921 * (1 - T0 / T)) * (T0 / T) ** 5.06
    else:
        raise ValueError(f"algorithm must be 0, 1 or 2, got {algorithm}")
    p = 100.0 * p  # hectopascals -> Pascals
    return float(p) if p.ndim == 0 else p


def dew_point_temperature(
    pressure: ArrayLike,
    algorithm: int = 0,
) -> Union[float, NDArray[np.floating]]:
    """
    Temperature at which a partial vapor pressure of water saturates.

    For a given partial pressure of water, find the temperature at which
    that pressure is the saturation pressure — the dew-point temperature.
    Inverse of :func:`dew_point_pressure`.

    Port of ``dewPointTemp4Pres.m``.

    Parameters
    ----------
    pressure : array_like
        Partial vapor pressure(s) of water in Pascals.
    algorithm : int, optional
        ``0`` (default) corrected Clausius-Clapeyron equation (inverted by
        the fixed-point iteration of Koutsoyiannis Eqs. 44-45, 27
        iterations); ``1`` Magnus-type over water; ``2`` Magnus-type over
        ice. See the module docstring.

    Returns
    -------
    temperature : float or ndarray
        Dew-point temperature(s) in Kelvin.

    Examples
    --------
    >>> round(float(dew_point_temperature(1706.632)), 4)
    288.15
    >>> round(float(dew_point_temperature(1701.9828, algorithm=1)), 4)
    288.15
    """
    p = np.asarray(pressure, dtype=np.float64)
    if algorithm == 1:
        # Inverse of Alduchov & Eskridge Eq. 21.
        log_rat = np.log(p / 100.0 / 6.1094)
        T = -243.04 * log_rat / (log_rat - 17.625) - _ABSOLUTE_ZERO_C
    elif algorithm == 2:
        # Inverse of Alduchov & Eskridge Eq. 23 (over ice).
        log_rat = np.log(p / 100.0 / 6.1121)
        T = -273.86 * log_rat / (log_rat - 22.587) - _ABSOLUTE_ZERO_C
    elif algorithm == 0:
        # Koutsoyiannis Eqs. 44-45 fixed-point iteration with the paper's
        # tweaked constants; 27 iterations as in the MATLAB source.
        p0 = 6.11657 * 100.0
        T0 = 273.16
        lp_rat = np.log(p0 / p)
        T_rat = 1 + 1 / (24.921 - 5.06) * lp_rat
        for _ in range(27):
            T_rat = 1 + (1 / 24.921) * lp_rat + (5.06 / 24.921) * np.log(T_rat)
        T = T0 / T_rat
    else:
        raise ValueError(f"algorithm must be 0, 1 or 2, got {algorithm}")
    return float(T) if T.ndim == 0 else T


def abs_humid_to_number_density(
    abs_humid: ArrayLike,
) -> Union[float, NDArray[np.floating]]:
    """
    Number density of water molecules from absolute humidity.

    Port of ``absHumid2NumberDensH2O.m``.

    Parameters
    ----------
    abs_humid : array_like
        Absolute humidity in kilograms of water per cubic meter of air.

    Returns
    -------
    number_density : float or ndarray
        Number of water molecules per cubic meter of air.

    Examples
    --------
    >>> nd = abs_humid_to_number_density(0.01)
    >>> print(f"{nd:.6e}")
    3.342783e+23
    """
    nd = np.asarray(abs_humid, dtype=np.float64) / (ATOMIC_MASS_UNIT * H2O_MOLAR_MASS)
    return float(nd) if nd.ndim == 0 else nd


def number_density_to_abs_humid(
    number_density: ArrayLike,
) -> Union[float, NDArray[np.floating]]:
    """
    Absolute humidity from the number density of water molecules.

    Port of ``numberDensH2O2AbsHumid.m``.

    Parameters
    ----------
    number_density : array_like
        Number of water molecules per cubic meter of air.

    Returns
    -------
    abs_humid : float or ndarray
        Absolute humidity in kilograms of water per cubic meter of air.

    Examples
    --------
    >>> round(number_density_to_abs_humid(3.342783e+23), 8)
    0.01
    """
    ah = (
        np.asarray(number_density, dtype=np.float64) * ATOMIC_MASS_UNIT * H2O_MOLAR_MASS
    )
    return float(ah) if ah.ndim == 0 else ah


def rel_humid_to_abs_humid(
    rel_humid: ArrayLike,
    temperature: ArrayLike,
    algorithm: int = 0,
) -> Union[float, NDArray[np.floating]]:
    """
    Convert relative humidity to absolute humidity.

    Assumes the Ideal Gas Law and Dalton's Law of Partial Pressures: the
    partial pressure of water is ``rel_humid`` times the saturation
    pressure at ``temperature``, and the corresponding mass density
    follows from the ideal gas law.

    Port of ``relHumid2AbsHumid.m``.

    Parameters
    ----------
    rel_humid : array_like
        Relative humidity as a fraction in [0, 1].
    temperature : array_like
        Temperature(s) in Kelvin.
    algorithm : int, optional
        Dew-point algorithm; see :func:`dew_point_pressure`.

    Returns
    -------
    abs_humid : float or ndarray
        Absolute humidity in kilograms of water per cubic meter of air.

    Examples
    --------
    >>> round(float(rel_humid_to_abs_humid(0.5, 288.15)), 8)
    0.00641652
    """
    p_sat = np.asarray(dew_point_pressure(temperature, algorithm))
    p_h2o = np.asarray(rel_humid, dtype=np.float64) * p_sat
    T = np.asarray(temperature, dtype=np.float64)
    # 1/1000 converts g/m^3 to kg/m^3 (molar mass is in g/mol).
    ah = (1 / 1000) * p_h2o * H2O_MOLAR_MASS / (UNIVERSAL_GAS_CONSTANT * T)
    return float(ah) if ah.ndim == 0 else ah


def abs_humid_to_rel_humid(
    abs_humid: ArrayLike,
    temperature: ArrayLike,
    algorithm: int = 0,
) -> Union[float, NDArray[np.floating]]:
    """
    Convert absolute humidity to relative humidity.

    Inverse of :func:`rel_humid_to_abs_humid`.

    Port of ``absHumid2RelHumid.m``.

    Parameters
    ----------
    abs_humid : array_like
        Absolute humidity in kilograms of water per cubic meter of air.
    temperature : array_like
        Temperature(s) in Kelvin.
    algorithm : int, optional
        Dew-point algorithm; see :func:`dew_point_pressure`.

    Returns
    -------
    rel_humid : float or ndarray
        Relative humidity as a fraction (0 to 1 for physical inputs).

    Examples
    --------
    >>> round(float(abs_humid_to_rel_humid(0.00641652, 288.15)), 6)
    0.5
    """
    T = np.asarray(temperature, dtype=np.float64)
    # Factor of 1000 converts kg/m^3 to g/m^3 (molar mass is in g/mol).
    p_h2o = (
        1000.0
        * np.asarray(abs_humid, dtype=np.float64)
        * UNIVERSAL_GAS_CONSTANT
        * T
        / H2O_MOLAR_MASS
    )
    rh = p_h2o / np.asarray(dew_point_pressure(temperature, algorithm))
    return float(rh) if rh.ndim == 0 else rh


def abs_humid_to_spec_humid(
    abs_humid: ArrayLike,
    dry_air_density: ArrayLike,
    definition: int = 0,
) -> Union[float, NDArray[np.floating]]:
    """
    Convert absolute humidity to specific humidity.

    Port of ``absHumid2SpecHumid.m``.

    Parameters
    ----------
    abs_humid : array_like
        Absolute humidity in kilograms of water per cubic meter of air.
    dry_air_density : array_like
        Mass density of the dry air (not counting the water) in kg/m^3.
    definition : int, optional
        ``0`` (default): specific humidity is the mass density of water
        over the mass density of dry air (mixing ratio). ``1``: mass
        density of water over the total mass density of the air.

    Returns
    -------
    spec_humid : float or ndarray
        Specific humidity under the chosen definition (dimensionless).

    Examples
    --------
    >>> round(float(abs_humid_to_spec_humid(0.00641652, 1.225)), 8)
    0.00523798
    >>> round(float(abs_humid_to_spec_humid(0.00641652, 1.225, definition=1)), 8)
    0.00521068
    """
    ah = np.asarray(abs_humid, dtype=np.float64)
    rho_dry = np.asarray(dry_air_density, dtype=np.float64)
    if definition != 0:
        sh = ah / (ah + rho_dry)
    else:
        sh = ah / rho_dry
    return float(sh) if sh.ndim == 0 else sh


def spec_humid_to_abs_humid(
    spec_humid: ArrayLike,
    dry_air_density: ArrayLike,
    definition: int = 0,
) -> Union[float, NDArray[np.floating]]:
    """
    Convert specific humidity to absolute humidity.

    Inverse of :func:`abs_humid_to_spec_humid`.

    Port of ``specHumid2AbsHumid.m``.

    Parameters
    ----------
    spec_humid : array_like
        Specific humidity (dimensionless).
    dry_air_density : array_like
        Mass density of the dry air (not counting the water) in kg/m^3.
    definition : int, optional
        ``0`` (default): specific humidity is water density over dry-air
        density. ``1``: water density over total air density.

    Returns
    -------
    abs_humid : float or ndarray
        Absolute humidity in kilograms of water per cubic meter of air.

    Examples
    --------
    >>> round(float(spec_humid_to_abs_humid(0.00523798, 1.225)), 8)
    0.00641653
    """
    sh = np.asarray(spec_humid, dtype=np.float64)
    rho_dry = np.asarray(dry_air_density, dtype=np.float64)
    if definition != 0:
        ah = rho_dry * sh / (1 - sh)
    else:
        ah = rho_dry * sh
    return float(ah) if ah.ndim == 0 else ah


def rel_humid_to_spec_humid(
    rel_humid: ArrayLike,
    temperature: ArrayLike,
    dry_air_density: ArrayLike,
    definition: int = 0,
    algorithm: int = 0,
) -> Union[float, NDArray[np.floating]]:
    """
    Convert relative humidity to specific humidity.

    Port of ``relHumid2SpecHumid.m``.

    Parameters
    ----------
    rel_humid : array_like
        Relative humidity as a fraction in [0, 1].
    temperature : array_like
        Temperature(s) in Kelvin.
    dry_air_density : array_like
        Mass density of the dry air (not counting the water) in kg/m^3.
    definition : int, optional
        Specific-humidity definition; see :func:`abs_humid_to_spec_humid`.
    algorithm : int, optional
        Dew-point algorithm; see :func:`dew_point_pressure`.

    Returns
    -------
    spec_humid : float or ndarray
        Specific humidity under the chosen definition (dimensionless).

    Examples
    --------
    >>> round(float(rel_humid_to_spec_humid(0.5, 288.15, 1.225)), 8)
    0.00523798
    """
    ah = rel_humid_to_abs_humid(rel_humid, temperature, algorithm)
    return abs_humid_to_spec_humid(ah, dry_air_density, definition)


def spec_humid_to_rel_humid(
    spec_humid: ArrayLike,
    temperature: ArrayLike,
    dry_air_density: ArrayLike,
    definition: int = 0,
    algorithm: int = 0,
) -> Union[float, NDArray[np.floating]]:
    """
    Convert specific humidity to relative humidity.

    Inverse of :func:`rel_humid_to_spec_humid`.

    Port of ``specHumid2RelHumid.m``.

    Parameters
    ----------
    spec_humid : array_like
        Specific humidity (dimensionless).
    temperature : array_like
        Temperature(s) in Kelvin.
    dry_air_density : array_like
        Mass density of the dry air (not counting the water) in kg/m^3.
    definition : int, optional
        Specific-humidity definition; see :func:`abs_humid_to_spec_humid`.
    algorithm : int, optional
        Dew-point algorithm; see :func:`dew_point_pressure`.

    Returns
    -------
    rel_humid : float or ndarray
        Relative humidity as a fraction (0 to 1 for physical inputs).

    Examples
    --------
    >>> round(float(spec_humid_to_rel_humid(0.00523798, 288.15, 1.225)), 6)
    0.5
    """
    ah = spec_humid_to_abs_humid(spec_humid, dry_air_density, definition)
    return abs_humid_to_rel_humid(ah, temperature, algorithm)
