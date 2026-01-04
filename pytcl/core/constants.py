"""
Physical and mathematical constants used throughout the Tracker Component Library.

This module provides standardized values for physical constants, with references
to their sources. Constants are provided as module-level variables for convenience
and as a PhysicalConstants class for documentation and grouping.

References
----------
.. [1] NIST CODATA 2018 - https://physics.nist.gov/cuu/Constants/
.. [2] IERS Conventions (2010) - IERS Technical Note 36
.. [3] WGS84 - https://earth-info.nga.mil/GandG/update/index.php?dir=wgs84&action=wgs84
"""

import math
from dataclasses import dataclass
from typing import Final

# =============================================================================
# Universal Physical Constants (CODATA 2018)
# =============================================================================

#: Speed of light in vacuum [m/s]
SPEED_OF_LIGHT: Final[float] = 299_792_458.0

#: Newtonian gravitational constant [m^3 kg^-1 s^-2]
GRAVITATIONAL_CONSTANT: Final[float] = 6.67430e-11

#: Planck constant [J s]
PLANCK_CONSTANT: Final[float] = 6.62607015e-34

#: Boltzmann constant [J K^-1]
BOLTZMANN_CONSTANT: Final[float] = 1.380649e-23

#: Stefan-Boltzmann constant [W m^-2 K^-4]
STEFAN_BOLTZMANN_CONSTANT: Final[float] = 5.670374419e-8

#: Elementary charge [C]
ELEMENTARY_CHARGE: Final[float] = 1.602176634e-19

#: Avogadro constant [mol^-1]
AVOGADRO_CONSTANT: Final[float] = 6.02214076e23

#: Universal gas constant [J mol^-1 K^-1]
UNIVERSAL_GAS_CONSTANT: Final[float] = 8.314462618

#: Standard atmosphere pressure [Pa]
STANDARD_ATMOSPHERE: Final[float] = 101_325.0

#: Absolute zero [K] (definition)
ABSOLUTE_ZERO_CELSIUS: Final[float] = -273.15


# =============================================================================
# Earth Parameters (WGS84)
# =============================================================================

#: Semi-major axis (equatorial radius) [m]
EARTH_SEMI_MAJOR_AXIS: Final[float] = 6_378_137.0

#: Semi-minor axis (polar radius) [m]
EARTH_SEMI_MINOR_AXIS: Final[float] = 6_356_752.314245

#: Flattening factor (dimensionless)
EARTH_FLATTENING: Final[float] = 1.0 / 298.257223563

#: First eccentricity squared
EARTH_ECCENTRICITY_SQ: Final[float] = 2 * EARTH_FLATTENING - EARTH_FLATTENING**2

#: First eccentricity
EARTH_ECCENTRICITY: Final[float] = math.sqrt(EARTH_ECCENTRICITY_SQ)

#: Second eccentricity squared
EARTH_ECCENTRICITY_PRIME_SQ: Final[float] = EARTH_ECCENTRICITY_SQ / (
    1 - EARTH_ECCENTRICITY_SQ
)

#: Earth rotation rate [rad/s] (IERS Conventions 2010)
EARTH_ROTATION_RATE: Final[float] = 7.292115e-5

#: Earth's gravitational parameter GM [m^3/s^2] (WGS84)
EARTH_GM: Final[float] = 3.986004418e14

#: Earth's gravitational parameter GM [m^3/s^2] (EGM2008, includes atmosphere)
EARTH_GM_EGM2008: Final[float] = 3.986004415e14

#: Mean angular velocity of Earth [rad/s]
EARTH_MEAN_ANGULAR_VELOCITY: Final[float] = 7.2921151467e-5

#: Nominal mean Earth radius [m] (IUGG)
EARTH_MEAN_RADIUS: Final[float] = 6_371_000.0

#: Standard gravitational acceleration at sea level [m/s^2]
STANDARD_GRAVITY: Final[float] = 9.80665


# =============================================================================
# Time Constants
# =============================================================================

#: Seconds per day
SECONDS_PER_DAY: Final[float] = 86_400.0

#: Seconds per Julian century
SECONDS_PER_JULIAN_CENTURY: Final[float] = 3_155_760_000.0

#: Days per Julian year
DAYS_PER_JULIAN_YEAR: Final[float] = 365.25

#: Days per Julian century
DAYS_PER_JULIAN_CENTURY: Final[float] = 36_525.0

#: J2000.0 epoch as Julian Date
J2000_EPOCH_JD: Final[float] = 2_451_545.0

#: Modified Julian Date offset (JD - MJD)
MJD_OFFSET: Final[float] = 2_400_000.5


# =============================================================================
# Mathematical Constants
# =============================================================================

#: Pi
PI: Final[float] = math.pi

#: 2*Pi
TWO_PI: Final[float] = 2.0 * math.pi

#: Pi/2
HALF_PI: Final[float] = math.pi / 2.0

#: Degrees to radians conversion factor
DEG_TO_RAD: Final[float] = math.pi / 180.0

#: Radians to degrees conversion factor
RAD_TO_DEG: Final[float] = 180.0 / math.pi

#: Arcseconds to radians conversion factor
ARCSEC_TO_RAD: Final[float] = math.pi / (180.0 * 3600.0)

#: Radians to arcseconds conversion factor
RAD_TO_ARCSEC: Final[float] = (180.0 * 3600.0) / math.pi


# =============================================================================
# Dataclasses for Grouped Constants
# =============================================================================


@dataclass(frozen=True)
class EllipsoidParameters:
    """
    Parameters defining a reference ellipsoid.

    Attributes
    ----------
    a : float
        Semi-major axis (equatorial radius) [m]
    f : float
        Flattening factor (dimensionless)
    GM : float
        Gravitational parameter [m^3/s^2]
    omega : float
        Angular rotation rate [rad/s]
    name : str
        Name of the ellipsoid

    Properties
    ----------
    b : float
        Semi-minor axis (polar radius) [m]
    e2 : float
        First eccentricity squared
    e : float
        First eccentricity
    ep2 : float
        Second eccentricity squared
    """

    a: float
    f: float
    GM: float
    omega: float
    name: str = "Custom"

    @property
    def b(self) -> float:
        """Semi-minor axis [m]."""
        return self.a * (1 - self.f)

    @property
    def e2(self) -> float:
        """First eccentricity squared."""
        return 2 * self.f - self.f**2

    @property
    def e(self) -> float:
        """First eccentricity."""
        return math.sqrt(self.e2)

    @property
    def ep2(self) -> float:
        """Second eccentricity squared."""
        return self.e2 / (1 - self.e2)

    @property
    def ep(self) -> float:
        """Second eccentricity."""
        return math.sqrt(self.ep2)


#: WGS84 ellipsoid parameters
WGS84: Final[EllipsoidParameters] = EllipsoidParameters(
    a=6_378_137.0,
    f=1.0 / 298.257223563,
    GM=3.986004418e14,
    omega=7.292115e-5,
    name="WGS84",
)

#: GRS80 ellipsoid parameters
GRS80: Final[EllipsoidParameters] = EllipsoidParameters(
    a=6_378_137.0,
    f=1.0 / 298.257222101,
    GM=3.986005e14,
    omega=7.292115e-5,
    name="GRS80",
)

#: Clarke 1866 ellipsoid (NAD27)
CLARKE1866: Final[EllipsoidParameters] = EllipsoidParameters(
    a=6_378_206.4,
    f=1.0 / 294.978698214,
    GM=3.986005e14,  # Approximate
    omega=7.292115e-5,
    name="Clarke1866",
)

#: Sphere with Earth mean radius
SPHERE_EARTH: Final[EllipsoidParameters] = EllipsoidParameters(
    a=6_371_000.0,
    f=0.0,
    GM=3.986004418e14,
    omega=7.292115e-5,
    name="Sphere",
)


@dataclass(frozen=True)
class PhysicalConstants:
    """
    Container for fundamental physical constants.

    This class groups physical constants for convenient access and documentation.
    All values follow CODATA 2018 recommendations.

    Examples
    --------
    >>> from pytcl.core.constants import PhysicalConstants
    >>> pc = PhysicalConstants()
    >>> print(f"Speed of light: {pc.c} m/s")
    Speed of light: 299792458.0 m/s
    """

    #: Speed of light in vacuum [m/s]
    c: float = SPEED_OF_LIGHT

    #: Gravitational constant [m^3 kg^-1 s^-2]
    G: float = GRAVITATIONAL_CONSTANT

    #: Planck constant [J s]
    h: float = PLANCK_CONSTANT

    #: Boltzmann constant [J K^-1]
    k_B: float = BOLTZMANN_CONSTANT

    #: Stefan-Boltzmann constant [W m^-2 K^-4]
    sigma: float = STEFAN_BOLTZMANN_CONSTANT

    #: Elementary charge [C]
    e: float = ELEMENTARY_CHARGE

    #: Avogadro constant [mol^-1]
    N_A: float = AVOGADRO_CONSTANT

    #: Universal gas constant [J mol^-1 K^-1]
    R: float = UNIVERSAL_GAS_CONSTANT

    #: Standard gravity [m/s^2]
    g_0: float = STANDARD_GRAVITY


# =============================================================================
# Solar System Parameters
# =============================================================================

#: Astronomical Unit [m] (IAU 2012)
ASTRONOMICAL_UNIT: Final[float] = 149_597_870_700.0

#: Sun gravitational parameter [m^3/s^2]
SUN_GM: Final[float] = 1.32712440018e20

#: Moon gravitational parameter [m^3/s^2]
MOON_GM: Final[float] = 4.9028695e12

#: Earth-Moon mass ratio
EARTH_MOON_MASS_RATIO: Final[float] = 81.30056


# =============================================================================
# Aliases for Common Usage
# =============================================================================

#: Alias for SPEED_OF_LIGHT
c = SPEED_OF_LIGHT

#: Alias for GRAVITATIONAL_CONSTANT
G = GRAVITATIONAL_CONSTANT


__all__ = [
    # Universal Physical Constants
    "SPEED_OF_LIGHT",
    "GRAVITATIONAL_CONSTANT",
    "PLANCK_CONSTANT",
    "BOLTZMANN_CONSTANT",
    "STEFAN_BOLTZMANN_CONSTANT",
    "ELEMENTARY_CHARGE",
    "AVOGADRO_CONSTANT",
    "UNIVERSAL_GAS_CONSTANT",
    "STANDARD_ATMOSPHERE",
    "ABSOLUTE_ZERO_CELSIUS",
    # Earth Parameters
    "EARTH_SEMI_MAJOR_AXIS",
    "EARTH_SEMI_MINOR_AXIS",
    "EARTH_FLATTENING",
    "EARTH_ECCENTRICITY_SQ",
    "EARTH_ECCENTRICITY",
    "EARTH_ECCENTRICITY_PRIME_SQ",
    "EARTH_ROTATION_RATE",
    "EARTH_GM",
    "EARTH_GM_EGM2008",
    "EARTH_MEAN_ANGULAR_VELOCITY",
    "EARTH_MEAN_RADIUS",
    "STANDARD_GRAVITY",
    # Time Constants
    "SECONDS_PER_DAY",
    "SECONDS_PER_JULIAN_CENTURY",
    "DAYS_PER_JULIAN_YEAR",
    "DAYS_PER_JULIAN_CENTURY",
    "J2000_EPOCH_JD",
    "MJD_OFFSET",
    # Mathematical Constants
    "PI",
    "TWO_PI",
    "HALF_PI",
    "DEG_TO_RAD",
    "RAD_TO_DEG",
    "ARCSEC_TO_RAD",
    "RAD_TO_ARCSEC",
    # Dataclasses
    "EllipsoidParameters",
    "PhysicalConstants",
    # Ellipsoid Instances
    "WGS84",
    "GRS80",
    "CLARKE1866",
    "SPHERE_EARTH",
    # Solar System Parameters
    "ASTRONOMICAL_UNIT",
    "SUN_GM",
    "MOON_GM",
    "EARTH_MOON_MASS_RATIO",
    # Aliases
    "c",
    "G",
]
