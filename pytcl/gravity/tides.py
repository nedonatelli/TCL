"""
Tidal Effects on Gravity and Position.

This module provides functions for computing tidal effects including:
- Solid Earth tides (body tides)
- Ocean tide loading
- Atmospheric pressure loading
- Pole tide effects

These effects cause periodic deformations of the Earth's surface and
variations in the gravity field.

References
----------
.. [1] Petit, G. and Luzum, B. (eds.), "IERS Conventions (2010),"
       IERS Technical Note No. 36, Frankfurt am Main, 2010.
.. [2] McCarthy, D.D. and Petit, G. (eds.), "IERS Conventions (2003),"
       IERS Technical Note No. 32, Frankfurt am Main, 2004.
.. [3] Mathews, P.M., Dehant, V., and Gipson, J.M., "Tidal station
       displacements," JGR, 102, B9, 20469-20477, 1997.
"""

from typing import NamedTuple, Tuple

import numpy as np
from numpy.typing import NDArray

from pytcl.gravity.models import WGS84


class TidalDisplacement(NamedTuple):
    """Result of tidal displacement computation.

    Attributes
    ----------
    radial : float
        Radial (up) displacement in meters.
    north : float
        North displacement in meters.
    east : float
        East displacement in meters.
    """

    radial: float
    north: float
    east: float


class TidalGravity(NamedTuple):
    """Result of tidal gravity computation.

    Attributes
    ----------
    delta_g : float
        Gravity change in m/s^2 (positive = increased gravity).
    delta_g_north : float
        North component of gravity change in m/s^2.
    delta_g_east : float
        East component of gravity change in m/s^2.
    """

    delta_g: float
    delta_g_north: float
    delta_g_east: float


class OceanTideLoading(NamedTuple):
    """Ocean tide loading parameters for a station.

    Attributes
    ----------
    amplitude : NDArray
        Amplitude for each constituent (m) for radial, west, south.
    phase : NDArray
        Phase for each constituent (radians).
    constituents : Tuple[str, ...]
        Names of tidal constituents.
    """

    amplitude: NDArray[np.floating]
    phase: NDArray[np.floating]
    constituents: tuple[str, ...]


# Love and Shida numbers for degree 2 (IERS 2010)
# Nominal values for solid Earth tides
LOVE_H2 = 0.6078  # Radial Love number
LOVE_K2 = 0.2980  # Potential Love number (gravitational)
SHIDA_L2 = 0.0847  # Shida number (horizontal)

# Degree 3 Love numbers (smaller effect)
LOVE_H3 = 0.291
LOVE_K3 = 0.093
SHIDA_L3 = 0.015

# Gravimetric factor (combination of Love numbers)
# delta = 1 + h - 3/2 * k
GRAVIMETRIC_FACTOR = 1.0 + LOVE_H2 - 1.5 * LOVE_K2

# Earth parameters
EARTH_RADIUS = WGS84.a  # Equatorial radius (m)
EARTH_GM = WGS84.GM  # Gravitational parameter (m^3/s^2)
MOON_GM = 4.902801e12  # Moon GM (m^3/s^2)
SUN_GM = 1.32712440018e20  # Sun GM (m^3/s^2)

# Mean distances
MOON_DISTANCE = 384400e3  # Moon mean distance (m)
SUN_DISTANCE = 1.496e11  # Earth-Sun distance (m)

# Tidal constituents frequencies (cycles per day)
# Principal lunar and solar constituents
TIDAL_CONSTITUENTS = {
    "M2": 1.9322736,  # Principal lunar semidiurnal
    "S2": 2.0000000,  # Principal solar semidiurnal
    "N2": 1.8959820,  # Larger lunar elliptic
    "K2": 2.0054758,  # Lunisolar semidiurnal
    "K1": 1.0027379,  # Lunisolar diurnal
    "O1": 0.9295357,  # Principal lunar diurnal
    "P1": 0.9972621,  # Principal solar diurnal
    "Q1": 0.8932441,  # Larger lunar elliptic diurnal
    "Mf": 0.0732167,  # Lunar fortnightly
    "Mm": 0.0362916,  # Lunar monthly
    "Ssa": 0.0054758,  # Solar semiannual
}


def _normalize_angle(angle: float) -> float:
    """Normalize angle to [0, 2*pi)."""
    return angle % (2 * np.pi)


def julian_centuries_j2000(mjd: float) -> float:
    """
    Convert Modified Julian Date to Julian centuries since J2000.0.

    Parameters
    ----------
    mjd : float
        Modified Julian Date.

    Returns
    -------
    T : float
        Julian centuries since J2000.0.
    """
    # J2000.0 = JD 2451545.0 = MJD 51544.5
    return (mjd - 51544.5) / 36525.0


def fundamental_arguments(T: float) -> Tuple[float, float, float, float, float]:
    """
    Compute fundamental astronomical arguments (Delaunay variables).

    Parameters
    ----------
    T : float
        Julian centuries since J2000.0.

    Returns
    -------
    l_moon : float
        Mean anomaly of the Moon (radians).
    l_sun : float
        Mean anomaly of the Sun (radians).
    F : float
        Mean argument of latitude of the Moon (radians).
    D : float
        Mean elongation of the Moon from the Sun (radians).
    Omega : float
        Mean longitude of the ascending node of the Moon (radians).

    Notes
    -----
    Based on IERS Conventions (2010) expressions.
    """
    # Convert degrees to radians
    deg2rad = np.pi / 180.0

    # Mean anomaly of the Moon (l)
    l_moon = (
        134.96340251
        + (1717915923.2178 * T + 31.8792 * T**2 + 0.051635 * T**3 - 0.00024470 * T**4)
        / 3600.0
    ) * deg2rad

    # Mean anomaly of the Sun (l')
    l_sun = (
        357.52910918
        + (129596581.0481 * T - 0.5532 * T**2 + 0.000136 * T**3 - 0.00001149 * T**4)
        / 3600.0
    ) * deg2rad

    # Mean argument of latitude of the Moon (F)
    F = (
        93.27209062
        + (1739527262.8478 * T - 12.7512 * T**2 - 0.001037 * T**3 + 0.00000417 * T**4)
        / 3600.0
    ) * deg2rad

    # Mean elongation of the Moon from the Sun (D)
    D = (
        297.85019547
        + (1602961601.2090 * T - 6.3706 * T**2 + 0.006593 * T**3 - 0.00003169 * T**4)
        / 3600.0
    ) * deg2rad

    # Mean longitude of the ascending node of the Moon (Omega)
    Omega = (
        125.04455501
        + (-6962890.5431 * T + 7.4722 * T**2 + 0.007702 * T**3 - 0.00005939 * T**4)
        / 3600.0
    ) * deg2rad

    return (
        _normalize_angle(l_moon),
        _normalize_angle(l_sun),
        _normalize_angle(F),
        _normalize_angle(D),
        _normalize_angle(Omega),
    )


def moon_position_approximate(mjd: float) -> Tuple[float, float, float]:
    """
    Compute approximate geocentric Moon position.

    Parameters
    ----------
    mjd : float
        Modified Julian Date.

    Returns
    -------
    r : float
        Distance from Earth center (m).
    lat : float
        Geocentric latitude (radians).
    lon : float
        Geocentric longitude (radians).

    Notes
    -----
    Low-precision formula adequate for tidal computations.
    """
    T = julian_centuries_j2000(mjd)

    # Mean elements (degrees) - normalize to [0, 360)
    L0 = (218.3164477 + 481267.88123421 * T) % 360  # Mean longitude
    D = (297.8501921 + 445267.1114034 * T) % 360  # Mean elongation
    M = (357.5291092 + 35999.0502909 * T) % 360  # Sun's mean anomaly
    M_prime = (134.9633964 + 477198.8675055 * T) % 360  # Moon's mean anomaly
    F = (93.2720950 + 483202.0175233 * T) % 360  # Argument of latitude

    # Convert to radians
    deg2rad = np.pi / 180.0
    L0 = L0 * deg2rad
    D = D * deg2rad
    M = M * deg2rad
    M_prime = M_prime * deg2rad
    F = F * deg2rad

    # Longitude perturbations (0.000001 degrees = micro-degrees)
    # These coefficients are from low-precision ephemeris in micro-degrees
    lon_pert = (
        6.288774 * np.sin(M_prime)
        + 1.274027 * np.sin(2 * D - M_prime)
        + 0.658314 * np.sin(2 * D)
        + 0.213618 * np.sin(2 * M_prime)
        - 0.185116 * np.sin(M)
    )

    # Latitude perturbations (degrees)
    lat_pert = (
        5.128122 * np.sin(F)
        + 0.280602 * np.sin(M_prime + F)
        + 0.277693 * np.sin(M_prime - F)
        + 0.173237 * np.sin(2 * D - F)
    )

    # Distance perturbations (km)
    r_pert = (
        -20.905355 * np.cos(M_prime)
        - 3.699111 * np.cos(2 * D - M_prime)
        - 2.955968 * np.cos(2 * D)
    ) * 1000  # Convert to km

    # Final position
    lon = L0 + lon_pert * deg2rad
    lat = lat_pert * deg2rad
    r = (385000.56 + r_pert) * 1000.0  # Convert km to m

    return r, lat, _normalize_angle(lon)


def sun_position_approximate(mjd: float) -> Tuple[float, float, float]:
    """
    Compute approximate geocentric Sun position.

    Parameters
    ----------
    mjd : float
        Modified Julian Date.

    Returns
    -------
    r : float
        Distance from Earth center (m).
    lat : float
        Geocentric latitude (radians).
    lon : float
        Geocentric longitude (radians).

    Notes
    -----
    Low-precision formula adequate for tidal computations.
    """
    T = julian_centuries_j2000(mjd)

    # Mean elements (degrees) - normalize to [0, 360)
    L0 = (280.46646 + 36000.76983 * T) % 360  # Mean longitude
    M = (357.52911 + 35999.05029 * T) % 360  # Mean anomaly
    e = 0.016708634 - 0.000042037 * T  # Eccentricity

    # Convert to radians
    deg2rad = np.pi / 180.0
    M_rad = M * deg2rad

    # Equation of center (degrees)
    C = (
        (1.914602 - 0.004817 * T) * np.sin(M_rad)
        + 0.019993 * np.sin(2 * M_rad)
        + 0.000289 * np.sin(3 * M_rad)
    )

    # True longitude (normalize)
    lon = ((L0 + C) % 360) * deg2rad

    # Distance (AU to meters)
    AU = 1.495978707e11
    r = AU * (1.000001018 * (1 - e**2)) / (1 + e * np.cos(M_rad + C * deg2rad))

    # Sun latitude is zero (on ecliptic)
    # For tidal purposes, we use geocentric ecliptic coordinates
    lat = 0.0

    return r, lat, _normalize_angle(lon)


def solid_earth_tide_displacement(
    lat: float,
    lon: float,
    mjd: float,
    h2: float = LOVE_H2,
    l2: float = SHIDA_L2,
) -> TidalDisplacement:
    """
    Compute solid Earth tide displacement at a station.

    Parameters
    ----------
    lat : float
        Geodetic latitude in radians.
    lon : float
        Longitude in radians.
    mjd : float
        Modified Julian Date.
    h2 : float, optional
        Love number h2. Default is IERS value.
    l2 : float, optional
        Shida number l2. Default is IERS value.

    Returns
    -------
    TidalDisplacement
        Displacement in radial, north, east directions (meters).

    Notes
    -----
    Computes the degree-2 solid Earth tide displacement caused by
    the Moon and Sun. The displacement follows the tide-generating
    potential with Love and Shida numbers.

    Examples
    --------
    >>> import numpy as np
    >>> # Displacement at 45N, 0E on MJD 58000
    >>> disp = solid_earth_tide_displacement(np.radians(45), 0, 58000)
    >>> abs(disp.radial) < 0.5  # Radial displacement typically < 50cm
    True
    """
    # Station geocentric position (approximate)
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)

    # Station unit vector in geocentric frame
    station_x = cos_lat * cos_lon
    station_y = cos_lat * sin_lon
    station_z = sin_lat

    # Get Moon and Sun positions
    r_moon, lat_moon, lon_moon = moon_position_approximate(mjd)
    r_sun, lat_sun, lon_sun = sun_position_approximate(mjd)

    # Moon unit vector
    moon_x = np.cos(lat_moon) * np.cos(lon_moon)
    moon_y = np.cos(lat_moon) * np.sin(lon_moon)
    moon_z = np.sin(lat_moon)

    # Sun unit vector
    sun_x = np.cos(lat_sun) * np.cos(lon_sun)
    sun_y = np.cos(lat_sun) * np.sin(lon_sun)
    sun_z = np.sin(lat_sun)

    # Zenith angle cosines
    cos_psi_moon = station_x * moon_x + station_y * moon_y + station_z * moon_z
    cos_psi_sun = station_x * sun_x + station_y * sun_y + sun_z * station_z

    # Tide-generating potential amplitude factors
    # Amplitude = (3/2) * (GM_body/GM_earth) * (R_earth/r_body)^3 * R_earth
    R = EARTH_RADIUS
    amp_moon = 1.5 * (MOON_GM / EARTH_GM) * (R / r_moon) ** 3 * R
    amp_sun = 1.5 * (SUN_GM / EARTH_GM) * (R / r_sun) ** 3 * R

    # Legendre polynomial P2(cos(psi))
    P2_moon = 0.5 * (3 * cos_psi_moon**2 - 1)
    P2_sun = 0.5 * (3 * cos_psi_sun**2 - 1)

    # Radial displacement: h2 * amplitude * P2
    radial_moon = h2 * amp_moon * P2_moon
    radial_sun = h2 * amp_sun * P2_sun
    radial = radial_moon + radial_sun

    # Horizontal displacement: l2 * amplitude * dP2/d(psi)
    # dP2/d(cos_psi) = 3 * cos_psi
    dP2_moon = 3 * cos_psi_moon
    dP2_sun = 3 * cos_psi_sun

    # Sin of zenith angle
    sin_psi_moon = np.sqrt(1 - cos_psi_moon**2)
    sin_psi_sun = np.sqrt(1 - cos_psi_sun**2)

    # Avoid division by zero at zenith/nadir
    if sin_psi_moon < 1e-10:
        sin_psi_moon = 1e-10
    if sin_psi_sun < 1e-10:
        sin_psi_sun = 1e-10

    # Horizontal displacement magnitude (toward the body)
    horiz_moon = l2 * amp_moon * dP2_moon * sin_psi_moon
    horiz_sun = l2 * amp_sun * dP2_sun * sin_psi_sun

    # Direction to Moon/Sun in local frame
    # Azimuth from station to body
    dx_moon = moon_x - station_x
    dy_moon = moon_y - station_y
    dz_moon = moon_z - station_z

    dx_sun = sun_x - station_x
    dy_sun = sun_y - station_y
    dz_sun = sun_z - station_z

    # Project to local north and east
    # Local north = -sin(lat)*cos(lon), -sin(lat)*sin(lon), cos(lat)
    # Local east = -sin(lon), cos(lon), 0
    north_x = -sin_lat * cos_lon
    north_y = -sin_lat * sin_lon
    north_z = cos_lat

    east_x = -sin_lon
    east_y = cos_lon
    east_z = 0.0

    # Direction cosines for Moon
    az_n_moon = dx_moon * north_x + dy_moon * north_y + dz_moon * north_z
    az_e_moon = dx_moon * east_x + dy_moon * east_y + dz_moon * east_z
    az_mag_moon = np.sqrt(az_n_moon**2 + az_e_moon**2)
    if az_mag_moon > 1e-10:
        az_n_moon /= az_mag_moon
        az_e_moon /= az_mag_moon
    else:
        az_n_moon, az_e_moon = 0.0, 0.0

    # Direction cosines for Sun
    az_n_sun = dx_sun * north_x + dy_sun * north_y + dz_sun * north_z
    az_e_sun = dx_sun * east_x + dy_sun * east_y + dz_sun * east_z
    az_mag_sun = np.sqrt(az_n_sun**2 + az_e_sun**2)
    if az_mag_sun > 1e-10:
        az_n_sun /= az_mag_sun
        az_e_sun /= az_mag_sun
    else:
        az_n_sun, az_e_sun = 0.0, 0.0

    # Horizontal components
    north = horiz_moon * az_n_moon + horiz_sun * az_n_sun
    east = horiz_moon * az_e_moon + horiz_sun * az_e_sun

    return TidalDisplacement(radial=radial, north=north, east=east)


def solid_earth_tide_gravity(
    lat: float,
    lon: float,
    mjd: float,
    delta: float = GRAVIMETRIC_FACTOR,
) -> TidalGravity:
    """
    Compute solid Earth tide effect on gravity.

    Parameters
    ----------
    lat : float
        Geodetic latitude in radians.
    lon : float
        Longitude in radians.
    mjd : float
        Modified Julian Date.
    delta : float, optional
        Gravimetric factor (1 + h - 3k/2). Default is IERS value.

    Returns
    -------
    TidalGravity
        Gravity change in vertical and horizontal components (m/s^2).

    Notes
    -----
    The tidal gravity change is computed using the tide-generating
    potential and the gravimetric factor delta = 1 + h - 3k/2.

    Examples
    --------
    >>> import numpy as np
    >>> grav = solid_earth_tide_gravity(np.radians(45), 0, 58000)
    >>> abs(grav.delta_g) < 3e-6  # Typically < 3 microGal
    True
    """
    # Station position
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    cos_lon = np.cos(lon)
    sin_lon = np.sin(lon)

    station_x = cos_lat * cos_lon
    station_y = cos_lat * sin_lon
    station_z = sin_lat

    # Get Moon and Sun positions
    r_moon, lat_moon, lon_moon = moon_position_approximate(mjd)
    r_sun, lat_sun, lon_sun = sun_position_approximate(mjd)

    # Body unit vectors
    moon_x = np.cos(lat_moon) * np.cos(lon_moon)
    moon_y = np.cos(lat_moon) * np.sin(lon_moon)
    moon_z = np.sin(lat_moon)

    sun_x = np.cos(lat_sun) * np.cos(lon_sun)
    sun_y = np.cos(lat_sun) * np.sin(lon_sun)
    sun_z = np.sin(lat_sun)

    # Zenith angle cosines
    cos_psi_moon = station_x * moon_x + station_y * moon_y + station_z * moon_z
    cos_psi_sun = station_x * sun_x + station_y * sun_y + station_z * sun_z

    # Gravity potential amplitude factors (m/s^2 equivalent)
    R = EARTH_RADIUS
    g = 9.80665  # Standard gravity

    amp_moon = 1.5 * (MOON_GM / EARTH_GM) * (R / r_moon) ** 3 * g
    amp_sun = 1.5 * (SUN_GM / EARTH_GM) * (R / r_sun) ** 3 * g

    # Vertical tidal gravity: delta * amplitude * dP2/dr
    # For gravity, we differentiate the potential
    # dP2/d(cos_psi) = 3 * cos_psi
    P2_moon = 0.5 * (3 * cos_psi_moon**2 - 1)
    P2_sun = 0.5 * (3 * cos_psi_sun**2 - 1)

    delta_g_moon = delta * amp_moon * 2 * P2_moon
    delta_g_sun = delta * amp_sun * 2 * P2_sun

    delta_g = delta_g_moon + delta_g_sun

    # Horizontal components (typically small)
    # Would need more detailed computation for tilt effects
    delta_g_north = 0.0
    delta_g_east = 0.0

    return TidalGravity(
        delta_g=delta_g, delta_g_north=delta_g_north, delta_g_east=delta_g_east
    )


def ocean_tide_loading_displacement(
    mjd: float,
    amplitude: NDArray[np.floating],
    phase: NDArray[np.floating],
    constituents: tuple[str, ...] = ("M2", "S2", "N2", "K2", "K1", "O1", "P1", "Q1"),
) -> TidalDisplacement:
    """
    Compute ocean tide loading displacement.

    Parameters
    ----------
    mjd : float
        Modified Julian Date.
    amplitude : NDArray
        Loading amplitudes (3, n_constituents) for radial, north, east.
        Units: meters.
    phase : NDArray
        Loading phases (3, n_constituents) in radians.
    constituents : Tuple[str, ...], optional
        Tidal constituent names. Default is major constituents.

    Returns
    -------
    TidalDisplacement
        Ocean loading displacement (meters).

    Notes
    -----
    Ocean tide loading parameters are typically obtained from services
    like IERS or computed from ocean tide models (e.g., FES2014, GOT4.10).

    The displacement is computed as:
    sum_i amplitude_i * cos(omega_i * t + chi_i - phase_i)

    where omega_i is the constituent frequency and chi_i is the
    astronomical argument.

    Examples
    --------
    >>> import numpy as np
    >>> # Example loading coefficients (typically from IERS)
    >>> amp = np.array([[0.01, 0.005], [0.002, 0.001], [0.002, 0.001]])
    >>> phase = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    >>> disp = ocean_tide_loading_displacement(58000, amp, phase, ("M2", "S2"))
    >>> isinstance(disp, TidalDisplacement)
    True
    """
    T = julian_centuries_j2000(mjd)
    l, l_prime, F, D, Omega = fundamental_arguments(T)

    # Days since J2000.0
    days = mjd - 51544.5

    radial = 0.0
    north = 0.0
    east = 0.0

    for i, const in enumerate(constituents):
        if const not in TIDAL_CONSTITUENTS:
            continue

        # Frequency (cycles per day)
        freq = TIDAL_CONSTITUENTS[const]

        # Astronomical argument (simplified)
        # Full computation would use Doodson numbers
        omega_t = 2 * np.pi * freq * days

        # Add fundamental argument contributions based on constituent type
        if const == "M2":
            chi = 2 * (F + Omega) - 2 * l
        elif const == "S2":
            chi = 0.0  # Solar time
        elif const == "N2":
            chi = 2 * (F + Omega) - 3 * l + l_prime
        elif const == "K2":
            chi = 2 * (F + Omega)
        elif const == "K1":
            chi = F + Omega
        elif const == "O1":
            chi = F + Omega - 2 * l
        elif const == "P1":
            chi = -F - Omega
        elif const == "Q1":
            chi = F + Omega - 3 * l
        elif const == "Mf":
            chi = 2 * F
        elif const == "Mm":
            chi = l
        elif const == "Ssa":
            chi = 2 * l_prime
        else:
            chi = 0.0

        # Total phase
        arg = omega_t + chi

        # Displacement components
        if amplitude.shape[0] >= 1:
            radial += amplitude[0, i] * np.cos(arg - phase[0, i])
        if amplitude.shape[0] >= 2:
            north += amplitude[1, i] * np.cos(arg - phase[1, i])
        if amplitude.shape[0] >= 3:
            east += amplitude[2, i] * np.cos(arg - phase[2, i])

    return TidalDisplacement(radial=radial, north=north, east=east)


def atmospheric_pressure_loading(
    lat: float,
    lon: float,
    pressure: float,
    reference_pressure: float = 101325.0,
    admittance: float = -0.35e-3,
) -> TidalDisplacement:
    """
    Compute atmospheric pressure loading displacement.

    Parameters
    ----------
    lat : float
        Geodetic latitude in radians.
    lon : float
        Longitude in radians.
    pressure : float
        Surface atmospheric pressure in Pascals.
    reference_pressure : float, optional
        Reference pressure in Pascals. Default is 101325 Pa (1 atm).
    admittance : float, optional
        Pressure admittance in m/Pa. Default is -0.35 mm/hPa.

    Returns
    -------
    TidalDisplacement
        Pressure loading displacement (meters).

    Notes
    -----
    Atmospheric pressure loading causes the Earth's surface to deform
    under changing atmospheric mass. The effect is primarily radial
    (vertical) with magnitude approximately -0.35 mm per hPa of
    pressure above the mean.

    For precise applications, use Green's functions from models like
    ECMWF or NCEP pressure fields.

    Examples
    --------
    >>> import numpy as np
    >>> # High pressure (1020 hPa) above reference
    >>> disp = atmospheric_pressure_loading(np.radians(45), 0, 102000)
    >>> disp.radial < 0  # Surface depressed under high pressure
    True
    """
    # Pressure anomaly
    delta_p = pressure - reference_pressure

    # Radial displacement (inverted barometer effect)
    # Negative admittance: high pressure -> surface depression
    radial = admittance * delta_p

    # Horizontal displacements are smaller and depend on pressure gradients
    # Simple approximation: assume no horizontal component
    north = 0.0
    east = 0.0

    return TidalDisplacement(radial=radial, north=north, east=east)


def pole_tide_displacement(
    lat: float,
    lon: float,
    xp: float,
    yp: float,
    xp_mean: float = 0.0,
    yp_mean: float = 0.0,
    h2: float = LOVE_H2,
    l2: float = SHIDA_L2,
) -> TidalDisplacement:
    """
    Compute pole tide (polar motion) displacement.

    Parameters
    ----------
    lat : float
        Geodetic latitude in radians.
    lon : float
        Longitude in radians.
    xp : float
        Pole x coordinate in arcseconds.
    yp : float
        Pole y coordinate in arcseconds.
    xp_mean : float, optional
        Mean pole x coordinate in arcseconds. Default 0.
    yp_mean : float, optional
        Mean pole y coordinate in arcseconds. Default 0.
    h2 : float, optional
        Love number. Default is IERS value.
    l2 : float, optional
        Shida number. Default is IERS value.

    Returns
    -------
    TidalDisplacement
        Pole tide displacement (meters).

    Notes
    -----
    The pole tide results from the centrifugal effect of polar motion.
    The Earth's rotation axis wobbles with a primary period of ~433 days
    (Chandler wobble) causing small displacements.

    Examples
    --------
    >>> import numpy as np
    >>> # Pole offset of 0.1 arcsec
    >>> disp = pole_tide_displacement(np.radians(45), 0, 0.1, 0.1)
    >>> abs(disp.radial) < 0.03  # Typically < 3cm
    True
    """
    # Convert arcseconds to radians
    arcsec2rad = np.pi / (180.0 * 3600.0)

    # Pole position anomaly
    m1 = (xp - xp_mean) * arcsec2rad
    m2 = (yp - yp_mean) * arcsec2rad

    # Angular velocity
    omega = WGS84.omega
    R = EARTH_RADIUS

    # Pole tide potential coefficient
    # Omega^2 * R^2 / g
    g = 9.80665
    coeff = omega**2 * R**2 / g

    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_2lat = np.sin(2 * lat)
    cos_lon = np.cos(lon)
    sin_lon = np.sin(lon)

    # Radial displacement
    radial = -h2 * coeff * sin_2lat * (m1 * cos_lon + m2 * sin_lon)

    # North displacement
    north = l2 * coeff * 2 * cos_lat * (m1 * cos_lon + m2 * sin_lon)

    # East displacement
    east = l2 * coeff * 2 * sin_lat * (-m1 * sin_lon + m2 * cos_lon)

    return TidalDisplacement(radial=radial, north=north, east=east)


def total_tidal_displacement(
    lat: float,
    lon: float,
    mjd: float,
    ocean_loading: OceanTideLoading | None = None,
    pressure: float | None = None,
    xp: float = 0.0,
    yp: float = 0.0,
) -> TidalDisplacement:
    """
    Compute total tidal displacement from all sources.

    Parameters
    ----------
    lat : float
        Geodetic latitude in radians.
    lon : float
        Longitude in radians.
    mjd : float
        Modified Julian Date.
    ocean_loading : OceanTideLoading, optional
        Ocean loading parameters. If None, ocean loading is not included.
    pressure : float, optional
        Surface pressure in Pascals. If None, pressure loading is not included.
    xp : float, optional
        Pole x coordinate in arcseconds. Default 0.
    yp : float, optional
        Pole y coordinate in arcseconds. Default 0.

    Returns
    -------
    TidalDisplacement
        Total tidal displacement (meters).

    Examples
    --------
    >>> import numpy as np
    >>> disp = total_tidal_displacement(np.radians(45), 0, 58000)
    >>> isinstance(disp, TidalDisplacement)
    True
    """
    # Solid Earth tides (always computed)
    solid = solid_earth_tide_displacement(lat, lon, mjd)

    radial = solid.radial
    north = solid.north
    east = solid.east

    # Ocean loading
    if ocean_loading is not None:
        ocean = ocean_tide_loading_displacement(
            mjd,
            ocean_loading.amplitude,
            ocean_loading.phase,
            ocean_loading.constituents,
        )
        radial += ocean.radial
        north += ocean.north
        east += ocean.east

    # Atmospheric pressure loading
    if pressure is not None:
        atm = atmospheric_pressure_loading(lat, lon, pressure)
        radial += atm.radial
        north += atm.north
        east += atm.east

    # Pole tide
    if xp != 0.0 or yp != 0.0:
        pole = pole_tide_displacement(lat, lon, xp, yp)
        radial += pole.radial
        north += pole.north
        east += pole.east

    return TidalDisplacement(radial=radial, north=north, east=east)


def tidal_gravity_correction(
    lat: float,
    lon: float,
    mjd: float,
) -> float:
    """
    Compute tidal correction to gravity observations.

    Parameters
    ----------
    lat : float
        Geodetic latitude in radians.
    lon : float
        Longitude in radians.
    mjd : float
        Modified Julian Date.

    Returns
    -------
    delta_g : float
        Gravity correction in m/s^2 (add to observed gravity).

    Notes
    -----
    This correction should be added to absolute gravity observations
    to remove the tidal signal.

    Examples
    --------
    >>> import numpy as np
    >>> corr = tidal_gravity_correction(np.radians(45), 0, 58000)
    >>> abs(corr) < 3e-6  # Typically < 300 microGal
    True
    """
    result = solid_earth_tide_gravity(lat, lon, mjd)
    return -result.delta_g  # Correction has opposite sign


__all__ = [
    # Result types
    "TidalDisplacement",
    "TidalGravity",
    "OceanTideLoading",
    # Constants
    "LOVE_H2",
    "LOVE_K2",
    "SHIDA_L2",
    "LOVE_H3",
    "LOVE_K3",
    "SHIDA_L3",
    "GRAVIMETRIC_FACTOR",
    "TIDAL_CONSTITUENTS",
    # Time and astronomy
    "julian_centuries_j2000",
    "fundamental_arguments",
    "moon_position_approximate",
    "sun_position_approximate",
    # Solid Earth tides
    "solid_earth_tide_displacement",
    "solid_earth_tide_gravity",
    # Ocean loading
    "ocean_tide_loading_displacement",
    # Atmospheric loading
    "atmospheric_pressure_loading",
    # Pole tide
    "pole_tide_displacement",
    # Combined
    "total_tidal_displacement",
    "tidal_gravity_correction",
]
