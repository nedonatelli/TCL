"""
Astronomical calculations for target tracking.

This module provides time system conversions, orbital mechanics,
Lambert problem solvers, reference frame transformations, and high-precision
ephemerides for celestial bodies.

Examples
--------
>>> from pytcl.astronomical import kepler_propagate, OrbitalElements
>>> import numpy as np

>>> # Propagate an orbit forward in time
>>> elements = OrbitalElements(a=7000, e=0.01, i=0.5, raan=0, omega=0, nu=0)
>>> new_elements = kepler_propagate(elements, 3600)  # 1 hour

>>> # Solve Lambert's problem
>>> from pytcl.astronomical import lambert_universal
>>> r1 = np.array([5000, 10000, 2100])
>>> r2 = np.array([-14600, 2500, 7000])
>>> sol = lambert_universal(r1, r2, 3600)

>>> # Query Sun position with high precision
>>> from pytcl.astronomical import sun_position
>>> r_sun, v_sun = sun_position(2451545.0)  # J2000.0
"""

from pytcl.astronomical.ephemerides import (
    DEEphemeris,
    barycenter_position,
    moon_position,
    planet_position,
    sun_position,
)
from pytcl.astronomical.lambert import (
    LambertSolution,
    bi_elliptic_transfer,
    hohmann_transfer,
    lambert_izzo,
    lambert_universal,
    minimum_energy_transfer,
)
from pytcl.astronomical.orbital_mechanics import (
    GM_EARTH,  # Constants; Types; Anomaly conversions; Element conversions; Propagation; Orbital quantities
)
from pytcl.astronomical.orbital_mechanics import (
    GM_JUPITER,
    GM_MARS,
    GM_MOON,
    GM_SUN,
    OrbitalElements,
    StateVector,
    apoapsis_radius,
    circular_velocity,
    eccentric_to_mean_anomaly,
    eccentric_to_true_anomaly,
    escape_velocity,
    flight_path_angle,
    hyperbolic_to_true_anomaly,
    kepler_propagate,
    kepler_propagate_state,
    mean_motion,
    mean_to_eccentric_anomaly,
    mean_to_hyperbolic_anomaly,
    mean_to_true_anomaly,
    orbit_radius,
    orbital_elements_to_state,
    orbital_period,
    periapsis_radius,
    specific_angular_momentum,
    specific_orbital_energy,
    state_to_orbital_elements,
    time_since_periapsis,
    true_to_eccentric_anomaly,
    true_to_hyperbolic_anomaly,
    true_to_mean_anomaly,
    vis_viva,
)
from pytcl.astronomical.reference_frames import (
    earth_rotation_angle,  # Time utilities; Precession; Nutation; Earth rotation; Polar motion; Full transformations; Ecliptic/equatorial
)
from pytcl.astronomical.reference_frames import (
    ecef_to_eci,
    eci_to_ecef,
    ecliptic_to_equatorial,
    equation_of_equinoxes,
    equatorial_to_ecliptic,
    gast_iau82,
    gcrf_to_itrf,
    gmst_iau82,
    itrf_to_gcrf,
    julian_centuries_j2000,
    mean_obliquity_iau80,
    nutation_angles_iau80,
    nutation_matrix,
    polar_motion_matrix,
    precession_angles_iau76,
    precession_matrix_iau76,
    sidereal_rotation_matrix,
    true_obliquity,
)
from pytcl.astronomical.relativity import (
    C_LIGHT,  # Physical constants; Schwarzschild metric; Time dilation; Shapiro delay; Precession; PN effects; Range corrections
)
from pytcl.astronomical.relativity import (
    G_GRAV,
    geodetic_precession,
    gravitational_time_dilation,
    lense_thirring_precession,
    post_newtonian_acceleration,
    proper_time_rate,
    relativistic_range_correction,
    schwarzschild_precession_per_orbit,
    schwarzschild_radius,
    shapiro_delay,
)
from pytcl.astronomical.time_systems import (
    JD_GPS_EPOCH,  # Julian dates; Time scales; Unix time; GPS week; Sidereal time; Leap seconds; Constants
)
from pytcl.astronomical.time_systems import (
    JD_J2000,
    JD_UNIX_EPOCH,
    MJD_OFFSET,
    TT_TAI_OFFSET,
    LeapSecondTable,
    cal_to_jd,
    gast,
    get_leap_seconds,
    gmst,
    gps_to_tai,
    gps_to_utc,
    gps_week_seconds,
    gps_week_to_utc,
    jd_to_cal,
    jd_to_mjd,
    jd_to_unix,
    mjd_to_jd,
    tai_to_gps,
    tai_to_tt,
    tai_to_utc,
    tt_to_tai,
    tt_to_utc,
    unix_to_jd,
    utc_to_gps,
    utc_to_tai,
    utc_to_tt,
)

__all__ = [
    # Time systems - Julian dates
    "cal_to_jd",
    "jd_to_cal",
    "mjd_to_jd",
    "jd_to_mjd",
    # Time systems - Time scales
    "utc_to_tai",
    "tai_to_utc",
    "tai_to_tt",
    "tt_to_tai",
    "utc_to_tt",
    "tt_to_utc",
    "tai_to_gps",
    "gps_to_tai",
    "utc_to_gps",
    "gps_to_utc",
    # Time systems - Unix time
    "unix_to_jd",
    "jd_to_unix",
    # Time systems - GPS week
    "gps_week_seconds",
    "gps_week_to_utc",
    # Time systems - Sidereal time
    "gmst",
    "gast",
    # Time systems - Leap seconds
    "get_leap_seconds",
    "LeapSecondTable",
    # Time systems - Constants
    "JD_J2000",
    "JD_UNIX_EPOCH",
    "JD_GPS_EPOCH",
    "MJD_OFFSET",
    "TT_TAI_OFFSET",
    # Orbital mechanics - Constants
    "GM_SUN",
    "GM_EARTH",
    "GM_MOON",
    "GM_MARS",
    "GM_JUPITER",
    # Orbital mechanics - Types
    "OrbitalElements",
    "StateVector",
    # Orbital mechanics - Anomaly conversions
    "mean_to_eccentric_anomaly",
    "mean_to_hyperbolic_anomaly",
    "eccentric_to_true_anomaly",
    "true_to_eccentric_anomaly",
    "hyperbolic_to_true_anomaly",
    "true_to_hyperbolic_anomaly",
    "eccentric_to_mean_anomaly",
    "mean_to_true_anomaly",
    "true_to_mean_anomaly",
    # Orbital mechanics - Element conversions
    "orbital_elements_to_state",
    "state_to_orbital_elements",
    # Orbital mechanics - Propagation
    "kepler_propagate",
    "kepler_propagate_state",
    # Orbital mechanics - Orbital quantities
    "orbital_period",
    "mean_motion",
    "vis_viva",
    "specific_angular_momentum",
    "specific_orbital_energy",
    "flight_path_angle",
    "periapsis_radius",
    "apoapsis_radius",
    "time_since_periapsis",
    "orbit_radius",
    "escape_velocity",
    "circular_velocity",
    # Lambert problem
    "LambertSolution",
    "lambert_universal",
    "lambert_izzo",
    "minimum_energy_transfer",
    "hohmann_transfer",
    "bi_elliptic_transfer",
    # Reference frames - Precession
    "julian_centuries_j2000",
    "precession_angles_iau76",
    "precession_matrix_iau76",
    # Reference frames - Nutation
    "nutation_angles_iau80",
    "nutation_matrix",
    "mean_obliquity_iau80",
    "true_obliquity",
    # Reference frames - Earth rotation
    "earth_rotation_angle",
    "gmst_iau82",
    "gast_iau82",
    "sidereal_rotation_matrix",
    "equation_of_equinoxes",
    # Reference frames - Polar motion
    "polar_motion_matrix",
    # Reference frames - Full transformations
    "gcrf_to_itrf",
    "itrf_to_gcrf",
    "eci_to_ecef",
    "ecef_to_eci",
    # Reference frames - Ecliptic/equatorial
    "ecliptic_to_equatorial",
    "equatorial_to_ecliptic",
    # Ephemerides - Classes
    "DEEphemeris",
    # Ephemerides - Functions
    "sun_position",
    "moon_position",
    "planet_position",
    "barycenter_position",
    # Relativity - Constants
    "C_LIGHT",
    "G_GRAV",
    # Relativity - Functions
    "schwarzschild_radius",
    "gravitational_time_dilation",
    "proper_time_rate",
    "shapiro_delay",
    "schwarzschild_precession_per_orbit",
    "post_newtonian_acceleration",
    "geodetic_precession",
    "lense_thirring_precession",
    "relativistic_range_correction",
]
