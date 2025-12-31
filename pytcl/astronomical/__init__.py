"""
Astronomical calculations for target tracking.

This module provides time system conversions, orbital mechanics,
Lambert problem solvers, and reference frame transformations.

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
"""

from pytcl.astronomical.time_systems import (
    # Julian dates
    cal_to_jd,
    jd_to_cal,
    mjd_to_jd,
    jd_to_mjd,
    # Time scales
    utc_to_tai,
    tai_to_utc,
    tai_to_tt,
    tt_to_tai,
    utc_to_tt,
    tt_to_utc,
    tai_to_gps,
    gps_to_tai,
    utc_to_gps,
    gps_to_utc,
    # Unix time
    unix_to_jd,
    jd_to_unix,
    # GPS week
    gps_week_seconds,
    gps_week_to_utc,
    # Sidereal time
    gmst,
    gast,
    # Leap seconds
    get_leap_seconds,
    LeapSecondTable,
    # Constants
    JD_J2000,
    JD_UNIX_EPOCH,
    JD_GPS_EPOCH,
    MJD_OFFSET,
    TT_TAI_OFFSET,
)

from pytcl.astronomical.orbital_mechanics import (
    # Constants
    GM_SUN,
    GM_EARTH,
    GM_MOON,
    GM_MARS,
    GM_JUPITER,
    # Types
    OrbitalElements,
    StateVector,
    # Anomaly conversions
    mean_to_eccentric_anomaly,
    mean_to_hyperbolic_anomaly,
    eccentric_to_true_anomaly,
    true_to_eccentric_anomaly,
    hyperbolic_to_true_anomaly,
    true_to_hyperbolic_anomaly,
    eccentric_to_mean_anomaly,
    mean_to_true_anomaly,
    true_to_mean_anomaly,
    # Element conversions
    orbital_elements_to_state,
    state_to_orbital_elements,
    # Propagation
    kepler_propagate,
    kepler_propagate_state,
    # Orbital quantities
    orbital_period,
    mean_motion,
    vis_viva,
    specific_angular_momentum,
    specific_orbital_energy,
    flight_path_angle,
    periapsis_radius,
    apoapsis_radius,
    time_since_periapsis,
    orbit_radius,
    escape_velocity,
    circular_velocity,
)

from pytcl.astronomical.lambert import (
    LambertSolution,
    lambert_universal,
    lambert_izzo,
    minimum_energy_transfer,
    hohmann_transfer,
    bi_elliptic_transfer,
)

from pytcl.astronomical.reference_frames import (
    # Time utilities
    julian_centuries_j2000,
    # Precession
    precession_angles_iau76,
    precession_matrix_iau76,
    # Nutation
    nutation_angles_iau80,
    nutation_matrix,
    mean_obliquity_iau80,
    true_obliquity,
    # Earth rotation
    earth_rotation_angle,
    gmst_iau82,
    gast_iau82,
    sidereal_rotation_matrix,
    equation_of_equinoxes,
    # Polar motion
    polar_motion_matrix,
    # Full transformations
    gcrf_to_itrf,
    itrf_to_gcrf,
    eci_to_ecef,
    ecef_to_eci,
    # Ecliptic/equatorial
    ecliptic_to_equatorial,
    equatorial_to_ecliptic,
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
]
