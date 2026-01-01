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

from pytcl.astronomical.ephemerides import DEEphemeris
from pytcl.astronomical.ephemerides import barycenter_position
from pytcl.astronomical.ephemerides import moon_position
from pytcl.astronomical.ephemerides import planet_position
from pytcl.astronomical.ephemerides import sun_position
from pytcl.astronomical.lambert import LambertSolution
from pytcl.astronomical.lambert import bi_elliptic_transfer
from pytcl.astronomical.lambert import hohmann_transfer
from pytcl.astronomical.lambert import lambert_izzo
from pytcl.astronomical.lambert import lambert_universal
from pytcl.astronomical.lambert import minimum_energy_transfer
from pytcl.astronomical.orbital_mechanics import (
    GM_EARTH,  # Constants; Types; Anomaly conversions; Element conversions; Propagation; Orbital quantities
)
from pytcl.astronomical.orbital_mechanics import GM_JUPITER
from pytcl.astronomical.orbital_mechanics import GM_MARS
from pytcl.astronomical.orbital_mechanics import GM_MOON
from pytcl.astronomical.orbital_mechanics import GM_SUN
from pytcl.astronomical.orbital_mechanics import OrbitalElements
from pytcl.astronomical.orbital_mechanics import StateVector
from pytcl.astronomical.orbital_mechanics import apoapsis_radius
from pytcl.astronomical.orbital_mechanics import circular_velocity
from pytcl.astronomical.orbital_mechanics import eccentric_to_mean_anomaly
from pytcl.astronomical.orbital_mechanics import eccentric_to_true_anomaly
from pytcl.astronomical.orbital_mechanics import escape_velocity
from pytcl.astronomical.orbital_mechanics import flight_path_angle
from pytcl.astronomical.orbital_mechanics import hyperbolic_to_true_anomaly
from pytcl.astronomical.orbital_mechanics import kepler_propagate
from pytcl.astronomical.orbital_mechanics import kepler_propagate_state
from pytcl.astronomical.orbital_mechanics import mean_motion
from pytcl.astronomical.orbital_mechanics import mean_to_eccentric_anomaly
from pytcl.astronomical.orbital_mechanics import mean_to_hyperbolic_anomaly
from pytcl.astronomical.orbital_mechanics import mean_to_true_anomaly
from pytcl.astronomical.orbital_mechanics import orbit_radius
from pytcl.astronomical.orbital_mechanics import orbital_elements_to_state
from pytcl.astronomical.orbital_mechanics import orbital_period
from pytcl.astronomical.orbital_mechanics import periapsis_radius
from pytcl.astronomical.orbital_mechanics import specific_angular_momentum
from pytcl.astronomical.orbital_mechanics import specific_orbital_energy
from pytcl.astronomical.orbital_mechanics import state_to_orbital_elements
from pytcl.astronomical.orbital_mechanics import time_since_periapsis
from pytcl.astronomical.orbital_mechanics import true_to_eccentric_anomaly
from pytcl.astronomical.orbital_mechanics import true_to_hyperbolic_anomaly
from pytcl.astronomical.orbital_mechanics import true_to_mean_anomaly
from pytcl.astronomical.orbital_mechanics import vis_viva
from pytcl.astronomical.reference_frames import (
    earth_rotation_angle,  # Time utilities; Precession; Nutation; Earth rotation; Polar motion; Full transformations; Ecliptic/equatorial
)
from pytcl.astronomical.reference_frames import ecef_to_eci
from pytcl.astronomical.reference_frames import eci_to_ecef
from pytcl.astronomical.reference_frames import ecliptic_to_equatorial
from pytcl.astronomical.reference_frames import equation_of_equinoxes
from pytcl.astronomical.reference_frames import equatorial_to_ecliptic
from pytcl.astronomical.reference_frames import gast_iau82
from pytcl.astronomical.reference_frames import gcrf_to_itrf
from pytcl.astronomical.reference_frames import gmst_iau82
from pytcl.astronomical.reference_frames import itrf_to_gcrf
from pytcl.astronomical.reference_frames import julian_centuries_j2000
from pytcl.astronomical.reference_frames import mean_obliquity_iau80
from pytcl.astronomical.reference_frames import nutation_angles_iau80
from pytcl.astronomical.reference_frames import nutation_matrix
from pytcl.astronomical.reference_frames import polar_motion_matrix
from pytcl.astronomical.reference_frames import precession_angles_iau76
from pytcl.astronomical.reference_frames import precession_matrix_iau76
from pytcl.astronomical.reference_frames import sidereal_rotation_matrix
from pytcl.astronomical.reference_frames import true_obliquity
from pytcl.astronomical.relativity import (
    C_LIGHT,  # Physical constants; Schwarzschild metric; Time dilation; Shapiro delay; Precession; PN effects; Range corrections
)
from pytcl.astronomical.relativity import G_GRAV
from pytcl.astronomical.relativity import geodetic_precession
from pytcl.astronomical.relativity import gravitational_time_dilation
from pytcl.astronomical.relativity import lense_thirring_precession
from pytcl.astronomical.relativity import post_newtonian_acceleration
from pytcl.astronomical.relativity import proper_time_rate
from pytcl.astronomical.relativity import relativistic_range_correction
from pytcl.astronomical.relativity import schwarzschild_precession_per_orbit
from pytcl.astronomical.relativity import schwarzschild_radius
from pytcl.astronomical.relativity import shapiro_delay
from pytcl.astronomical.time_systems import (
    JD_GPS_EPOCH,  # Julian dates; Time scales; Unix time; GPS week; Sidereal time; Leap seconds; Constants
)
from pytcl.astronomical.time_systems import JD_J2000
from pytcl.astronomical.time_systems import JD_UNIX_EPOCH
from pytcl.astronomical.time_systems import MJD_OFFSET
from pytcl.astronomical.time_systems import TT_TAI_OFFSET
from pytcl.astronomical.time_systems import LeapSecondTable
from pytcl.astronomical.time_systems import cal_to_jd
from pytcl.astronomical.time_systems import gast
from pytcl.astronomical.time_systems import get_leap_seconds
from pytcl.astronomical.time_systems import gmst
from pytcl.astronomical.time_systems import gps_to_tai
from pytcl.astronomical.time_systems import gps_to_utc
from pytcl.astronomical.time_systems import gps_week_seconds
from pytcl.astronomical.time_systems import gps_week_to_utc
from pytcl.astronomical.time_systems import jd_to_cal
from pytcl.astronomical.time_systems import jd_to_mjd
from pytcl.astronomical.time_systems import jd_to_unix
from pytcl.astronomical.time_systems import mjd_to_jd
from pytcl.astronomical.time_systems import tai_to_gps
from pytcl.astronomical.time_systems import tai_to_tt
from pytcl.astronomical.time_systems import tai_to_utc
from pytcl.astronomical.time_systems import tt_to_tai
from pytcl.astronomical.time_systems import tt_to_utc
from pytcl.astronomical.time_systems import unix_to_jd
from pytcl.astronomical.time_systems import utc_to_gps
from pytcl.astronomical.time_systems import utc_to_tai
from pytcl.astronomical.time_systems import utc_to_tt

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
