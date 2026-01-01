"""
Navigation utilities for target tracking.

This module provides geodetic and navigation calculations commonly
needed in tracking applications, including:
- Geodetic coordinate conversions
- Inertial Navigation System (INS) mechanization
- Alignment algorithms
- INS/GNSS integration (loosely and tightly coupled)
- Great circle navigation
- Rhumb line navigation
"""

from pytcl.navigation.geodesy import (
    GRS80,  # Ellipsoids; Coordinate conversions; Geodetic problems
)
from pytcl.navigation.geodesy import SPHERE
from pytcl.navigation.geodesy import WGS84
from pytcl.navigation.geodesy import Ellipsoid
from pytcl.navigation.geodesy import direct_geodetic
from pytcl.navigation.geodesy import ecef_to_enu
from pytcl.navigation.geodesy import ecef_to_geodetic
from pytcl.navigation.geodesy import ecef_to_ned
from pytcl.navigation.geodesy import enu_to_ecef
from pytcl.navigation.geodesy import geodetic_to_ecef
from pytcl.navigation.geodesy import haversine_distance
from pytcl.navigation.geodesy import inverse_geodetic
from pytcl.navigation.geodesy import ned_to_ecef
from pytcl.navigation.great_circle import EARTH_RADIUS  # Great circle navigation
from pytcl.navigation.great_circle import CrossTrackResult
from pytcl.navigation.great_circle import GreatCircleResult
from pytcl.navigation.great_circle import IntersectionResult
from pytcl.navigation.great_circle import WaypointResult
from pytcl.navigation.great_circle import angular_distance
from pytcl.navigation.great_circle import cross_track_distance
from pytcl.navigation.great_circle import destination_point
from pytcl.navigation.great_circle import great_circle_azimuth
from pytcl.navigation.great_circle import great_circle_direct
from pytcl.navigation.great_circle import great_circle_distance
from pytcl.navigation.great_circle import great_circle_intersect
from pytcl.navigation.great_circle import great_circle_inverse
from pytcl.navigation.great_circle import great_circle_path_intersect
from pytcl.navigation.great_circle import great_circle_tdoa_loc
from pytcl.navigation.great_circle import great_circle_waypoint
from pytcl.navigation.great_circle import great_circle_waypoints
from pytcl.navigation.ins import (
    A_EARTH,  # Constants; State representation; Gravity and Earth rate
)
from pytcl.navigation.ins import B_EARTH
from pytcl.navigation.ins import E2_EARTH
from pytcl.navigation.ins import F_EARTH
from pytcl.navigation.ins import GM_EARTH
from pytcl.navigation.ins import OMEGA_EARTH
from pytcl.navigation.ins import IMUData
from pytcl.navigation.ins import INSErrorState
from pytcl.navigation.ins import INSState
from pytcl.navigation.ins import coarse_alignment
from pytcl.navigation.ins import compensate_imu_data
from pytcl.navigation.ins import coning_correction
from pytcl.navigation.ins import earth_rate_ned
from pytcl.navigation.ins import gravity_ned
from pytcl.navigation.ins import gyrocompass_alignment
from pytcl.navigation.ins import initialize_ins_state
from pytcl.navigation.ins import ins_error_state_matrix
from pytcl.navigation.ins import ins_process_noise_matrix
from pytcl.navigation.ins import mechanize_ins_ned
from pytcl.navigation.ins import normal_gravity
from pytcl.navigation.ins import radii_of_curvature
from pytcl.navigation.ins import sculling_correction
from pytcl.navigation.ins import skew_symmetric
from pytcl.navigation.ins import transport_rate_ned
from pytcl.navigation.ins import update_attitude_ned
from pytcl.navigation.ins import update_quaternion
from pytcl.navigation.ins_gnss import GPS_L1_FREQ  # INS/GNSS integration
from pytcl.navigation.ins_gnss import GPS_L1_WAVELENGTH
from pytcl.navigation.ins_gnss import SPEED_OF_LIGHT
from pytcl.navigation.ins_gnss import GNSSMeasurement
from pytcl.navigation.ins_gnss import INSGNSSState
from pytcl.navigation.ins_gnss import LooseCoupledResult
from pytcl.navigation.ins_gnss import SatelliteInfo
from pytcl.navigation.ins_gnss import TightCoupledResult
from pytcl.navigation.ins_gnss import compute_dop
from pytcl.navigation.ins_gnss import compute_line_of_sight
from pytcl.navigation.ins_gnss import gnss_outage_detection
from pytcl.navigation.ins_gnss import initialize_ins_gnss
from pytcl.navigation.ins_gnss import loose_coupled_predict
from pytcl.navigation.ins_gnss import loose_coupled_update
from pytcl.navigation.ins_gnss import loose_coupled_update_position
from pytcl.navigation.ins_gnss import loose_coupled_update_velocity
from pytcl.navigation.ins_gnss import position_measurement_matrix
from pytcl.navigation.ins_gnss import position_velocity_measurement_matrix
from pytcl.navigation.ins_gnss import pseudorange_measurement_matrix
from pytcl.navigation.ins_gnss import satellite_elevation_azimuth
from pytcl.navigation.ins_gnss import tight_coupled_measurement_matrix
from pytcl.navigation.ins_gnss import tight_coupled_pseudorange_innovation
from pytcl.navigation.ins_gnss import tight_coupled_update
from pytcl.navigation.ins_gnss import velocity_measurement_matrix
from pytcl.navigation.rhumb import RhumbDirectResult  # Rhumb line navigation
from pytcl.navigation.rhumb import RhumbIntersectionResult
from pytcl.navigation.rhumb import RhumbResult
from pytcl.navigation.rhumb import compare_great_circle_rhumb
from pytcl.navigation.rhumb import direct_rhumb
from pytcl.navigation.rhumb import direct_rhumb_spherical
from pytcl.navigation.rhumb import indirect_rhumb
from pytcl.navigation.rhumb import indirect_rhumb_spherical
from pytcl.navigation.rhumb import rhumb_bearing
from pytcl.navigation.rhumb import rhumb_distance_ellipsoidal
from pytcl.navigation.rhumb import rhumb_distance_spherical
from pytcl.navigation.rhumb import rhumb_intersect
from pytcl.navigation.rhumb import rhumb_midpoint
from pytcl.navigation.rhumb import rhumb_waypoints

__all__ = [
    # Ellipsoids
    "Ellipsoid",
    "WGS84",
    "GRS80",
    "SPHERE",
    # Coordinate conversions
    "geodetic_to_ecef",
    "ecef_to_geodetic",
    "ecef_to_enu",
    "enu_to_ecef",
    "ecef_to_ned",
    "ned_to_ecef",
    # Geodetic problems
    "direct_geodetic",
    "inverse_geodetic",
    "haversine_distance",
    # INS Constants
    "OMEGA_EARTH",
    "GM_EARTH",
    "A_EARTH",
    "F_EARTH",
    "B_EARTH",
    "E2_EARTH",
    # INS State representation
    "INSState",
    "IMUData",
    "INSErrorState",
    # INS Gravity and Earth rate
    "normal_gravity",
    "gravity_ned",
    "earth_rate_ned",
    "transport_rate_ned",
    "radii_of_curvature",
    # INS Coning and sculling
    "coning_correction",
    "sculling_correction",
    "compensate_imu_data",
    # INS Attitude
    "skew_symmetric",
    "update_quaternion",
    "update_attitude_ned",
    # INS Mechanization
    "mechanize_ins_ned",
    "initialize_ins_state",
    # INS Alignment
    "coarse_alignment",
    "gyrocompass_alignment",
    # INS Error state model
    "ins_error_state_matrix",
    "ins_process_noise_matrix",
    # GNSS Constants
    "SPEED_OF_LIGHT",
    "GPS_L1_FREQ",
    "GPS_L1_WAVELENGTH",
    # GNSS State representation
    "GNSSMeasurement",
    "SatelliteInfo",
    "INSGNSSState",
    "LooseCoupledResult",
    "TightCoupledResult",
    # GNSS Measurement models
    "position_measurement_matrix",
    "velocity_measurement_matrix",
    "position_velocity_measurement_matrix",
    "compute_line_of_sight",
    "pseudorange_measurement_matrix",
    "compute_dop",
    "satellite_elevation_azimuth",
    # Loosely-coupled integration
    "initialize_ins_gnss",
    "loose_coupled_predict",
    "loose_coupled_update_position",
    "loose_coupled_update_velocity",
    "loose_coupled_update",
    # Tightly-coupled integration
    "tight_coupled_pseudorange_innovation",
    "tight_coupled_measurement_matrix",
    "tight_coupled_update",
    # Fault detection
    "gnss_outage_detection",
    # Great circle navigation
    "EARTH_RADIUS",
    "GreatCircleResult",
    "WaypointResult",
    "IntersectionResult",
    "CrossTrackResult",
    "great_circle_distance",
    "great_circle_azimuth",
    "great_circle_inverse",
    "great_circle_waypoint",
    "great_circle_waypoints",
    "great_circle_direct",
    "cross_track_distance",
    "great_circle_intersect",
    "great_circle_path_intersect",
    "great_circle_tdoa_loc",
    "angular_distance",
    "destination_point",
    # Rhumb line navigation
    "RhumbResult",
    "RhumbDirectResult",
    "RhumbIntersectionResult",
    "rhumb_distance_spherical",
    "rhumb_bearing",
    "indirect_rhumb_spherical",
    "direct_rhumb_spherical",
    "rhumb_distance_ellipsoidal",
    "indirect_rhumb",
    "direct_rhumb",
    "rhumb_intersect",
    "rhumb_midpoint",
    "rhumb_waypoints",
    "compare_great_circle_rhumb",
]
