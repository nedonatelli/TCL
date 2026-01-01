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
from pytcl.navigation.geodesy import (
    SPHERE,
    WGS84,
    Ellipsoid,
    direct_geodetic,
    ecef_to_enu,
    ecef_to_geodetic,
    ecef_to_ned,
    enu_to_ecef,
    geodetic_to_ecef,
    haversine_distance,
    inverse_geodetic,
    ned_to_ecef,
)
from pytcl.navigation.great_circle import EARTH_RADIUS  # Great circle navigation
from pytcl.navigation.great_circle import (
    CrossTrackResult,
    GreatCircleResult,
    IntersectionResult,
    WaypointResult,
    angular_distance,
    cross_track_distance,
    destination_point,
    great_circle_azimuth,
    great_circle_direct,
    great_circle_distance,
    great_circle_intersect,
    great_circle_inverse,
    great_circle_path_intersect,
    great_circle_tdoa_loc,
    great_circle_waypoint,
    great_circle_waypoints,
)
from pytcl.navigation.ins import (
    A_EARTH,  # Constants; State representation; Gravity and Earth rate
)
from pytcl.navigation.ins import (
    B_EARTH,
    E2_EARTH,
    F_EARTH,
    GM_EARTH,
    OMEGA_EARTH,
    IMUData,
    INSErrorState,
    INSState,
    coarse_alignment,
    compensate_imu_data,
    coning_correction,
    earth_rate_ned,
    gravity_ned,
    gyrocompass_alignment,
    initialize_ins_state,
    ins_error_state_matrix,
    ins_process_noise_matrix,
    mechanize_ins_ned,
    normal_gravity,
    radii_of_curvature,
    sculling_correction,
    skew_symmetric,
    transport_rate_ned,
    update_attitude_ned,
    update_quaternion,
)
from pytcl.navigation.ins_gnss import GPS_L1_FREQ  # INS/GNSS integration
from pytcl.navigation.ins_gnss import (
    GPS_L1_WAVELENGTH,
    SPEED_OF_LIGHT,
    GNSSMeasurement,
    INSGNSSState,
    LooseCoupledResult,
    SatelliteInfo,
    TightCoupledResult,
    compute_dop,
    compute_line_of_sight,
    gnss_outage_detection,
    initialize_ins_gnss,
    loose_coupled_predict,
    loose_coupled_update,
    loose_coupled_update_position,
    loose_coupled_update_velocity,
    position_measurement_matrix,
    position_velocity_measurement_matrix,
    pseudorange_measurement_matrix,
    satellite_elevation_azimuth,
    tight_coupled_measurement_matrix,
    tight_coupled_pseudorange_innovation,
    tight_coupled_update,
    velocity_measurement_matrix,
)
from pytcl.navigation.rhumb import RhumbDirectResult  # Rhumb line navigation
from pytcl.navigation.rhumb import (
    RhumbIntersectionResult,
    RhumbResult,
    compare_great_circle_rhumb,
    direct_rhumb,
    direct_rhumb_spherical,
    indirect_rhumb,
    indirect_rhumb_spherical,
    rhumb_bearing,
    rhumb_distance_ellipsoidal,
    rhumb_distance_spherical,
    rhumb_intersect,
    rhumb_midpoint,
    rhumb_waypoints,
)

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
