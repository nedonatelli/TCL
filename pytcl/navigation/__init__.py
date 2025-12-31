"""
Navigation utilities for target tracking.

This module provides geodetic and navigation calculations commonly
needed in tracking applications, including:
- Geodetic coordinate conversions
- Inertial Navigation System (INS) mechanization
- Alignment algorithms
"""

from pytcl.navigation.geodesy import (  # Ellipsoids; Coordinate conversions; Geodetic problems
    GRS80,
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
from pytcl.navigation.ins import (  # Constants; State representation; Gravity and Earth rate
    A_EARTH,
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
]
