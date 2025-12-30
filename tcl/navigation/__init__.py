"""
Navigation utilities for target tracking.

This module provides geodetic and navigation calculations commonly
needed in tracking applications.
"""

from tracker_component_library.navigation.geodesy import (
    # Ellipsoids
    Ellipsoid,
    WGS84,
    GRS80,
    SPHERE,
    # Coordinate conversions
    geodetic_to_ecef,
    ecef_to_geodetic,
    ecef_to_enu,
    enu_to_ecef,
    ecef_to_ned,
    ned_to_ecef,
    # Geodetic problems
    direct_geodetic,
    inverse_geodetic,
    haversine_distance,
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
]
