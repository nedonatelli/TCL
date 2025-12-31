"""
Navigation utilities for target tracking.

This module provides geodetic and navigation calculations commonly
needed in tracking applications.
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
