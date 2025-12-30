"""
Coordinate conversions.

This module provides:
- Spherical/polar coordinate conversions
- Geodetic (lat/lon/alt) to ECEF conversions
- Local tangent plane frames (ENU, NED)
- Direction cosine representations (r-u-v)
"""

from pytcl.coordinate_systems.conversions.spherical import (
    cart2sphere,
    sphere2cart,
    cart2pol,
    pol2cart,
    cart2cyl,
    cyl2cart,
    ruv2cart,
    cart2ruv,
)

from pytcl.coordinate_systems.conversions.geodetic import (
    geodetic2ecef,
    ecef2geodetic,
    geodetic2enu,
    ecef2enu,
    enu2ecef,
    ecef2ned,
    ned2ecef,
    enu2ned,
    ned2enu,
    geocentric_radius,
    prime_vertical_radius,
    meridional_radius,
)

__all__ = [
    # Spherical/polar
    "cart2sphere",
    "sphere2cart",
    "cart2pol",
    "pol2cart",
    "cart2cyl",
    "cyl2cart",
    "ruv2cart",
    "cart2ruv",
    # Geodetic
    "geodetic2ecef",
    "ecef2geodetic",
    "geodetic2enu",
    "ecef2enu",
    "enu2ecef",
    "ecef2ned",
    "ned2ecef",
    "enu2ned",
    "ned2enu",
    "geocentric_radius",
    "prime_vertical_radius",
    "meridional_radius",
]
