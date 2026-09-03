"""
Coordinate conversions.

This module provides:
- Spherical/polar coordinate conversions
- Geodetic (lat/lon/alt) to ECEF conversions
- Local tangent plane frames (ENU, NED)
- Direction cosine representations (r-u-v)
"""

from pytcl.coordinate_systems.conversions.geodetic import (
    ecef2enu,
    ecef2geodetic,
    ecef2ned,
    ecef2sez,
    enu2ecef,
    enu2ned,
    geocentric_radius,
    geodetic2ecef,
    geodetic2enu,
    geodetic2sez,
    meridional_radius,
    ned2ecef,
    ned2enu,
    prime_vertical_radius,
    sez2ecef,
    sez2geodetic,
)
from pytcl.coordinate_systems.conversions.spherical import (
    cart2cyl,
    cart2pol,
    cart2ruv,
    cart2sphere,
    cyl2cart,
    pol2cart,
    ruv2cart,
    sphere2cart,
)
from pytcl.coordinate_systems.conversions.uv import (
    camera_coords2uv,
    cart2ruv_bistatic,
    ruv2cart_bistatic,
    ruv2ruv,
    spher_ang2uv,
    state_ruv2cart,
    uv2spher_ang,
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
    # u-v direction cosines
    "camera_coords2uv",
    "cart2ruv_bistatic",
    "ruv2cart_bistatic",
    "ruv2ruv",
    "spher_ang2uv",
    "state_ruv2cart",
    "uv2spher_ang",
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
    # SEZ (south-east-zenith) is a standard topocentric frame and its four
    # conversions were the only ones in this module left unexported, while
    # every ENU and NED sibling beside them was public.
    "geodetic2sez",
    "sez2geodetic",
    "ecef2sez",
    "sez2ecef",
    "geocentric_radius",
    "prime_vertical_radius",
    "meridional_radius",
]
