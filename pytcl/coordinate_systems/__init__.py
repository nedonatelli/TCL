"""
Coordinate system conversions and transformations.

This module provides functions for converting between different coordinate
systems commonly used in tracking applications:

- Cartesian coordinates (x, y, z)
- Spherical coordinates (range, azimuth, elevation)
- Polar and cylindrical coordinates
- Geodetic coordinates (latitude, longitude, altitude)
- Various local tangent plane frames (ENU, NED)
- Direction cosine representations (r-u-v)
- Rotation representations (matrices, quaternions, Euler angles)
- Jacobian matrices for error propagation
"""

# Import submodules for easy access
from pytcl.coordinate_systems import conversions, jacobians, projections, rotations

# Geodetic conversions
# Spherical/polar conversions
from pytcl.coordinate_systems.conversions import (
    cart2cyl,
    cart2pol,
    cart2ruv,
    cart2sphere,
    cyl2cart,
    ecef2enu,
    ecef2geodetic,
    ecef2ned,
    enu2ecef,
    enu2ned,
    geocentric_radius,
    geodetic2ecef,
    geodetic2enu,
    meridional_radius,
    ned2ecef,
    ned2enu,
    pol2cart,
    prime_vertical_radius,
    ruv2cart,
    sphere2cart,
)

# Jacobians
from pytcl.coordinate_systems.jacobians import (
    cross_covariance_transform,
    enu_jacobian,
    geodetic_jacobian,
    ned_jacobian,
    numerical_jacobian,
    polar_jacobian,
    polar_jacobian_inv,
    ruv_jacobian,
    spherical_jacobian,
    spherical_jacobian_inv,
)

# Projections
from pytcl.coordinate_systems.projections import (
    azimuthal_equidistant,
    azimuthal_equidistant_inverse,
    geodetic2utm,
    lambert_conformal_conic,
    lambert_conformal_conic_inverse,
    mercator,
    mercator_inverse,
    polar_stereographic,
    stereographic,
    stereographic_inverse,
    transverse_mercator,
    transverse_mercator_inverse,
    utm2geodetic,
    utm_central_meridian,
    utm_zone,
)

# Rotation operations
from pytcl.coordinate_systems.rotations import (
    axisangle2rotmat,
    dcm_rate,
    euler2quat,
    euler2rotmat,
    is_rotation_matrix,
    quat2euler,
    quat2rotmat,
    quat_conjugate,
    quat_inverse,
    quat_multiply,
    quat_rotate,
    rodrigues2rotmat,
    rotmat2axisangle,
    rotmat2euler,
    rotmat2quat,
    rotmat2rodrigues,
    rotx,
    roty,
    rotz,
    slerp,
)

# Re-export commonly used functions at the top level


__all__ = [
    # Submodules
    "conversions",
    "rotations",
    "jacobians",
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
    # Rotations
    "rotx",
    "roty",
    "rotz",
    "euler2rotmat",
    "rotmat2euler",
    "axisangle2rotmat",
    "rotmat2axisangle",
    "quat2rotmat",
    "rotmat2quat",
    "euler2quat",
    "quat2euler",
    "quat_multiply",
    "quat_conjugate",
    "quat_inverse",
    "quat_rotate",
    "slerp",
    "rodrigues2rotmat",
    "rotmat2rodrigues",
    "dcm_rate",
    "is_rotation_matrix",
    # Projections
    "projections",
    "mercator",
    "mercator_inverse",
    "transverse_mercator",
    "transverse_mercator_inverse",
    "utm_zone",
    "utm_central_meridian",
    "geodetic2utm",
    "utm2geodetic",
    "stereographic",
    "stereographic_inverse",
    "polar_stereographic",
    "lambert_conformal_conic",
    "lambert_conformal_conic_inverse",
    "azimuthal_equidistant",
    "azimuthal_equidistant_inverse",
    # Jacobians
    "spherical_jacobian",
    "spherical_jacobian_inv",
    "polar_jacobian",
    "polar_jacobian_inv",
    "ruv_jacobian",
    "enu_jacobian",
    "ned_jacobian",
    "geodetic_jacobian",
    "cross_covariance_transform",
    "numerical_jacobian",
]
