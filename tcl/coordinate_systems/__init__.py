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
from tcl.coordinate_systems import conversions
from tcl.coordinate_systems import rotations
from tcl.coordinate_systems import jacobians

# Re-export commonly used functions at the top level

# Spherical/polar conversions
from tcl.coordinate_systems.conversions import (
    cart2sphere,
    sphere2cart,
    cart2pol,
    pol2cart,
    cart2cyl,
    cyl2cart,
    ruv2cart,
    cart2ruv,
)

# Geodetic conversions
from tcl.coordinate_systems.conversions import (
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

# Rotation operations
from tcl.coordinate_systems.rotations import (
    rotx,
    roty,
    rotz,
    euler2rotmat,
    rotmat2euler,
    axisangle2rotmat,
    rotmat2axisangle,
    quat2rotmat,
    rotmat2quat,
    euler2quat,
    quat2euler,
    quat_multiply,
    quat_conjugate,
    quat_inverse,
    quat_rotate,
    slerp,
    rodrigues2rotmat,
    rotmat2rodrigues,
    dcm_rate,
    is_rotation_matrix,
)

# Jacobians
from tcl.coordinate_systems.jacobians import (
    spherical_jacobian,
    spherical_jacobian_inv,
    polar_jacobian,
    polar_jacobian_inv,
    ruv_jacobian,
    enu_jacobian,
    ned_jacobian,
    geodetic_jacobian,
    cross_covariance_transform,
    numerical_jacobian,
)

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
