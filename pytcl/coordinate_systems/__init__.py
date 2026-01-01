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
from pytcl.coordinate_systems import conversions
from pytcl.coordinate_systems import jacobians
from pytcl.coordinate_systems import projections
from pytcl.coordinate_systems import rotations

# Geodetic conversions
# Spherical/polar conversions
from pytcl.coordinate_systems.conversions import cart2cyl
from pytcl.coordinate_systems.conversions import cart2pol
from pytcl.coordinate_systems.conversions import cart2ruv
from pytcl.coordinate_systems.conversions import cart2sphere
from pytcl.coordinate_systems.conversions import cyl2cart
from pytcl.coordinate_systems.conversions import ecef2enu
from pytcl.coordinate_systems.conversions import ecef2geodetic
from pytcl.coordinate_systems.conversions import ecef2ned
from pytcl.coordinate_systems.conversions import enu2ecef
from pytcl.coordinate_systems.conversions import enu2ned
from pytcl.coordinate_systems.conversions import geocentric_radius
from pytcl.coordinate_systems.conversions import geodetic2ecef
from pytcl.coordinate_systems.conversions import geodetic2enu
from pytcl.coordinate_systems.conversions import meridional_radius
from pytcl.coordinate_systems.conversions import ned2ecef
from pytcl.coordinate_systems.conversions import ned2enu
from pytcl.coordinate_systems.conversions import pol2cart
from pytcl.coordinate_systems.conversions import prime_vertical_radius
from pytcl.coordinate_systems.conversions import ruv2cart
from pytcl.coordinate_systems.conversions import sphere2cart

# Jacobians
from pytcl.coordinate_systems.jacobians import cross_covariance_transform
from pytcl.coordinate_systems.jacobians import enu_jacobian
from pytcl.coordinate_systems.jacobians import geodetic_jacobian
from pytcl.coordinate_systems.jacobians import ned_jacobian
from pytcl.coordinate_systems.jacobians import numerical_jacobian
from pytcl.coordinate_systems.jacobians import polar_jacobian
from pytcl.coordinate_systems.jacobians import polar_jacobian_inv
from pytcl.coordinate_systems.jacobians import ruv_jacobian
from pytcl.coordinate_systems.jacobians import spherical_jacobian
from pytcl.coordinate_systems.jacobians import spherical_jacobian_inv

# Projections
from pytcl.coordinate_systems.projections import azimuthal_equidistant
from pytcl.coordinate_systems.projections import azimuthal_equidistant_inverse
from pytcl.coordinate_systems.projections import geodetic2utm
from pytcl.coordinate_systems.projections import lambert_conformal_conic
from pytcl.coordinate_systems.projections import lambert_conformal_conic_inverse
from pytcl.coordinate_systems.projections import mercator
from pytcl.coordinate_systems.projections import mercator_inverse
from pytcl.coordinate_systems.projections import polar_stereographic
from pytcl.coordinate_systems.projections import stereographic
from pytcl.coordinate_systems.projections import stereographic_inverse
from pytcl.coordinate_systems.projections import transverse_mercator
from pytcl.coordinate_systems.projections import transverse_mercator_inverse
from pytcl.coordinate_systems.projections import utm2geodetic
from pytcl.coordinate_systems.projections import utm_central_meridian
from pytcl.coordinate_systems.projections import utm_zone

# Rotation operations
from pytcl.coordinate_systems.rotations import axisangle2rotmat
from pytcl.coordinate_systems.rotations import dcm_rate
from pytcl.coordinate_systems.rotations import euler2quat
from pytcl.coordinate_systems.rotations import euler2rotmat
from pytcl.coordinate_systems.rotations import is_rotation_matrix
from pytcl.coordinate_systems.rotations import quat2euler
from pytcl.coordinate_systems.rotations import quat2rotmat
from pytcl.coordinate_systems.rotations import quat_conjugate
from pytcl.coordinate_systems.rotations import quat_inverse
from pytcl.coordinate_systems.rotations import quat_multiply
from pytcl.coordinate_systems.rotations import quat_rotate
from pytcl.coordinate_systems.rotations import rodrigues2rotmat
from pytcl.coordinate_systems.rotations import rotmat2axisangle
from pytcl.coordinate_systems.rotations import rotmat2euler
from pytcl.coordinate_systems.rotations import rotmat2quat
from pytcl.coordinate_systems.rotations import rotmat2rodrigues
from pytcl.coordinate_systems.rotations import rotx
from pytcl.coordinate_systems.rotations import roty
from pytcl.coordinate_systems.rotations import rotz
from pytcl.coordinate_systems.rotations import slerp

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
