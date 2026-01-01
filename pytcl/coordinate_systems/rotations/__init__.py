"""
Rotation representations and conversions.

This module provides:
- Basic rotation matrices (rotx, roty, rotz)
- Euler angle conversions
- Quaternion operations
- Axis-angle and Rodrigues representations
- Rotation interpolation (SLERP)
"""

from pytcl.coordinate_systems.rotations.rotations import axisangle2rotmat
from pytcl.coordinate_systems.rotations.rotations import dcm_rate
from pytcl.coordinate_systems.rotations.rotations import euler2quat
from pytcl.coordinate_systems.rotations.rotations import euler2rotmat
from pytcl.coordinate_systems.rotations.rotations import is_rotation_matrix
from pytcl.coordinate_systems.rotations.rotations import quat2euler
from pytcl.coordinate_systems.rotations.rotations import quat2rotmat
from pytcl.coordinate_systems.rotations.rotations import quat_conjugate
from pytcl.coordinate_systems.rotations.rotations import quat_inverse
from pytcl.coordinate_systems.rotations.rotations import quat_multiply
from pytcl.coordinate_systems.rotations.rotations import quat_rotate
from pytcl.coordinate_systems.rotations.rotations import rodrigues2rotmat
from pytcl.coordinate_systems.rotations.rotations import rotmat2axisangle
from pytcl.coordinate_systems.rotations.rotations import rotmat2euler
from pytcl.coordinate_systems.rotations.rotations import rotmat2quat
from pytcl.coordinate_systems.rotations.rotations import rotmat2rodrigues
from pytcl.coordinate_systems.rotations.rotations import rotx
from pytcl.coordinate_systems.rotations.rotations import roty
from pytcl.coordinate_systems.rotations.rotations import rotz
from pytcl.coordinate_systems.rotations.rotations import slerp

__all__ = [
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
]
