"""
Rotation representations and conversions.

This module provides:
- Basic rotation matrices (rotx, roty, rotz)
- Euler angle conversions
- Quaternion operations
- Axis-angle and Rodrigues representations
- Rotation interpolation (SLERP)
"""

from pytcl.coordinate_systems.rotations.rotations import (
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
