"""
Rotation representations and conversions.

This module provides functions for working with different rotation
representations including rotation matrices, quaternions, Euler angles,
axis-angle, and Rodrigues parameters.
"""

from typing import Any, Tuple

import numpy as np
from numba import njit
from numpy.typing import ArrayLike, NDArray


@njit(cache=True, fastmath=True)
def _rotx_inplace(angle: float, R: np.ndarray[Any, Any]) -> None:
    """JIT-compiled rotation about x-axis (fills existing matrix)."""
    c = np.cos(angle)
    s = np.sin(angle)
    R[0, 0] = 1.0
    R[0, 1] = 0.0
    R[0, 2] = 0.0
    R[1, 0] = 0.0
    R[1, 1] = c
    R[1, 2] = -s
    R[2, 0] = 0.0
    R[2, 1] = s
    R[2, 2] = c


@njit(cache=True, fastmath=True)
def _roty_inplace(angle: float, R: np.ndarray[Any, Any]) -> None:
    """JIT-compiled rotation about y-axis (fills existing matrix)."""
    c = np.cos(angle)
    s = np.sin(angle)
    R[0, 0] = c
    R[0, 1] = 0.0
    R[0, 2] = s
    R[1, 0] = 0.0
    R[1, 1] = 1.0
    R[1, 2] = 0.0
    R[2, 0] = -s
    R[2, 1] = 0.0
    R[2, 2] = c


@njit(cache=True, fastmath=True)
def _rotz_inplace(angle: float, R: np.ndarray[Any, Any]) -> None:
    """JIT-compiled rotation about z-axis (fills existing matrix)."""
    c = np.cos(angle)
    s = np.sin(angle)
    R[0, 0] = c
    R[0, 1] = -s
    R[0, 2] = 0.0
    R[1, 0] = s
    R[1, 1] = c
    R[1, 2] = 0.0
    R[2, 0] = 0.0
    R[2, 1] = 0.0
    R[2, 2] = 1.0


@njit(cache=True, fastmath=True)
def _euler_zyx_to_rotmat(
    yaw: float, pitch: float, roll: float, R: np.ndarray[Any, Any]
) -> None:
    """JIT-compiled ZYX Euler angles to rotation matrix."""
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cr = np.cos(roll)
    sr = np.sin(roll)

    # R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    R[0, 0] = cy * cp
    R[0, 1] = cy * sp * sr - sy * cr
    R[0, 2] = cy * sp * cr + sy * sr
    R[1, 0] = sy * cp
    R[1, 1] = sy * sp * sr + cy * cr
    R[1, 2] = sy * sp * cr - cy * sr
    R[2, 0] = -sp
    R[2, 1] = cp * sr
    R[2, 2] = cp * cr


@njit(cache=True, fastmath=True)
def _matmul_3x3(
    A: np.ndarray[Any, Any], B: np.ndarray[Any, Any], C: np.ndarray[Any, Any]
) -> None:
    """JIT-compiled 3x3 matrix multiplication C = A @ B."""
    for i in range(3):
        for j in range(3):
            C[i, j] = 0.0
            for k in range(3):
                C[i, j] += A[i, k] * B[k, j]


def rotx(angle: float) -> NDArray[np.floating]:
    """
    Create rotation matrix for rotation about the x-axis.

    Parameters
    ----------
    angle : float
        Rotation angle in radians.

    Returns
    -------
    R : ndarray
        3x3 rotation matrix.

    Examples
    --------
    >>> R = rotx(np.pi/2)  # 90 degree rotation about x
    >>> R @ [0, 1, 0]  # y-axis maps to z-axis
    array([0., 0., 1.])
    """
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def roty(angle: float) -> NDArray[np.floating]:
    """
    Create rotation matrix for rotation about the y-axis.

    Parameters
    ----------
    angle : float
        Rotation angle in radians.

    Returns
    -------
    R : ndarray
        3x3 rotation matrix.
    """
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def rotz(angle: float) -> NDArray[np.floating]:
    """
    Create rotation matrix for rotation about the z-axis.

    Parameters
    ----------
    angle : float
        Rotation angle in radians.

    Returns
    -------
    R : ndarray
        3x3 rotation matrix.
    """
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def euler2rotmat(
    angles: ArrayLike,
    sequence: str = "ZYX",
) -> NDArray[np.floating]:
    """
    Convert Euler angles to rotation matrix.

    Parameters
    ----------
    angles : array_like
        Three Euler angles in radians [angle1, angle2, angle3].
    sequence : str, optional
        Rotation sequence (e.g., 'ZYX', 'XYZ', 'ZXZ').
        Default is 'ZYX' (aerospace convention: yaw-pitch-roll).

    Returns
    -------
    R : ndarray
        3x3 rotation matrix.

    Examples
    --------
    >>> yaw, pitch, roll = np.radians([45, 30, 15])
    >>> R = euler2rotmat([yaw, pitch, roll], 'ZYX')

    Notes
    -----
    The rotation is applied in the order specified by the sequence,
    from right to left. For 'ZYX': R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
    """
    angles = np.asarray(angles, dtype=np.float64)

    rot_funcs = {"X": rotx, "Y": roty, "Z": rotz}

    if len(sequence) != 3:
        raise ValueError("Sequence must have exactly 3 characters")

    R = np.eye(3, dtype=np.float64)
    for i, axis in enumerate(sequence):
        if axis.upper() not in rot_funcs:
            raise ValueError(f"Invalid axis: {axis}")
        R = R @ rot_funcs[axis.upper()](angles[i])

    return R


def rotmat2euler(
    R: ArrayLike,
    sequence: str = "ZYX",
) -> NDArray[np.floating]:
    """
    Convert rotation matrix to Euler angles.

    Parameters
    ----------
    R : array_like
        3x3 rotation matrix.
    sequence : str, optional
        Rotation sequence. Default is 'ZYX'.

    Returns
    -------
    angles : ndarray
        Three Euler angles in radians.

    Notes
    -----
    May have singularities (gimbal lock) at certain angles.
    """
    R = np.asarray(R, dtype=np.float64)

    if sequence == "ZYX":
        # Aerospace convention: yaw-pitch-roll
        # R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)

        if sy > 1e-6:
            roll = np.arctan2(R[2, 1], R[2, 2])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            # Gimbal lock
            roll = np.arctan2(-R[1, 2], R[1, 1])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = 0

        return np.array([yaw, pitch, roll], dtype=np.float64)

    elif sequence == "XYZ":
        sy = np.sqrt(R[0, 0] ** 2 + R[0, 1] ** 2)

        if sy > 1e-6:
            x = np.arctan2(R[1, 2], R[2, 2])
            y = np.arctan2(-R[0, 2], sy)
            z = np.arctan2(R[0, 1], R[0, 0])
        else:
            x = np.arctan2(-R[2, 1], R[1, 1])
            y = np.arctan2(-R[0, 2], sy)
            z = 0

        return np.array([x, y, z], dtype=np.float64)

    elif sequence == "ZXZ":
        # Classic Euler angles
        if np.abs(R[2, 2]) < 1 - 1e-6:
            alpha = np.arctan2(R[0, 2], -R[1, 2])
            beta = np.arccos(R[2, 2])
            gamma = np.arctan2(R[2, 0], R[2, 1])
        else:
            alpha = np.arctan2(-R[0, 1], R[0, 0])
            beta = 0 if R[2, 2] > 0 else np.pi
            gamma = 0

        return np.array([alpha, beta, gamma], dtype=np.float64)

    else:
        raise ValueError(f"Unsupported sequence: {sequence}")


def axisangle2rotmat(
    axis: ArrayLike,
    angle: float,
) -> NDArray[np.floating]:
    """
    Convert axis-angle representation to rotation matrix.

    Parameters
    ----------
    axis : array_like
        Unit vector defining the rotation axis [ax, ay, az].
    angle : float
        Rotation angle in radians.

    Returns
    -------
    R : ndarray
        3x3 rotation matrix.

    Notes
    -----
    Uses Rodrigues' rotation formula.
    """
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)

    c = np.cos(angle)
    s = np.sin(angle)
    t = 1 - c

    x, y, z = axis

    R = np.array(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ],
        dtype=np.float64,
    )

    return R


def rotmat2axisangle(
    R: ArrayLike,
) -> Tuple[NDArray[np.floating], float]:
    """
    Convert rotation matrix to axis-angle representation.

    Parameters
    ----------
    R : array_like
        3x3 rotation matrix.

    Returns
    -------
    axis : ndarray
        Unit vector rotation axis.
    angle : float
        Rotation angle in radians [0, π].
    """
    R = np.asarray(R, dtype=np.float64)

    angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))

    if np.abs(angle) < 1e-10:
        # No rotation
        return np.array([0, 0, 1], dtype=np.float64), 0.0

    if np.abs(angle - np.pi) < 1e-10:
        # 180 degree rotation - axis from eigenvector
        eigenvalues, eigenvectors = np.linalg.eig(R)
        idx = np.argmin(np.abs(eigenvalues - 1))
        axis = np.real(eigenvectors[:, idx])
        return axis / np.linalg.norm(axis), float(angle)

    # General case
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (
        2 * np.sin(angle)
    )

    return axis, float(angle)


def quat2rotmat(q: ArrayLike) -> NDArray[np.floating]:
    """
    Convert quaternion to rotation matrix.

    Parameters
    ----------
    q : array_like
        Quaternion [qw, qx, qy, qz] (scalar-first convention).

    Returns
    -------
    R : ndarray
        3x3 rotation matrix.

    Examples
    --------
    >>> q = [1, 0, 0, 0]  # Identity quaternion
    >>> quat2rotmat(q)
    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])
    """
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q)

    qw, qx, qy, qz = q

    R = np.array(
        [
            [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)],
        ],
        dtype=np.float64,
    )

    return R


def rotmat2quat(R: ArrayLike) -> NDArray[np.floating]:
    """
    Convert rotation matrix to quaternion.

    Parameters
    ----------
    R : array_like
        3x3 rotation matrix.

    Returns
    -------
    q : ndarray
        Quaternion [qw, qx, qy, qz] (scalar-first, positive qw).

    Notes
    -----
    Uses Shepperd's method for numerical stability.
    """
    R = np.asarray(R, dtype=np.float64)

    trace = np.trace(R)

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2 * np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2 * np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2 * np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    q = np.array([qw, qx, qy, qz], dtype=np.float64)

    # Ensure positive scalar part (canonical form)
    if qw < 0:
        q = -q

    return q / np.linalg.norm(q)


def euler2quat(
    angles: ArrayLike,
    sequence: str = "ZYX",
) -> NDArray[np.floating]:
    """
    Convert Euler angles to quaternion.

    Parameters
    ----------
    angles : array_like
        Three Euler angles in radians.
    sequence : str, optional
        Rotation sequence. Default is 'ZYX'.

    Returns
    -------
    q : ndarray
        Quaternion [qw, qx, qy, qz].
    """
    R = euler2rotmat(angles, sequence)
    return rotmat2quat(R)


def quat2euler(
    q: ArrayLike,
    sequence: str = "ZYX",
) -> NDArray[np.floating]:
    """
    Convert quaternion to Euler angles.

    Parameters
    ----------
    q : array_like
        Quaternion [qw, qx, qy, qz].
    sequence : str, optional
        Rotation sequence. Default is 'ZYX'.

    Returns
    -------
    angles : ndarray
        Three Euler angles in radians.
    """
    R = quat2rotmat(q)
    return rotmat2euler(R, sequence)


def quat_multiply(q1: ArrayLike, q2: ArrayLike) -> NDArray[np.floating]:
    """
    Multiply two quaternions.

    Parameters
    ----------
    q1 : array_like
        First quaternion [qw, qx, qy, qz].
    q2 : array_like
        Second quaternion.

    Returns
    -------
    q : ndarray
        Product quaternion q1 * q2.

    Notes
    -----
    Quaternion multiplication represents composition of rotations.
    q1 * q2 applies q2 first, then q1.
    """
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)

    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def quat_conjugate(q: ArrayLike) -> NDArray[np.floating]:
    """
    Compute quaternion conjugate.

    Parameters
    ----------
    q : array_like
        Quaternion [qw, qx, qy, qz].

    Returns
    -------
    q_conj : ndarray
        Conjugate quaternion [qw, -qx, -qy, -qz].
    """
    q = np.asarray(q, dtype=np.float64)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_inverse(q: ArrayLike) -> NDArray[np.floating]:
    """
    Compute quaternion inverse.

    Parameters
    ----------
    q : array_like
        Quaternion [qw, qx, qy, qz].

    Returns
    -------
    q_inv : ndarray
        Inverse quaternion.

    Notes
    -----
    For unit quaternions, inverse equals conjugate.
    """
    q = np.asarray(q, dtype=np.float64)
    return quat_conjugate(q) / np.dot(q, q)


def quat_rotate(q: ArrayLike, v: ArrayLike) -> NDArray[np.floating]:
    """
    Rotate a vector using a quaternion.

    Parameters
    ----------
    q : array_like
        Quaternion [qw, qx, qy, qz].
    v : array_like
        Vector to rotate [x, y, z].

    Returns
    -------
    v_rot : ndarray
        Rotated vector.

    Notes
    -----
    Computes q * v * q^(-1) where v is treated as a pure quaternion.
    """
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    # Use rotation matrix for efficiency
    R = quat2rotmat(q)
    return R @ v


def slerp(
    q1: ArrayLike,
    q2: ArrayLike,
    t: float,
) -> NDArray[np.floating]:
    """
    Spherical linear interpolation between two quaternions.

    Parameters
    ----------
    q1 : array_like
        Start quaternion.
    q2 : array_like
        End quaternion.
    t : float
        Interpolation parameter in [0, 1].

    Returns
    -------
    q : ndarray
        Interpolated quaternion.
    """
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)

    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)

    dot = np.dot(q1, q2)

    # Take shorter path
    if dot < 0:
        q2 = -q2
        dot = -dot

    if dot > 0.9995:
        # Linear interpolation for very close quaternions
        result = q1 + t * (q2 - q1)
        return result / np.linalg.norm(result)

    theta = np.arccos(dot)
    sin_theta = np.sin(theta)

    s1 = np.sin((1 - t) * theta) / sin_theta
    s2 = np.sin(t * theta) / sin_theta

    return s1 * q1 + s2 * q2


def rodrigues2rotmat(rvec: ArrayLike) -> NDArray[np.floating]:
    """
    Convert Rodrigues vector to rotation matrix.

    Parameters
    ----------
    rvec : array_like
        Rodrigues vector (axis * angle).

    Returns
    -------
    R : ndarray
        3x3 rotation matrix.

    Notes
    -----
    The Rodrigues vector encodes both the rotation axis and angle:
    rvec = axis * angle, where |rvec| = angle.
    """
    rvec = np.asarray(rvec, dtype=np.float64)
    angle = np.linalg.norm(rvec)

    if angle < 1e-10:
        return np.eye(3, dtype=np.float64)

    axis = rvec / angle
    return axisangle2rotmat(axis, angle)


def rotmat2rodrigues(R: ArrayLike) -> NDArray[np.floating]:
    """
    Convert rotation matrix to Rodrigues vector.

    Parameters
    ----------
    R : array_like
        3x3 rotation matrix.

    Returns
    -------
    rvec : ndarray
        Rodrigues vector (axis * angle).
    """
    axis, angle = rotmat2axisangle(R)
    return axis * angle


def dcm_rate(
    R: ArrayLike,
    omega: ArrayLike,
) -> NDArray[np.floating]:
    """
    Compute the time derivative of a rotation matrix.

    Parameters
    ----------
    R : array_like
        Current rotation matrix.
    omega : array_like
        Angular velocity vector [wx, wy, wz] in body frame.

    Returns
    -------
    R_dot : ndarray
        Time derivative of R.

    Notes
    -----
    R_dot = R @ skew(omega)
    """
    R = np.asarray(R, dtype=np.float64)
    omega = np.asarray(omega, dtype=np.float64)

    omega_skew = np.array(
        [[0, -omega[2], omega[1]], [omega[2], 0, -omega[0]], [-omega[1], omega[0], 0]],
        dtype=np.float64,
    )

    return R @ omega_skew


def is_rotation_matrix(R: ArrayLike, tol: float = 1e-6) -> bool:
    """
    Check if a matrix is a valid rotation matrix.

    Parameters
    ----------
    R : array_like
        Matrix to check.
    tol : float, optional
        Tolerance for numerical checks.

    Returns
    -------
    valid : bool
        True if R is a valid rotation matrix.
    """
    R = np.asarray(R, dtype=np.float64)

    if R.shape != (3, 3):
        return False

    # Check orthogonality: R @ R.T = I
    if not np.allclose(R @ R.T, np.eye(3), atol=tol):
        return False

    # Check determinant = 1
    if not np.isclose(np.linalg.det(R), 1.0, atol=tol):
        return False

    return True


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
