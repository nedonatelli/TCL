"""
Inertial Navigation System (INS) mechanization.

This module provides strapdown INS mechanization equations for navigation,
including:
- INS state representation (position, velocity, attitude)
- Strapdown mechanization equations (NED and ECEF frames)
- Coning and sculling corrections for IMU data
- Gravity and transport rate computations
- Attitude integration using quaternions

References
----------
.. [1] D. Titterton and J. Weston, "Strapdown Inertial Navigation Technology",
       2nd ed., IEE, 2004.
.. [2] P. Groves, "Principles of GNSS, Inertial, and Multisensor Integrated
       Navigation Systems", 2nd ed., Artech House, 2013.
.. [3] J. Farrell, "Aided Navigation: GPS with High Rate Sensors", McGraw-Hill, 2008.
"""

from typing import NamedTuple, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pytcl.coordinate_systems.rotations import quat2rotmat, quat_multiply, rotmat2quat
from pytcl.navigation.geodesy import WGS84, Ellipsoid

# =============================================================================
# Physical Constants
# =============================================================================

# WGS84 Earth rotation rate (rad/s)
OMEGA_EARTH = 7.2921150e-5

# WGS84 gravitational constant (m^3/s^2)
GM_EARTH = 3.986004418e14

# WGS84 semi-major axis (m)
A_EARTH = 6378137.0

# WGS84 flattening
F_EARTH = 1.0 / 298.257223563

# WGS84 semi-minor axis (m)
B_EARTH = A_EARTH * (1 - F_EARTH)

# WGS84 first eccentricity squared
E2_EARTH = F_EARTH * (2 - F_EARTH)

# Somigliana's formula constants for normal gravity
GAMMA_A = 9.7803253359  # Gravity at equator (m/s^2)
GAMMA_B = 9.8321849378  # Gravity at pole (m/s^2)
K_SOMIGLIANA = (B_EARTH * GAMMA_B - A_EARTH * GAMMA_A) / (A_EARTH * GAMMA_A)


# =============================================================================
# State Representation
# =============================================================================


class INSState(NamedTuple):
    """
    Inertial Navigation System state.

    Attributes
    ----------
    position : ndarray
        Position in geodetic coordinates [latitude (rad), longitude (rad), altitude (m)].
    velocity : ndarray
        Velocity in NED frame [vN, vE, vD] (m/s).
    quaternion : ndarray
        Attitude quaternion [qw, qx, qy, qz] from body to NED frame (scalar-first).
    time : float
        Time associated with this state (seconds).
    """

    position: NDArray[np.floating]  # [lat, lon, alt]
    velocity: NDArray[np.floating]  # [vN, vE, vD]
    quaternion: NDArray[np.floating]  # [qw, qx, qy, qz]
    time: float

    @property
    def latitude(self) -> float:
        """Geodetic latitude in radians."""
        return float(self.position[0])

    @property
    def longitude(self) -> float:
        """Geodetic longitude in radians."""
        return float(self.position[1])

    @property
    def altitude(self) -> float:
        """Altitude above ellipsoid in meters."""
        return float(self.position[2])

    @property
    def velocity_north(self) -> float:
        """North velocity in m/s."""
        return float(self.velocity[0])

    @property
    def velocity_east(self) -> float:
        """East velocity in m/s."""
        return float(self.velocity[1])

    @property
    def velocity_down(self) -> float:
        """Down velocity in m/s."""
        return float(self.velocity[2])

    @property
    def dcm(self) -> NDArray[np.floating]:
        """Direction cosine matrix from body to NED frame."""
        return quat2rotmat(self.quaternion)

    def euler_angles(self) -> NDArray[np.floating]:
        """
        Get Euler angles (roll, pitch, yaw) in radians.

        Returns
        -------
        euler : ndarray
            [roll, pitch, yaw] in radians, ZYX convention.
        """
        R = self.dcm
        # Extract Euler angles from DCM (ZYX convention)
        pitch = np.arcsin(-R[2, 0])
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
        return np.array([roll, pitch, yaw], dtype=np.float64)


class IMUData(NamedTuple):
    """
    IMU measurement data.

    Attributes
    ----------
    accel : ndarray
        Specific force measurements [ax, ay, az] in body frame (m/s^2).
    gyro : ndarray
        Angular rate measurements [wx, wy, wz] in body frame (rad/s).
    dt : float
        Time interval between measurements (seconds).
    """

    accel: NDArray[np.floating]  # [ax, ay, az]
    gyro: NDArray[np.floating]  # [wx, wy, wz]
    dt: float


class INSErrorState(NamedTuple):
    """
    INS error state for Kalman filtering.

    Attributes
    ----------
    position_error : ndarray
        Position error [delta_lat, delta_lon, delta_alt] (rad, rad, m).
    velocity_error : ndarray
        Velocity error in NED [delta_vN, delta_vE, delta_vD] (m/s).
    attitude_error : ndarray
        Attitude error angles [phi_N, phi_E, phi_D] (rad).
    accel_bias : ndarray
        Accelerometer bias [bax, bay, baz] (m/s^2).
    gyro_bias : ndarray
        Gyroscope bias [bwx, bwy, bwz] (rad/s).
    """

    position_error: NDArray[np.floating]
    velocity_error: NDArray[np.floating]
    attitude_error: NDArray[np.floating]
    accel_bias: NDArray[np.floating]
    gyro_bias: NDArray[np.floating]

    @staticmethod
    def zeros() -> "INSErrorState":
        """Create zero error state."""
        return INSErrorState(
            position_error=np.zeros(3, dtype=np.float64),
            velocity_error=np.zeros(3, dtype=np.float64),
            attitude_error=np.zeros(3, dtype=np.float64),
            accel_bias=np.zeros(3, dtype=np.float64),
            gyro_bias=np.zeros(3, dtype=np.float64),
        )

    def to_vector(self) -> NDArray[np.floating]:
        """Convert to 15-element state vector."""
        return np.concatenate(
            [
                self.position_error,
                self.velocity_error,
                self.attitude_error,
                self.accel_bias,
                self.gyro_bias,
            ]
        )

    @staticmethod
    def from_vector(x: ArrayLike) -> "INSErrorState":
        """Create from 15-element state vector."""
        x = np.asarray(x, dtype=np.float64)
        return INSErrorState(
            position_error=x[0:3],
            velocity_error=x[3:6],
            attitude_error=x[6:9],
            accel_bias=x[9:12],
            gyro_bias=x[12:15],
        )


# =============================================================================
# Gravity and Earth Rate Computations
# =============================================================================


def normal_gravity(lat: float, alt: float = 0.0) -> float:
    """
    Compute normal gravity using Somigliana's formula with free-air correction.

    Parameters
    ----------
    lat : float
        Geodetic latitude in radians.
    alt : float, optional
        Altitude above ellipsoid in meters (default: 0).

    Returns
    -------
    g : float
        Normal gravity magnitude in m/s^2.

    Notes
    -----
    Uses the WGS84 gravity formula with first-order altitude correction.
    """
    sin_lat = np.sin(lat)
    sin2_lat = sin_lat**2

    # Somigliana's formula for gravity at ellipsoid surface
    g0 = GAMMA_A * (1 + K_SOMIGLIANA * sin2_lat) / np.sqrt(1 - E2_EARTH * sin2_lat)

    # Free-air correction (first-order)
    g = g0 * (
        1
        - 2
        * alt
        / A_EARTH
        * (1 + F_EARTH + (OMEGA_EARTH**2 * A_EARTH**2 * B_EARTH) / GM_EARTH)
    )

    return g


def gravity_ned(lat: float, alt: float = 0.0) -> NDArray[np.floating]:
    """
    Compute gravity vector in NED frame.

    Parameters
    ----------
    lat : float
        Geodetic latitude in radians.
    alt : float, optional
        Altitude above ellipsoid in meters.

    Returns
    -------
    g_ned : ndarray
        Gravity vector [gN, gE, gD] in m/s^2.
    """
    g = normal_gravity(lat, alt)
    return np.array([0.0, 0.0, g], dtype=np.float64)


def earth_rate_ned(lat: float) -> NDArray[np.floating]:
    """
    Compute Earth rotation rate in NED frame.

    Parameters
    ----------
    lat : float
        Geodetic latitude in radians.

    Returns
    -------
    omega_ie_n : ndarray
        Earth rotation rate in NED frame [wN, wE, wD] (rad/s).
    """
    return np.array(
        [OMEGA_EARTH * np.cos(lat), 0.0, -OMEGA_EARTH * np.sin(lat)],
        dtype=np.float64,
    )


def transport_rate_ned(
    lat: float,
    alt: float,
    vN: float,
    vE: float,
    ellipsoid: Ellipsoid = WGS84,
) -> NDArray[np.floating]:
    """
    Compute transport rate (navigation frame rotation) in NED frame.

    Parameters
    ----------
    lat : float
        Geodetic latitude in radians.
    alt : float
        Altitude above ellipsoid in meters.
    vN : float
        North velocity in m/s.
    vE : float
        East velocity in m/s.
    ellipsoid : Ellipsoid, optional
        Reference ellipsoid (default: WGS84).

    Returns
    -------
    omega_en_n : ndarray
        Transport rate in NED frame [wN, wE, wD] (rad/s).

    Notes
    -----
    The transport rate accounts for navigation frame rotation as the
    vehicle moves over the curved Earth surface.
    """
    # Radii of curvature
    RN, RE = radii_of_curvature(lat, ellipsoid)

    # Transport rate components
    omega_N = vE / (RE + alt)
    omega_E = -vN / (RN + alt)
    omega_D = -vE * np.tan(lat) / (RE + alt)

    return np.array([omega_N, omega_E, omega_D], dtype=np.float64)


def radii_of_curvature(
    lat: float,
    ellipsoid: Ellipsoid = WGS84,
) -> Tuple[float, float]:
    """
    Compute principal radii of curvature.

    Parameters
    ----------
    lat : float
        Geodetic latitude in radians.
    ellipsoid : Ellipsoid, optional
        Reference ellipsoid (default: WGS84).

    Returns
    -------
    RN : float
        Meridian radius of curvature (m).
    RE : float
        Transverse radius of curvature (prime vertical) (m).
    """
    sin_lat = np.sin(lat)
    sin2_lat = sin_lat**2

    a = ellipsoid.a
    e2 = ellipsoid.e2

    # Prime vertical radius
    RE = a / np.sqrt(1 - e2 * sin2_lat)

    # Meridian radius
    RN = a * (1 - e2) / (1 - e2 * sin2_lat) ** 1.5

    return RN, RE


# =============================================================================
# Coning and Sculling Corrections
# =============================================================================


def coning_correction(
    gyro_prev: ArrayLike,
    gyro_curr: ArrayLike,
) -> NDArray[np.floating]:
    """
    Compute first-order coning correction for angular increment.

    Parameters
    ----------
    gyro_prev : array_like
        Previous angular rate [wx, wy, wz] (rad/s).
    gyro_curr : array_like
        Current angular rate [wx, wy, wz] (rad/s).

    Returns
    -------
    delta_theta_coning : ndarray
        Coning correction to angular increment (rad).

    Notes
    -----
    Coning error occurs when the rotation axis itself rotates (coning motion).
    This first-order correction assumes trapezoidal integration of gyro data.

    References
    ----------
    .. [1] Savage, P.G., "Strapdown Inertial Navigation Integration Algorithm
           Design Part 1: Attitude Algorithms", AIAA Journal of Guidance, 1998.
    """
    gyro_prev = np.asarray(gyro_prev, dtype=np.float64)
    gyro_curr = np.asarray(gyro_curr, dtype=np.float64)

    # First-order coning correction: (1/12) * (theta_prev x theta_curr)
    return (1.0 / 12.0) * np.cross(gyro_prev, gyro_curr)


def sculling_correction(
    accel_prev: ArrayLike,
    accel_curr: ArrayLike,
    gyro_prev: ArrayLike,
    gyro_curr: ArrayLike,
) -> NDArray[np.floating]:
    """
    Compute first-order sculling correction for velocity increment.

    Parameters
    ----------
    accel_prev : array_like
        Previous specific force [ax, ay, az] (m/s^2).
    accel_curr : array_like
        Current specific force [ax, ay, az] (m/s^2).
    gyro_prev : array_like
        Previous angular rate [wx, wy, wz] (rad/s).
    gyro_curr : array_like
        Current angular rate [wx, wy, wz] (rad/s).

    Returns
    -------
    delta_v_sculling : ndarray
        Sculling correction to velocity increment (m/s).

    Notes
    -----
    Sculling error occurs when linear vibration and angular vibration
    are correlated (e.g., on a flexible structure).

    References
    ----------
    .. [1] Savage, P.G., "Strapdown Inertial Navigation Integration Algorithm
           Design Part 2: Velocity and Position Algorithms", AIAA, 1998.
    """
    accel_prev = np.asarray(accel_prev, dtype=np.float64)
    accel_curr = np.asarray(accel_curr, dtype=np.float64)
    gyro_prev = np.asarray(gyro_prev, dtype=np.float64)
    gyro_curr = np.asarray(gyro_curr, dtype=np.float64)

    # First-order sculling correction
    term1 = np.cross(gyro_prev, accel_curr)
    term2 = np.cross(accel_prev, gyro_curr)

    return (1.0 / 12.0) * (term1 + term2)


def compensate_imu_data(
    accel_prev: ArrayLike,
    accel_curr: ArrayLike,
    gyro_prev: ArrayLike,
    gyro_curr: ArrayLike,
    dt: float,
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Compute compensated angular and velocity increments with coning/sculling corrections.

    Parameters
    ----------
    accel_prev : array_like
        Previous specific force [ax, ay, az] (m/s^2).
    accel_curr : array_like
        Current specific force [ax, ay, az] (m/s^2).
    gyro_prev : array_like
        Previous angular rate [wx, wy, wz] (rad/s).
    gyro_curr : array_like
        Current angular rate [wx, wy, wz] (rad/s).
    dt : float
        Time interval (seconds).

    Returns
    -------
    delta_theta : ndarray
        Compensated angular increment (rad).
    delta_v : ndarray
        Compensated velocity increment (m/s).
    """
    gyro_prev = np.asarray(gyro_prev, dtype=np.float64)
    gyro_curr = np.asarray(gyro_curr, dtype=np.float64)
    accel_prev = np.asarray(accel_prev, dtype=np.float64)
    accel_curr = np.asarray(accel_curr, dtype=np.float64)

    # Trapezoidal integration for angular increment
    delta_theta_raw = 0.5 * (gyro_prev + gyro_curr) * dt
    delta_theta_coning = coning_correction(gyro_prev * dt, gyro_curr * dt)
    delta_theta = delta_theta_raw + delta_theta_coning

    # Trapezoidal integration for velocity increment
    delta_v_raw = 0.5 * (accel_prev + accel_curr) * dt
    delta_v_sculling = sculling_correction(
        accel_prev * dt, accel_curr * dt, gyro_prev * dt, gyro_curr * dt
    )
    # Rotation compensation for velocity increment
    delta_v_rotation = 0.5 * np.cross(delta_theta, delta_v_raw)
    delta_v = delta_v_raw + delta_v_sculling + delta_v_rotation

    return delta_theta, delta_v


# =============================================================================
# Attitude Update
# =============================================================================


def skew_symmetric(v: ArrayLike) -> NDArray[np.floating]:
    """
    Create skew-symmetric matrix from vector.

    Parameters
    ----------
    v : array_like
        3-element vector.

    Returns
    -------
    S : ndarray
        3x3 skew-symmetric matrix.
    """
    v = np.asarray(v, dtype=np.float64)
    return np.array(
        [[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]],
        dtype=np.float64,
    )


def update_quaternion(
    q: ArrayLike,
    delta_theta: ArrayLike,
) -> NDArray[np.floating]:
    """
    Update quaternion using angular increment.

    Parameters
    ----------
    q : array_like
        Current quaternion [qw, qx, qy, qz].
    delta_theta : array_like
        Angular increment in body frame (rad).

    Returns
    -------
    q_new : ndarray
        Updated quaternion (normalized).

    Notes
    -----
    Uses first-order quaternion update: q_new = q * delta_q
    where delta_q represents the incremental rotation.
    """
    q = np.asarray(q, dtype=np.float64)
    delta_theta = np.asarray(delta_theta, dtype=np.float64)

    # Magnitude of rotation
    angle = np.linalg.norm(delta_theta)

    if angle < 1e-10:
        # Small angle approximation
        delta_q = np.array(
            [1.0, 0.5 * delta_theta[0], 0.5 * delta_theta[1], 0.5 * delta_theta[2]]
        )
    else:
        # Exact quaternion for rotation
        axis = delta_theta / angle
        half_angle = 0.5 * angle
        sin_half = np.sin(half_angle)
        delta_q = np.array(
            [
                np.cos(half_angle),
                sin_half * axis[0],
                sin_half * axis[1],
                sin_half * axis[2],
            ]
        )

    # Quaternion multiplication
    q_new = quat_multiply(q, delta_q)

    # Normalize
    return q_new / np.linalg.norm(q_new)


def update_attitude_ned(
    q_b_n: ArrayLike,
    omega_ib_b: ArrayLike,
    omega_in_n: ArrayLike,
    dt: float,
) -> NDArray[np.floating]:
    """
    Update attitude quaternion (body to NED) for one time step.

    Parameters
    ----------
    q_b_n : array_like
        Current quaternion from body to NED frame.
    omega_ib_b : array_like
        Angular rate of body w.r.t. inertial, in body frame (rad/s).
    omega_in_n : array_like
        Angular rate of NED w.r.t. inertial, in NED frame (rad/s).
    dt : float
        Time step (seconds).

    Returns
    -------
    q_new : ndarray
        Updated quaternion (body to NED).

    Notes
    -----
    The attitude update accounts for both body rotation and navigation frame rotation.
    omega_in_n = omega_ie_n + omega_en_n (Earth rate + transport rate)
    """
    q_b_n = np.asarray(q_b_n, dtype=np.float64)
    omega_ib_b = np.asarray(omega_ib_b, dtype=np.float64)
    omega_in_n = np.asarray(omega_in_n, dtype=np.float64)

    # Transform navigation frame rate to body frame
    R_b_n = quat2rotmat(q_b_n)
    R_n_b = R_b_n.T
    omega_in_b = R_n_b @ omega_in_n

    # Angular rate of body w.r.t. navigation frame, in body frame
    omega_nb_b = omega_ib_b - omega_in_b

    # Angular increment
    delta_theta = omega_nb_b * dt

    # Update quaternion
    return update_quaternion(q_b_n, delta_theta)


# =============================================================================
# Strapdown Mechanization
# =============================================================================


def mechanize_ins_ned(
    state: INSState,
    imu: IMUData,
    ellipsoid: Ellipsoid = WGS84,
    accel_prev: Optional[ArrayLike] = None,
    gyro_prev: Optional[ArrayLike] = None,
) -> INSState:
    """
    Perform one step of strapdown INS mechanization in NED frame.

    Parameters
    ----------
    state : INSState
        Current INS state.
    imu : IMUData
        IMU measurements for this time step.
    ellipsoid : Ellipsoid, optional
        Reference ellipsoid (default: WGS84).
    accel_prev : array_like, optional
        Previous accelerometer reading for coning/sculling corrections.
    gyro_prev : array_like, optional
        Previous gyroscope reading for coning/sculling corrections.

    Returns
    -------
    new_state : INSState
        Updated INS state after mechanization.

    Notes
    -----
    This implements the standard NED-frame strapdown mechanization:
    1. Attitude update (quaternion integration)
    2. Specific force transformation to NED
    3. Velocity update (with gravity and Coriolis)
    4. Position update

    The algorithm follows Groves (2013), Chapter 5.
    """
    dt = imu.dt
    lat, lon, alt = state.position
    vN, vE, vD = state.velocity
    q = state.quaternion

    # Get compensated increments if previous data available
    if accel_prev is not None and gyro_prev is not None:
        delta_theta_b, delta_v_b = compensate_imu_data(
            accel_prev, imu.accel, gyro_prev, imu.gyro, dt
        )
        # Average angular rate for attitude update
        omega_ib_b = delta_theta_b / dt
    else:
        # Simple integration without corrections
        delta_v_b = imu.accel * dt
        omega_ib_b = imu.gyro

    # ==========
    # 1. Attitude update
    # ==========

    # Earth rate and transport rate in NED
    omega_ie_n = earth_rate_ned(lat)
    omega_en_n = transport_rate_ned(lat, alt, vN, vE, ellipsoid)
    omega_in_n = omega_ie_n + omega_en_n

    # Update attitude
    q_new = update_attitude_ned(q, omega_ib_b, omega_in_n, dt)

    # Average DCM for velocity transformation
    R_b_n = quat2rotmat(q)
    R_b_n_new = quat2rotmat(q_new)
    R_b_n_avg = 0.5 * (R_b_n + R_b_n_new)

    # ==========
    # 2. Velocity update
    # ==========

    # Transform specific force to NED frame
    f_n = R_b_n_avg @ delta_v_b / dt

    # Gravity
    g_n = gravity_ned(lat, alt)

    # Coriolis and transport rate correction
    v_n = np.array([vN, vE, vD], dtype=np.float64)
    coriolis = np.cross(2 * omega_ie_n + omega_en_n, v_n)

    # Velocity rate
    v_dot = f_n + g_n - coriolis

    # Update velocity (trapezoidal would need v_dot at both ends)
    v_new = v_n + v_dot * dt
    vN_new, vE_new, vD_new = v_new

    # ==========
    # 3. Position update
    # ==========

    # Radii of curvature
    RN, RE = radii_of_curvature(lat, ellipsoid)

    # Average velocities
    vN_avg = 0.5 * (vN + vN_new)
    vE_avg = 0.5 * (vE + vE_new)
    vD_avg = 0.5 * (vD + vD_new)

    # Position rates
    lat_dot = vN_avg / (RN + alt)
    lon_dot = vE_avg / ((RE + alt) * np.cos(lat))
    alt_dot = -vD_avg

    # Update position
    lat_new = lat + lat_dot * dt
    lon_new = lon + lon_dot * dt
    alt_new = alt + alt_dot * dt

    return INSState(
        position=np.array([lat_new, lon_new, alt_new], dtype=np.float64),
        velocity=np.array([vN_new, vE_new, vD_new], dtype=np.float64),
        quaternion=q_new,
        time=state.time + dt,
    )


def initialize_ins_state(
    lat: float,
    lon: float,
    alt: float,
    vN: float = 0.0,
    vE: float = 0.0,
    vD: float = 0.0,
    roll: float = 0.0,
    pitch: float = 0.0,
    yaw: float = 0.0,
    time: float = 0.0,
) -> INSState:
    """
    Initialize INS state from position, velocity, and attitude.

    Parameters
    ----------
    lat : float
        Geodetic latitude in radians.
    lon : float
        Geodetic longitude in radians.
    alt : float
        Altitude above ellipsoid in meters.
    vN : float, optional
        North velocity in m/s (default: 0).
    vE : float, optional
        East velocity in m/s (default: 0).
    vD : float, optional
        Down velocity in m/s (default: 0).
    roll : float, optional
        Roll angle in radians (default: 0).
    pitch : float, optional
        Pitch angle in radians (default: 0).
    yaw : float, optional
        Yaw/heading angle in radians (default: 0).
    time : float, optional
        Initial time in seconds (default: 0).

    Returns
    -------
    state : INSState
        Initialized INS state.
    """
    # Build rotation matrix from Euler angles (ZYX: yaw-pitch-roll)
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)

    R = np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )

    q = rotmat2quat(R)

    return INSState(
        position=np.array([lat, lon, alt], dtype=np.float64),
        velocity=np.array([vN, vE, vD], dtype=np.float64),
        quaternion=q,
        time=time,
    )


def coarse_alignment(
    accel: ArrayLike,
    lat: float,
) -> Tuple[float, float]:
    """
    Perform coarse leveling alignment from accelerometer readings.

    Parameters
    ----------
    accel : array_like
        Averaged accelerometer readings [ax, ay, az] in body frame (m/s^2).
        Vehicle should be stationary. For a level vehicle, accel = [0, 0, -g].
    lat : float
        Approximate latitude in radians.

    Returns
    -------
    roll : float
        Estimated roll angle in radians.
    pitch : float
        Estimated pitch angle in radians.

    Notes
    -----
    This assumes the vehicle is stationary so the accelerometer measures
    only the reaction to gravity (specific force = -g when level).
    Does not estimate heading (yaw).

    For a level vehicle pointing north:
    - Accelerometer reads [0, 0, -g] (z-axis up measures negative gravity)
    - Roll = 0, Pitch = 0

    Sign convention: positive roll is right wing down, positive pitch is nose up.
    """
    accel = np.asarray(accel, dtype=np.float64)
    ax, ay, az = accel

    # Roll and pitch from specific force vector
    # For level vehicle: ax=0, ay=0, az=-g
    # Roll from Y and Z components (arctan2(-ay, -az) gives 0 when level)
    roll = np.arctan2(-ay, -az)

    # Pitch from X and magnitude of Y-Z plane
    pitch = np.arctan2(ax, np.sqrt(ay**2 + az**2))

    return float(roll), float(pitch)


def gyrocompass_alignment(
    gyro: ArrayLike,
    roll: float,
    pitch: float,
    lat: float,
) -> float:
    """
    Perform gyrocompass alignment to estimate heading.

    Parameters
    ----------
    gyro : array_like
        Averaged gyroscope readings [wx, wy, wz] in body frame (rad/s).
        Vehicle should be stationary.
    roll : float
        Known roll angle in radians.
    pitch : float
        Known pitch angle in radians.
    lat : float
        Latitude in radians.

    Returns
    -------
    yaw : float
        Estimated heading (yaw) angle in radians.

    Notes
    -----
    Uses the horizontal component of Earth's rotation rate to determine heading.
    Requires accurate roll and pitch (from coarse_alignment).
    """
    gyro = np.asarray(gyro, dtype=np.float64)

    # Build partial DCM (without yaw)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)

    # Expected horizontal Earth rate
    omega_h = OMEGA_EARTH * np.cos(lat)

    # Gyro components in horizontal plane
    # After removing pitch and roll effects
    gy_h = gyro[1] * cr - gyro[2] * sr
    gx_h = gyro[0] * cp + gyro[1] * sp * sr + gyro[2] * sp * cr

    # Heading from horizontal gyro components
    yaw = np.arctan2(-gy_h, gx_h / omega_h) if abs(omega_h) > 1e-10 else 0.0

    return float(yaw)


# =============================================================================
# Error State Model
# =============================================================================


def ins_error_state_matrix(
    state: INSState,
    ellipsoid: Ellipsoid = WGS84,
) -> NDArray[np.floating]:
    """
    Compute the INS error state transition matrix (continuous-time F matrix).

    Parameters
    ----------
    state : INSState
        Current INS state.
    ellipsoid : Ellipsoid, optional
        Reference ellipsoid (default: WGS84).

    Returns
    -------
    F : ndarray
        15x15 error state transition matrix.

    Notes
    -----
    The 15-state error model includes:
    - Position errors (3): delta_lat, delta_lon, delta_alt
    - Velocity errors (3): delta_vN, delta_vE, delta_vD
    - Attitude errors (3): phi_N, phi_E, phi_D
    - Accelerometer biases (3): bax, bay, baz
    - Gyroscope biases (3): bwx, bwy, bwz
    """
    lat, lon, alt = state.position
    vN, vE, vD = state.velocity
    R_b_n = state.dcm

    # Radii of curvature
    RN, RE = radii_of_curvature(lat, ellipsoid)

    # Gravity gradient
    g = normal_gravity(lat, alt)

    # Initialize F matrix
    F = np.zeros((15, 15), dtype=np.float64)

    # Position error dynamics (Fpp, Fpv)
    F[0, 2] = -vN / (RN + alt) ** 2  # d(lat_dot)/d(alt)
    F[0, 3] = 1.0 / (RN + alt)  # d(lat_dot)/d(vN)

    F[1, 0] = vE * np.tan(lat) / ((RE + alt) * np.cos(lat))
    F[1, 2] = -vE / ((RE + alt) ** 2 * np.cos(lat))
    F[1, 4] = 1.0 / ((RE + alt) * np.cos(lat))

    F[2, 5] = -1.0  # d(alt_dot)/d(vD)

    # Velocity error dynamics (Fvp, Fvv, Fva)
    omega_ie = OMEGA_EARTH

    # Gravity and Coriolis terms (simplified)
    F[3, 0] = -2 * vE * omega_ie * np.cos(lat)
    F[3, 4] = -2 * omega_ie * np.sin(lat) - vE * np.tan(lat) / (RE + alt)
    F[3, 5] = vN / (RN + alt)

    F[4, 0] = 2 * omega_ie * (vN * np.cos(lat) + vD * np.sin(lat))
    F[4, 3] = 2 * omega_ie * np.sin(lat) + vE * np.tan(lat) / (RE + alt)
    F[4, 5] = 2 * omega_ie * np.cos(lat) + vE / (RE + alt)

    F[5, 0] = -2 * vE * omega_ie * np.sin(lat)
    F[5, 2] = 2 * g / A_EARTH  # Gravity gradient
    F[5, 4] = -2 * omega_ie * np.cos(lat) - 2 * vE / (RE + alt)

    # Velocity-attitude coupling (specific force skew-symmetric)
    # F_va = -[f^n x] where f^n is specific force in NED
    # For simplicity, using gravity as dominant term
    F[3, 7] = -g  # Cross-coupling with attitude
    F[4, 6] = g

    # Velocity-accel bias coupling
    F[3:6, 9:12] = R_b_n

    # Attitude error dynamics (Fap, Fav, Faa)
    F[6, 4] = 1.0 / (RE + alt)
    F[7, 3] = -1.0 / (RN + alt)
    F[7, 0] = omega_ie * np.sin(lat)
    F[8, 0] = omega_ie * np.cos(lat) + vE / ((RE + alt) * np.cos(lat) ** 2)
    F[8, 4] = -np.tan(lat) / (RE + alt)

    # Attitude-gyro bias coupling
    F[6:9, 12:15] = -R_b_n

    return F


def ins_process_noise_matrix(
    accel_noise_std: float,
    gyro_noise_std: float,
    accel_bias_std: float,
    gyro_bias_std: float,
    state: INSState,
) -> NDArray[np.floating]:
    """
    Compute the INS process noise covariance matrix (continuous-time Q matrix).

    Parameters
    ----------
    accel_noise_std : float
        Accelerometer white noise standard deviation (m/s^2).
    gyro_noise_std : float
        Gyroscope white noise standard deviation (rad/s).
    accel_bias_std : float
        Accelerometer bias random walk standard deviation (m/s^2/sqrt(s)).
    gyro_bias_std : float
        Gyroscope bias random walk standard deviation (rad/s/sqrt(s)).
    state : INSState
        Current INS state (for DCM).

    Returns
    -------
    Q : ndarray
        15x15 process noise covariance matrix.
    """
    R_b_n = state.dcm

    Q = np.zeros((15, 15), dtype=np.float64)

    # Velocity noise from accelerometer
    Q[3:6, 3:6] = accel_noise_std**2 * R_b_n @ R_b_n.T

    # Attitude noise from gyroscope
    Q[6:9, 6:9] = gyro_noise_std**2 * R_b_n @ R_b_n.T

    # Bias random walks
    Q[9:12, 9:12] = accel_bias_std**2 * np.eye(3)
    Q[12:15, 12:15] = gyro_bias_std**2 * np.eye(3)

    return Q


__all__ = [
    # Constants
    "OMEGA_EARTH",
    "GM_EARTH",
    "A_EARTH",
    "F_EARTH",
    "B_EARTH",
    "E2_EARTH",
    # State representation
    "INSState",
    "IMUData",
    "INSErrorState",
    # Gravity and Earth rate
    "normal_gravity",
    "gravity_ned",
    "earth_rate_ned",
    "transport_rate_ned",
    "radii_of_curvature",
    # Coning and sculling
    "coning_correction",
    "sculling_correction",
    "compensate_imu_data",
    # Attitude
    "skew_symmetric",
    "update_quaternion",
    "update_attitude_ned",
    # Mechanization
    "mechanize_ins_ned",
    "initialize_ins_state",
    # Alignment
    "coarse_alignment",
    "gyrocompass_alignment",
    # Error state model
    "ins_error_state_matrix",
    "ins_process_noise_matrix",
]
