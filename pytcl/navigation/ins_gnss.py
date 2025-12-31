"""
INS/GNSS Integration algorithms.

This module provides integrated navigation solutions combining Inertial
Navigation System (INS) and Global Navigation Satellite System (GNSS):
- GNSS measurement models and state representation
- Loosely-coupled INS/GNSS integration (position/velocity aiding)
- Tightly-coupled INS/GNSS integration (pseudorange/Doppler aiding)
- Error state Kalman filter for INS/GNSS fusion

References
----------
.. [1] P. Groves, "Principles of GNSS, Inertial, and Multisensor Integrated
       Navigation Systems", 2nd ed., Artech House, 2013.
.. [2] J. Farrell, "Aided Navigation: GPS with High Rate Sensors", McGraw-Hill, 2008.
.. [3] R. Brown and P. Hwang, "Introduction to Random Signals and Applied
       Kalman Filtering", 4th ed., Wiley, 2012.
"""

from typing import List, NamedTuple, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pytcl.dynamic_estimation.kalman import kf_predict, kf_update
from pytcl.navigation.geodesy import WGS84, Ellipsoid, geodetic_to_ecef
from pytcl.navigation.ins import (
    IMUData,
    INSState,
    ins_error_state_matrix,
    ins_process_noise_matrix,
    mechanize_ins_ned,
)

# =============================================================================
# Constants
# =============================================================================

# Speed of light (m/s)
SPEED_OF_LIGHT = 299792458.0

# GPS L1 frequency (Hz)
GPS_L1_FREQ = 1575.42e6

# GPS L1 wavelength (m)
GPS_L1_WAVELENGTH = SPEED_OF_LIGHT / GPS_L1_FREQ


# =============================================================================
# GNSS State Representation
# =============================================================================


class GNSSMeasurement(NamedTuple):
    """
    GNSS measurement data.

    Attributes
    ----------
    position : ndarray, optional
        GNSS position [lat, lon, alt] in (rad, rad, m).
    velocity : ndarray, optional
        GNSS velocity in NED frame [vN, vE, vD] (m/s).
    position_cov : ndarray, optional
        Position covariance (3x3) in geodetic frame.
    velocity_cov : ndarray, optional
        Velocity covariance (3x3) in NED frame.
    time : float
        GPS time of measurement (seconds).
    valid : bool
        Whether the measurement is valid.
    """

    position: Optional[NDArray[np.floating]]
    velocity: Optional[NDArray[np.floating]]
    position_cov: Optional[NDArray[np.floating]]
    velocity_cov: Optional[NDArray[np.floating]]
    time: float
    valid: bool = True


class SatelliteInfo(NamedTuple):
    """
    Satellite information for tightly-coupled integration.

    Attributes
    ----------
    prn : int
        Satellite PRN number.
    position : ndarray
        Satellite ECEF position [x, y, z] (m).
    velocity : ndarray
        Satellite ECEF velocity [vx, vy, vz] (m/s).
    pseudorange : float
        Measured pseudorange (m).
    doppler : float, optional
        Measured Doppler shift (Hz).
    cn0 : float, optional
        Carrier-to-noise ratio (dB-Hz).
    elevation : float, optional
        Satellite elevation angle (rad).
    azimuth : float, optional
        Satellite azimuth angle (rad).
    """

    prn: int
    position: NDArray[np.floating]
    velocity: NDArray[np.floating]
    pseudorange: float
    doppler: Optional[float] = None
    cn0: Optional[float] = None
    elevation: Optional[float] = None
    azimuth: Optional[float] = None


class INSGNSSState(NamedTuple):
    """
    Combined INS/GNSS navigation state.

    Attributes
    ----------
    ins_state : INSState
        Current INS navigation state.
    error_state : ndarray
        Error state vector (15 or 17 elements).
    error_cov : ndarray
        Error state covariance matrix.
    clock_bias : float
        Receiver clock bias (m).
    clock_drift : float
        Receiver clock drift (m/s).
    """

    ins_state: INSState
    error_state: NDArray[np.floating]
    error_cov: NDArray[np.floating]
    clock_bias: float = 0.0
    clock_drift: float = 0.0


class LooseCoupledResult(NamedTuple):
    """
    Result from loosely-coupled INS/GNSS update.

    Attributes
    ----------
    state : INSGNSSState
        Updated navigation state.
    innovation : ndarray
        Measurement innovation (residual).
    innovation_cov : ndarray
        Innovation covariance.
    """

    state: INSGNSSState
    innovation: NDArray[np.floating]
    innovation_cov: NDArray[np.floating]


class TightCoupledResult(NamedTuple):
    """
    Result from tightly-coupled INS/GNSS update.

    Attributes
    ----------
    state : INSGNSSState
        Updated navigation state.
    innovations : ndarray
        Pseudorange/Doppler innovations.
    dop : tuple
        Dilution of precision (GDOP, PDOP, HDOP, VDOP).
    """

    state: INSGNSSState
    innovations: NDArray[np.floating]
    dop: Tuple[float, float, float, float]


# =============================================================================
# Measurement Models
# =============================================================================


def position_measurement_matrix() -> NDArray[np.floating]:
    """
    Create measurement matrix for position-only GNSS update.

    Returns
    -------
    H : ndarray
        3x15 measurement matrix mapping error state to position measurement.
    """
    H = np.zeros((3, 15), dtype=np.float64)
    H[0, 0] = 1.0  # latitude error
    H[1, 1] = 1.0  # longitude error
    H[2, 2] = 1.0  # altitude error
    return H


def velocity_measurement_matrix() -> NDArray[np.floating]:
    """
    Create measurement matrix for velocity-only GNSS update.

    Returns
    -------
    H : ndarray
        3x15 measurement matrix mapping error state to velocity measurement.
    """
    H = np.zeros((3, 15), dtype=np.float64)
    H[0, 3] = 1.0  # vN error
    H[1, 4] = 1.0  # vE error
    H[2, 5] = 1.0  # vD error
    return H


def position_velocity_measurement_matrix() -> NDArray[np.floating]:
    """
    Create measurement matrix for position+velocity GNSS update.

    Returns
    -------
    H : ndarray
        6x15 measurement matrix.
    """
    H = np.zeros((6, 15), dtype=np.float64)
    H[0, 0] = 1.0  # latitude error
    H[1, 1] = 1.0  # longitude error
    H[2, 2] = 1.0  # altitude error
    H[3, 3] = 1.0  # vN error
    H[4, 4] = 1.0  # vE error
    H[5, 5] = 1.0  # vD error
    return H


def compute_line_of_sight(
    user_pos: ArrayLike,
    sat_pos: ArrayLike,
) -> Tuple[NDArray[np.floating], float]:
    """
    Compute line-of-sight unit vector and range from user to satellite.

    Parameters
    ----------
    user_pos : array_like
        User ECEF position [x, y, z] (m).
    sat_pos : array_like
        Satellite ECEF position [x, y, z] (m).

    Returns
    -------
    los : ndarray
        Line-of-sight unit vector from user to satellite.
    range : float
        Geometric range (m).
    """
    user_pos = np.asarray(user_pos, dtype=np.float64)
    sat_pos = np.asarray(sat_pos, dtype=np.float64)

    delta = sat_pos - user_pos
    range_val = np.linalg.norm(delta)
    los = delta / range_val

    return los, float(range_val)


def pseudorange_measurement_matrix(
    user_pos: ArrayLike,
    satellites: List[SatelliteInfo],
    include_clock: bool = True,
) -> NDArray[np.floating]:
    """
    Create measurement matrix for pseudorange observations.

    Parameters
    ----------
    user_pos : array_like
        User ECEF position [x, y, z] (m).
    satellites : list of SatelliteInfo
        List of satellite information.
    include_clock : bool, optional
        Whether to include clock bias column (default: True).

    Returns
    -------
    H : ndarray
        (n_sats x 4) or (n_sats x 3) geometry matrix.
        Columns are [dx, dy, dz, clock_bias] or [dx, dy, dz].
    """
    user_pos = np.asarray(user_pos, dtype=np.float64)
    n_sats = len(satellites)

    if include_clock:
        H = np.zeros((n_sats, 4), dtype=np.float64)
    else:
        H = np.zeros((n_sats, 3), dtype=np.float64)

    for i, sat in enumerate(satellites):
        los, _ = compute_line_of_sight(user_pos, sat.position)
        H[i, 0:3] = -los  # Negative because positive user displacement decreases range
        if include_clock:
            H[i, 3] = 1.0  # Clock bias contribution

    return H


def compute_dop(H: ArrayLike) -> Tuple[float, float, float, float]:
    """
    Compute Dilution of Precision from geometry matrix.

    Parameters
    ----------
    H : array_like
        Geometry matrix (n_sats x 4) with columns [dx, dy, dz, clock].

    Returns
    -------
    GDOP : float
        Geometric DOP.
    PDOP : float
        Position DOP.
    HDOP : float
        Horizontal DOP.
    VDOP : float
        Vertical DOP.
    """
    H = np.asarray(H, dtype=np.float64)

    try:
        Q = np.linalg.inv(H.T @ H)
    except np.linalg.LinAlgError:
        return float("inf"), float("inf"), float("inf"), float("inf")

    GDOP = np.sqrt(np.trace(Q))
    PDOP = np.sqrt(Q[0, 0] + Q[1, 1] + Q[2, 2])
    HDOP = np.sqrt(Q[0, 0] + Q[1, 1])
    VDOP = np.sqrt(Q[2, 2])

    return float(GDOP), float(PDOP), float(HDOP), float(VDOP)


def satellite_elevation_azimuth(
    user_lla: ArrayLike,
    sat_ecef: ArrayLike,
    ellipsoid: Ellipsoid = WGS84,
) -> Tuple[float, float]:
    """
    Compute satellite elevation and azimuth angles from user position.

    Parameters
    ----------
    user_lla : array_like
        User position [lat, lon, alt] in (rad, rad, m).
    sat_ecef : array_like
        Satellite ECEF position [x, y, z] (m).
    ellipsoid : Ellipsoid, optional
        Reference ellipsoid (default: WGS84).

    Returns
    -------
    elevation : float
        Elevation angle in radians.
    azimuth : float
        Azimuth angle in radians (from north, clockwise).
    """
    user_lla = np.asarray(user_lla, dtype=np.float64)
    sat_ecef = np.asarray(sat_ecef, dtype=np.float64)

    lat, lon, alt = user_lla

    # User ECEF position
    user_x, user_y, user_z = geodetic_to_ecef(lat, lon, alt, ellipsoid)
    user_ecef = np.array([user_x, user_y, user_z])

    # Vector from user to satellite
    delta = sat_ecef - user_ecef

    # Rotation matrix from ECEF to ENU
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)

    R_ecef_to_enu = np.array(
        [
            [-sin_lon, cos_lon, 0],
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
        ]
    )

    # Transform to ENU
    enu = R_ecef_to_enu @ delta
    e, n, u = enu

    # Elevation and azimuth
    horizontal_range = np.sqrt(e**2 + n**2)
    elevation = np.arctan2(u, horizontal_range)
    azimuth = np.arctan2(e, n)

    if azimuth < 0:
        azimuth += 2 * np.pi

    return float(elevation), float(azimuth)


# =============================================================================
# Loosely-Coupled Integration
# =============================================================================


def initialize_ins_gnss(
    ins_state: INSState,
    position_std: float = 10.0,
    velocity_std: float = 1.0,
    attitude_std: float = 0.1,
    accel_bias_std: float = 0.1,
    gyro_bias_std: float = 0.01,
) -> INSGNSSState:
    """
    Initialize INS/GNSS integration state.

    Parameters
    ----------
    ins_state : INSState
        Initial INS navigation state.
    position_std : float, optional
        Initial position uncertainty (m). Default: 10.0.
    velocity_std : float, optional
        Initial velocity uncertainty (m/s). Default: 1.0.
    attitude_std : float, optional
        Initial attitude uncertainty (rad). Default: 0.1.
    accel_bias_std : float, optional
        Initial accelerometer bias uncertainty (m/s^2). Default: 0.1.
    gyro_bias_std : float, optional
        Initial gyroscope bias uncertainty (rad/s). Default: 0.01.

    Returns
    -------
    state : INSGNSSState
        Initialized INS/GNSS state.
    """
    # 15-state error vector (zeros initially)
    error_state = np.zeros(15, dtype=np.float64)

    # Initial covariance
    P = np.diag(
        [
            position_std**2,
            position_std**2,
            position_std**2,
            velocity_std**2,
            velocity_std**2,
            velocity_std**2,
            attitude_std**2,
            attitude_std**2,
            attitude_std**2,
            accel_bias_std**2,
            accel_bias_std**2,
            accel_bias_std**2,
            gyro_bias_std**2,
            gyro_bias_std**2,
            gyro_bias_std**2,
        ]
    ).astype(np.float64)

    return INSGNSSState(
        ins_state=ins_state,
        error_state=error_state,
        error_cov=P,
        clock_bias=0.0,
        clock_drift=0.0,
    )


def loose_coupled_predict(
    state: INSGNSSState,
    imu: IMUData,
    accel_noise_std: float = 0.01,
    gyro_noise_std: float = 1e-4,
    accel_bias_std: float = 1e-5,
    gyro_bias_std: float = 1e-7,
    accel_prev: Optional[ArrayLike] = None,
    gyro_prev: Optional[ArrayLike] = None,
) -> INSGNSSState:
    """
    Perform prediction step for loosely-coupled INS/GNSS.

    Parameters
    ----------
    state : INSGNSSState
        Current INS/GNSS state.
    imu : IMUData
        IMU measurements for this time step.
    accel_noise_std : float, optional
        Accelerometer noise std (m/s^2). Default: 0.01.
    gyro_noise_std : float, optional
        Gyroscope noise std (rad/s). Default: 1e-4.
    accel_bias_std : float, optional
        Accelerometer bias random walk std. Default: 1e-5.
    gyro_bias_std : float, optional
        Gyroscope bias random walk std. Default: 1e-7.
    accel_prev : array_like, optional
        Previous accelerometer reading for coning/sculling.
    gyro_prev : array_like, optional
        Previous gyroscope reading for coning/sculling.

    Returns
    -------
    state : INSGNSSState
        Predicted state.
    """
    dt = imu.dt

    # Propagate INS mechanization
    ins_new = mechanize_ins_ned(
        state.ins_state, imu, accel_prev=accel_prev, gyro_prev=gyro_prev
    )

    # Get error state transition matrix (continuous-time)
    F_cont = ins_error_state_matrix(state.ins_state)

    # Discretize F (first-order approximation)
    F = np.eye(15) + F_cont * dt

    # Get process noise (continuous-time)
    Q_cont = ins_process_noise_matrix(
        accel_noise_std, gyro_noise_std, accel_bias_std, gyro_bias_std, state.ins_state
    )

    # Discretize Q (first-order approximation)
    Q = Q_cont * dt

    # Propagate error state and covariance using linear KF
    result = kf_predict(state.error_state, state.error_cov, F, Q)

    return INSGNSSState(
        ins_state=ins_new,
        error_state=result.x,
        error_cov=result.P,
        clock_bias=state.clock_bias,
        clock_drift=state.clock_drift,
    )


def loose_coupled_update_position(
    state: INSGNSSState,
    gnss: GNSSMeasurement,
) -> LooseCoupledResult:
    """
    Update INS/GNSS state with GNSS position measurement.

    Parameters
    ----------
    state : INSGNSSState
        Current INS/GNSS state.
    gnss : GNSSMeasurement
        GNSS position measurement.

    Returns
    -------
    result : LooseCoupledResult
        Updated state and innovation statistics.
    """
    if gnss.position is None or not gnss.valid:
        # No valid measurement, return unchanged
        return LooseCoupledResult(
            state=state,
            innovation=np.zeros(3),
            innovation_cov=np.eye(3) * 1e10,
        )

    # Measurement matrix (position only)
    H = position_measurement_matrix()

    # Measurement noise covariance
    if gnss.position_cov is not None:
        R = gnss.position_cov
    else:
        R = np.diag(
            [10.0**2, 10.0**2, 15.0**2]
        )  # Default: 10m horizontal, 15m vertical

    # Innovation: measured position - INS predicted position
    z = gnss.position - state.ins_state.position

    # Kalman update
    result = kf_update(state.error_state, state.error_cov, z, H, R)

    # Apply correction to INS state
    corrected_ins = _apply_error_correction(state.ins_state, result.x)

    # Reset error state (closed-loop)
    new_error_state = np.zeros(15, dtype=np.float64)

    new_state = INSGNSSState(
        ins_state=corrected_ins,
        error_state=new_error_state,
        error_cov=result.P,
        clock_bias=state.clock_bias,
        clock_drift=state.clock_drift,
    )

    return LooseCoupledResult(
        state=new_state,
        innovation=result.y,
        innovation_cov=result.S,
    )


def loose_coupled_update_velocity(
    state: INSGNSSState,
    gnss: GNSSMeasurement,
) -> LooseCoupledResult:
    """
    Update INS/GNSS state with GNSS velocity measurement.

    Parameters
    ----------
    state : INSGNSSState
        Current INS/GNSS state.
    gnss : GNSSMeasurement
        GNSS velocity measurement.

    Returns
    -------
    result : LooseCoupledResult
        Updated state and innovation statistics.
    """
    if gnss.velocity is None or not gnss.valid:
        return LooseCoupledResult(
            state=state,
            innovation=np.zeros(3),
            innovation_cov=np.eye(3) * 1e10,
        )

    H = velocity_measurement_matrix()

    if gnss.velocity_cov is not None:
        R = gnss.velocity_cov
    else:
        R = np.diag([0.1**2, 0.1**2, 0.1**2])  # Default: 0.1 m/s

    z = gnss.velocity - state.ins_state.velocity

    result = kf_update(state.error_state, state.error_cov, z, H, R)

    corrected_ins = _apply_error_correction(state.ins_state, result.x)
    new_error_state = np.zeros(15, dtype=np.float64)

    new_state = INSGNSSState(
        ins_state=corrected_ins,
        error_state=new_error_state,
        error_cov=result.P,
        clock_bias=state.clock_bias,
        clock_drift=state.clock_drift,
    )

    return LooseCoupledResult(
        state=new_state,
        innovation=result.y,
        innovation_cov=result.S,
    )


def loose_coupled_update(
    state: INSGNSSState,
    gnss: GNSSMeasurement,
) -> LooseCoupledResult:
    """
    Update INS/GNSS state with GNSS position and velocity measurements.

    Parameters
    ----------
    state : INSGNSSState
        Current INS/GNSS state.
    gnss : GNSSMeasurement
        GNSS measurement with position and/or velocity.

    Returns
    -------
    result : LooseCoupledResult
        Updated state and innovation statistics.
    """
    if not gnss.valid:
        return LooseCoupledResult(
            state=state,
            innovation=np.zeros(6),
            innovation_cov=np.eye(6) * 1e10,
        )

    has_pos = gnss.position is not None
    has_vel = gnss.velocity is not None

    if has_pos and has_vel:
        # Full position + velocity update
        H = position_velocity_measurement_matrix()

        R_pos = (
            gnss.position_cov
            if gnss.position_cov is not None
            else np.diag([10.0**2] * 3)
        )
        R_vel = (
            gnss.velocity_cov
            if gnss.velocity_cov is not None
            else np.diag([0.1**2] * 3)
        )
        R = np.block([[R_pos, np.zeros((3, 3))], [np.zeros((3, 3)), R_vel]])

        z = np.concatenate(
            [
                gnss.position - state.ins_state.position,
                gnss.velocity - state.ins_state.velocity,
            ]
        )

        result = kf_update(state.error_state, state.error_cov, z, H, R)

        corrected_ins = _apply_error_correction(state.ins_state, result.x)
        new_error_state = np.zeros(15, dtype=np.float64)

        new_state = INSGNSSState(
            ins_state=corrected_ins,
            error_state=new_error_state,
            error_cov=result.P,
            clock_bias=state.clock_bias,
            clock_drift=state.clock_drift,
        )

        return LooseCoupledResult(
            state=new_state,
            innovation=result.y,
            innovation_cov=result.S,
        )

    elif has_pos:
        return loose_coupled_update_position(state, gnss)
    elif has_vel:
        return loose_coupled_update_velocity(state, gnss)
    else:
        return LooseCoupledResult(
            state=state,
            innovation=np.zeros(6),
            innovation_cov=np.eye(6) * 1e10,
        )


# =============================================================================
# Tightly-Coupled Integration
# =============================================================================


def tight_coupled_pseudorange_innovation(
    state: INSGNSSState,
    satellites: List[SatelliteInfo],
    ellipsoid: Ellipsoid = WGS84,
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Compute pseudorange innovations for tightly-coupled update.

    Parameters
    ----------
    state : INSGNSSState
        Current INS/GNSS state.
    satellites : list of SatelliteInfo
        Satellite observations.
    ellipsoid : Ellipsoid, optional
        Reference ellipsoid.

    Returns
    -------
    innovations : ndarray
        Pseudorange innovations (measured - predicted).
    predicted : ndarray
        Predicted pseudoranges.
    """
    ins = state.ins_state
    lat, lon, alt = ins.position

    # User ECEF position
    user_x, user_y, user_z = geodetic_to_ecef(lat, lon, alt, ellipsoid)
    user_ecef = np.array([user_x, user_y, user_z])

    n_sats = len(satellites)
    innovations = np.zeros(n_sats, dtype=np.float64)
    predicted = np.zeros(n_sats, dtype=np.float64)

    for i, sat in enumerate(satellites):
        _, geo_range = compute_line_of_sight(user_ecef, sat.position)

        # Predicted pseudorange = geometric range + clock bias
        pred_pr = geo_range + state.clock_bias
        predicted[i] = pred_pr

        # Innovation
        innovations[i] = sat.pseudorange - pred_pr

    return innovations, predicted


def tight_coupled_measurement_matrix(
    state: INSGNSSState,
    satellites: List[SatelliteInfo],
    ellipsoid: Ellipsoid = WGS84,
) -> NDArray[np.floating]:
    """
    Compute measurement matrix for tightly-coupled pseudorange update.

    Maps 17-state error (15 INS + 2 clock) to pseudorange measurements.

    Parameters
    ----------
    state : INSGNSSState
        Current INS/GNSS state.
    satellites : list of SatelliteInfo
        Satellite observations.
    ellipsoid : Ellipsoid, optional
        Reference ellipsoid.

    Returns
    -------
    H : ndarray
        (n_sats x 17) measurement matrix.
    """
    ins = state.ins_state
    lat, lon, alt = ins.position

    # User ECEF position
    user_x, user_y, user_z = geodetic_to_ecef(lat, lon, alt, ellipsoid)
    user_ecef = np.array([user_x, user_y, user_z])

    n_sats = len(satellites)
    H = np.zeros((n_sats, 17), dtype=np.float64)

    # Jacobian of ECEF w.r.t. geodetic (simplified)
    cos_lat = np.cos(lat)
    sin_lat = np.sin(lat)
    cos_lon = np.cos(lon)
    sin_lon = np.sin(lon)

    # Approximate Jacobian (position derivatives)
    # This is a simplified linearization
    N = ellipsoid.a / np.sqrt(1 - ellipsoid.e2 * sin_lat**2)

    for i, sat in enumerate(satellites):
        los, _ = compute_line_of_sight(user_ecef, sat.position)

        # LOS components in ECEF
        los_x, los_y, los_z = (
            -los
        )  # Negative because increase in user pos decreases range

        # Transform LOS to geodetic derivatives
        # d(range)/d(lat), d(range)/d(lon), d(range)/d(alt)
        H[i, 0] = (
            los_x * (-sin_lat * cos_lon * N)
            + los_y * (-sin_lat * sin_lon * N)
            + los_z * (cos_lat * N * (1 - ellipsoid.e2))
        )
        H[i, 1] = los_x * (-cos_lat * sin_lon * N) + los_y * (cos_lat * cos_lon * N)
        H[i, 2] = (
            los_x * cos_lat * cos_lon + los_y * cos_lat * sin_lon + los_z * sin_lat
        )

        # Clock bias (state 15)
        H[i, 15] = 1.0

    return H


def tight_coupled_update(
    state: INSGNSSState,
    satellites: List[SatelliteInfo],
    pseudorange_std: float = 3.0,
    ellipsoid: Ellipsoid = WGS84,
) -> TightCoupledResult:
    """
    Perform tightly-coupled INS/GNSS update using pseudoranges.

    Parameters
    ----------
    state : INSGNSSState
        Current INS/GNSS state.
    satellites : list of SatelliteInfo
        Satellite observations with pseudoranges.
    pseudorange_std : float, optional
        Pseudorange measurement noise std (m). Default: 3.0.
    ellipsoid : Ellipsoid, optional
        Reference ellipsoid.

    Returns
    -------
    result : TightCoupledResult
        Updated state and DOP values.
    """
    if len(satellites) < 4:
        # Not enough satellites for 3D fix + clock
        return TightCoupledResult(
            state=state,
            innovations=np.array([]),
            dop=(float("inf"), float("inf"), float("inf"), float("inf")),
        )

    # Compute innovations
    innovations, _ = tight_coupled_pseudorange_innovation(state, satellites, ellipsoid)

    # Measurement matrix (17-state: 15 INS + clock bias + clock drift)
    H_full = tight_coupled_measurement_matrix(state, satellites, ellipsoid)

    # For simplicity, only use clock bias (not drift) - so 16-state
    # Extend error state with clock bias
    n_sats = len(satellites)
    x_extended = np.zeros(16, dtype=np.float64)
    x_extended[:15] = state.error_state
    x_extended[15] = 0.0  # Clock bias error (relative to current estimate)

    P_extended = np.zeros((16, 16), dtype=np.float64)
    P_extended[:15, :15] = state.error_cov
    P_extended[15, 15] = 1e6  # Large initial clock uncertainty

    # Use only first 16 columns of H
    H = H_full[:, :16]

    # Measurement noise
    R = np.eye(n_sats) * pseudorange_std**2

    # Kalman update
    result = kf_update(x_extended, P_extended, innovations, H, R)

    # Apply corrections
    corrected_ins = _apply_error_correction(state.ins_state, result.x[:15])
    new_clock_bias = state.clock_bias + result.x[15]

    new_state = INSGNSSState(
        ins_state=corrected_ins,
        error_state=np.zeros(15, dtype=np.float64),
        error_cov=result.P[:15, :15],
        clock_bias=new_clock_bias,
        clock_drift=state.clock_drift,
    )

    # Compute DOP from geometry
    ins = state.ins_state
    lat, lon, alt = ins.position
    user_x, user_y, user_z = geodetic_to_ecef(lat, lon, alt, ellipsoid)
    user_ecef = np.array([user_x, user_y, user_z])
    H_geom = pseudorange_measurement_matrix(user_ecef, satellites, include_clock=True)
    dop = compute_dop(H_geom)

    return TightCoupledResult(
        state=new_state,
        innovations=innovations,
        dop=dop,
    )


# =============================================================================
# Helper Functions
# =============================================================================


def _apply_error_correction(
    ins_state: INSState,
    error: ArrayLike,
) -> INSState:
    """
    Apply error state correction to INS state.

    Parameters
    ----------
    ins_state : INSState
        Current INS state.
    error : array_like
        15-element error state vector.

    Returns
    -------
    corrected : INSState
        Corrected INS state.
    """
    error = np.asarray(error, dtype=np.float64)

    # Position correction
    new_position = ins_state.position + error[0:3]

    # Velocity correction
    new_velocity = ins_state.velocity + error[3:6]

    # Attitude correction (small angle approximation)
    phi = error[6:9]  # Attitude error angles

    # Apply small angle rotation to quaternion
    q = ins_state.quaternion
    delta_q = np.array(
        [1.0, 0.5 * phi[0], 0.5 * phi[1], 0.5 * phi[2]], dtype=np.float64
    )
    delta_q = delta_q / np.linalg.norm(delta_q)

    # Quaternion multiplication (body frame correction)
    qw, qx, qy, qz = q
    dw, dx, dy, dz = delta_q

    new_q = np.array(
        [
            qw * dw - qx * dx - qy * dy - qz * dz,
            qw * dx + qx * dw + qy * dz - qz * dy,
            qw * dy - qx * dz + qy * dw + qz * dx,
            qw * dz + qx * dy - qy * dx + qz * dw,
        ],
        dtype=np.float64,
    )
    new_q = new_q / np.linalg.norm(new_q)

    return INSState(
        position=new_position,
        velocity=new_velocity,
        quaternion=new_q,
        time=ins_state.time,
    )


def gnss_outage_detection(
    innovations: ArrayLike,
    innovation_cov: ArrayLike,
    threshold: float = 5.991,
) -> bool:
    """
    Detect potential GNSS measurement faults using chi-squared test.

    Parameters
    ----------
    innovations : array_like
        Measurement innovations.
    innovation_cov : array_like
        Innovation covariance matrix.
    threshold : float, optional
        Chi-squared threshold (default: 5.991 for 2 DOF, 95% confidence).

    Returns
    -------
    fault_detected : bool
        True if measurement appears faulty.
    """
    innovations = np.asarray(innovations, dtype=np.float64)
    innovation_cov = np.asarray(innovation_cov, dtype=np.float64)

    try:
        nis = innovations @ np.linalg.solve(innovation_cov, innovations)
    except np.linalg.LinAlgError:
        return True  # Singular covariance indicates problem

    return float(nis) > threshold


__all__ = [
    # Constants
    "SPEED_OF_LIGHT",
    "GPS_L1_FREQ",
    "GPS_L1_WAVELENGTH",
    # State representation
    "GNSSMeasurement",
    "SatelliteInfo",
    "INSGNSSState",
    "LooseCoupledResult",
    "TightCoupledResult",
    # Measurement models
    "position_measurement_matrix",
    "velocity_measurement_matrix",
    "position_velocity_measurement_matrix",
    "compute_line_of_sight",
    "pseudorange_measurement_matrix",
    "compute_dop",
    "satellite_elevation_azimuth",
    # Loosely-coupled integration
    "initialize_ins_gnss",
    "loose_coupled_predict",
    "loose_coupled_update_position",
    "loose_coupled_update_velocity",
    "loose_coupled_update",
    # Tightly-coupled integration
    "tight_coupled_pseudorange_innovation",
    "tight_coupled_measurement_matrix",
    "tight_coupled_update",
    # Utilities
    "gnss_outage_detection",
]
