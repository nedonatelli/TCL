"""
Tests for Inertial Navigation System (INS) mechanization.

Tests cover:
- INS state representation
- Gravity and Earth rate computations
- Coning and sculling corrections
- Strapdown mechanization
- Alignment algorithms
- Error state model
"""

import numpy as np
import pytest

from pytcl.navigation.ins import (
    A_EARTH,
    OMEGA_EARTH,
    IMUData,
    INSErrorState,
    INSState,
    coarse_alignment,
    compensate_imu_data,
    coning_correction,
    earth_rate_ned,
    gravity_ned,
    gyrocompass_alignment,
    initialize_ins_state,
    ins_error_state_matrix,
    ins_process_noise_matrix,
    mechanize_ins_ned,
    normal_gravity,
    radii_of_curvature,
    sculling_correction,
    skew_symmetric,
    transport_rate_ned,
    update_quaternion,
)


class TestINSState:
    """Tests for INS state representation."""

    def test_ins_state_creation(self):
        """Test INS state creation."""
        state = INSState(
            position=np.array([np.radians(45), np.radians(-75), 100.0]),
            velocity=np.array([10.0, 5.0, -1.0]),
            quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
            time=0.0,
        )

        assert np.isclose(state.latitude, np.radians(45))
        assert np.isclose(state.longitude, np.radians(-75))
        assert np.isclose(state.altitude, 100.0)
        assert np.isclose(state.velocity_north, 10.0)
        assert np.isclose(state.velocity_east, 5.0)
        assert np.isclose(state.velocity_down, -1.0)

    def test_ins_state_dcm_identity(self):
        """Test DCM computation for identity quaternion."""
        state = INSState(
            position=np.zeros(3),
            velocity=np.zeros(3),
            quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
            time=0.0,
        )

        dcm = state.dcm
        assert dcm.shape == (3, 3)
        np.testing.assert_allclose(dcm, np.eye(3), atol=1e-10)

    def test_ins_state_euler_angles(self):
        """Test Euler angle extraction."""
        # Create state with known roll/pitch/yaw
        roll, pitch, yaw = np.radians([10, 5, 30])
        state = initialize_ins_state(
            lat=0, lon=0, alt=0, roll=roll, pitch=pitch, yaw=yaw
        )

        euler = state.euler_angles()
        np.testing.assert_allclose(euler, [roll, pitch, yaw], atol=1e-10)

    def test_initialize_ins_state(self):
        """Test INS state initialization."""
        lat = np.radians(45)
        lon = np.radians(-75)
        alt = 1000.0
        vN, vE, vD = 100.0, 50.0, -5.0
        roll, pitch, yaw = np.radians([5, 10, 45])

        state = initialize_ins_state(
            lat=lat,
            lon=lon,
            alt=alt,
            vN=vN,
            vE=vE,
            vD=vD,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            time=1.5,
        )

        assert np.isclose(state.latitude, lat)
        assert np.isclose(state.longitude, lon)
        assert np.isclose(state.altitude, alt)
        np.testing.assert_allclose(state.velocity, [vN, vE, vD])
        assert state.time == 1.5


class TestIMUData:
    """Tests for IMU data representation."""

    def test_imu_data_creation(self):
        """Test IMU data creation."""
        imu = IMUData(
            accel=np.array([0.1, -0.2, -9.8]),
            gyro=np.array([0.01, -0.005, 0.02]),
            dt=0.01,
        )

        np.testing.assert_allclose(imu.accel, [0.1, -0.2, -9.8])
        np.testing.assert_allclose(imu.gyro, [0.01, -0.005, 0.02])
        assert imu.dt == 0.01


class TestINSErrorState:
    """Tests for INS error state representation."""

    def test_error_state_zeros(self):
        """Test zero error state creation."""
        error = INSErrorState.zeros()

        np.testing.assert_allclose(error.position_error, np.zeros(3))
        np.testing.assert_allclose(error.velocity_error, np.zeros(3))
        np.testing.assert_allclose(error.attitude_error, np.zeros(3))
        np.testing.assert_allclose(error.accel_bias, np.zeros(3))
        np.testing.assert_allclose(error.gyro_bias, np.zeros(3))

    def test_error_state_to_from_vector(self):
        """Test error state vector conversion."""
        error = INSErrorState(
            position_error=np.array([1e-6, 2e-6, 10.0]),
            velocity_error=np.array([0.1, 0.2, 0.3]),
            attitude_error=np.array([1e-3, 2e-3, 3e-3]),
            accel_bias=np.array([0.01, 0.02, 0.03]),
            gyro_bias=np.array([1e-4, 2e-4, 3e-4]),
        )

        vec = error.to_vector()
        assert vec.shape == (15,)

        error2 = INSErrorState.from_vector(vec)
        np.testing.assert_allclose(error2.position_error, error.position_error)
        np.testing.assert_allclose(error2.velocity_error, error.velocity_error)
        np.testing.assert_allclose(error2.attitude_error, error.attitude_error)
        np.testing.assert_allclose(error2.accel_bias, error.accel_bias)
        np.testing.assert_allclose(error2.gyro_bias, error.gyro_bias)


class TestGravity:
    """Tests for gravity computations."""

    def test_normal_gravity_equator(self):
        """Test gravity at equator, sea level."""
        g = normal_gravity(0.0, 0.0)
        # WGS84 equatorial gravity is ~9.780 m/s^2
        assert 9.78 < g < 9.79

    def test_normal_gravity_pole(self):
        """Test gravity at pole, sea level."""
        g = normal_gravity(np.pi / 2, 0.0)
        # WGS84 polar gravity is ~9.832 m/s^2
        assert 9.83 < g < 9.84

    def test_normal_gravity_altitude_decrease(self):
        """Test gravity decreases with altitude."""
        g_sea = normal_gravity(np.radians(45), 0.0)
        g_high = normal_gravity(np.radians(45), 10000.0)
        assert g_high < g_sea

    def test_gravity_ned_direction(self):
        """Test gravity vector points down in NED."""
        g_ned = gravity_ned(np.radians(45), 1000.0)

        # North and East components should be zero
        assert g_ned[0] == 0.0
        assert g_ned[1] == 0.0
        # Down component should be positive (gravity is positive down)
        assert g_ned[2] > 9.0


class TestEarthRate:
    """Tests for Earth rotation rate computations."""

    def test_earth_rate_equator(self):
        """Test Earth rate at equator."""
        omega = earth_rate_ned(0.0)

        # At equator: wN = omega_e, wE = 0, wD = 0
        np.testing.assert_allclose(omega[0], OMEGA_EARTH, atol=1e-12)
        assert omega[1] == 0.0
        np.testing.assert_allclose(omega[2], 0.0, atol=1e-12)

    def test_earth_rate_pole(self):
        """Test Earth rate at North pole."""
        omega = earth_rate_ned(np.pi / 2)

        # At pole: wN = 0, wE = 0, wD = -omega_e
        np.testing.assert_allclose(omega[0], 0.0, atol=1e-12)
        assert omega[1] == 0.0
        np.testing.assert_allclose(omega[2], -OMEGA_EARTH, atol=1e-12)

    def test_earth_rate_magnitude(self):
        """Test Earth rate magnitude is constant."""
        for lat_deg in [0, 30, 45, 60, 90]:
            omega = earth_rate_ned(np.radians(lat_deg))
            mag = np.linalg.norm(omega)
            np.testing.assert_allclose(mag, OMEGA_EARTH, atol=1e-12)


class TestTransportRate:
    """Tests for transport rate computations."""

    def test_transport_rate_stationary(self):
        """Test transport rate is zero when stationary."""
        omega = transport_rate_ned(np.radians(45), 1000.0, 0.0, 0.0)
        np.testing.assert_allclose(omega, np.zeros(3), atol=1e-12)

    def test_transport_rate_north_motion(self):
        """Test transport rate for northward motion."""
        vN = 100.0  # 100 m/s north
        omega = transport_rate_ned(np.radians(45), 0.0, vN, 0.0)

        # Moving north causes negative rotation about East axis
        assert omega[1] < 0

    def test_transport_rate_east_motion(self):
        """Test transport rate for eastward motion."""
        vE = 100.0  # 100 m/s east
        omega = transport_rate_ned(np.radians(45), 0.0, 0.0, vE)

        # Moving east causes positive rotation about North axis
        assert omega[0] > 0


class TestRadiiOfCurvature:
    """Tests for radii of curvature."""

    def test_radii_equator(self):
        """Test radii of curvature at equator."""
        RN, RE = radii_of_curvature(0.0)

        # At equator, RE = a (semi-major), RN = b^2/a
        np.testing.assert_allclose(RE, A_EARTH, rtol=1e-10)
        assert RN < RE  # Meridian radius is smaller at equator

    def test_radii_pole(self):
        """Test radii of curvature at pole."""
        RN, RE = radii_of_curvature(np.pi / 2)

        # At pole, RN = RE
        np.testing.assert_allclose(RN, RE, rtol=1e-6)


class TestConingAndSculling:
    """Tests for coning and sculling corrections."""

    def test_coning_correction_parallel(self):
        """Test coning correction for parallel angular rates."""
        # When rates are parallel, cross product is zero
        gyro = np.array([0.1, 0.0, 0.0])
        correction = coning_correction(gyro, gyro * 1.1)

        # For parallel vectors, y and z components of cross product are zero
        # Only x component may be non-zero due to magnitude difference
        assert np.linalg.norm(correction) < 1e-10

    def test_coning_correction_perpendicular(self):
        """Test coning correction for perpendicular angular rates."""
        gyro1 = np.array([0.1, 0.0, 0.0])
        gyro2 = np.array([0.0, 0.1, 0.0])

        correction = coning_correction(gyro1, gyro2)

        # Cross product of perpendicular vectors is non-zero
        assert np.linalg.norm(correction) > 0

    def test_sculling_correction_stationary(self):
        """Test sculling correction for constant inputs."""
        accel = np.array([0.0, 0.0, -9.8])
        gyro = np.array([0.0, 0.0, 0.0])

        correction = sculling_correction(accel, accel, gyro, gyro)
        np.testing.assert_allclose(correction, np.zeros(3), atol=1e-12)

    def test_compensate_imu_data_stationary(self):
        """Test compensated IMU data for stationary case."""
        accel = np.array([0.0, 0.0, -9.8])
        gyro = np.array([0.0, 0.0, 0.0])
        dt = 0.01

        delta_theta, delta_v = compensate_imu_data(accel, accel, gyro, gyro, dt)

        # For constant inputs, should be simple integration
        np.testing.assert_allclose(delta_theta, gyro * dt, atol=1e-12)
        # Delta_v should be close to accel * dt
        np.testing.assert_allclose(delta_v, accel * dt, atol=1e-6)


class TestAttitudeUpdate:
    """Tests for attitude update functions."""

    def test_skew_symmetric(self):
        """Test skew-symmetric matrix construction."""
        v = np.array([1.0, 2.0, 3.0])
        S = skew_symmetric(v)

        # Check skew-symmetry
        np.testing.assert_allclose(S, -S.T, atol=1e-10)

        # Check that S @ w = v x w for any vector w
        w = np.array([4.0, 5.0, 6.0])
        np.testing.assert_allclose(S @ w, np.cross(v, w), atol=1e-10)

    def test_update_quaternion_identity(self):
        """Test quaternion update with zero rotation."""
        q = np.array([1.0, 0.0, 0.0, 0.0])
        delta_theta = np.array([0.0, 0.0, 0.0])

        q_new = update_quaternion(q, delta_theta)
        np.testing.assert_allclose(q_new, q, atol=1e-10)

    def test_update_quaternion_small_rotation(self):
        """Test quaternion update with small rotation."""
        q = np.array([1.0, 0.0, 0.0, 0.0])
        delta_theta = np.array([0.01, 0.0, 0.0])  # Small roll

        q_new = update_quaternion(q, delta_theta)

        # Check normalization
        np.testing.assert_allclose(np.linalg.norm(q_new), 1.0, atol=1e-10)

        # Check that rotation was applied
        assert q_new[1] != 0  # qx should be non-zero

    def test_update_quaternion_90deg(self):
        """Test quaternion update with 90 degree rotation."""
        q = np.array([1.0, 0.0, 0.0, 0.0])
        delta_theta = np.array([np.pi / 2, 0.0, 0.0])  # 90 deg roll

        q_new = update_quaternion(q, delta_theta)

        # Expected: qw = cos(45deg), qx = sin(45deg)
        expected = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0])
        np.testing.assert_allclose(q_new, expected, atol=1e-10)


class TestMechanization:
    """Tests for INS mechanization."""

    def test_mechanize_stationary(self):
        """Test mechanization for stationary vehicle."""
        lat = np.radians(45)
        state = initialize_ins_state(lat=lat, lon=0, alt=0)

        g = normal_gravity(lat)
        # IMU measures only gravity (pointing up in body frame for level vehicle)
        imu = IMUData(
            accel=np.array([0.0, 0.0, -g]),
            gyro=np.array([0.0, 0.0, 0.0]),
            dt=0.01,
        )

        new_state = mechanize_ins_ned(state, imu)

        # Position should barely change
        np.testing.assert_allclose(new_state.position, state.position, atol=1e-6)

        # Velocity should remain near zero
        np.testing.assert_allclose(new_state.velocity, np.zeros(3), atol=0.01)

    def test_mechanize_time_update(self):
        """Test that mechanization updates time correctly."""
        state = initialize_ins_state(lat=0, lon=0, alt=0, time=10.0)

        imu = IMUData(
            accel=np.array([0.0, 0.0, -9.8]),
            gyro=np.array([0.0, 0.0, 0.0]),
            dt=0.1,
        )

        new_state = mechanize_ins_ned(state, imu)
        assert new_state.time == 10.1

    def test_mechanize_free_fall(self):
        """Test mechanization for free-fall (zero specific force)."""
        state = initialize_ins_state(lat=np.radians(45), lon=0, alt=10000, vD=0.0)

        # Zero specific force means accelerometer reads zero
        imu = IMUData(
            accel=np.array([0.0, 0.0, 0.0]),
            gyro=np.array([0.0, 0.0, 0.0]),
            dt=0.1,
        )

        new_state = mechanize_ins_ned(state, imu)

        # Vehicle should accelerate downward (positive vD)
        assert new_state.velocity_down > state.velocity_down

    def test_mechanize_with_coning_sculling(self):
        """Test mechanization with coning/sculling corrections."""
        state = initialize_ins_state(lat=np.radians(45), lon=0, alt=0)

        g = normal_gravity(np.radians(45))
        accel = np.array([0.0, 0.0, -g])
        gyro = np.array([0.01, 0.0, 0.0])  # Small rotation

        imu = IMUData(accel=accel, gyro=gyro, dt=0.01)

        # Run with and without corrections
        new_state_no_corr = mechanize_ins_ned(state, imu)
        new_state_with_corr = mechanize_ins_ned(
            state, imu, accel_prev=accel * 0.99, gyro_prev=gyro * 0.99
        )

        # Both should give similar results for small perturbations
        # Use atol for positions that may be very close to zero
        np.testing.assert_allclose(
            new_state_no_corr.position, new_state_with_corr.position, atol=1e-5
        )


class TestAlignment:
    """Tests for INS alignment algorithms."""

    def test_coarse_alignment_level(self):
        """Test coarse alignment for level vehicle."""
        # For level vehicle, accelerometer measures gravity straight down
        accel = np.array([0.0, 0.0, -9.8])
        lat = np.radians(45)

        roll, pitch = coarse_alignment(accel, lat)

        np.testing.assert_allclose(roll, 0.0, atol=1e-6)
        np.testing.assert_allclose(pitch, 0.0, atol=1e-6)

    def test_coarse_alignment_tilted(self):
        """Test coarse alignment for tilted vehicle."""
        # Simulate 10 degree roll (positive roll = right wing down)
        # For positive roll, the Y-axis tips down, so gravity has +Y component
        # and smaller -Z component
        roll_true = np.radians(10)
        g = 9.8
        # When rolled right by roll_true, the -Z gravity vector projects as:
        # ay = -g * sin(roll_true) (negative because roll is right wing down)
        # az = -g * cos(roll_true)
        accel = np.array([0.0, -g * np.sin(roll_true), -g * np.cos(roll_true)])
        lat = np.radians(45)

        roll, pitch = coarse_alignment(accel, lat)

        np.testing.assert_allclose(roll, roll_true, atol=1e-3)
        np.testing.assert_allclose(pitch, 0.0, atol=1e-6)

    def test_gyrocompass_north_heading(self):
        """Test gyrocompass alignment for north heading."""
        lat = np.radians(45)
        roll, pitch = 0.0, 0.0

        # For north heading, gyro measures Earth rate in x (forward) direction
        omega_h = OMEGA_EARTH * np.cos(lat)
        gyro = np.array([omega_h, 0.0, -OMEGA_EARTH * np.sin(lat)])

        yaw = gyrocompass_alignment(gyro, roll, pitch, lat)

        # Should detect north heading (yaw = 0)
        np.testing.assert_allclose(yaw, 0.0, atol=0.01)


class TestErrorStateModel:
    """Tests for INS error state model."""

    def test_error_state_matrix_shape(self):
        """Test error state matrix has correct shape."""
        state = initialize_ins_state(lat=np.radians(45), lon=0, alt=1000)
        F = ins_error_state_matrix(state)

        assert F.shape == (15, 15)

    def test_error_state_matrix_structure(self):
        """Test error state matrix has expected structure."""
        state = initialize_ins_state(lat=np.radians(45), lon=0, alt=1000, vN=100, vE=50)
        F = ins_error_state_matrix(state)

        # Position-velocity coupling should be non-zero
        assert np.any(F[0:3, 3:6] != 0)

        # Velocity-attitude coupling should be non-zero
        assert np.any(F[3:6, 6:9] != 0)

        # Attitude-gyro bias coupling should be non-zero
        assert np.any(F[6:9, 12:15] != 0)

    def test_process_noise_matrix_shape(self):
        """Test process noise matrix has correct shape."""
        state = initialize_ins_state(lat=np.radians(45), lon=0, alt=1000)
        Q = ins_process_noise_matrix(
            accel_noise_std=0.01,
            gyro_noise_std=1e-4,
            accel_bias_std=1e-5,
            gyro_bias_std=1e-7,
            state=state,
        )

        assert Q.shape == (15, 15)

    def test_process_noise_matrix_symmetry(self):
        """Test process noise matrix is symmetric positive semi-definite."""
        state = initialize_ins_state(lat=np.radians(45), lon=0, alt=1000)
        Q = ins_process_noise_matrix(
            accel_noise_std=0.01,
            gyro_noise_std=1e-4,
            accel_bias_std=1e-5,
            gyro_bias_std=1e-7,
            state=state,
        )

        # Check symmetry
        np.testing.assert_allclose(Q, Q.T, atol=1e-15)

        # Check positive semi-definiteness (all eigenvalues >= 0)
        eigenvalues = np.linalg.eigvalsh(Q)
        assert np.all(eigenvalues >= -1e-15)


class TestIntegration:
    """Integration tests for INS mechanization."""

    def test_circular_motion(self):
        """Test mechanization for vehicle in circular motion."""
        # Start at equator, heading east
        lat0 = 0.0
        lon0 = 0.0
        alt0 = 0.0
        vE = 100.0  # 100 m/s east

        state = initialize_ins_state(lat=lat0, lon=lon0, alt=alt0, vE=vE, yaw=np.pi / 2)

        g = normal_gravity(lat0)

        # For circular motion, need centripetal acceleration
        # and gyro rate corresponding to Earth rate + transport rate
        omega_ie = earth_rate_ned(lat0)
        omega_en = transport_rate_ned(lat0, alt0, 0.0, vE)

        # Total navigation frame rate
        omega_in_n = omega_ie + omega_en

        # Run for several steps
        dt = 0.01
        for _ in range(100):
            imu = IMUData(
                accel=np.array([0.0, 0.0, -g]),
                gyro=state.dcm.T @ omega_in_n,  # Transform to body frame
                dt=dt,
            )
            state = mechanize_ins_ned(state, imu)

        # Check that altitude stayed roughly constant
        np.testing.assert_allclose(state.altitude, alt0, atol=1.0)

        # Check that speed stayed roughly constant
        speed = np.sqrt(state.velocity_north**2 + state.velocity_east**2)
        np.testing.assert_allclose(speed, vE, rtol=0.01)

    @pytest.mark.slow
    def test_long_duration_stability(self):
        """Test mechanization stability over longer duration."""
        state = initialize_ins_state(lat=np.radians(45), lon=0, alt=10000, vN=200)

        g = normal_gravity(state.latitude, state.altitude)

        dt = 0.01
        n_steps = 10000  # 100 seconds

        for _ in range(n_steps):
            imu = IMUData(
                accel=np.array([0.0, 0.0, -g]),
                gyro=np.array([0.0, 0.0, 0.0]),
                dt=dt,
            )
            state = mechanize_ins_ned(state, imu)

            # Check for numerical stability
            assert np.isfinite(state.position).all()
            assert np.isfinite(state.velocity).all()
            assert np.isfinite(state.quaternion).all()

        # Quaternion should still be normalized
        np.testing.assert_allclose(np.linalg.norm(state.quaternion), 1.0, atol=1e-6)
