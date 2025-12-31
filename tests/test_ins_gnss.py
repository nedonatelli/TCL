"""
Tests for INS/GNSS Integration.

Tests cover:
- GNSS measurement models
- Loosely-coupled integration
- Tightly-coupled integration
- DOP computation
- Fault detection
"""

import numpy as np
import pytest

from pytcl.navigation.geodesy import geodetic_to_ecef
from pytcl.navigation.ins import IMUData, initialize_ins_state, normal_gravity
from pytcl.navigation.ins_gnss import (
    GPS_L1_FREQ,
    GPS_L1_WAVELENGTH,
    SPEED_OF_LIGHT,
    GNSSMeasurement,
    INSGNSSState,
    SatelliteInfo,
    compute_dop,
    compute_line_of_sight,
    gnss_outage_detection,
    initialize_ins_gnss,
    loose_coupled_predict,
    loose_coupled_update,
    loose_coupled_update_position,
    loose_coupled_update_velocity,
    position_measurement_matrix,
    position_velocity_measurement_matrix,
    pseudorange_measurement_matrix,
    satellite_elevation_azimuth,
    tight_coupled_pseudorange_innovation,
    tight_coupled_update,
    velocity_measurement_matrix,
)


class TestConstants:
    """Tests for GNSS constants."""

    def test_speed_of_light(self):
        """Test speed of light constant."""
        assert SPEED_OF_LIGHT == pytest.approx(299792458.0, rel=1e-10)

    def test_gps_l1_frequency(self):
        """Test GPS L1 frequency."""
        assert GPS_L1_FREQ == pytest.approx(1575.42e6, rel=1e-6)

    def test_gps_l1_wavelength(self):
        """Test GPS L1 wavelength consistency."""
        expected = SPEED_OF_LIGHT / GPS_L1_FREQ
        assert GPS_L1_WAVELENGTH == pytest.approx(expected, rel=1e-10)


class TestMeasurementMatrices:
    """Tests for measurement matrix construction."""

    def test_position_measurement_matrix_shape(self):
        """Test position measurement matrix shape."""
        H = position_measurement_matrix()
        assert H.shape == (3, 15)

    def test_position_measurement_matrix_values(self):
        """Test position measurement matrix extracts position errors."""
        H = position_measurement_matrix()

        # First 3 elements of error state are position errors
        error_state = np.zeros(15)
        error_state[0] = 1e-6  # lat error
        error_state[1] = 2e-6  # lon error
        error_state[2] = 10.0  # alt error

        result = H @ error_state
        np.testing.assert_allclose(result, [1e-6, 2e-6, 10.0])

    def test_velocity_measurement_matrix_shape(self):
        """Test velocity measurement matrix shape."""
        H = velocity_measurement_matrix()
        assert H.shape == (3, 15)

    def test_velocity_measurement_matrix_values(self):
        """Test velocity measurement matrix extracts velocity errors."""
        H = velocity_measurement_matrix()

        error_state = np.zeros(15)
        error_state[3] = 1.0  # vN error
        error_state[4] = 2.0  # vE error
        error_state[5] = 0.5  # vD error

        result = H @ error_state
        np.testing.assert_allclose(result, [1.0, 2.0, 0.5])

    def test_position_velocity_measurement_matrix_shape(self):
        """Test combined measurement matrix shape."""
        H = position_velocity_measurement_matrix()
        assert H.shape == (6, 15)

    def test_position_velocity_measurement_matrix_values(self):
        """Test combined matrix extracts both position and velocity."""
        H = position_velocity_measurement_matrix()

        error_state = np.zeros(15)
        error_state[0] = 1e-6
        error_state[3] = 1.0

        result = H @ error_state
        assert result[0] == pytest.approx(1e-6)
        assert result[3] == pytest.approx(1.0)


class TestLineOfSight:
    """Tests for line-of-sight computation."""

    def test_line_of_sight_vertical(self):
        """Test LOS for satellite directly above."""
        user = np.array([6378137.0, 0.0, 0.0])  # On equator
        sat = np.array([6378137.0 + 20200000.0, 0.0, 0.0])  # Directly above

        los, range_val = compute_line_of_sight(user, sat)

        # LOS should point radially outward
        np.testing.assert_allclose(los, [1.0, 0.0, 0.0], atol=1e-10)
        assert range_val == pytest.approx(20200000.0, rel=1e-10)

    def test_line_of_sight_unit_vector(self):
        """Test LOS is a unit vector."""
        user = np.array([6378137.0, 0.0, 0.0])
        sat = np.array([6378137.0 + 20000000.0, 5000000.0, 3000000.0])

        los, _ = compute_line_of_sight(user, sat)

        assert np.linalg.norm(los) == pytest.approx(1.0, rel=1e-10)


class TestPseudorangeGeometry:
    """Tests for pseudorange measurement geometry."""

    def test_pseudorange_matrix_shape(self):
        """Test pseudorange measurement matrix shape."""
        user = np.array([6378137.0, 0.0, 0.0])
        satellites = [
            SatelliteInfo(
                prn=i,
                position=np.array([6378137.0 + 20200000.0, i * 1e6, 0.0]),
                velocity=np.zeros(3),
                pseudorange=22000000.0,
            )
            for i in range(4)
        ]

        H = pseudorange_measurement_matrix(user, satellites, include_clock=True)
        assert H.shape == (4, 4)

        H_no_clock = pseudorange_measurement_matrix(
            user, satellites, include_clock=False
        )
        assert H_no_clock.shape == (4, 3)

    def test_pseudorange_matrix_clock_column(self):
        """Test clock bias column is all ones."""
        user = np.array([6378137.0, 0.0, 0.0])
        satellites = [
            SatelliteInfo(
                prn=i,
                position=np.array([6378137.0 + 20200000.0, i * 1e6, 0.0]),
                velocity=np.zeros(3),
                pseudorange=22000000.0,
            )
            for i in range(4)
        ]

        H = pseudorange_measurement_matrix(user, satellites, include_clock=True)

        # Clock column should be all ones
        np.testing.assert_allclose(H[:, 3], np.ones(4), atol=1e-10)


class TestDOP:
    """Tests for Dilution of Precision computation."""

    def test_dop_good_geometry(self):
        """Test DOP with good satellite geometry."""
        # Create geometry matrix with good coverage (diverse LOS vectors)
        # Satellites at different azimuths and elevations
        H = np.array(
            [
                [0.5, 0.5, 0.7071, 1.0],  # NE, high
                [-0.5, 0.5, 0.7071, 1.0],  # NW, high
                [0.5, -0.5, 0.7071, 1.0],  # SE, high
                [-0.5, -0.5, 0.7071, 1.0],  # SW, high
                [0.0, 0.0, 1.0, 1.0],  # Directly overhead (adds vertical diversity)
            ]
        )

        gdop, pdop, hdop, vdop = compute_dop(H)

        # All DOP values should be finite and reasonable
        assert np.isfinite(gdop)
        assert np.isfinite(pdop)
        assert np.isfinite(hdop)
        assert np.isfinite(vdop)

        # PDOP should be less than GDOP (GDOP includes clock)
        assert pdop < gdop

    def test_dop_poor_geometry(self):
        """Test DOP with poor (coplanar) geometry."""
        # Coplanar satellites - poor vertical geometry
        H = np.array(
            [
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0, 1.0],
                [0.0, -1.0, 0.0, 1.0],
            ]
        )

        gdop, pdop, hdop, vdop = compute_dop(H)

        # VDOP should be very large (poor vertical geometry)
        assert vdop > 10 or np.isinf(vdop)


class TestSatelliteElevationAzimuth:
    """Tests for satellite elevation/azimuth computation."""

    def test_satellite_directly_above(self):
        """Test elevation for satellite directly above."""
        user_lla = np.array([0.0, 0.0, 0.0])  # Equator, prime meridian

        # Satellite directly above on x-axis
        sat_ecef = np.array([6378137.0 + 20200000.0, 0.0, 0.0])

        elev, az = satellite_elevation_azimuth(user_lla, sat_ecef)

        # Should be at 90 degrees elevation
        assert elev == pytest.approx(np.pi / 2, abs=0.01)

    def test_satellite_on_horizon(self):
        """Test elevation for satellite on horizon."""
        user_lla = np.array([0.0, 0.0, 0.0])  # Equator, prime meridian

        # Satellite to the east, same altitude as user (approximately on horizon)
        # This is a simplified test - actual horizon geometry is more complex
        sat_ecef = np.array([6378137.0, 20200000.0, 0.0])

        elev, az = satellite_elevation_azimuth(user_lla, sat_ecef)

        # Should be close to 90 degrees azimuth (east)
        assert az == pytest.approx(np.pi / 2, abs=0.1)

    def test_azimuth_range(self):
        """Test azimuth is in valid range [0, 2pi)."""
        user_lla = np.array([np.radians(45), np.radians(-75), 100.0])

        for angle in np.linspace(0, 2 * np.pi, 8):
            sat_ecef = np.array(
                [
                    6378137.0 + 20200000.0 * np.cos(angle),
                    20200000.0 * np.sin(angle),
                    10000000.0,
                ]
            )
            _, az = satellite_elevation_azimuth(user_lla, sat_ecef)
            assert 0 <= az < 2 * np.pi


class TestINSGNSSInitialization:
    """Tests for INS/GNSS state initialization."""

    def test_initialize_ins_gnss(self):
        """Test basic initialization."""
        ins_state = initialize_ins_state(lat=np.radians(45), lon=0, alt=100)

        state = initialize_ins_gnss(ins_state)

        assert isinstance(state, INSGNSSState)
        assert state.ins_state == ins_state
        np.testing.assert_allclose(state.error_state, np.zeros(15))
        assert state.error_cov.shape == (15, 15)
        assert state.clock_bias == 0.0
        assert state.clock_drift == 0.0

    def test_initialize_covariance_diagonal(self):
        """Test initial covariance is diagonal."""
        ins_state = initialize_ins_state(lat=0, lon=0, alt=0)

        state = initialize_ins_gnss(
            ins_state,
            position_std=5.0,
            velocity_std=0.5,
        )

        # Check diagonal elements match expected variances
        assert state.error_cov[0, 0] == pytest.approx(25.0)  # position variance
        assert state.error_cov[3, 3] == pytest.approx(0.25)  # velocity variance

        # Off-diagonal should be zero
        assert state.error_cov[0, 3] == 0.0


class TestLooseCoupledIntegration:
    """Tests for loosely-coupled INS/GNSS integration."""

    def test_loose_coupled_predict(self):
        """Test prediction step."""
        ins_state = initialize_ins_state(lat=np.radians(45), lon=0, alt=1000)
        state = initialize_ins_gnss(ins_state)

        g = normal_gravity(np.radians(45), 1000)
        imu = IMUData(
            accel=np.array([0.0, 0.0, -g]),
            gyro=np.array([0.0, 0.0, 0.0]),
            dt=0.01,
        )

        new_state = loose_coupled_predict(state, imu)

        # State should be updated
        assert new_state.ins_state.time == state.ins_state.time + 0.01

        # Covariance should grow during prediction
        assert np.trace(new_state.error_cov) >= np.trace(state.error_cov)

    def test_loose_coupled_update_position(self):
        """Test position update reduces position uncertainty."""
        ins_state = initialize_ins_state(lat=np.radians(45), lon=0, alt=1000)
        state = initialize_ins_gnss(ins_state, position_std=100.0)

        # GNSS measurement at true position
        gnss = GNSSMeasurement(
            position=ins_state.position,
            velocity=None,
            position_cov=np.diag([1.0, 1.0, 2.0]),
            velocity_cov=None,
            time=0.0,
            valid=True,
        )

        result = loose_coupled_update_position(state, gnss)

        # Position covariance should decrease
        pos_var_before = np.trace(state.error_cov[0:3, 0:3])
        pos_var_after = np.trace(result.state.error_cov[0:3, 0:3])
        assert pos_var_after < pos_var_before

    def test_loose_coupled_update_velocity(self):
        """Test velocity update reduces velocity uncertainty."""
        ins_state = initialize_ins_state(lat=np.radians(45), lon=0, alt=1000, vN=10)
        state = initialize_ins_gnss(ins_state, velocity_std=10.0)

        gnss = GNSSMeasurement(
            position=None,
            velocity=ins_state.velocity,
            position_cov=None,
            velocity_cov=np.diag([0.1, 0.1, 0.1]),
            time=0.0,
            valid=True,
        )

        result = loose_coupled_update_velocity(state, gnss)

        vel_var_before = np.trace(state.error_cov[3:6, 3:6])
        vel_var_after = np.trace(result.state.error_cov[3:6, 3:6])
        assert vel_var_after < vel_var_before

    def test_loose_coupled_update_combined(self):
        """Test combined position+velocity update."""
        ins_state = initialize_ins_state(lat=np.radians(45), lon=0, alt=1000, vN=10)
        state = initialize_ins_gnss(ins_state)

        gnss = GNSSMeasurement(
            position=ins_state.position + np.array([1e-6, 0, 5]),  # Small offset
            velocity=ins_state.velocity + np.array([0.5, 0, 0]),
            position_cov=np.diag([1.0, 1.0, 2.0]),
            velocity_cov=np.diag([0.1, 0.1, 0.1]),
            time=0.0,
            valid=True,
        )

        result = loose_coupled_update(state, gnss)

        # Innovation should reflect the measurement offset
        assert result.innovation.shape == (6,)
        assert result.innovation_cov.shape == (6, 6)

    def test_loose_coupled_invalid_measurement(self):
        """Test handling of invalid GNSS measurement."""
        ins_state = initialize_ins_state(lat=np.radians(45), lon=0, alt=1000)
        state = initialize_ins_gnss(ins_state)

        gnss = GNSSMeasurement(
            position=None,
            velocity=None,
            position_cov=None,
            velocity_cov=None,
            time=0.0,
            valid=False,
        )

        result = loose_coupled_update(state, gnss)

        # State should be unchanged
        np.testing.assert_allclose(
            result.state.ins_state.position, state.ins_state.position
        )


class TestTightCoupledIntegration:
    """Tests for tightly-coupled INS/GNSS integration."""

    def _create_test_satellites(self, user_ecef):
        """Create test satellite constellation with good geometry."""
        # Create satellites at various positions with diverse geometry
        # Including vertical diversity for good VDOP
        satellites = []
        sat_positions = [
            (20e6, 5e6, 10e6),  # NE, high
            (20e6, -5e6, 10e6),  # SE, high
            (-10e6, 10e6, 15e6),  # NW, higher
            (-10e6, -10e6, 15e6),  # SW, higher
            (5e6, 0, 25e6),  # Near zenith
        ]
        for i, (dx, dy, dz) in enumerate(sat_positions):
            sat_pos = user_ecef + np.array([dx, dy, dz])
            geo_range = np.linalg.norm(sat_pos - user_ecef)

            satellites.append(
                SatelliteInfo(
                    prn=i + 1,
                    position=sat_pos,
                    velocity=np.zeros(3),
                    pseudorange=geo_range + 100.0,  # Include clock bias
                )
            )
        return satellites

    def test_tight_coupled_innovation(self):
        """Test pseudorange innovation computation."""
        ins_state = initialize_ins_state(lat=np.radians(45), lon=0, alt=1000)
        state = initialize_ins_gnss(ins_state)

        # Get user ECEF position
        lat, lon, alt = ins_state.position
        user_x, user_y, user_z = geodetic_to_ecef(lat, lon, alt)
        user_ecef = np.array([user_x, user_y, user_z])

        satellites = self._create_test_satellites(user_ecef)

        innovations, predicted = tight_coupled_pseudorange_innovation(state, satellites)

        assert innovations.shape == (5,)
        assert predicted.shape == (5,)

        # Innovations should be approximately the clock bias (100m in our test)
        for innov in innovations:
            assert innov == pytest.approx(100.0, abs=1.0)

    def test_tight_coupled_update(self):
        """Test tightly-coupled update with pseudoranges."""
        ins_state = initialize_ins_state(lat=np.radians(45), lon=0, alt=1000)
        state = initialize_ins_gnss(ins_state)

        lat, lon, alt = ins_state.position
        user_x, user_y, user_z = geodetic_to_ecef(lat, lon, alt)
        user_ecef = np.array([user_x, user_y, user_z])

        satellites = self._create_test_satellites(user_ecef)

        result = tight_coupled_update(state, satellites, pseudorange_std=3.0)

        # Should have valid DOP values
        gdop, pdop, hdop, vdop = result.dop
        assert np.isfinite(gdop)
        assert np.isfinite(pdop)

        # Clock bias should be estimated
        assert result.state.clock_bias != 0.0

    def test_tight_coupled_insufficient_satellites(self):
        """Test handling of insufficient satellites."""
        ins_state = initialize_ins_state(lat=np.radians(45), lon=0, alt=1000)
        state = initialize_ins_gnss(ins_state)

        # Only 3 satellites (need at least 4)
        satellites = [
            SatelliteInfo(
                prn=i,
                position=np.array([20e6, i * 1e6, 0]),
                velocity=np.zeros(3),
                pseudorange=22e6,
            )
            for i in range(3)
        ]

        result = tight_coupled_update(state, satellites)

        # Should return infinite DOP
        assert np.isinf(result.dop[0])


class TestFaultDetection:
    """Tests for GNSS fault detection."""

    def test_no_fault_detected(self):
        """Test no fault for consistent measurements."""
        innovations = np.array([0.5, 0.3, -0.2])
        innovation_cov = np.eye(3) * 1.0

        fault = gnss_outage_detection(innovations, innovation_cov, threshold=10.0)
        assert not fault

    def test_fault_detected_large_innovation(self):
        """Test fault detected for large innovations."""
        innovations = np.array([10.0, 10.0, 10.0])  # Large innovations
        innovation_cov = np.eye(3) * 0.1  # Small expected variance

        fault = gnss_outage_detection(innovations, innovation_cov, threshold=5.991)
        assert fault

    def test_fault_singular_covariance(self):
        """Test fault returned for singular covariance."""
        innovations = np.array([0.1, 0.1])
        innovation_cov = np.zeros((2, 2))  # Singular

        fault = gnss_outage_detection(innovations, innovation_cov)
        assert fault


class TestIntegration:
    """Integration tests for INS/GNSS."""

    def test_predict_update_cycle(self):
        """Test complete predict-update cycle."""
        ins_state = initialize_ins_state(
            lat=np.radians(45), lon=np.radians(-75), alt=100, vN=10
        )
        state = initialize_ins_gnss(ins_state)

        g = normal_gravity(np.radians(45), 100)

        # Run several predict-update cycles
        for i in range(10):
            # IMU prediction at 100 Hz
            for _ in range(10):
                imu = IMUData(
                    accel=np.array([0.0, 0.0, -g]),
                    gyro=np.array([0.0, 0.0, 0.0]),
                    dt=0.01,
                )
                state = loose_coupled_predict(state, imu)

            # GNSS update at 10 Hz
            gnss = GNSSMeasurement(
                position=state.ins_state.position,  # Perfect measurement
                velocity=state.ins_state.velocity,
                position_cov=np.diag([1.0, 1.0, 2.0]),
                velocity_cov=np.diag([0.1, 0.1, 0.1]),
                time=float(i) * 0.1,
                valid=True,
            )
            result = loose_coupled_update(state, gnss)
            state = result.state

        # Final state should be stable
        assert np.isfinite(state.ins_state.position).all()
        assert np.isfinite(state.error_cov).all()

        # Covariance should be bounded
        assert np.trace(state.error_cov) < 1e6

    def test_gnss_outage_recovery(self):
        """Test system behavior during GNSS outage."""
        ins_state = initialize_ins_state(lat=np.radians(45), lon=0, alt=100)
        state = initialize_ins_gnss(ins_state)

        initial_cov = np.trace(state.error_cov)

        g = normal_gravity(np.radians(45), 100)

        # Simulate GNSS outage (predictions only)
        for _ in range(100):
            imu = IMUData(
                accel=np.array([0.0, 0.0, -g]),
                gyro=np.array([0.0, 0.0, 0.0]),
                dt=0.01,
            )
            state = loose_coupled_predict(state, imu)

        # Covariance should grow during outage
        assert np.trace(state.error_cov) > initial_cov

        # GNSS recovery
        gnss = GNSSMeasurement(
            position=state.ins_state.position,
            velocity=state.ins_state.velocity,
            position_cov=np.diag([1.0, 1.0, 2.0]),
            velocity_cov=np.diag([0.1, 0.1, 0.1]),
            time=1.0,
            valid=True,
        )
        result = loose_coupled_update(state, gnss)

        # Covariance should decrease after GNSS update
        assert np.trace(result.state.error_cov) < np.trace(state.error_cov)
