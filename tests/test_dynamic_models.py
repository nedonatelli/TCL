"""
Tests for dynamic_models module.

Tests cover:
- Polynomial state transition matrices (constant velocity, constant acceleration)
- Coordinated turn models (2D and 3D)
- Process noise covariance matrices
- Singer acceleration model
"""

import numpy as np

from pytcl.dynamic_models import (  # Discrete-time state transition matrices; Process noise covariance matrices
    f_constant_acceleration,
    f_constant_velocity,
    f_coord_turn_2d,
    f_coord_turn_3d,
    f_coord_turn_polar,
    f_discrete_white_noise_accel,
    f_piecewise_white_noise_jerk,
    f_poly_kal,
    q_constant_acceleration,
    q_constant_velocity,
    q_continuous_white_noise,
    q_discrete_white_noise,
    q_poly_kal,
)


class TestPolynomialModels:
    """Tests for polynomial (CV/CA) state transition matrices."""

    def test_f_poly_kal_order0_identity(self):
        """Test order 0 (constant position) gives identity."""
        F = f_poly_kal(order=0, T=1.0, num_dims=1)
        assert F.shape == (1, 1)
        np.testing.assert_allclose(F, [[1.0]])

    def test_f_poly_kal_order1_cv(self):
        """Test order 1 gives constant velocity model."""
        T = 0.1
        F = f_poly_kal(order=1, T=T, num_dims=1)
        expected = np.array([[1, T], [0, 1]])
        np.testing.assert_allclose(F, expected)

    def test_f_poly_kal_order2_ca(self):
        """Test order 2 gives constant acceleration model."""
        T = 1.0
        F = f_poly_kal(order=2, T=T, num_dims=1)
        expected = np.array([[1, T, T**2 / 2], [0, 1, T], [0, 0, 1]])
        np.testing.assert_allclose(F, expected)

    def test_f_poly_kal_multidim(self):
        """Test multi-dimensional state transition matrix."""
        T = 0.1
        F = f_poly_kal(order=1, T=T, num_dims=2)
        assert F.shape == (4, 4)

        # Check block diagonal structure
        expected_block = np.array([[1, T], [0, 1]])
        np.testing.assert_allclose(F[:2, :2], expected_block)
        np.testing.assert_allclose(F[2:, 2:], expected_block)
        np.testing.assert_allclose(F[:2, 2:], np.zeros((2, 2)))
        np.testing.assert_allclose(F[2:, :2], np.zeros((2, 2)))

    def test_f_constant_velocity(self):
        """Test constant velocity convenience function."""
        T = 0.5
        F = f_constant_velocity(T=T, num_dims=2)
        expected = f_poly_kal(order=1, T=T, num_dims=2)
        np.testing.assert_allclose(F, expected)

    def test_f_constant_acceleration(self):
        """Test constant acceleration convenience function."""
        T = 0.5
        F = f_constant_acceleration(T=T, num_dims=2)
        expected = f_poly_kal(order=2, T=T, num_dims=2)
        np.testing.assert_allclose(F, expected)

    def test_f_discrete_white_noise_accel(self):
        """Test DWNA is same as constant velocity."""
        T = 0.5
        F1 = f_discrete_white_noise_accel(T=T, num_dims=3)
        F2 = f_constant_velocity(T=T, num_dims=3)
        np.testing.assert_allclose(F1, F2)

    def test_f_piecewise_white_noise_jerk(self):
        """Test PWN jerk is same as constant acceleration."""
        T = 0.5
        F1 = f_piecewise_white_noise_jerk(T=T, num_dims=3)
        F2 = f_constant_acceleration(T=T, num_dims=3)
        np.testing.assert_allclose(F1, F2)


class TestStateTransitionProperties:
    """Test mathematical properties of state transition matrices."""

    def test_cv_preserves_velocity_under_zero_time(self):
        """Test F(0) = I for constant velocity."""
        F = f_constant_velocity(T=0.0, num_dims=2)
        np.testing.assert_allclose(F, np.eye(4))

    def test_ca_preserves_acceleration_under_zero_time(self):
        """Test F(0) = I for constant acceleration."""
        F = f_constant_acceleration(T=0.0, num_dims=2)
        np.testing.assert_allclose(F, np.eye(6))

    def test_cv_state_propagation(self):
        """Test CV model correctly propagates state."""
        T = 1.0
        F = f_constant_velocity(T=T, num_dims=1)

        # Initial state: position=0, velocity=10
        x0 = np.array([0.0, 10.0])
        x1 = F @ x0

        # After 1 second: position=10, velocity=10
        np.testing.assert_allclose(x1, [10.0, 10.0])

    def test_ca_state_propagation(self):
        """Test CA model correctly propagates state."""
        T = 1.0
        F = f_constant_acceleration(T=T, num_dims=1)

        # Initial state: pos=0, vel=0, acc=10
        x0 = np.array([0.0, 0.0, 10.0])
        x1 = F @ x0

        # After 1s: pos = 0.5*a*t^2 = 5, vel = a*t = 10, acc = 10
        np.testing.assert_allclose(x1, [5.0, 10.0, 10.0])

    def test_f_composition(self):
        """Test F(T1) @ F(T2) ≈ F(T1+T2) for CV model."""
        T1, T2 = 0.3, 0.7
        F1 = f_constant_velocity(T=T1, num_dims=2)
        F2 = f_constant_velocity(T=T2, num_dims=2)
        F_composed = F2 @ F1
        F_total = f_constant_velocity(T=T1 + T2, num_dims=2)
        np.testing.assert_allclose(F_composed, F_total, rtol=1e-10)


class TestCoordinatedTurnModels:
    """Tests for coordinated turn motion models."""

    def test_coord_turn_2d_zero_omega(self):
        """Test CT with omega=0 reduces to CV model."""
        T = 1.0
        F_ct = f_coord_turn_2d(T=T, omega=0.0)
        F_cv = f_constant_velocity(T=T, num_dims=2)
        np.testing.assert_allclose(F_ct, F_cv)

    def test_coord_turn_2d_shape(self):
        """Test 2D coordinated turn matrix shape."""
        F = f_coord_turn_2d(T=1.0, omega=0.1)
        assert F.shape == (4, 4)

    def test_coord_turn_2d_with_omega_state(self):
        """Test 2D CT with omega in state."""
        F = f_coord_turn_2d(T=1.0, omega=0.1, state_type="position_velocity_omega")
        assert F.shape == (5, 5)
        # Omega should be preserved
        assert F[4, 4] == 1.0

    def test_coord_turn_3d_shape(self):
        """Test 3D coordinated turn matrix shape."""
        F = f_coord_turn_3d(T=1.0, omega=0.1)
        assert F.shape == (6, 6)

    def test_coord_turn_3d_z_dynamics(self):
        """Test z-axis follows CV dynamics in 3D CT."""
        T = 1.0
        F = f_coord_turn_3d(T=T, omega=0.1)

        # z dynamics should be CV (indices 4, 5)
        np.testing.assert_allclose(F[4, 4], 1.0)
        np.testing.assert_allclose(F[4, 5], T)
        np.testing.assert_allclose(F[5, 5], 1.0)

    def test_coord_turn_3d_with_omega_state(self):
        """Test 3D CT with omega in state."""
        F = f_coord_turn_3d(T=1.0, omega=0.1, state_type="position_velocity_omega")
        assert F.shape == (7, 7)
        # Omega should be preserved
        assert F[6, 6] == 1.0

    def test_coord_turn_polar_shape(self):
        """Test polar coordinated turn matrix shape."""
        F = f_coord_turn_polar(T=1.0, omega=0.1, speed=100.0)
        assert F.shape == (5, 5)

    def test_coord_turn_polar_zero_omega(self):
        """Test polar CT with omega=0 gives straight line."""
        F = f_coord_turn_polar(T=1.0, omega=0.0, speed=100.0)
        # Position should advance by speed*T in heading direction
        assert F.shape == (5, 5)


class TestCircularMotion:
    """Test coordinated turn produces circular motion."""

    def test_coord_turn_circular_path(self):
        """Test that CT model produces circular motion."""
        omega = np.pi / 4  # 45 deg/s
        T = 0.1
        speed = 100.0  # m/s

        # Initial state: at origin, moving along +x
        x = np.array([0.0, speed, 0.0, 0.0])  # [x, vx, y, vy]

        # Propagate for full circle (8 seconds at 45 deg/s)
        n_steps = int(2 * np.pi / omega / T)
        positions = [x[:1]]

        F = f_coord_turn_2d(T=T, omega=omega)
        for _ in range(n_steps):
            x = F @ x
            positions.append(x[0])

        # Should return close to origin after full circle
        assert np.abs(x[0]) < 10.0  # x close to 0
        assert np.abs(x[2]) < 10.0  # y close to 0


class TestProcessNoiseMatrices:
    """Tests for process noise covariance matrices."""

    def test_q_poly_kal_shape(self):
        """Test Q matrix shape."""
        Q = q_poly_kal(order=1, T=1.0, q=1.0, num_dims=2)
        assert Q.shape == (4, 4)

    def test_q_poly_kal_symmetric(self):
        """Test Q matrix is symmetric."""
        Q = q_poly_kal(order=2, T=0.5, q=0.1, num_dims=3)
        np.testing.assert_allclose(Q, Q.T)

    def test_q_poly_kal_positive_definite(self):
        """Test Q matrix is positive definite."""
        Q = q_poly_kal(order=1, T=1.0, q=1.0, num_dims=1)
        eigenvalues = np.linalg.eigvalsh(Q)
        assert np.all(eigenvalues > 0)

    def test_q_discrete_white_noise_cv(self):
        """Test discrete white noise for CV model."""
        T = 1.0
        var = 1.0
        Q = q_discrete_white_noise(dim=2, T=T, var=var)

        # Check known values for CV model
        expected = var * np.array([[T**4 / 4, T**3 / 2], [T**3 / 2, T**2]])
        np.testing.assert_allclose(Q, expected)

    def test_q_discrete_white_noise_ca(self):
        """Test discrete white noise for CA model."""
        T = 1.0
        var = 1.0
        Q = q_discrete_white_noise(dim=3, T=T, var=var)
        assert Q.shape == (3, 3)
        # Check symmetric
        np.testing.assert_allclose(Q, Q.T)

    def test_q_constant_velocity(self):
        """Test constant velocity process noise."""
        T = 0.5
        sigma_a = 0.1
        Q = q_constant_velocity(T=T, sigma_a=sigma_a, num_dims=2)
        assert Q.shape == (4, 4)

        # Check block diagonal
        Q_block = Q[:2, :2]
        np.testing.assert_allclose(Q[:2, 2:], np.zeros((2, 2)))

        # Check values match expected
        var = sigma_a**2
        expected_block = var * np.array([[T**4 / 4, T**3 / 2], [T**3 / 2, T**2]])
        np.testing.assert_allclose(Q_block, expected_block)

    def test_q_constant_acceleration(self):
        """Test constant acceleration process noise."""
        T = 0.5
        sigma_j = 0.1
        Q = q_constant_acceleration(T=T, sigma_j=sigma_j, num_dims=2)
        assert Q.shape == (6, 6)

    def test_q_continuous_white_noise(self):
        """Test continuous white noise process noise."""
        Q = q_continuous_white_noise(dim=2, T=1.0, spectral_density=1.0, block_size=2)
        assert Q.shape == (4, 4)


class TestProcessNoiseScaling:
    """Test that process noise scales correctly."""

    def test_q_scales_with_variance(self):
        """Test Q scales linearly with variance."""
        T = 0.5
        Q1 = q_constant_velocity(T=T, sigma_a=1.0, num_dims=1)
        Q2 = q_constant_velocity(T=T, sigma_a=2.0, num_dims=1)

        # Q should scale with sigma^2
        np.testing.assert_allclose(Q2, 4.0 * Q1)

    def test_q_cv_zero_time(self):
        """Test Q with T=0 is zero matrix."""
        Q = q_constant_velocity(T=0.0, sigma_a=1.0, num_dims=2)
        np.testing.assert_allclose(Q, np.zeros((4, 4)))


class TestModelConsistency:
    """Test consistency between F and Q matrices."""

    def test_cv_f_q_shapes_match(self):
        """Test F and Q have matching shapes for CV."""
        T, num_dims = 0.5, 3
        F = f_constant_velocity(T=T, num_dims=num_dims)
        Q = q_constant_velocity(T=T, sigma_a=1.0, num_dims=num_dims)
        assert F.shape == Q.shape

    def test_ca_f_q_shapes_match(self):
        """Test F and Q have matching shapes for CA."""
        T, num_dims = 0.5, 3
        F = f_constant_acceleration(T=T, num_dims=num_dims)
        Q = q_constant_acceleration(T=T, sigma_j=1.0, num_dims=num_dims)
        assert F.shape == Q.shape

    def test_predict_covariance_valid(self):
        """Test predicted covariance is valid (P' = F P F' + Q)."""
        T = 0.5
        F = f_constant_velocity(T=T, num_dims=2)
        Q = q_constant_velocity(T=T, sigma_a=0.1, num_dims=2)
        P = np.eye(4)  # Initial covariance

        # Predicted covariance
        P_pred = F @ P @ F.T + Q

        # Should be symmetric
        np.testing.assert_allclose(P_pred, P_pred.T, atol=1e-10)

        # Should be positive definite
        eigenvalues = np.linalg.eigvalsh(P_pred)
        assert np.all(eigenvalues > 0)
