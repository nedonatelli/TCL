"""
Tests for continuous-time dynamics functions.

Tests cover:
- Drift functions (constant velocity, constant acceleration, Singer, coordinated turn)
- Diffusion matrix functions
- State Jacobian matrices
- Continuous-to-discrete conversion
- LTI discretization
"""

import numpy as np

from pytcl.dynamic_models.continuous_time.dynamics import (
    continuous_to_discrete,
    diffusion_constant_acceleration,
    diffusion_constant_velocity,
    diffusion_singer,
    discretize_lti,
    drift_constant_acceleration,
    drift_constant_velocity,
    drift_coordinated_turn_2d,
    drift_singer,
    state_jacobian_ca,
    state_jacobian_cv,
    state_jacobian_singer,
)


class TestDriftFunctions:
    """Tests for continuous-time drift functions."""

    def test_drift_constant_velocity_1d(self):
        """Test 1D constant velocity drift."""
        x = np.array([0.0, 5.0])
        a = drift_constant_velocity(x, num_dims=1)
        np.testing.assert_allclose(a, [5.0, 0.0])

    def test_drift_constant_velocity_3d(self):
        """Test 3D constant velocity drift."""
        x = np.array([0, 1, 0, 2, 0, 3])
        a = drift_constant_velocity(x, num_dims=3)
        expected = np.array([1, 0, 2, 0, 3, 0])
        np.testing.assert_allclose(a, expected)

    def test_drift_constant_acceleration_1d(self):
        """Test 1D constant acceleration drift."""
        x = np.array([0.0, 5.0, 2.0])
        a = drift_constant_acceleration(x, num_dims=1)
        np.testing.assert_allclose(a, [5.0, 2.0, 0.0])

    def test_drift_singer(self):
        """Test Singer model drift."""
        x = np.array([0.0, 0.0, 10.0])
        tau = 5.0
        a = drift_singer(x, tau=tau, num_dims=1)
        np.testing.assert_allclose(a, [0.0, 10.0, -2.0])

    def test_drift_coordinated_turn_2d(self):
        """Test 2D coordinated turn drift."""
        vx, vy, omega = 10.0, 5.0, 0.1
        x = np.array([0, vx, 0, vy, omega])
        a = drift_coordinated_turn_2d(x)
        expected = np.array([vx, -omega * vy, vy, omega * vx, 0])
        np.testing.assert_allclose(a, expected)


class TestDiffusionFunctions:
    """Tests for diffusion matrix functions."""

    def test_diffusion_constant_velocity(self):
        """Test constant velocity diffusion matrix."""
        x = np.zeros(6)
        D = diffusion_constant_velocity(x, sigma_a=1.0, num_dims=3)
        assert D.shape == (6, 3)
        assert D[1, 0] == 1.0
        assert D[3, 1] == 1.0
        assert D[5, 2] == 1.0

    def test_diffusion_constant_acceleration(self):
        """Test constant acceleration diffusion matrix."""
        x = np.zeros(9)
        D = diffusion_constant_acceleration(x, sigma_j=2.0, num_dims=3)
        assert D.shape == (9, 3)
        assert D[2, 0] == 2.0
        assert D[5, 1] == 2.0
        assert D[8, 2] == 2.0

    def test_diffusion_singer(self):
        """Test Singer model diffusion."""
        x = np.zeros(3)
        D = diffusion_singer(x, sigma_m=1.0, tau=10.0, num_dims=1)
        assert D.shape == (3, 1)
        expected_sigma = np.sqrt(2 * 1.0**2 / 10.0)
        assert np.isclose(D[2, 0], expected_sigma)


class TestStateJacobians:
    """Tests for state Jacobian matrices."""

    def test_state_jacobian_cv_1d(self):
        """Test 1D constant velocity Jacobian."""
        A = state_jacobian_cv(None, num_dims=1)
        expected = np.array([[0, 1], [0, 0]])
        np.testing.assert_allclose(A, expected)

    def test_state_jacobian_cv_3d(self):
        """Test 3D constant velocity Jacobian."""
        A = state_jacobian_cv(None, num_dims=3)
        assert A.shape == (6, 6)
        for d in range(3):
            assert A[d * 2, d * 2 + 1] == 1.0

    def test_state_jacobian_ca_1d(self):
        """Test 1D constant acceleration Jacobian."""
        A = state_jacobian_ca(None, num_dims=1)
        expected = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        np.testing.assert_allclose(A, expected)

    def test_state_jacobian_singer(self):
        """Test Singer model Jacobian."""
        tau = 5.0
        A = state_jacobian_singer(None, tau=tau, num_dims=1)
        expected = np.array([[0, 1, 0], [0, 0, 1], [0, 0, -1 / tau]])
        np.testing.assert_allclose(A, expected)


class TestContinuousToDiscrete:
    """Tests for continuous to discrete conversion."""

    def test_continuous_to_discrete_cv(self):
        """Test C2D for constant velocity."""
        A = np.array([[0, 1], [0, 0]])
        G = np.array([[0], [1]])
        Q_c = np.array([[1.0]])
        T = 0.1

        F, Q_d = continuous_to_discrete(A, G, Q_c, T)

        expected_F = np.array([[1, T], [0, 1]])
        np.testing.assert_allclose(F, expected_F, rtol=1e-10)
        np.testing.assert_allclose(Q_d, Q_d.T)
        assert np.all(np.linalg.eigvalsh(Q_d) >= -1e-10)


class TestDiscretizeLTI:
    """Tests for LTI discretization."""

    def test_discretize_lti_no_input(self):
        """Test discretization without input matrix."""
        A = np.array([[0, 1], [0, 0]])
        F, G = discretize_lti(A, T=0.1)
        assert G is None
        expected_F = np.array([[1, 0.1], [0, 1]])
        np.testing.assert_allclose(F, expected_F, rtol=1e-10)

    def test_discretize_lti_with_input(self):
        """Test discretization with input matrix."""
        A = np.array([[0, 1], [0, 0]])
        B = np.array([[0], [1]])
        T = 0.1

        F, G = discretize_lti(A, B, T=T)
        assert G is not None
        assert G.shape == (2, 1)
