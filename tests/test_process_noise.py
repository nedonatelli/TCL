"""
Tests for process noise models.

Tests cover:
- Singer process noise covariance (1D, 2D, 3D)
- Coordinated turn process noise (2D, 3D, polar)
"""

import numpy as np

from pytcl.dynamic_models.process_noise.coordinated_turn import (
    q_coord_turn_2d,
    q_coord_turn_3d,
    q_coord_turn_polar,
)
from pytcl.dynamic_models.process_noise.singer import (
    q_singer,
    q_singer_2d,
    q_singer_3d,
)


class TestSingerProcessNoise:
    """Tests for Singer process noise model."""

    def test_q_singer_1d(self):
        """Test Singer process noise covariance 1D."""
        T = 0.1
        tau = 10.0
        sigma_m = 5.0
        Q = q_singer(T, tau, sigma_m, num_dims=1)
        assert Q.shape == (3, 3)
        np.testing.assert_allclose(Q, Q.T)
        assert np.all(np.diag(Q) >= 0)

    def test_q_singer_2d(self):
        """Test Singer process noise covariance 2D."""
        Q = q_singer_2d(T=0.1, tau=10.0, sigma_m=5.0)
        assert Q.shape == (6, 6)
        np.testing.assert_allclose(Q, Q.T)

    def test_q_singer_3d(self):
        """Test Singer process noise covariance 3D."""
        Q = q_singer_3d(T=0.1, tau=10.0, sigma_m=5.0)
        assert Q.shape == (9, 9)
        np.testing.assert_allclose(Q, Q.T)


class TestCoordinatedTurnProcessNoise:
    """Tests for coordinated turn process noise."""

    def test_q_coord_turn_2d_position_velocity(self):
        """Test 2D coordinated turn process noise (pos/vel only)."""
        Q = q_coord_turn_2d(T=0.1, sigma_a=1.0)
        assert Q.shape == (4, 4)
        np.testing.assert_allclose(Q, Q.T)

    def test_q_coord_turn_2d_with_omega(self):
        """Test 2D coordinated turn process noise with turn rate."""
        Q = q_coord_turn_2d(
            T=0.1,
            sigma_a=1.0,
            sigma_omega=0.01,
            state_type="position_velocity_omega",
        )
        assert Q.shape == (5, 5)
        np.testing.assert_allclose(Q, Q.T)

    def test_q_coord_turn_3d(self):
        """Test 3D coordinated turn process noise."""
        Q = q_coord_turn_3d(T=0.1, sigma_a=1.0)
        assert Q.shape == (6, 6)
        np.testing.assert_allclose(Q, Q.T)

    def test_q_coord_turn_polar(self):
        """Test polar form coordinated turn process noise."""
        Q = q_coord_turn_polar(T=0.1, sigma_a=1.0, sigma_omega_dot=0.01)
        assert Q.shape == (5, 5)
        np.testing.assert_allclose(Q, Q.T)
