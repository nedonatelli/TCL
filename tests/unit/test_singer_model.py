"""
Tests for Singer discrete-time acceleration model.

Tests cover:
- 1D, 2D, 3D Singer transition matrices
- State propagation behavior
"""

import numpy as np

from pytcl.dynamic_models.discrete_time.singer import (
    f_singer,
    f_singer_2d,
    f_singer_3d,
)


class TestSingerModel:
    """Tests for Singer acceleration model."""

    def test_f_singer_1d(self):
        """Test 1D Singer model."""
        T = 1.0
        tau = 10.0
        F = f_singer(T, tau)

        assert F.shape == (3, 3)
        assert F[0, 0] == 1.0
        assert F[0, 1] == T
        assert F[1, 0] == 0.0
        assert F[1, 1] == 1.0

        alpha = np.exp(-T / tau)
        assert np.isclose(F[2, 2], alpha)

    def test_f_singer_2d(self):
        """Test 2D Singer model."""
        T = 0.5
        tau = 5.0
        F = f_singer_2d(T, tau)
        F_expected = f_singer(T, tau, num_dims=2)

        assert F.shape == (6, 6)
        assert np.allclose(F, F_expected)
        assert np.allclose(F[:3, :3], F[3:6, 3:6])
        assert np.allclose(F[:3, 3:6], 0)
        assert np.allclose(F[3:6, :3], 0)

    def test_f_singer_3d(self):
        """Test 3D Singer model."""
        T = 0.1
        tau = 20.0
        F = f_singer_3d(T, tau)
        F_expected = f_singer(T, tau, num_dims=3)

        assert F.shape == (9, 9)
        assert np.allclose(F, F_expected)

    def test_f_singer_state_propagation(self):
        """Test state propagation with Singer model."""
        T = 0.1
        tau = 5.0
        F = f_singer(T, tau)

        x = np.array([0.0, 10.0, 2.0])
        x_next = F @ x

        expected_pos_approx = x[0] + x[1] * T + 0.5 * x[2] * T**2
        assert x_next[0] > x[0]
        assert np.isclose(x_next[0], expected_pos_approx, atol=0.1)

        expected_vel_approx = x[1] + x[2] * T
        assert np.isclose(x_next[1], expected_vel_approx, atol=0.5)

        assert abs(x_next[2]) < abs(x[2])
