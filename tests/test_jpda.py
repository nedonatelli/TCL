"""
Tests for Joint Probabilistic Data Association (JPDA) algorithms.

This module contains tests for JPDA-based multi-target association and tracking.
Tests are migrated from v0.3.0 comprehensive test suite.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

# =============================================================================
# JPDA Basic Tests
# =============================================================================


class TestJPDA:
    """Tests for Joint Probabilistic Data Association."""

    def test_jpda_probabilities_single_track(self):
        """Test JPDA with single track."""
        from pytcl.assignment_algorithms.jpda import jpda_probabilities

        # Single track, two measurements
        likelihood = np.array([[0.8, 0.1]])
        gated = np.array([[True, True]])

        beta = jpda_probabilities(
            likelihood, gated, detection_prob=0.9, clutter_density=0.01
        )

        assert beta.shape == (1, 3)  # 1 track, 2 meas + 1 for no-meas
        # Probabilities should sum to 1 for each track
        assert_allclose(np.sum(beta[0, :]), 1.0, rtol=1e-6)
        # Higher likelihood measurement should have higher probability
        assert beta[0, 0] > beta[0, 1]

    def test_jpda_update_basic(self):
        """Test basic JPDA update."""
        from pytcl.assignment_algorithms.jpda import jpda_update

        x1 = np.array([0.0, 1.0])
        x2 = np.array([5.0, -1.0])
        P = np.eye(2) * 0.5

        measurements = np.array([[0.1], [5.2], [10.0]])
        H = np.array([[1.0, 0.0]])
        R = np.array([[0.1]])

        result = jpda_update([x1, x2], [P, P], measurements, H, R)

        assert len(result.states) == 2
        assert len(result.covariances) == 2
        assert result.association_probs.shape == (2, 4)  # 2 tracks, 3 meas + no-meas

    def test_jpda_no_measurements(self):
        """Test JPDA with no measurements."""
        from pytcl.assignment_algorithms.jpda import jpda_update

        x = np.array([0.0, 1.0])
        P = np.eye(2) * 0.5

        measurements = np.array([]).reshape(0, 1)
        H = np.array([[1.0, 0.0]])
        R = np.array([[0.1]])

        result = jpda_update([x], [P], measurements, H, R)

        # With no measurements, state should be unchanged
        assert_allclose(result.states[0], x)

    def test_jpda_result_convenience(self):
        """Test JPDA convenience function."""
        from pytcl.assignment_algorithms import jpda

        x1 = np.array([0.0, 1.0])
        x2 = np.array([5.0, -1.0])
        P = np.eye(2) * 0.5

        measurements = np.array([[0.1], [5.2]])
        H = np.array([[1.0, 0.0]])
        R = np.array([[0.1]])

        result = jpda([x1, x2], [P, P], measurements, H, R)

        assert result.association_probs.shape == (2, 3)
        assert len(result.marginal_probs) == 2
        assert result.likelihood_matrix.shape == (2, 2)


# =============================================================================
# JPDA Comprehensive Tests (v0.3.0)
# =============================================================================


class TestJPDAComprehensive:
    """Comprehensive tests for JPDA algorithm robustness and edge cases."""

    def test_jpda_probabilities_normalization(self):
        """Test JPDA probabilities sum to 1."""
        from pytcl.assignment_algorithms.jpda import jpda_probabilities

        # Multiple tracks, multiple measurements
        likelihood = np.array([[0.8, 0.1, 0.05], [0.1, 0.7, 0.1], [0.05, 0.1, 0.6]])
        gated = np.ones_like(likelihood, dtype=bool)

        beta = jpda_probabilities(likelihood, gated, detection_prob=0.9)

        # Each track's probabilities should sum to 1
        for i in range(3):
            assert_allclose(np.sum(beta[i, :]), 1.0, rtol=1e-6)

    def test_jpda_high_clutter(self):
        """Test JPDA behavior with high clutter density."""
        from pytcl.assignment_algorithms.jpda import jpda_probabilities

        likelihood = np.array([[0.5, 0.3]])
        gated = np.array([[True, True]])

        # High clutter should increase miss probability
        beta_low_clutter = jpda_probabilities(
            likelihood, gated, detection_prob=0.9, clutter_density=0.001
        )
        beta_high_clutter = jpda_probabilities(
            likelihood, gated, detection_prob=0.9, clutter_density=0.1
        )

        # With high clutter, more probability goes to "no measurement"
        assert beta_high_clutter[0, -1] > beta_low_clutter[0, -1]

    def test_jpda_low_detection_probability(self):
        """Test JPDA with low detection probability."""
        from pytcl.assignment_algorithms.jpda import jpda_probabilities

        likelihood = np.array([[0.8, 0.1]])
        gated = np.array([[True, True]])

        beta_high_pd = jpda_probabilities(
            likelihood, gated, detection_prob=0.99, clutter_density=0.01
        )
        beta_low_pd = jpda_probabilities(
            likelihood, gated, detection_prob=0.5, clutter_density=0.01
        )

        # Low detection prob should increase miss probability
        assert beta_low_pd[0, -1] > beta_high_pd[0, -1]

    def test_jpda_gating_effect(self):
        """Test that gating correctly excludes measurements."""
        from pytcl.assignment_algorithms.jpda import jpda_probabilities

        likelihood = np.array([[0.8, 0.7, 0.1]])
        gated = np.array([[True, False, True]])  # Middle measurement not gated

        beta = jpda_probabilities(likelihood, gated, detection_prob=0.9)

        # Gated-out measurement should have zero probability
        assert_allclose(beta[0, 1], 0.0)

    def test_jpda_update_with_ambiguous_measurements(self):
        """Test JPDA update handles ambiguous measurements."""
        from pytcl.assignment_algorithms.jpda import jpda_update

        x = np.array([0.0, 1.0])
        P = np.eye(2) * 0.1

        # Two measurements, both plausible
        measurements = np.array([[0.1], [-0.1]])
        H = np.array([[1.0, 0.0]])
        R = np.array([[0.1]])

        result = jpda_update([x], [P], measurements, H, R)

        # JPDA should produce valid state and covariance
        assert result.states[0].shape == (2,)
        assert result.covariances[0].shape == (2, 2)
        # Covariance should remain positive definite
        eigvals = np.linalg.eigvalsh(result.covariances[0])
        assert np.all(eigvals > 0)

    def test_jpda_multiple_tracks(self):
        """Test JPDA with multiple tracks."""
        from pytcl.assignment_algorithms.jpda import jpda_update

        # 3 tracks at different positions
        tracks = [
            np.array([0.0, 1.0]),
            np.array([5.0, 0.0]),
            np.array([10.0, -1.0]),
        ]
        covs = [np.eye(2) * 0.1 for _ in range(3)]

        # 3 measurements near each track
        measurements = np.array([[0.1], [5.1], [9.9]])
        H = np.array([[1.0, 0.0]])
        R = np.array([[0.1]])

        result = jpda_update(tracks, covs, measurements, H, R)

        assert len(result.states) == 3
        assert result.association_probs.shape == (3, 4)  # 3 tracks, 3+1 columns

    def test_jpda_empty_tracks(self):
        """Test JPDA handles empty track list."""
        from pytcl.assignment_algorithms.jpda import jpda_update

        measurements = np.array([[0.1], [0.2]])
        H = np.array([[1.0, 0.0]])
        R = np.array([[0.1]])

        result = jpda_update([], [], measurements, H, R)

        assert len(result.states) == 0
        assert len(result.covariances) == 0

    def test_jpda_single_measurement_per_track(self):
        """Test JPDA when each track has exactly one measurement."""
        from pytcl.assignment_algorithms.jpda import jpda_update

        tracks = [np.array([0.0, 1.0]), np.array([10.0, -1.0])]
        covs = [np.eye(2) * 0.01, np.eye(2) * 0.01]  # Very small covariance

        # Measurements clearly associated with each track
        measurements = np.array([[0.01], [10.01]])
        H = np.array([[1.0, 0.0]])
        R = np.array([[0.001]])

        result = jpda_update(tracks, covs, measurements, H, R)

        # Each track should strongly associate with its measurement
        assert result.association_probs[0, 0] > 0.9
        assert result.association_probs[1, 1] > 0.9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
