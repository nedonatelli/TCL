"""
Tests for Joint Probabilistic Data Association (JPDA) algorithms.

This module contains tests for JPDA-based multi-target association and tracking.
Tests are migrated from v0.3.0 comprehensive test suite.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.stats import chi2

from pytcl.assignment_algorithms.gating import mahalanobis_distance
from pytcl.assignment_algorithms.jpda import (
    compute_likelihood_matrix,
    compute_measurement_likelihood,
)

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


# =============================================================================
# compute_likelihood_matrix behavior-equality (perf-levers Task 1)
# =============================================================================


def _reference_likelihood_matrix(
    track_states, track_covariances, measurements, H, R, detection_prob, gate_threshold
):
    """Frozen copy of the pre-restructure `compute_likelihood_matrix` loop
    body: per (track, measurement) pair, one `mahalanobis_distance` solve
    (gating.py's own dispatched solve/inv path) plus one independent
    `compute_measurement_likelihood` solve, and a per-pair `np.linalg.det`
    of a matrix that only depends on the track. This is the defect the
    Task 1 restructure removes (single per-track `cho_factor` reused across
    measurements); this reference is deliberately never touched by that
    restructure so it stays the ground truth for behavior equality.

    `mahalanobis_distance` returns the *squared* Mahalanobis distance (see
    its docstring), and the loop below gates on that squared value directly
    against `gate_threshold` (itself on the chi-squared/squared-distance
    scale, e.g. `chi2.ppf(...)`) -- there is no separate square root step.
    """
    n_tracks = len(track_states)
    n_meas = len(measurements)

    likelihood_matrix = np.zeros((n_tracks, n_meas))
    gated = np.zeros((n_tracks, n_meas), dtype=bool)

    for i in range(n_tracks):
        z_pred = H @ track_states[i]
        S = H @ track_covariances[i] @ H.T + R

        for j in range(n_meas):
            innovation = measurements[j] - z_pred
            mahal_dist = mahalanobis_distance(innovation, S)

            if gate_threshold is None or mahal_dist <= gate_threshold:
                gated[i, j] = True
                likelihood_matrix[i, j] = compute_measurement_likelihood(
                    innovation, S, detection_prob
                )

    return likelihood_matrix, gated


_N_TRACKS_CHOICES = [1, 3, 10, 40]
_N_MEAS_CHOICES = [0, 1, 7, 25]
_M_CHOICES = [2, 3, 4, 6]


def _make_likelihood_matrix_scenario(seed):
    """Build one seeded (rng spawn key `seed`) scenario for the
    behavior-equality sweep.

    n_tracks/n_meas/m are drawn from the plan's grids
    ({1,3,10,40}/{0,1,7,25}/{2,3,4,6}); state dimension is set equal to `m`
    (H = identity) so track-covariance eigenvalues drive S's scale/
    conditioning directly. Each track's covariance is `(A @ A.T + m*I) *
    scale` with `scale` log-uniform over [1e-6, 1e6] and R fixed at
    `0.1 * I` -- this mixes S's that are dominated by the (tiny) noise floor
    (near-singular relative to typical gate thresholds) with S's many
    orders of magnitude larger (well-conditioned but numerically large),
    exercising both ends of the 1e-6..1e6 range named in the plan.

    Odd seeds get a finite gate threshold set to 1.5x
    chi2.ppf(0.999, df=m) -- comfortably above the mass of any mahal_sq
    this scenario can produce -- so no scenario's gate decisions sit close
    enough to the threshold for the two solve routes' ULP-level
    disagreement (see test_gating_mahalanobis_dispatch.py) to flip one.
    Even seeds leave gate_threshold unset (ungated).
    """
    rng = np.random.default_rng(seed)
    n_tracks = int(rng.choice(_N_TRACKS_CHOICES))
    n_meas = int(rng.choice(_N_MEAS_CHOICES))
    m = int(rng.choice(_M_CHOICES))

    H = np.eye(m)
    R = 0.1 * np.eye(m)

    track_states = [rng.normal(size=m) for _ in range(n_tracks)]
    track_covariances = []
    for _ in range(n_tracks):
        A = rng.normal(size=(m, m))
        scale = 10.0 ** rng.uniform(-6.0, 6.0)
        track_covariances.append((A @ A.T + m * np.eye(m)) * scale)

    measurements = rng.normal(size=(n_meas, m)) if n_meas else np.zeros((0, m))

    gate_threshold = None
    if seed % 2 == 1:
        gate_threshold = float(chi2.ppf(0.999, df=m)) * 1.5

    return {
        "track_states": track_states,
        "track_covariances": track_covariances,
        "measurements": measurements,
        "H": H,
        "R": R,
        "detection_prob": 0.9,
        "gate_threshold": gate_threshold,
    }


class TestLikelihoodMatrixEquality:
    """`compute_likelihood_matrix` must match `_reference_likelihood_matrix`
    (the frozen pre-restructure loop) across 20 seeded scenarios.

    Written and passing against the unmodified code path (the live function
    *is* the reference's logic at this point, so the match is exact).
    After the Task 1 restructure (single per-track `cho_factor` + one
    `cho_solve` per measurement, replacing the per-pair double solve and
    per-pair `np.linalg.det`), likelihoods are compared at rtol=1e-9 --
    `cho_solve`'s roundings are not expected to exactly reproduce
    `mahalanobis_distance`'s dispatched solve/inv path -- while `gated` is
    asserted exactly equal (see `_make_likelihood_matrix_scenario` for why
    gate decisions cannot flip: thresholds are deliberately far from any
    scenario's mahal_sq mass).
    """

    @pytest.mark.parametrize("seed", list(range(20)))
    def test_matches_reference(self, seed):
        scenario = _make_likelihood_matrix_scenario(seed)

        expected_likelihood, expected_gated = _reference_likelihood_matrix(
            scenario["track_states"],
            scenario["track_covariances"],
            scenario["measurements"],
            scenario["H"],
            scenario["R"],
            scenario["detection_prob"],
            scenario["gate_threshold"],
        )
        actual_likelihood, actual_gated = compute_likelihood_matrix(
            scenario["track_states"],
            scenario["track_covariances"],
            scenario["measurements"],
            scenario["H"],
            scenario["R"],
            scenario["detection_prob"],
            scenario["gate_threshold"],
        )

        assert_allclose(actual_gated, expected_gated)
        assert_allclose(actual_likelihood, expected_likelihood, rtol=1e-9, atol=1e-12)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
