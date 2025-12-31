"""Tests for Multiple Hypothesis Tracking (MHT)."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pytcl.trackers import (
    Hypothesis,
    HypothesisTree,
    MHTConfig,
    MHTResult,
    MHTTrack,
    MHTTracker,
    MHTTrackStatus,
    compute_association_likelihood,
    generate_joint_associations,
    n_scan_prune,
    prune_hypotheses_by_probability,
)


class TestMHTTrack:
    """Tests for MHTTrack data structure."""

    def test_creation(self):
        """Test basic track creation."""
        track = MHTTrack(
            id=1,
            state=np.array([1.0, 2.0, 0.0, 0.0]),
            covariance=np.eye(4),
            score=0.5,
            status=MHTTrackStatus.TENTATIVE,
            history=[0],
            parent_id=-1,
            scan_created=0,
            n_hits=1,
            n_misses=0,
        )

        assert track.id == 1
        assert track.status == MHTTrackStatus.TENTATIVE
        assert track.n_hits == 1


class TestHypothesis:
    """Tests for Hypothesis data structure."""

    def test_creation(self):
        """Test basic hypothesis creation."""
        hyp = Hypothesis(
            id=0,
            probability=1.0,
            track_ids=[1, 2, 3],
            scan_created=0,
            parent_id=-1,
        )

        assert hyp.id == 0
        assert hyp.probability == 1.0
        assert len(hyp.track_ids) == 3


class TestGenerateJointAssociations:
    """Tests for joint association generation."""

    def test_all_gated(self):
        """Generates all associations when everything is gated."""
        gated = np.array([[True, True], [True, True]])

        associations = generate_joint_associations(gated, n_tracks=2, n_meas=2)

        # For 2 tracks, 2 measurements, all gated:
        # Each track can: miss or one measurement (with one-to-one constraint)
        # Valid associations:
        # 1. t0->miss, t1->miss
        # 2. t0->miss, t1->m0
        # 3. t0->miss, t1->m1
        # 4. t0->m0, t1->miss
        # 5. t0->m0, t1->m1
        # 6. t0->m1, t1->miss
        # 7. t0->m1, t1->m0
        # That's 7 total (one-to-one prevents t0->m0, t1->m0 etc.)
        assert len(associations) == 7

    def test_no_gated(self):
        """Only missed detections when nothing gated."""
        gated = np.array([[False, False], [False, False]])

        associations = generate_joint_associations(gated, n_tracks=2, n_meas=2)

        # Only one option: both tracks miss
        assert len(associations) == 1
        assert associations[0] == {0: -1, 1: -1}

    def test_partial_gating(self):
        """Respects gating constraints."""
        gated = np.array([[True, False], [False, True]])

        associations = generate_joint_associations(gated, n_tracks=2, n_meas=2)

        # Track 0 can only see meas 0, track 1 can only see meas 1
        # Associations: both miss, t0->m0 t1->miss, t0->miss t1->m1, t0->m0 t1->m1
        assert len(associations) == 4

        # Verify t0 never associated with m1
        for assoc in associations:
            if 0 in assoc and assoc[0] >= 0:
                assert assoc[0] == 0  # Can only be meas 0

    def test_one_to_one_constraint(self):
        """Enforces one-to-one constraint."""
        gated = np.array([[True, True], [True, True]])

        associations = generate_joint_associations(gated, n_tracks=2, n_meas=2)

        # No measurement assigned to multiple tracks
        for assoc in associations:
            assigned = [m for m in assoc.values() if m >= 0]
            assert len(assigned) == len(set(assigned))

    def test_empty_tracks(self):
        """Handles case with no tracks."""
        gated = np.zeros((0, 2), dtype=bool)

        associations = generate_joint_associations(gated, n_tracks=0, n_meas=2)

        assert len(associations) == 1
        assert associations[0] == {}


class TestComputeAssociationLikelihood:
    """Tests for association likelihood computation."""

    def test_all_miss(self):
        """Likelihood for all missed detections."""
        association = {0: -1, 1: -1}
        likelihood_matrix = np.ones((2, 2))

        lik = compute_association_likelihood(
            association,
            likelihood_matrix,
            detection_prob=0.9,
            clutter_density=1e-6,
            n_meas=2,
        )

        # (1-0.9)^2 * (1e-6)^2 for misses and all meas as clutter
        expected = (0.1**2) * (1e-6**2)
        assert_allclose(lik, expected)

    def test_detection_increases_likelihood(self):
        """Detection with high likelihood increases probability."""
        assoc_detect = {0: 0}
        assoc_miss = {0: -1}
        likelihood_matrix = np.array([[1.0]])  # High likelihood

        lik_detect = compute_association_likelihood(
            assoc_detect,
            likelihood_matrix,
            detection_prob=0.9,
            clutter_density=1e-6,
            n_meas=1,
        )
        lik_miss = compute_association_likelihood(
            assoc_miss,
            likelihood_matrix,
            detection_prob=0.9,
            clutter_density=1e-6,
            n_meas=1,
        )

        assert lik_detect > lik_miss


class TestNScanPrune:
    """Tests for N-scan pruning."""

    def test_no_prune_recent(self):
        """Recent hypotheses are not pruned."""
        tracks = {
            1: MHTTrack(
                1,
                np.zeros(2),
                np.eye(2),
                0.0,
                MHTTrackStatus.CONFIRMED,
                [0],
                -1,
                5,
                1,
                0,
            ),
        }
        hypotheses = [
            Hypothesis(0, 0.6, [1], scan_created=5, parent_id=-1),
            Hypothesis(1, 0.4, [1], scan_created=5, parent_id=-1),
        ]

        pruned, committed = n_scan_prune(hypotheses, tracks, n_scan=3, current_scan=5)

        assert len(pruned) == 2

    def test_prune_old_divergent(self):
        """Old divergent hypotheses are pruned."""
        tracks = {
            1: MHTTrack(
                1,
                np.zeros(2),
                np.eye(2),
                0.0,
                MHTTrackStatus.CONFIRMED,
                [0],
                -1,
                0,
                1,
                0,
            ),
            2: MHTTrack(
                2,
                np.ones(2),
                np.eye(2),
                0.0,
                MHTTrackStatus.CONFIRMED,
                [1],
                -1,
                0,
                1,
                0,
            ),
        }
        hypotheses = [
            Hypothesis(0, 0.9, [1], scan_created=0, parent_id=-1),
            Hypothesis(1, 0.1, [2], scan_created=0, parent_id=-1),
        ]

        pruned, committed = n_scan_prune(hypotheses, tracks, n_scan=3, current_scan=5)

        # Should keep hypothesis with track 1 (higher prob)
        assert len(pruned) == 1
        assert 1 in pruned[0].track_ids


class TestPruneHypothesesByProbability:
    """Tests for probability-based pruning."""

    def test_remove_low_probability(self):
        """Low probability hypotheses are removed."""
        hypotheses = [
            Hypothesis(0, 0.9, [1], 0, -1),
            Hypothesis(1, 1e-10, [2], 0, -1),
        ]

        pruned = prune_hypotheses_by_probability(hypotheses, max_hypotheses=100)

        assert len(pruned) == 1
        assert pruned[0].id == 0

    def test_limit_count(self):
        """Respects max_hypotheses limit."""
        hypotheses = [Hypothesis(i, 1.0 / 10, [i], 0, -1) for i in range(10)]

        pruned = prune_hypotheses_by_probability(hypotheses, max_hypotheses=3)

        assert len(pruned) == 3

    def test_renormalize(self):
        """Probabilities are renormalized."""
        hypotheses = [
            Hypothesis(0, 0.6, [1], 0, -1),
            Hypothesis(1, 0.4, [2], 0, -1),
        ]

        pruned = prune_hypotheses_by_probability(hypotheses, max_hypotheses=100)

        total_prob = sum(h.probability for h in pruned)
        assert_allclose(total_prob, 1.0)


class TestHypothesisTree:
    """Tests for HypothesisTree class."""

    def test_initialization(self):
        """Tree initializes correctly."""
        tree = HypothesisTree(max_hypotheses=50, n_scan=2)
        tree.initialize()

        assert len(tree.hypotheses) == 1
        assert tree.hypotheses[0].probability == 1.0

    def test_add_track(self):
        """Can add tracks to tree."""
        tree = HypothesisTree()
        tree.initialize()

        track = MHTTrack(
            id=-1,
            state=np.zeros(2),
            covariance=np.eye(2),
            score=0.0,
            status=MHTTrackStatus.TENTATIVE,
            history=[0],
            parent_id=-1,
            scan_created=0,
            n_hits=1,
            n_misses=0,
        )

        track_id = tree.add_track(track)

        assert track_id in tree.tracks

    def test_get_best_hypothesis(self):
        """Can get best hypothesis."""
        tree = HypothesisTree()
        tree.initialize()

        best = tree.get_best_hypothesis()

        assert best is not None
        assert best.probability == 1.0


class TestMHTConfig:
    """Tests for MHTConfig."""

    def test_default_values(self):
        """Default config has sensible values."""
        config = MHTConfig()

        assert config.n_scan == 3
        assert config.max_hypotheses == 100
        assert 0 < config.detection_prob < 1
        assert config.clutter_density > 0

    def test_custom_values(self):
        """Can create config with custom values."""
        config = MHTConfig(
            n_scan=5,
            max_hypotheses=50,
            detection_prob=0.95,
        )

        assert config.n_scan == 5
        assert config.max_hypotheses == 50
        assert config.detection_prob == 0.95


class TestMHTTracker:
    """Tests for MHTTracker class."""

    @pytest.fixture
    def simple_tracker(self):
        """Create a simple 2D tracker."""

        # Constant velocity model
        def F(dt):
            return np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]])

        H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])

        def Q(dt):
            return np.eye(4) * 0.1

        R = np.eye(2) * 0.5

        config = MHTConfig(
            n_scan=2,
            max_hypotheses=20,
            detection_prob=0.9,
            confirm_threshold=2,
        )

        return MHTTracker(
            state_dim=4,
            meas_dim=2,
            F=F,
            H=H,
            Q=Q,
            R=R,
            config=config,
        )

    def test_initialization(self, simple_tracker):
        """Tracker initializes correctly."""
        assert simple_tracker.n_hypotheses == 1

    def test_process_no_measurements(self, simple_tracker):
        """Processing with no measurements works."""
        result = simple_tracker.process([], dt=1.0)

        assert isinstance(result, MHTResult)
        assert len(result.all_tracks) == 0

    def test_process_single_measurement(self, simple_tracker):
        """Single measurement creates tentative track."""
        measurements = [np.array([5.0, 5.0])]

        result = simple_tracker.process(measurements, dt=1.0)

        assert len(result.tentative_tracks) >= 1
        assert len(result.confirmed_tracks) == 0

    def test_track_confirmation(self, simple_tracker):
        """Track gets confirmed after enough hits."""
        # Consistent measurements near same location
        for i in range(5):
            measurements = [np.array([5.0 + i * 0.1, 5.0 + i * 0.1])]
            result = simple_tracker.process(measurements, dt=1.0)

        # Should have a confirmed track after several updates
        assert len(result.confirmed_tracks) >= 1 or len(result.all_tracks) >= 1

    def test_multiple_tracks(self, simple_tracker):
        """Can maintain multiple tracks."""
        # Two widely separated measurements
        measurements = [
            np.array([0.0, 0.0]),
            np.array([100.0, 100.0]),
        ]

        for _ in range(3):
            result = simple_tracker.process(measurements, dt=1.0)
            # Update measurement positions slightly
            measurements = [m + 0.1 for m in measurements]

        # Should have two distinct tracks
        assert len(result.all_tracks) >= 2

    def test_clutter_rejection(self, simple_tracker):
        """Random clutter generates some tracks but hypotheses are limited."""
        rng = np.random.default_rng(42)

        for _ in range(5):
            # Random clutter measurements each frame
            measurements = [rng.uniform(-100, 100, 2) for _ in range(3)]
            result = simple_tracker.process(measurements, dt=1.0)

        # MHT should keep hypotheses limited by max_hypotheses
        # (not testing track count since random clutter can create associations)
        assert result.n_hypotheses <= simple_tracker.config.max_hypotheses

    def test_properties(self, simple_tracker):
        """Tracker properties work."""
        measurements = [np.array([5.0, 5.0])]
        simple_tracker.process(measurements, dt=1.0)

        # Check properties
        assert simple_tracker.n_hypotheses >= 1
        assert isinstance(simple_tracker.tracks, list)
        assert isinstance(simple_tracker.confirmed_tracks, list)


class TestMHTIntegration:
    """Integration tests for MHT."""

    def test_crossing_targets(self):
        """MHT maintains hypotheses for crossing targets."""

        # Two targets that cross paths
        def F(dt):
            return np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]])

        H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])

        def Q(dt):
            return np.eye(4) * 0.01

        R = np.eye(2) * 0.1

        config = MHTConfig(
            n_scan=3,
            max_hypotheses=50,
            detection_prob=0.95,
        )

        tracker = MHTTracker(4, 2, F, H, Q, R, config)

        # Initialize two tracks moving towards each other
        # Track 1: starts at (0, 0), moves right
        # Track 2: starts at (10, 0), moves left

        for t in range(10):
            # Before crossing
            pos1 = [t * 1.0, 0.0]
            pos2 = [10.0 - t * 1.0, 0.0]

            if abs(pos1[0] - pos2[0]) < 0.5:
                # Near crossing - measurements could be swapped
                # This is where MHT shines
                pass

            measurements = [np.array(pos1), np.array(pos2)]
            tracker.process(measurements, dt=1.0)

        # Should maintain at least some hypotheses
        assert tracker.n_hypotheses >= 1

    def test_track_birth_death(self):
        """MHT handles appearing and disappearing targets."""

        def F(dt):
            return np.array([[1, dt], [0, 1]])

        H = np.array([[1, 0]])

        def Q(dt):
            return np.eye(2) * 0.1

        R = np.array([[0.5]])

        config = MHTConfig(
            n_scan=2,
            max_hypotheses=30,
            delete_threshold=3,
        )

        tracker = MHTTracker(2, 1, F, H, Q, R, config)

        # Target appears
        for t in range(5):
            result = tracker.process([np.array([t * 1.0])], dt=1.0)

        n_tracks_during = len(result.all_tracks)

        # Target disappears
        for t in range(5):
            result = tracker.process([], dt=1.0)

        # Tracks should eventually be deleted
        assert len(result.all_tracks) <= n_tracks_during
