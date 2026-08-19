"""Tests for tracker implementations."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pytcl.trackers import (
    MultiTargetTracker,
    SingleTargetTracker,
    Track,
    TrackState,
    TrackStatus,
)


class TestSingleTargetTracker:
    """Tests for SingleTargetTracker."""

    def setup_method(self):
        """Set up test fixtures."""
        # Simple 2D position-velocity model
        self.F = np.array([[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]])
        self.H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
        self.Q = np.eye(4) * 0.1
        self.R = np.eye(2) * 1.0

    def test_initialization(self):
        """Test tracker initialization."""
        tracker = SingleTargetTracker(4, 2, self.F, self.H, self.Q, self.R)

        assert not tracker.is_initialized
        assert tracker.state is None

        tracker.initialize(np.array([0, 1, 0, 1]), np.eye(4))

        assert tracker.is_initialized
        assert tracker.state is not None
        assert_allclose(tracker.state.state, [0, 1, 0, 1])

    def test_predict(self):
        """Test prediction step."""
        tracker = SingleTargetTracker(4, 2, self.F, self.H, self.Q, self.R)
        tracker.initialize(np.array([0, 1, 0, 1]), np.eye(4) * 0.1)

        state = tracker.predict(1.0)

        # State should propagate with constant velocity
        assert_allclose(state.state[:2], [1, 1], atol=0.1)
        assert_allclose(state.state[2:], [1, 1], atol=0.1)

    def test_update(self):
        """Test update step."""
        tracker = SingleTargetTracker(4, 2, self.F, self.H, self.Q, self.R)
        tracker.initialize(np.array([0, 1, 0, 1]), np.eye(4) * 10)

        # Predict then update
        tracker.predict(1.0)
        state, d2 = tracker.update(np.array([1.0, 1.0]))

        # State should be close to measurement
        assert abs(state.state[0] - 1.0) < 1.0
        assert abs(state.state[2] - 1.0) < 1.0

        # Mahalanobis distance should be reasonable
        assert d2 >= 0

    def test_gating(self):
        """Test measurement gating."""
        tracker = SingleTargetTracker(
            4,
            2,
            self.F,
            self.H,
            self.Q,
            self.R,
            gate_threshold=9.21,  # 99% chi2 for 2D
        )
        tracker.initialize(np.array([0, 1, 0, 1]), np.eye(4) * 0.1)
        tracker.predict(1.0)

        # Close measurement should pass gate
        state1, d2_1 = tracker.update(np.array([1.0, 1.0]))
        assert d2_1 < 9.21

        # Distant measurement should fail gate
        tracker.initialize(np.array([0, 1, 0, 1]), np.eye(4) * 0.1)
        tracker.predict(1.0)
        state2, d2_2 = tracker.update(np.array([100.0, 100.0]))
        assert d2_2 > 9.21

    def test_gate_distance_is_the_squared_mahalanobis_distance(self):
        """Pin what ``update``'s second return value actually is.

        Its docstring called it a "likelihood". It is the squared
        Mahalanobis distance -- so smaller means a better match, and a
        caller thresholding it as though larger were better inverts the
        decision. ``MHTTracker`` keeps the two quantities distinct, so the
        naming was against the house convention as well as against the code.
        """
        tracker = SingleTargetTracker(4, 2, self.F, self.H, self.Q, self.R)
        tracker.initialize(np.array([0, 1, 0, 1]), np.eye(4) * 10)
        predicted = tracker.predict(1.0)

        z = np.array([1.4, 0.7])
        S = self.H @ predicted.covariance @ self.H.T + self.R
        innovation = z - self.H @ predicted.state
        expected = float(innovation @ np.linalg.inv(S) @ innovation)

        _, gate_distance = tracker.update(z)
        assert_allclose(gate_distance, expected, rtol=1e-12)

    def test_gate_distance_grows_as_the_measurement_moves_away(self):
        """A distance, not a likelihood: a likelihood would fall here."""
        distances = []
        for offset in (0.0, 0.5, 2.0, 5.0):
            tracker = SingleTargetTracker(4, 2, self.F, self.H, self.Q, self.R)
            tracker.initialize(np.array([0, 1, 0, 1]), np.eye(4) * 10)
            predicted = tracker.predict(1.0)
            z = self.H @ predicted.state + np.array([offset, 0.0])
            distances.append(tracker.update(z)[1])

        assert distances == sorted(distances)
        assert distances[0] == pytest.approx(0.0, abs=1e-12)
        # A Gaussian likelihood is bounded above by its normalisation and
        # decreases with distance; this does neither.
        assert distances[-1] > 1.0

    def test_a_rejected_measurement_leaves_the_state_untouched(self):
        """Documented in Notes, because the return value is the only signal."""
        tracker = SingleTargetTracker(
            4, 2, self.F, self.H, self.Q, self.R, gate_threshold=9.21
        )
        tracker.initialize(np.array([0, 1, 0, 1]), np.eye(4) * 0.1)
        before = tracker.predict(1.0)

        after, gate_distance = tracker.update(np.array([100.0, 100.0]))

        assert gate_distance > 9.21
        assert_allclose(after.state, before.state, rtol=0, atol=0)
        assert_allclose(after.covariance, before.covariance, rtol=0, atol=0)

    def test_callable_dynamics(self):
        """Test with callable F and Q."""

        def F_func(dt):
            return np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]])

        def Q_func(dt):
            return np.eye(4) * dt * 0.1

        tracker = SingleTargetTracker(4, 2, F_func, self.H, Q_func, self.R)
        tracker.initialize(np.array([0, 1, 0, 1]), np.eye(4))

        # Different dt values
        state1 = tracker.predict(0.5)
        tracker.initialize(np.array([0, 1, 0, 1]), np.eye(4))
        state2 = tracker.predict(2.0)

        # Longer prediction should move state further
        assert abs(state2.state[0]) > abs(state1.state[0])


class TestMultiTargetTracker:
    """Tests for MultiTargetTracker."""

    def setup_method(self):
        """Set up test fixtures."""
        # Simple 2D position-velocity model
        self.F = lambda dt: np.array(
            [[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]]
        )
        self.H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
        self.Q = lambda dt: np.eye(4) * 0.1
        self.R = np.eye(2) * 1.0
        self.P0 = np.eye(4) * 10.0

    def test_initialization(self):
        """Test tracker initialization."""
        tracker = MultiTargetTracker(
            4, 2, self.F, self.H, self.Q, self.R, init_covariance=self.P0
        )

        assert len(tracker.tracks) == 0
        assert len(tracker.confirmed_tracks) == 0

    def test_track_initiation(self):
        """Test new track initiation."""
        tracker = MultiTargetTracker(
            4, 2, self.F, self.H, self.Q, self.R, init_covariance=self.P0
        )

        # Process single measurement
        measurements = [np.array([10.0, 20.0])]
        tracks = tracker.process(measurements, dt=1.0)

        # Should have one tentative track
        assert len(tracks) == 1
        assert tracks[0].status == TrackStatus.TENTATIVE

    def test_track_confirmation(self):
        """Test track confirmation after multiple hits."""
        tracker = MultiTargetTracker(
            4,
            2,
            self.F,
            self.H,
            self.Q,
            self.R,
            init_covariance=self.P0,
            confirm_hits=3,
        )

        # Process consistent measurements
        for i in range(5):
            measurements = [np.array([10.0 + i, 20.0 + i])]
            tracker.process(measurements, dt=1.0)

        # Should have one confirmed track
        assert len(tracker.confirmed_tracks) == 1
        assert tracker.confirmed_tracks[0].status == TrackStatus.CONFIRMED

    def test_track_deletion(self):
        """Test track deletion after misses."""
        tracker = MultiTargetTracker(
            4,
            2,
            self.F,
            self.H,
            self.Q,
            self.R,
            init_covariance=self.P0,
            max_misses=3,
        )

        # Create track
        for i in range(3):
            measurements = [np.array([10.0 + i, 20.0 + i])]
            tracker.process(measurements, dt=1.0)

        initial_tracks = len(tracker.tracks)

        # Miss detections
        for _ in range(5):
            tracker.process([], dt=1.0)

        # Track should be deleted
        assert len(tracker.tracks) < initial_tracks

    def test_multiple_targets(self):
        """Test tracking multiple targets."""
        tracker = MultiTargetTracker(
            4,
            2,
            self.F,
            self.H,
            self.Q,
            self.R,
            init_covariance=self.P0,
            confirm_hits=2,
        )

        # Two targets moving in different directions
        for i in range(5):
            measurements = [
                np.array([10.0 + i * 2, 20.0 + i]),  # Target 1
                np.array([50.0 - i * 2, 30.0 + i * 0.5]),  # Target 2
            ]
            tracker.process(measurements, dt=1.0)

        # Should have two confirmed tracks
        assert len(tracker.confirmed_tracks) == 2

    def test_data_association(self):
        """Test correct data association with crossing targets."""
        tracker = MultiTargetTracker(
            4,
            2,
            self.F,
            self.H,
            self.Q,
            self.R,
            init_covariance=self.P0,
            confirm_hits=2,
        )

        # Two targets that cross paths
        for i in range(10):
            t1_x = 10.0 + i * 5  # Moving right
            t2_x = 60.0 - i * 5  # Moving left
            y = 25.0

            measurements = [np.array([t1_x, y]), np.array([t2_x, y])]
            tracker.process(measurements, dt=1.0)

        # Should maintain two separate tracks
        confirmed = tracker.confirmed_tracks
        assert len(confirmed) == 2

        # Tracks should have different positions
        positions = [t.state[0] for t in confirmed]
        assert abs(positions[0] - positions[1]) > 10  # Separated

    def test_false_alarm_rejection(self):
        """Test that isolated false alarms don't become confirmed tracks."""
        tracker = MultiTargetTracker(
            4,
            2,
            self.F,
            self.H,
            self.Q,
            self.R,
            init_covariance=self.P0,
            confirm_hits=3,
        )

        # One consistent target + random false alarms
        np.random.seed(42)
        for i in range(10):
            measurements = [np.array([10.0 + i, 20.0 + i])]  # Real target

            # Add random false alarm occasionally
            if i % 3 == 0:
                fa = np.array([np.random.uniform(50, 100), np.random.uniform(50, 100)])
                measurements.append(fa)

            tracker.process(measurements, dt=1.0)

        # Should have only one confirmed track (the real target)
        assert len(tracker.confirmed_tracks) == 1


class TestMofNConfirmation:
    """``confirm_window`` must actually affect confirmation.

    It was accepted, stored, and never read: confirmation compared a
    cumulative lifetime ``hits`` count against ``confirm_hits``, so a track
    that scraped together enough detections over any span eventually
    confirmed, no matter how sparse. The class docstring, the parameter
    docstring and ``MultiTargetConfig`` all said M-of-N. Passing a different
    ``confirm_window`` changed nothing, which is the property these tests
    pin.
    """

    F = np.array([[1.0, 1.0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 1.0], [0, 0, 0, 1.0]])
    H = np.array([[1.0, 0, 0, 0], [0, 0, 1.0, 0]])
    Q = np.eye(4) * 0.01
    R = np.eye(2) * 0.1

    def _run(self, window, n_scans, skip=()):
        tracker = MultiTargetTracker(
            4,
            2,
            self.F,
            self.H,
            self.Q,
            self.R,
            confirm_hits=3,
            confirm_window=window,
            max_misses=99,
        )
        for k in range(n_scans):
            detections = [] if k in skip else [np.array([float(k), float(k)])]
            tracker.process(detections, 1.0)
        return tracker.tracks

    def test_dense_hits_confirm(self):
        tracks = self._run(window=5, n_scans=3)
        assert [t.status for t in tracks] == [TrackStatus.CONFIRMED]

    def test_the_initiating_detection_counts_toward_confirmation(self):
        """confirm_hits=3 must mean three detections, not four.

        The initiating detection is already counted in ``hits``; omitting it
        from the window would silently require confirm_hits + 1.
        """
        tracks = self._run(window=5, n_scans=3)
        assert tracks[0].hits == 3
        assert tracks[0].status == TrackStatus.CONFIRMED

    def test_sparse_hits_do_not_confirm_within_a_short_window(self):
        """Three hits at scans 0, 3, 6 -- only two fall inside a 5-scan window."""
        tracks = self._run(window=5, n_scans=7, skip={1, 2, 4, 5})
        assert tracks[0].hits == 3
        assert tracks[0].status == TrackStatus.TENTATIVE

    def test_the_same_pattern_confirms_with_a_long_enough_window(self):
        """The direct proof that confirm_window is read at all.

        Identical detections, identical confirm_hits, different window,
        different outcome. Under the old cumulative rule both confirmed.
        """
        tracks = self._run(window=99, n_scans=7, skip={1, 2, 4, 5})
        assert tracks[0].hits == 3
        assert tracks[0].status == TrackStatus.CONFIRMED

    def test_hits_remains_a_cumulative_lifetime_count(self):
        """``Track.hits`` is documented as the number of updates; unchanged."""
        tracks = self._run(window=2, n_scans=6)
        assert tracks[0].hits == 6


class TestTrackState:
    """Tests for TrackState named tuple."""

    def test_track_state_creation(self):
        """Test creating a TrackState."""
        state = np.array([1.0, 2.0])
        cov = np.eye(2)
        time = 1.5

        ts = TrackState(state=state, covariance=cov, time=time)

        assert_allclose(ts.state, state)
        assert_allclose(ts.covariance, cov)
        assert ts.time == time


class TestTrack:
    """Tests for Track named tuple."""

    def test_track_creation(self):
        """Test creating a Track."""
        track = Track(
            id=1,
            state=np.array([1.0, 2.0, 3.0, 4.0]),
            covariance=np.eye(4),
            status=TrackStatus.CONFIRMED,
            hits=5,
            misses=0,
            time=10.0,
        )

        assert track.id == 1
        assert track.status == TrackStatus.CONFIRMED
        assert track.hits == 5
        assert track.misses == 0


class TestIntegration:
    """Integration tests for complete tracking scenarios."""

    def test_complete_tracking_scenario(self):
        """Test a complete tracking scenario with track lifecycle."""

        # Set up tracker
        def F(dt):
            return np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]])

        H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])

        def Q(dt):
            return np.eye(4) * 0.1

        R = np.eye(2) * 0.5
        P0 = np.eye(4) * 10.0

        tracker = MultiTargetTracker(
            4, 2, F, H, Q, R, init_covariance=P0, confirm_hits=3, max_misses=3
        )

        # Phase 1: Target appears
        for i in range(5):
            tracker.process([np.array([i * 2.0, i * 1.0])], dt=1.0)

        assert len(tracker.confirmed_tracks) == 1, "Target should be confirmed"

        # Phase 2: Target continues with some noise
        for i in range(5, 15):
            noise = np.random.randn(2) * 0.1
            tracker.process([np.array([i * 2.0, i * 1.0]) + noise], dt=1.0)

        assert len(tracker.confirmed_tracks) == 1, "Target should still be tracked"

        # Phase 3: Target disappears
        for _ in range(5):
            tracker.process([], dt=1.0)

        assert len(tracker.confirmed_tracks) == 0, (
            "Target should be deleted after misses"
        )


class TestPerDetectionMeasurementCovariance:
    """Per-detection measurement covariance in gating and update.

    Both trackers took a single fixed R. That is wrong for any sensor whose
    error varies between detections, and converted polar measurements are the
    common case: the Cartesian covariance is ``J R_polar J^T``, anisotropic and
    growing with range, so no scalar R describes it. Forcing one makes the gate
    too tight at long range -- true detections fall outside it and the tracker
    starts duplicate tracks -- or too loose at short range.
    """

    @staticmethod
    def _model():
        F = np.array(
            [
                [1.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
        Q = np.eye(4) * 0.01
        R = np.eye(2) * 4.0
        return F, H, Q, R

    def test_uniform_per_detection_matches_the_fixed_path(self):
        """Passing the tracker's own R for every detection changes nothing.

        The two code paths must agree exactly, or the fast path and the
        per-detection path have diverged.
        """
        F, H, Q, R = self._model()
        detections = [
            [np.array([1.0, 2.0]), np.array([40.0, 41.0])],
            [np.array([2.0, 3.0]), np.array([41.0, 42.0])],
            [np.array([3.0, 4.0]), np.array([42.0, 43.0])],
            [np.array([4.0, 5.0]), np.array([43.0, 44.0])],
        ]

        fixed = MultiTargetTracker(4, 2, F, H, Q, R)
        supplied = MultiTargetTracker(4, 2, F, H, Q, R)
        for scan in detections:
            fixed.process(scan, dt=1.0)
            supplied.process(scan, dt=1.0, measurement_covariances=[R] * len(scan))

        assert len(fixed.tracks) == len(supplied.tracks)
        for a, b in zip(fixed.tracks, supplied.tracks):
            assert a.id == b.id
            assert a.status is b.status
            assert_allclose(a.state, b.state, rtol=1e-12, atol=1e-12)
            assert_allclose(a.covariance, b.covariance, rtol=1e-12, atol=1e-12)

    def test_a_wide_covariance_widens_the_gate(self):
        """A detection declared uncertain must be reachable when a tight one is not.

        Same geometry, same track, same offset -- only the declared covariance
        differs. With a tight covariance the detection is outside the 99% gate
        and starts its own track; with a wide one it is associated instead.
        """
        F, H, Q, R = self._model()
        # Taken from the measured gate boundary rather than guessed: with the
        # track covariance settled at P_xx ~ 3.7, a 30-unit offset lies outside
        # the 99% gate for R = 4 and inside it for R = 400.
        offset = np.array([30.0, 0.0])

        def run(covariance):
            tracker = MultiTargetTracker(4, 2, F, H, Q, R, confirm_hits=2, max_misses=5)
            for _ in range(3):
                tracker.process([np.array([0.0, 0.0])], dt=1.0)
            n_before = len(tracker.tracks)
            tracker.process(
                [offset],
                dt=1.0,
                measurement_covariances=None if covariance is None else [covariance],
            )
            return n_before, len(tracker.tracks)

        before_tight, after_tight = run(None)
        before_wide, after_wide = run(np.eye(2) * 400.0)

        assert before_tight == before_wide == 1
        assert after_tight == 2, "a tight covariance should reject the offset detection"
        assert after_wide == 1, "a wide covariance should admit it into the track"

    def test_length_mismatch_is_rejected(self):
        F, H, Q, R = self._model()
        tracker = MultiTargetTracker(4, 2, F, H, Q, R)
        with pytest.raises(ValueError, match="2 entries for 1 measurements"):
            tracker.process(
                [np.array([0.0, 0.0])], dt=1.0, measurement_covariances=[R, R]
            )

    def test_wrong_shape_is_rejected(self):
        F, H, Q, R = self._model()
        tracker = MultiTargetTracker(4, 2, F, H, Q, R)
        with pytest.raises(ValueError, match=r"shape \(3, 3\), expected \(2, 2\)"):
            tracker.process(
                [np.array([0.0, 0.0])], dt=1.0, measurement_covariances=[np.eye(3)]
            )

    def test_single_target_update_uses_the_supplied_covariance(self):
        """A wider declared covariance must move the state less."""
        F, H, Q, R = self._model()
        z = np.array([10.0, 0.0])

        def run(covariance):
            tracker = SingleTargetTracker(4, 2, F, H, Q, R)
            tracker.initialize(np.zeros(4), np.eye(4) * 10.0)
            tracker.predict(dt=1.0)
            state, _ = tracker.update(z, measurement_covariance=covariance)
            return state.state[0]

        tight = run(np.eye(2) * 1.0)
        default = run(None)
        wide = run(np.eye(2) * 1000.0)

        assert tight > default > wide, (
            f"gain should shrink as the declared covariance grows: "
            f"{tight:.3f}, {default:.3f}, {wide:.3f}"
        )

    def test_single_target_rejects_a_wrong_shape(self):
        F, H, Q, R = self._model()
        tracker = SingleTargetTracker(4, 2, F, H, Q, R)
        tracker.initialize(np.zeros(4), np.eye(4))
        with pytest.raises(ValueError, match=r"shape \(3, 3\), expected \(2, 2\)"):
            tracker.update(np.array([1.0, 1.0]), measurement_covariance=np.eye(3))

    def test_predict_measurement_accounts_for_the_covariance(self):
        F, H, Q, R = self._model()
        tracker = SingleTargetTracker(4, 2, F, H, Q, R)
        tracker.initialize(np.zeros(4), np.eye(4) * 5.0)

        _, S_default = tracker.predict_measurement()
        _, S_wide = tracker.predict_measurement(np.eye(2) * 100.0)

        assert np.trace(S_wide) > np.trace(S_default)
        assert_allclose(S_wide - S_default, np.eye(2) * 96.0, rtol=1e-12)
