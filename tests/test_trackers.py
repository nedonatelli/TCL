"""Tests for tracker implementations."""

import numpy as np
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
            4, 2, self.F, self.H, self.Q, self.R, gate_threshold=9.21  # 99% chi2 for 2D
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

        assert (
            len(tracker.confirmed_tracks) == 0
        ), "Target should be deleted after misses"
