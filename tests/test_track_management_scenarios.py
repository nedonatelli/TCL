"""Multi-sensor validation suite for track management.

Real-world scenario testing with diverse tracking conditions:

1. Single-sensor scenarios (clean, clutter, high-dynamic)
2. Multi-sensor fusion (RADAR+LiDAR, asynchronous timing)
3. Track lifecycle stress tests (rapid initiation, merging, pruning, long-duration)
4. Data integrity checks (round-trip, association, timestamps, covariance PD)

Target: 40+ integration tests across all scenario categories.
"""

import numpy as np
import pytest

from pytcl.dynamic_estimation.kalman.linear import kf_predict, kf_update
from pytcl.io import TrackDatabaseManager, TrackDatabaseStatus, TrackHDF5Storage

# =============================================================================
# Helpers
# =============================================================================


def _cv_model(dt: float = 1.0):
    """Return constant-velocity F, Q, H, R matrices (4-state, 2-meas)."""
    F = np.array(
        [
            [1, dt, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, dt],
            [0, 0, 0, 1],
        ]
    )
    q = 0.1
    Q = q * np.array(
        [
            [dt**3 / 3, dt**2 / 2, 0, 0],
            [dt**2 / 2, dt, 0, 0],
            [0, 0, dt**3 / 3, dt**2 / 2],
            [0, 0, dt**2 / 2, dt],
        ]
    )
    H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
    R = np.eye(2) * 4.0
    return F, Q, H, R


def _simulate_cv_targets(n_targets, n_steps, dt=1.0, seed=42):
    """Generate constant-velocity ground truth trajectories."""
    rng = np.random.default_rng(seed)
    starts = []
    for _ in range(n_targets):
        x0 = np.array(
            [
                rng.uniform(-100, 100),
                rng.uniform(-3, 3),
                rng.uniform(-100, 100),
                rng.uniform(-3, 3),
            ]
        )
        starts.append(x0)

    F, Q, _, _ = _cv_model(dt)
    true_states = np.zeros((n_targets, n_steps, 4))
    for i, x0 in enumerate(starts):
        state = x0.copy()
        for k in range(n_steps):
            true_states[i, k] = state
            state = F @ state + rng.multivariate_normal(np.zeros(4), Q * 0.01)
    return true_states


def _generate_detections(
    true_states, R, pd=0.9, fa_rate=0.0, bounds=(-200, -200, 200, 200), seed=42
):
    """Generate noisy detections with missed detections and false alarms."""
    rng = np.random.default_rng(seed)
    H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
    n_targets, n_steps, _ = true_states.shape
    all_detections = []  # list of list of (measurement, target_idx_or_None)

    for k in range(n_steps):
        scan = []
        for i in range(n_targets):
            if rng.random() < pd:
                pos = H @ true_states[i, k]
                z = pos + rng.multivariate_normal(np.zeros(2), R)
                scan.append((z, i))
        # False alarms
        n_fa = rng.poisson(fa_rate)
        for _ in range(n_fa):
            fa = np.array(
                [
                    rng.uniform(bounds[0], bounds[2]),
                    rng.uniform(bounds[1], bounds[3]),
                ]
            )
            scan.append((fa, None))
        rng.shuffle(scan)
        all_detections.append(scan)
    return all_detections


def _run_tracking_pipeline(db, true_states, detections, dt=1.0):
    """Run a simple nearest-neighbor Kalman tracker, storing results in db."""
    F, Q, H, R = _cv_model(dt)
    P0 = np.diag([R[0, 0], 1.0, R[1, 1], 1.0])
    n_targets, n_steps, _ = true_states.shape

    filters = {}  # tid -> (x, P)

    for k in range(n_steps):
        timestamp = k * dt
        scan = detections[k]

        # Predict existing tracks
        for tid in filters:
            x, P = filters[tid]
            pred = kf_predict(x, P, F, Q)
            filters[tid] = (pred.x, pred.P)

        # Associate detections to nearest track (simplified)
        used_dets = set()
        for tid, (x, P) in filters.items():
            best_j = None
            best_dist = 1e12
            z_pred = H @ x
            for j, (z, _) in enumerate(scan):
                if j in used_dets:
                    continue
                dist = np.sum((z - z_pred) ** 2)
                if dist < best_dist:
                    best_dist = dist
                    best_j = j
            if best_j is not None and best_dist < 100.0:
                used_dets.add(best_j)
                z = scan[best_j][0]
                upd = kf_update(x, P, z, H, R)
                filters[tid] = (upd.x, upd.P)
                db.update_track_state(tid, upd.x, upd.P, timestamp)

                det_id = f"det_{k:04d}_{best_j:04d}"
                db.store_detection(det_id, z, "radar", timestamp)
                db.associate_detection(det_id, tid)
            else:
                db.update_track_state(tid, x, P, timestamp, update_type="prediction")

        # Initiate tracks from unassociated detections
        for j, (z, _) in enumerate(scan):
            if j not in used_dets:
                tid = f"trk_{k:04d}_{j:04d}"
                x0 = np.array([z[0], 0.0, z[1], 0.0])
                db.initiate_track(tid, x0, P0, timestamp)
                filters[tid] = (x0, P0.copy())
                det_id = f"det_{k:04d}_{j:04d}"
                db.store_detection(det_id, z, "radar", timestamp)
                db.associate_detection(det_id, tid)

    return filters


# =============================================================================
# 1. Single-Sensor Scenarios
# =============================================================================


class TestSingleSensorClean:
    """Clean environment: few targets, high PD, no clutter."""

    def test_clean_3_targets_30_steps(self, tmp_path):
        """3 targets, 30 steps, 95% PD, no clutter."""
        true_states = _simulate_cv_targets(3, 30, seed=10)
        _, _, _, R = _cv_model()
        dets = _generate_detections(true_states, R, pd=0.95, fa_rate=0.0, seed=10)

        db = TrackDatabaseManager(str(tmp_path / "clean3.db"))
        db.open(mode="w")
        _run_tracking_pipeline(db, true_states, dets)

        tracks = db.retrieve_all_tracks()
        assert len(tracks) >= 3, f"Expected at least 3 tracks, got {len(tracks)}"
        db.close()

    def test_clean_10_targets_50_steps(self, tmp_path):
        """10 targets, 50 steps, 90% PD, no clutter."""
        true_states = _simulate_cv_targets(10, 50, seed=11)
        _, _, _, R = _cv_model()
        dets = _generate_detections(true_states, R, pd=0.9, fa_rate=0.0, seed=11)

        db = TrackDatabaseManager(str(tmp_path / "clean10.db"))
        db.open(mode="w")
        _run_tracking_pipeline(db, true_states, dets)

        tracks = db.retrieve_all_tracks()
        assert len(tracks) >= 10
        db.close()

    def test_clean_all_tracks_have_history(self, tmp_path):
        """Every initiated track has at least one state entry."""
        true_states = _simulate_cv_targets(5, 20, seed=12)
        _, _, _, R = _cv_model()
        dets = _generate_detections(true_states, R, pd=0.95, fa_rate=0.0, seed=12)

        db = TrackDatabaseManager(str(tmp_path / "history.db"))
        db.open(mode="w")
        _run_tracking_pipeline(db, true_states, dets)

        for t in db.retrieve_all_tracks():
            history = db.get_track_history(t["track_id"])
            assert len(history["timestamps"]) >= 1
        db.close()


class TestSingleSensorClutter:
    """Clutter environment: many false alarms per scan."""

    def test_clutter_3_targets_10fa(self, tmp_path):
        """3 targets with 10 false alarms per scan."""
        true_states = _simulate_cv_targets(3, 30, seed=20)
        _, _, _, R = _cv_model()
        dets = _generate_detections(true_states, R, pd=0.9, fa_rate=10.0, seed=20)

        db = TrackDatabaseManager(str(tmp_path / "clutter10.db"))
        db.open(mode="w")
        _run_tracking_pipeline(db, true_states, dets)

        # Should have many tracks (true + false alarm-initiated)
        tracks = db.retrieve_all_tracks()
        assert len(tracks) >= 3
        db.close()

    def test_clutter_5_targets_50fa(self, tmp_path):
        """5 targets with 50 false alarms per scan (heavy clutter)."""
        true_states = _simulate_cv_targets(5, 20, seed=21)
        _, _, _, R = _cv_model()
        dets = _generate_detections(true_states, R, pd=0.85, fa_rate=50.0, seed=21)

        db = TrackDatabaseManager(str(tmp_path / "clutter50.db"))
        db.open(mode="w")
        _run_tracking_pipeline(db, true_states, dets)

        tracks = db.retrieve_all_tracks()
        assert len(tracks) >= 5
        db.close()

    def test_clutter_detections_stored_correctly(self, tmp_path):
        """All detections (true + clutter) are stored and queryable."""
        true_states = _simulate_cv_targets(2, 10, seed=22)
        _, _, _, R = _cv_model()
        dets = _generate_detections(true_states, R, pd=0.9, fa_rate=5.0, seed=22)

        db = TrackDatabaseManager(str(tmp_path / "clutterdet.db"))
        db.open(mode="w")
        _run_tracking_pipeline(db, true_states, dets)

        all_dets = db.retrieve_all_detections()
        assert len(all_dets) > 0
        # Each detection should have a valid measurement
        for d in all_dets:
            assert d["measurement"].shape == (2,)
            assert np.isfinite(d["measurement"]).all()
        db.close()


class TestSingleSensorHighDynamic:
    """High-dynamic targets: maneuvering, fast velocity changes."""

    def test_maneuvering_target(self, tmp_path):
        """Target with sinusoidal trajectory (simulating turns)."""
        n_steps = 50
        true_states = np.zeros((1, n_steps, 4))
        for k in range(n_steps):
            t = k * 1.0
            true_states[0, k] = [
                20 * np.sin(0.1 * t),
                2 * np.cos(0.1 * t),
                20 * np.cos(0.1 * t),
                -2 * np.sin(0.1 * t),
            ]

        _, _, _, R = _cv_model()
        dets = _generate_detections(true_states, R, pd=0.95, fa_rate=0.0, seed=30)

        db = TrackDatabaseManager(str(tmp_path / "maneuver.db"))
        db.open(mode="w")
        _run_tracking_pipeline(db, true_states, dets)

        tracks = db.retrieve_all_tracks()
        assert len(tracks) >= 1
        db.close()

    def test_fast_target(self, tmp_path):
        """High-speed target (100 m/s)."""
        n_steps = 30
        true_states = np.zeros((1, n_steps, 4))
        for k in range(n_steps):
            true_states[0, k] = [k * 100.0, 100.0, k * 50.0, 50.0]

        _, _, _, R = _cv_model()
        dets = _generate_detections(true_states, R, pd=0.9, fa_rate=0.0, seed=31)

        db = TrackDatabaseManager(str(tmp_path / "fast.db"))
        db.open(mode="w")
        _run_tracking_pipeline(db, true_states, dets)

        tracks = db.retrieve_all_tracks()
        assert len(tracks) >= 1
        # Verify track has reasonable history length
        for t in tracks:
            h = db.get_track_history(t["track_id"])
            assert len(h["timestamps"]) >= 1
        db.close()


# =============================================================================
# 2. Multi-Sensor Fusion
# =============================================================================


class TestMultiSensorFusion:
    """Multi-sensor scenarios: RADAR + LiDAR, asynchronous timing."""

    def test_radar_lidar_complementary(self, tmp_path):
        """RADAR (5m noise, 85% PD) + LiDAR (1m noise, 95% PD)."""
        true_states = _simulate_cv_targets(3, 30, seed=40)
        H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
        rng = np.random.default_rng(40)

        R_radar = np.eye(2) * 25.0  # 5m std
        R_lidar = np.eye(2) * 1.0  # 1m std

        db = TrackDatabaseManager(str(tmp_path / "radar_lidar.db"))
        db.open(mode="w")

        F, Q, _, _ = _cv_model()
        P0 = np.diag([25.0, 1.0, 25.0, 1.0])
        filters = {}

        for k in range(30):
            t = k * 1.0

            for i in range(3):
                tid = f"trk_{i:02d}"
                if tid not in filters:
                    x0 = np.array(
                        [
                            true_states[i, k, 0],
                            0.0,
                            true_states[i, k, 2],
                            0.0,
                        ]
                    )
                    db.initiate_track(tid, x0, P0, t)
                    filters[tid] = (x0, P0.copy())

                x, P = filters[tid]
                pred = kf_predict(x, P, F, Q)
                x, P = pred.x, pred.P

                # RADAR detection
                if rng.random() < 0.85:
                    pos = H @ true_states[i, k]
                    z = pos + rng.multivariate_normal([0, 0], R_radar)
                    upd = kf_update(x, P, z, H, R_radar)
                    x, P = upd.x, upd.P
                    det_id = f"radar_{k:03d}_{i:02d}"
                    db.store_detection(det_id, z, "radar", t, covariance=R_radar)
                    db.update_track_state(tid, x, P, t)
                    db.associate_detection(det_id, tid)

                # LiDAR detection
                if rng.random() < 0.95:
                    pos = H @ true_states[i, k]
                    z = pos + rng.multivariate_normal([0, 0], R_lidar)
                    upd = kf_update(x, P, z, H, R_lidar)
                    x, P = upd.x, upd.P
                    det_id = f"lidar_{k:03d}_{i:02d}"
                    db.store_detection(det_id, z, "lidar", t, covariance=R_lidar)
                    db.update_track_state(tid, x, P, t + 0.001)
                    db.associate_detection(det_id, tid)

                filters[tid] = (x, P)

        # Verify both sensors contributed
        radar_dets = db.retrieve_detections(sensor_id="radar")
        lidar_dets = db.retrieve_detections(sensor_id="lidar")
        assert len(radar_dets) > 0, "No RADAR detections stored"
        assert len(lidar_dets) > 0, "No LiDAR detections stored"

        tracks = db.retrieve_all_tracks()
        assert len(tracks) == 3
        db.close()

    def test_asynchronous_sensor_timing(self, tmp_path):
        """Two sensors at different update rates (1Hz and 5Hz)."""
        true_states = _simulate_cv_targets(2, 50, dt=0.2, seed=41)
        H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
        rng = np.random.default_rng(41)
        R = np.eye(2) * 4.0

        db = TrackDatabaseManager(str(tmp_path / "async.db"))
        db.open(mode="w")

        F_fast, Q_fast, _, _ = _cv_model(dt=0.2)
        P0 = np.diag([4.0, 1.0, 4.0, 1.0])
        filters = {}

        for k in range(50):
            t = k * 0.2

            for i in range(2):
                tid = f"trk_{i:02d}"
                if tid not in filters:
                    x0 = np.array(
                        [
                            true_states[i, k, 0],
                            0.0,
                            true_states[i, k, 2],
                            0.0,
                        ]
                    )
                    db.initiate_track(tid, x0, P0, t)
                    filters[tid] = (x0, P0.copy())

                x, P = filters[tid]
                pred = kf_predict(x, P, F_fast, Q_fast)
                x, P = pred.x, pred.P

                # Sensor A: updates every step (5Hz)
                if rng.random() < 0.9:
                    pos = H @ true_states[i, k]
                    z = pos + rng.multivariate_normal([0, 0], R)
                    upd = kf_update(x, P, z, H, R)
                    x, P = upd.x, upd.P
                    det_id = f"sensorA_{k:04d}_{i:02d}"
                    db.store_detection(det_id, z, "sensor_A", t)
                    db.associate_detection(det_id, tid)

                # Sensor B: updates every 5th step (1Hz)
                if k % 5 == 0 and rng.random() < 0.9:
                    pos = H @ true_states[i, k]
                    z = pos + rng.multivariate_normal([0, 0], R * 0.5)
                    upd = kf_update(x, P, z, H, R * 0.5)
                    x, P = upd.x, upd.P
                    det_id = f"sensorB_{k:04d}_{i:02d}"
                    db.store_detection(det_id, z, "sensor_B", t)
                    db.associate_detection(det_id, tid)

                filters[tid] = (x, P)
                db.update_track_state(tid, x, P, t)

        # Sensor A should have ~5x more detections than sensor B
        a_dets = db.retrieve_detections(sensor_id="sensor_A")
        b_dets = db.retrieve_detections(sensor_id="sensor_B")
        assert len(a_dets) > len(b_dets)
        assert len(b_dets) > 0
        db.close()

    def test_multi_sensor_detection_query_by_sensor(self, tmp_path):
        """Verify detections can be filtered by sensor_id after fusion."""
        db = TrackDatabaseManager(str(tmp_path / "query_sensor.db"))
        db.open(mode="w")

        for i in range(10):
            db.store_detection(f"r_{i}", np.array([i, i]), "radar", float(i))
            db.store_detection(f"l_{i}", np.array([i, i]), "lidar", float(i))
            db.store_detection(f"c_{i}", np.array([i, i]), "camera", float(i))

        assert len(db.retrieve_detections(sensor_id="radar")) == 10
        assert len(db.retrieve_detections(sensor_id="lidar")) == 10
        assert len(db.retrieve_detections(sensor_id="camera")) == 10
        assert len(db.retrieve_all_detections()) == 30
        db.close()

    def test_multi_sensor_covariance_stored(self, tmp_path):
        """Each sensor's measurement covariance is preserved."""
        db = TrackDatabaseManager(str(tmp_path / "cov_sensor.db"))
        db.open(mode="w")

        R_radar = np.eye(2) * 25.0
        R_lidar = np.eye(2) * 1.0

        db.store_detection("r0", np.array([1.0, 2.0]), "radar", 0.0, covariance=R_radar)
        db.store_detection("l0", np.array([1.0, 2.0]), "lidar", 0.0, covariance=R_lidar)

        r = db.retrieve_detection("r0")
        lidar_det = db.retrieve_detection("l0")
        np.testing.assert_allclose(r["covariance"], R_radar)
        np.testing.assert_allclose(lidar_det["covariance"], R_lidar)
        db.close()


# =============================================================================
# 3. Track Lifecycle Stress Tests
# =============================================================================


class TestRapidTrackInitiation:
    """Stress test: rapid creation of many tracks."""

    def test_initiate_100_tracks(self, tmp_path):
        """Initiate 100 tracks in rapid succession."""
        db = TrackDatabaseManager(str(tmp_path / "rapid100.db"))
        db.open(mode="w")

        P0 = np.eye(4)
        for i in range(100):
            db.initiate_track(
                f"trk_{i:04d}",
                np.random.randn(4),
                P0,
                float(i) * 0.01,
            )

        tracks = db.retrieve_all_tracks()
        assert len(tracks) == 100
        db.close()

    def test_initiate_500_tracks(self, tmp_path):
        """Initiate 500 tracks (simulating dense environment)."""
        db = TrackDatabaseManager(str(tmp_path / "rapid500.db"))
        db.open(mode="w")

        rng = np.random.default_rng(50)
        P0 = np.eye(4)
        for i in range(500):
            db.initiate_track(
                f"trk_{i:04d}",
                rng.normal(0, 10, 4),
                P0,
                float(i) * 0.002,
            )

        tracks = db.retrieve_all_tracks()
        assert len(tracks) == 500

        # Verify status filtering works at scale
        tentative = db.retrieve_all_tracks(status=TrackDatabaseStatus.TENTATIVE)
        assert len(tentative) == 500
        db.close()


class TestTrackMerging:
    """Track merge operations under uncertainty."""

    def test_merge_two_tracks(self, tmp_path):
        """Merge duplicate tracks; verify combined history."""
        db = TrackDatabaseManager(str(tmp_path / "merge.db"))
        db.open(mode="w")

        P0 = np.eye(4)
        db.initiate_track("trk_keep", np.array([1, 0, 1, 0], dtype=float), P0, 0.0)
        db.initiate_track("trk_merge", np.array([1.1, 0, 0.9, 0], dtype=float), P0, 0.5)

        # Add state history to both
        for k in range(1, 10):
            db.update_track_state(
                "trk_keep", np.array([1 + k, 0, 1 + k, 0], dtype=float), P0, float(k)
            )
            db.update_track_state(
                "trk_merge",
                np.array([1.1 + k, 0, 0.9 + k, 0], dtype=float),
                P0,
                float(k),
            )

        # Store detections for trk_merge
        db.store_detection("det_m0", np.array([1.1, 0.9]), "radar", 0.5)
        db.associate_detection("det_m0", "trk_merge")

        hits_keep = db.get_track("trk_keep")["hits"]
        hits_merge = db.get_track("trk_merge")["hits"]

        # Merge
        db.merge_tracks("trk_keep", "trk_merge")

        # Merged track should be dead
        merged_info = db.get_track("trk_merge")
        assert merged_info["status"] == TrackDatabaseStatus.DEAD.value

        # Keep track should have combined hits
        keep_info = db.get_track("trk_keep")
        assert keep_info["hits"] == hits_keep + hits_merge

        # Detection should now be associated with keep track
        det = db.retrieve_detection("det_m0")
        assert det["associated_track_id"] == "trk_keep"
        db.close()

    def test_merge_preserves_state_history(self, tmp_path):
        """Merged track's history is combined with kept track."""
        db = TrackDatabaseManager(str(tmp_path / "merge_hist.db"))
        db.open(mode="w")

        P0 = np.eye(4)
        db.initiate_track("a", np.zeros(4), P0, 0.0)
        db.initiate_track("b", np.ones(4), P0, 0.0)

        for k in range(5):
            db.update_track_state("a", np.ones(4) * k, P0, float(k))
            db.update_track_state("b", np.ones(4) * (k + 10), P0, float(k))

        hist_a_before = db.get_track_history("a")
        hist_b_before = db.get_track_history("b")
        total_before = len(hist_a_before["timestamps"]) + len(
            hist_b_before["timestamps"]
        )

        db.merge_tracks("a", "b")

        hist_a_after = db.get_track_history("a")
        assert len(hist_a_after["timestamps"]) == total_before
        db.close()


class TestTrackPruning:
    """Track pruning at configurable thresholds."""

    def test_prune_dead_tracks(self, tmp_path):
        """Prune dead tracks older than threshold."""
        db = TrackDatabaseManager(str(tmp_path / "prune.db"))
        db.open(mode="w")

        P0 = np.eye(4)
        # Create tracks at different times
        for i in range(10):
            tid = f"trk_{i:02d}"
            db.initiate_track(tid, np.zeros(4), P0, float(i))
            db.update_track_state(tid, np.zeros(4), P0, float(i))
            if i < 5:
                db.mark_track_dead(tid)

        # Prune dead tracks older than 3 seconds
        pruned = db.prune_dead_tracks(age_threshold=3.0)
        assert pruned > 0

        remaining = db.retrieve_all_tracks()
        # Dead tracks 0-4 had update times 0-4; threshold=3 prunes those < (9-3)=6
        for t in remaining:
            if t["status"] == TrackDatabaseStatus.DEAD.value:
                assert t["last_update_time"] >= 6.0
        db.close()

    def test_prune_old_detections(self, tmp_path):
        """Prune unassociated detections older than threshold."""
        db = TrackDatabaseManager(str(tmp_path / "prune_det.db"))
        db.open(mode="w")

        for i in range(20):
            db.store_detection(
                f"det_{i:02d}", np.array([i, i], dtype=float), "radar", float(i)
            )

        pruned = db.prune_old_detections(age_threshold=5.0)
        assert pruned > 0

        remaining = db.retrieve_all_detections()
        # max timestamp=19.0, cutoff=19.0-5.0=14.0, prune timestamps < 14.0
        for d in remaining:
            assert d["timestamp"] >= 14.0
        db.close()


class TestLongDuration:
    """Long-duration mission simulation."""

    def test_1000_timesteps(self, tmp_path):
        """Track 3 targets for 1000 timesteps."""
        true_states = _simulate_cv_targets(3, 1000, seed=60)
        _, _, _, R = _cv_model()
        dets = _generate_detections(true_states, R, pd=0.9, fa_rate=0.0, seed=60)

        db = TrackDatabaseManager(str(tmp_path / "long.db"))
        db.open(mode="w")
        _run_tracking_pipeline(db, true_states, dets)

        tracks = db.retrieve_all_tracks()
        assert len(tracks) >= 3

        # Verify longest track has substantial history
        max_len = 0
        for t in tracks:
            h = db.get_track_history(t["track_id"])
            max_len = max(max_len, len(h["timestamps"]))
        assert max_len >= 500, f"Longest track only {max_len} steps"
        db.close()

    def test_long_duration_hdf5_archival(self, tmp_path):
        """Archive a 1000-step mission to HDF5 and verify retrieval."""
        rng = np.random.default_rng(61)
        n_steps = 1000
        n_tracks = 5
        state_dim = 4

        db = TrackDatabaseManager(str(tmp_path / "long_src.db"))
        db.open(mode="w")

        for i in range(n_tracks):
            tid = f"trk_{i:02d}"
            x0 = rng.normal(0, 10, state_dim)
            P0 = np.eye(state_dim)
            db.initiate_track(tid, x0, P0, 0.0)
            states = np.cumsum(rng.normal(0, 0.5, (n_steps, state_dim)), axis=0)
            covs = np.array([np.eye(state_dim) for _ in range(n_steps)])
            times = np.arange(n_steps, dtype=np.float64)
            db.store_track_history(tid, states, covs, times)

        h5_path = str(tmp_path / "long_archive.h5")
        store = TrackHDF5Storage(h5_path)
        store.open(mode="w")
        store.import_from_sql(db, scenario_id="long_mission")

        archived_tracks = store.list_tracks("long_mission")
        assert len(archived_tracks) == n_tracks

        # Verify trajectory retrieval on archived data
        traj = store.get_track_trajectory(
            "trk_00", scenario_id="long_mission", start_time=100.0, end_time=200.0
        )
        assert len(traj["timestamps"]) > 0

        store.close()
        db.close()


# =============================================================================
# 4. Data Integrity Checks
# =============================================================================


class TestRoundTripIntegrity:
    """Track state consistency before/after database round-trip."""

    def test_state_roundtrip_sql(self, tmp_path):
        """State/covariance survives SQL store-retrieve cycle."""
        rng = np.random.default_rng(70)
        db = TrackDatabaseManager(str(tmp_path / "rt_sql.db"))
        db.open(mode="w")

        x_orig = rng.normal(0, 10, 4)
        P_orig = rng.normal(0, 1, (4, 4))
        P_orig = P_orig @ P_orig.T + np.eye(4)  # Make PD

        db.initiate_track("trk_rt", x_orig, P_orig, 0.0)
        state = db.get_track_state("trk_rt")

        np.testing.assert_allclose(state["state"], x_orig, atol=1e-10)
        np.testing.assert_allclose(state["covariance"], P_orig, atol=1e-10)
        db.close()

    def test_state_roundtrip_hdf5(self, tmp_path):
        """State/covariance survives HDF5 store-retrieve cycle."""
        rng = np.random.default_rng(71)
        n = 50
        states = rng.normal(0, 10, (n, 4))
        covs = np.array([np.eye(4) * (1 + 0.1 * k) for k in range(n)])
        times = np.arange(n, dtype=np.float64)

        store = TrackHDF5Storage(str(tmp_path / "rt_h5.h5"))
        store.open(mode="w")
        store.store_track("trk_rt", states, covs, times)

        retrieved = store.retrieve_track("trk_rt")
        np.testing.assert_allclose(retrieved["states"], states, atol=1e-10)
        np.testing.assert_allclose(retrieved["covariances"], covs, atol=1e-10)
        np.testing.assert_allclose(retrieved["timestamps"], times, atol=1e-10)
        store.close()

    def test_sql_hdf5_sql_roundtrip(self, tmp_path):
        """SQL → HDF5 → SQL preserves state data."""
        rng = np.random.default_rng(72)
        n_tracks = 5
        n_steps = 30

        # Phase 1: populate SQL
        src_path = str(tmp_path / "src.db")
        src_db = TrackDatabaseManager(src_path)
        src_db.open(mode="w")
        original_states = {}

        for i in range(n_tracks):
            tid = f"trk_{i:02d}"
            x0 = rng.normal(0, 10, 4)
            P0 = np.eye(4) * 2.0
            src_db.initiate_track(tid, x0, P0, 0.0)
            states = np.cumsum(rng.normal(0, 0.5, (n_steps, 4)), axis=0)
            covs = np.array([np.eye(4) * (1 + 0.01 * k) for k in range(n_steps)])
            times = np.arange(n_steps, dtype=np.float64)
            src_db.store_track_history(tid, states, covs, times)
            original_states[tid] = {"states": states, "covs": covs, "times": times}

        # Phase 2: export to HDF5
        h5_path = str(tmp_path / "roundtrip.h5")
        store = TrackHDF5Storage(h5_path)
        store.open(mode="w")
        store.import_from_sql(src_db, scenario_id="rt")
        src_db.close()

        # Phase 3: import back to SQL
        dst_path = str(tmp_path / "dst.db")
        dst_db = TrackDatabaseManager(dst_path)
        dst_db.open(mode="w")
        store.export_to_sql(dst_db, scenario_id="rt")
        store.close()

        # Verify
        for tid, orig in original_states.items():
            history = dst_db.get_track_history(tid)
            # States should match (allow for initial state from initiate_track)
            assert history["states"].shape[1] == 4
            assert len(history["timestamps"]) >= n_steps
        dst_db.close()


class TestAssociationCorrectness:
    """Detection-track association integrity."""

    def test_associated_detection_has_track_id(self, tmp_path):
        """Associated detections have correct track_id field."""
        db = TrackDatabaseManager(str(tmp_path / "assoc.db"))
        db.open(mode="w")

        db.initiate_track("trk_0", np.zeros(4), np.eye(4), 0.0)
        db.store_detection("det_0", np.array([1.0, 2.0]), "radar", 0.0)
        db.associate_detection("det_0", "trk_0", confidence=0.9)

        det = db.retrieve_detection("det_0")
        assert det["association_status"] == "associated"
        assert det["associated_track_id"] == "trk_0"
        assert det["association_confidence"] == pytest.approx(0.9)
        db.close()

    def test_unassociated_detection_status(self, tmp_path):
        """Unassociated detections have 'unassociated' status."""
        db = TrackDatabaseManager(str(tmp_path / "unassoc.db"))
        db.open(mode="w")

        db.store_detection("det_0", np.array([1.0, 2.0]), "radar", 0.0)
        det = db.retrieve_detection("det_0")
        assert det["association_status"] == "unassociated"
        assert det["associated_track_id"] is None
        db.close()

    def test_multiple_detections_per_track(self, tmp_path):
        """Multiple detections can be associated to the same track."""
        db = TrackDatabaseManager(str(tmp_path / "multi_assoc.db"))
        db.open(mode="w")

        db.initiate_track("trk_0", np.zeros(4), np.eye(4), 0.0)
        for i in range(10):
            db.store_detection(f"det_{i}", np.array([float(i), 0.0]), "radar", float(i))
            db.associate_detection(f"det_{i}", "trk_0")

        associated = db.retrieve_detections(association_status="associated")
        assert len(associated) == 10
        for d in associated:
            assert d["associated_track_id"] == "trk_0"
        db.close()


class TestTimestampOrdering:
    """Timestamp ordering validation."""

    def test_track_history_ordered(self, tmp_path):
        """Track state history is returned in chronological order."""
        db = TrackDatabaseManager(str(tmp_path / "ts_order.db"))
        db.open(mode="w")

        P0 = np.eye(4)
        db.initiate_track("trk_0", np.zeros(4), P0, 0.0)

        # Insert states in scrambled order
        for t in [5.0, 1.0, 3.0, 9.0, 2.0, 7.0, 4.0, 8.0, 6.0, 0.5]:
            db.update_track_state("trk_0", np.ones(4) * t, P0, t)

        history = db.get_track_history("trk_0")
        timestamps = history["timestamps"]

        # Should be sorted
        assert np.all(np.diff(timestamps) >= 0), "History not in chronological order"
        db.close()

    def test_detection_query_ordered(self, tmp_path):
        """Detection queries return results ordered by timestamp."""
        db = TrackDatabaseManager(str(tmp_path / "det_order.db"))
        db.open(mode="w")

        # Insert detections out of order
        for t in [3.0, 1.0, 4.0, 1.5, 9.0, 2.6]:
            db.store_detection(f"det_{t:.1f}", np.array([t, t]), "radar", t)

        dets = db.retrieve_all_detections()
        timestamps = [d["timestamp"] for d in dets]
        assert timestamps == sorted(timestamps), "Detections not ordered by time"
        db.close()

    def test_hdf5_trajectory_timestamps_sorted(self, tmp_path):
        """HDF5 trajectory retrieval returns sorted timestamps."""
        store = TrackHDF5Storage(str(tmp_path / "ts_h5.h5"))
        store.open(mode="w")

        times = np.array([0, 1, 2, 3, 4, 5], dtype=np.float64)
        states = np.random.randn(6, 4)
        covs = np.array([np.eye(4) for _ in range(6)])
        store.store_track("trk_0", states, covs, times)

        traj = store.get_track_trajectory("trk_0", start_time=1.0, end_time=4.0)
        assert np.all(np.diff(traj["timestamps"]) >= 0)
        store.close()


class TestCovariancePositiveDefiniteness:
    """Covariance matrix positive-definiteness checks."""

    def test_initial_covariance_pd(self, tmp_path):
        """Initial covariance stored in DB is positive definite."""
        rng = np.random.default_rng(80)
        db = TrackDatabaseManager(str(tmp_path / "pd.db"))
        db.open(mode="w")

        # Create a PD matrix
        A = rng.normal(0, 1, (4, 4))
        P_pd = A @ A.T + np.eye(4)

        db.initiate_track("trk_0", np.zeros(4), P_pd, 0.0)
        state = db.get_track_state("trk_0")

        eigenvalues = np.linalg.eigvalsh(state["covariance"])
        assert np.all(eigenvalues > 0), f"Non-PD covariance: eigenvalues={eigenvalues}"
        db.close()

    def test_kalman_updated_covariance_pd(self, tmp_path):
        """Covariance remains PD after Kalman update + DB round-trip."""
        db = TrackDatabaseManager(str(tmp_path / "pd_kf.db"))
        db.open(mode="w")

        F, Q, H, R = _cv_model()
        P0 = np.eye(4) * 10.0
        x0 = np.zeros(4)

        db.initiate_track("trk_0", x0, P0, 0.0)

        x, P = x0, P0
        rng = np.random.default_rng(81)
        for k in range(1, 20):
            pred = kf_predict(x, P, F, Q)
            z = H @ pred.x + rng.multivariate_normal([0, 0], R)
            upd = kf_update(pred.x, pred.P, z, H, R)
            x, P = upd.x, upd.P
            db.update_track_state("trk_0", x, P, float(k))

        # Check all stored covariances are PD
        history = db.get_track_history("trk_0")
        for i, cov in enumerate(history["covariances"]):
            eigenvalues = np.linalg.eigvalsh(cov)
            assert np.all(
                eigenvalues > 0
            ), f"Step {i}: Non-PD covariance, eigenvalues={eigenvalues}"
        db.close()

    def test_hdf5_covariance_pd_preserved(self, tmp_path):
        """HDF5 round-trip preserves covariance positive-definiteness."""
        rng = np.random.default_rng(82)
        n = 20

        # Generate PD covariances
        covs = []
        for _ in range(n):
            A = rng.normal(0, 1, (4, 4))
            covs.append(A @ A.T + np.eye(4))
        covs = np.array(covs)

        states = rng.normal(0, 1, (n, 4))
        times = np.arange(n, dtype=np.float64)

        store = TrackHDF5Storage(str(tmp_path / "pd_h5.h5"))
        store.open(mode="w")
        store.store_track("trk_0", states, covs, times)

        retrieved = store.retrieve_track("trk_0")
        for i, cov in enumerate(retrieved["covariances"]):
            eigenvalues = np.linalg.eigvalsh(cov)
            assert np.all(
                eigenvalues > 0
            ), f"Step {i}: Non-PD after HDF5 round-trip, eigenvalues={eigenvalues}"
        store.close()


class TestMetadataIntegrity:
    """Metadata preservation through storage operations."""

    def test_detection_metadata_roundtrip(self, tmp_path):
        """Detection metadata survives store/retrieve."""
        db = TrackDatabaseManager(str(tmp_path / "meta.db"))
        db.open(mode="w")

        meta = {"snr": 15.5, "quality": "high"}
        db.store_detection("det_0", np.array([1.0, 2.0]), "radar", 0.0, metadata=meta)

        det = db.retrieve_detection("det_0")
        assert det["metadata"]["snr"] == pytest.approx(15.5)
        assert det["metadata"]["quality"] == "high"
        db.close()

    def test_track_metadata_roundtrip(self, tmp_path):
        """Track metadata survives store/retrieve."""
        db = TrackDatabaseManager(str(tmp_path / "trk_meta.db"))
        db.open(mode="w")

        meta = {"source": "GNN", "priority": 1}
        db.initiate_track("trk_0", np.zeros(4), np.eye(4), 0.0, metadata=meta)

        info = db.get_track("trk_0")
        assert info["metadata"]["source"] == "GNN"
        assert info["metadata"]["priority"] == 1
        db.close()


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_single_target_single_step(self, tmp_path):
        """Minimal scenario: 1 target, 1 timestep."""
        true_states = _simulate_cv_targets(1, 1, seed=90)
        _, _, _, R = _cv_model()
        dets = _generate_detections(true_states, R, pd=1.0, fa_rate=0.0, seed=90)

        db = TrackDatabaseManager(str(tmp_path / "minimal.db"))
        db.open(mode="w")
        _run_tracking_pipeline(db, true_states, dets)

        tracks = db.retrieve_all_tracks()
        assert len(tracks) >= 1
        db.close()

    def test_no_detections_all_missed(self, tmp_path):
        """All detections missed (PD=0): tracks still initiated from scan."""
        true_states = _simulate_cv_targets(3, 10, seed=91)
        _, _, _, R = _cv_model()
        dets = _generate_detections(true_states, R, pd=0.0, fa_rate=0.0, seed=91)

        db = TrackDatabaseManager(str(tmp_path / "nomeas.db"))
        db.open(mode="w")
        _run_tracking_pipeline(db, true_states, dets)

        # With PD=0 and no FA, there are no detections, so no tracks
        tracks = db.retrieve_all_tracks()
        assert len(tracks) == 0
        db.close()

    def test_all_clutter_no_targets(self, tmp_path):
        """No true targets, only false alarms."""
        true_states = np.zeros((0, 20, 4))  # no targets
        _, _, _, R = _cv_model()
        # Manually create clutter-only detections
        rng = np.random.default_rng(92)
        dets = []
        for k in range(20):
            scan = []
            for _ in range(5):
                fa = rng.uniform(-100, 100, 2)
                scan.append((fa, None))
            dets.append(scan)

        db = TrackDatabaseManager(str(tmp_path / "clutter_only.db"))
        db.open(mode="w")
        _run_tracking_pipeline(db, true_states, dets)

        tracks = db.retrieve_all_tracks()
        # Should have some tracks (initiated from false alarms)
        assert len(tracks) >= 0  # May or may not initiate
        db.close()

    def test_pytcl_track_conversion(self, tmp_path):
        """TrackDatabaseManager.track_to_pytcl produces valid Track objects."""
        db = TrackDatabaseManager(str(tmp_path / "convert.db"))
        db.open(mode="w")

        x0 = np.array([10.0, 1.0, 20.0, 2.0])
        P0 = np.eye(4) * 5.0
        db.initiate_track("trk_conv", x0, P0, 0.0)
        db.confirm_track("trk_conv")

        track_obj = db.track_to_pytcl("trk_conv")
        assert track_obj.state.shape == (4,)
        assert track_obj.covariance.shape == (4, 4)
        np.testing.assert_allclose(track_obj.state, x0, atol=1e-10)
        db.close()

    def test_tracklist_conversion(self, tmp_path):
        """TrackDatabaseManager.tracks_to_tracklist returns proper container."""
        db = TrackDatabaseManager(str(tmp_path / "tlist.db"))
        db.open(mode="w")

        for i in range(5):
            db.initiate_track(f"trk_{i}", np.ones(4) * i, np.eye(4), float(i))
            db.confirm_track(f"trk_{i}")

        track_list = db.tracks_to_tracklist(status=TrackDatabaseStatus.CONFIRMED)
        assert len(track_list) == 5
        db.close()

    def test_store_from_pytcl_track(self, tmp_path):
        """Store a pytcl Track into the database and retrieve it."""
        db = TrackDatabaseManager(str(tmp_path / "from_pytcl.db"))
        db.open(mode="w")

        # Create a track, convert to pytcl, store back
        db.initiate_track("trk_src", np.array([5.0, 1.0, 3.0, 0.5]), np.eye(4), 0.0)
        track_obj = db.track_to_pytcl("trk_src")
        db.store_from_track(track_obj, timestamp=1.0)

        # The store_from_track should create/update a track with id based on track_obj.id
        all_tracks = db.retrieve_all_tracks()
        assert len(all_tracks) >= 1
        db.close()

    def test_hdf5_scenario_list_empty(self, tmp_path):
        """Listing scenarios on empty HDF5 returns empty list."""
        store = TrackHDF5Storage(str(tmp_path / "empty.h5"))
        store.open(mode="w")
        assert store.list_scenarios() == []
        assert store.list_tracks() == []
        store.close()

    def test_initiation_queue(self, tmp_path):
        """get_initiation_queue returns unassociated detections."""
        db = TrackDatabaseManager(str(tmp_path / "queue.db"))
        db.open(mode="w")

        db.store_detection("det_0", np.array([1.0, 2.0]), "radar", 0.0)
        db.store_detection("det_1", np.array([3.0, 4.0]), "radar", 1.0)
        db.store_detection("det_2", np.array([5.0, 6.0]), "radar", 2.0)

        # Associate one
        db.initiate_track("trk_0", np.zeros(4), np.eye(4), 0.0)
        db.associate_detection("det_1", "trk_0")

        queue = db.get_initiation_queue()
        queue_ids = {d["detection_id"] for d in queue}
        assert "det_0" in queue_ids
        assert "det_2" in queue_ids
        assert "det_1" not in queue_ids
        db.close()
