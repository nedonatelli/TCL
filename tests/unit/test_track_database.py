"""Tests for TrackDatabaseManager SQL track lifecycle management."""

import tempfile

import numpy as np
import pytest

from pytcl.io.track_database import TrackDatabaseManager, TrackDatabaseStatus


class TestTrackDatabaseManager:
    """Tests for TrackDatabaseManager."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database file."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            yield f.name

    @pytest.fixture
    def populated_db(self, temp_db):
        """Create database with sample tracks and detections."""
        with TrackDatabaseManager(temp_db) as db:
            db.open(mode="w")

            # Store detections
            for i in range(5):
                db.store_detection(
                    f"det_{i:03d}",
                    np.array([float(i), float(i) * 2]),
                    "radar",
                    float(i),
                )

            # Initiate tracks
            db.initiate_track(
                "trk_001",
                np.array([1.0, 2.0, 0.0, 0.0]),
                np.eye(4),
                0.0,
            )
            db.initiate_track(
                "trk_002",
                np.array([5.0, 6.0, 0.0, 0.0]),
                np.eye(4),
                0.0,
            )

            # Add some state updates
            db.update_track_state(
                "trk_001",
                np.array([1.1, 2.1, 0.1, 0.1]),
                np.eye(4) * 0.9,
                1.0,
            )
            db.update_track_state(
                "trk_001",
                np.array([1.2, 2.2, 0.1, 0.1]),
                np.eye(4) * 0.8,
                2.0,
            )

            # Associate some detections
            db.associate_detection("det_000", "trk_001")
            db.associate_detection("det_001", "trk_001")

        yield temp_db

    # === Context Manager & Lifecycle ===

    def test_context_manager(self, temp_db):
        """Test context manager opens and closes properly."""
        with TrackDatabaseManager(temp_db) as db:
            db.open(mode="w")
            db.store_detection("d1", np.array([1.0]), "sensor", 0.0)

        assert db._connection is None

    def test_open_close(self, temp_db):
        """Test explicit open and close."""
        db = TrackDatabaseManager(temp_db)
        db.open(mode="w")
        assert db._connection is not None
        db.close()
        assert db._connection is None

    def test_not_open_raises(self, temp_db):
        """Test operations on closed database raise RuntimeError."""
        db = TrackDatabaseManager(temp_db)
        with pytest.raises(RuntimeError, match="not open"):
            db.store_detection("d1", np.array([1.0]), "s", 0.0)

    # === Detection Management ===

    def test_store_detection(self, temp_db):
        """Test storing a detection."""
        with TrackDatabaseManager(temp_db) as db:
            db.open(mode="w")
            db.store_detection("det_001", np.array([1.0, 2.0]), "radar", 10.0)

            det = db.retrieve_detection("det_001")
            np.testing.assert_array_equal(det["measurement"], [1.0, 2.0])
            assert det["sensor_id"] == "radar"
            assert det["timestamp"] == 10.0
            assert det["association_status"] == "unassociated"

    def test_store_detection_with_covariance(self, temp_db):
        """Test storing a detection with covariance."""
        with TrackDatabaseManager(temp_db) as db:
            db.open(mode="w")
            cov = np.array([[1.0, 0.1], [0.1, 1.0]])
            db.store_detection(
                "det_001",
                np.array([1.0, 2.0]),
                "lidar",
                5.0,
                covariance=cov,
            )

            det = db.retrieve_detection("det_001")
            np.testing.assert_array_almost_equal(det["measurement"], [1.0, 2.0])

    def test_retrieve_detections_by_time(self, populated_db):
        """Test filtering detections by time range."""
        with TrackDatabaseManager(populated_db) as db:
            db.open(mode="r")
            dets = db.retrieve_detections(start_time=1.0, end_time=3.0)
            timestamps = [d["timestamp"] for d in dets]
            assert all(1.0 <= t <= 3.0 for t in timestamps)

    def test_retrieve_detections_by_sensor(self, populated_db):
        """Test filtering detections by sensor."""
        with TrackDatabaseManager(populated_db) as db:
            db.open(mode="r")
            dets = db.retrieve_detections(sensor_id="radar")
            assert len(dets) == 5
            assert all(d["sensor_id"] == "radar" for d in dets)

    def test_retrieve_detections_by_status(self, populated_db):
        """Test filtering detections by association status."""
        with TrackDatabaseManager(populated_db) as db:
            db.open(mode="r")
            unassoc = db.retrieve_detections(association_status="unassociated")
            assoc = db.retrieve_detections(association_status="associated")
            assert len(unassoc) == 3
            assert len(assoc) == 2

    def test_associate_detection(self, temp_db):
        """Test linking a detection to a track."""
        with TrackDatabaseManager(temp_db) as db:
            db.open(mode="w")
            db.store_detection("det_001", np.array([1.0]), "radar", 0.0)
            db.initiate_track("trk_001", np.array([1.0, 0.0]), np.eye(2), 0.0)
            db.associate_detection("det_001", "trk_001", confidence=0.95)

            det = db.retrieve_detection("det_001")
            assert det["association_status"] == "associated"
            assert det["associated_track_id"] == "trk_001"
            assert det["association_confidence"] == 0.95

    # === Track Initiation ===

    def test_initiate_track(self, temp_db):
        """Test creating a new tentative track."""
        with TrackDatabaseManager(temp_db) as db:
            db.open(mode="w")
            db.initiate_track(
                "trk_001",
                np.array([1.0, 2.0, 0.0, 0.0]),
                np.eye(4),
                0.0,
                metadata={"source": "radar"},
            )

            track = db.get_track("trk_001")
            assert track["status"] == TrackDatabaseStatus.TENTATIVE.value
            assert track["birth_time"] == 0.0
            assert track["state_dim"] == 4
            assert track["metadata"]["source"] == "radar"

    def test_get_initiation_queue(self, temp_db):
        """Test getting unassociated detections."""
        with TrackDatabaseManager(temp_db) as db:
            db.open(mode="w")
            db.store_detection("d1", np.array([1.0]), "s", 0.0)
            db.store_detection("d2", np.array([2.0]), "s", 1.0)
            db.store_detection("d3", np.array([3.0]), "s", 2.0)

            queue = db.get_initiation_queue()
            assert len(queue) == 3

    def test_get_initiation_queue_with_age(self, temp_db):
        """Test initiation queue with age filter."""
        with TrackDatabaseManager(temp_db) as db:
            db.open(mode="w")
            db.store_detection("d1", np.array([1.0]), "s", 0.0)
            db.store_detection("d2", np.array([2.0]), "s", 5.0)
            db.store_detection("d3", np.array([3.0]), "s", 10.0)

            queue = db.get_initiation_queue(max_age=3.0)
            assert len(queue) == 1
            assert queue[0]["detection_id"] == "d3"

    def test_confirm_track(self, temp_db):
        """Test promoting tentative to confirmed."""
        with TrackDatabaseManager(temp_db) as db:
            db.open(mode="w")
            db.initiate_track("trk_001", np.array([1.0]), np.eye(1), 0.0)
            db.confirm_track("trk_001")

            track = db.get_track("trk_001")
            assert track["status"] == TrackDatabaseStatus.CONFIRMED.value

    # === State Maintenance ===

    def test_update_track_state(self, temp_db):
        """Test recording a state update."""
        with TrackDatabaseManager(temp_db) as db:
            db.open(mode="w")
            db.initiate_track("trk_001", np.array([1.0, 0.0]), np.eye(2), 0.0)
            db.update_track_state("trk_001", np.array([1.1, 0.1]), np.eye(2) * 0.9, 1.0)

            state = db.get_track_state("trk_001")
            np.testing.assert_array_almost_equal(state["state"], [1.1, 0.1])
            assert state["timestamp"] == 1.0
            assert state["hits"] == 1

    def test_update_track_state_prediction(self, temp_db):
        """Test prediction update increments misses."""
        with TrackDatabaseManager(temp_db) as db:
            db.open(mode="w")
            db.initiate_track("trk_001", np.array([1.0]), np.eye(1), 0.0)
            db.update_track_state(
                "trk_001",
                np.array([1.1]),
                np.eye(1),
                1.0,
                update_type="prediction",
            )

            track = db.get_track("trk_001")
            assert track["misses"] == 1
            assert track["total_misses"] == 1

    def test_store_track_history_batch(self, temp_db):
        """Test batch storing state history."""
        with TrackDatabaseManager(temp_db) as db:
            db.open(mode="w")
            db.initiate_track("trk_001", np.array([0.0, 0.0]), np.eye(2), 0.0)

            n = 10
            states = np.random.randn(n, 2)
            covs = np.array([np.eye(2) for _ in range(n)])
            times = np.arange(1.0, n + 1.0)

            db.store_track_history("trk_001", states, covs, times)

            history = db.get_track_history("trk_001")
            # 1 initial + 10 batch = 11 total
            assert len(history["timestamps"]) == 11

    def test_get_track_state_latest(self, populated_db):
        """Test getting most recent state."""
        with TrackDatabaseManager(populated_db) as db:
            db.open(mode="r")
            state = db.get_track_state("trk_001")
            assert state["timestamp"] == 2.0
            np.testing.assert_array_almost_equal(state["state"], [1.2, 2.2, 0.1, 0.1])

    def test_get_track_history_full(self, populated_db):
        """Test getting full state history."""
        with TrackDatabaseManager(populated_db) as db:
            db.open(mode="r")
            history = db.get_track_history("trk_001")
            assert history["states"].shape[1] == 4
            assert len(history["timestamps"]) == 3  # init + 2 updates

    def test_get_track_history_time_range(self, populated_db):
        """Test getting state history within time range."""
        with TrackDatabaseManager(populated_db) as db:
            db.open(mode="r")
            history = db.get_track_history("trk_001", start_time=1.0, end_time=2.0)
            assert len(history["timestamps"]) == 2
            assert history["timestamps"][0] == 1.0

    # === Lifecycle Management ===

    def test_mark_track_tentative(self, temp_db):
        """Test setting status to tentative."""
        with TrackDatabaseManager(temp_db) as db:
            db.open(mode="w")
            db.initiate_track("trk_001", np.array([1.0]), np.eye(1), 0.0)
            db.confirm_track("trk_001")
            db.mark_track_tentative("trk_001")
            assert (
                db.get_track("trk_001")["status"] == TrackDatabaseStatus.TENTATIVE.value
            )

    def test_mark_track_confirmed(self, temp_db):
        """Test setting status to confirmed."""
        with TrackDatabaseManager(temp_db) as db:
            db.open(mode="w")
            db.initiate_track("trk_001", np.array([1.0]), np.eye(1), 0.0)
            db.mark_track_confirmed("trk_001")
            assert (
                db.get_track("trk_001")["status"] == TrackDatabaseStatus.CONFIRMED.value
            )

    def test_mark_track_coasting(self, temp_db):
        """Test setting status to coasting."""
        with TrackDatabaseManager(temp_db) as db:
            db.open(mode="w")
            db.initiate_track("trk_001", np.array([1.0]), np.eye(1), 0.0)
            db.mark_track_coasting("trk_001")
            assert (
                db.get_track("trk_001")["status"] == TrackDatabaseStatus.COASTING.value
            )

    def test_mark_track_dead(self, temp_db):
        """Test setting status to dead."""
        with TrackDatabaseManager(temp_db) as db:
            db.open(mode="w")
            db.initiate_track("trk_001", np.array([1.0]), np.eye(1), 0.0)
            db.mark_track_dead("trk_001")
            assert db.get_track("trk_001")["status"] == TrackDatabaseStatus.DEAD.value

    def test_prune_old_detections(self, temp_db):
        """Test pruning old unassociated detections."""
        with TrackDatabaseManager(temp_db) as db:
            db.open(mode="w")
            db.store_detection("d1", np.array([1.0]), "s", 0.0)
            db.store_detection("d2", np.array([2.0]), "s", 5.0)
            db.store_detection("d3", np.array([3.0]), "s", 10.0)

            pruned = db.prune_old_detections(age_threshold=3.0)
            assert pruned == 2

            remaining = db.retrieve_all_detections()
            assert len(remaining) == 1
            assert remaining[0]["detection_id"] == "d3"

    def test_prune_dead_tracks(self, temp_db):
        """Test pruning dead tracks."""
        with TrackDatabaseManager(temp_db) as db:
            db.open(mode="w")
            db.initiate_track("trk_001", np.array([1.0]), np.eye(1), 0.0)
            db.initiate_track("trk_002", np.array([2.0]), np.eye(1), 10.0)
            db.mark_track_dead("trk_001")

            pruned = db.prune_dead_tracks(age_threshold=5.0)
            assert pruned == 1

            tracks = db.retrieve_all_tracks()
            assert len(tracks) == 1
            assert tracks[0]["track_id"] == "trk_002"

    # === Merge Operations ===

    def test_merge_tracks(self, temp_db):
        """Test merging two tracks."""
        with TrackDatabaseManager(temp_db) as db:
            db.open(mode="w")
            db.initiate_track("trk_001", np.array([1.0]), np.eye(1), 0.0)
            db.initiate_track("trk_002", np.array([2.0]), np.eye(1), 0.0)
            db.update_track_state("trk_002", np.array([2.1]), np.eye(1), 1.0)

            db.merge_tracks("trk_001", "trk_002")

            # Merged track should be dead
            assert db.get_track("trk_002")["status"] == TrackDatabaseStatus.DEAD.value

            # Keep track should have merged history
            history = db.get_track_history("trk_001")
            assert len(history["timestamps"]) >= 3

    def test_merge_tracks_detection_reassignment(self, temp_db):
        """Test that merge reassigns detections."""
        with TrackDatabaseManager(temp_db) as db:
            db.open(mode="w")
            db.store_detection("d1", np.array([1.0]), "s", 0.0)
            db.initiate_track("trk_001", np.array([1.0]), np.eye(1), 0.0)
            db.initiate_track("trk_002", np.array([2.0]), np.eye(1), 0.0)
            db.associate_detection("d1", "trk_002")

            db.merge_tracks("trk_001", "trk_002")

            det = db.retrieve_detection("d1")
            assert det["associated_track_id"] == "trk_001"

    # === Conversion Helpers ===

    def test_track_to_pytcl(self, populated_db):
        """Test converting to pytcl Track NamedTuple."""
        with TrackDatabaseManager(populated_db) as db:
            db.open(mode="r")
            track = db.track_to_pytcl("trk_001")

            assert track.state.shape == (4,)
            assert track.covariance.shape == (4, 4)
            assert isinstance(track.hits, int)

    def test_tracks_to_tracklist(self, populated_db):
        """Test converting to pytcl TrackList."""
        with TrackDatabaseManager(populated_db) as db:
            db.open(mode="r")
            tl = db.tracks_to_tracklist()

            assert len(tl) == 2

    def test_store_from_track(self, temp_db):
        """Test storing a pytcl Track."""
        from pytcl.trackers.multi_target import Track, TrackStatus

        track = Track(
            id=42,
            state=np.array([1.0, 2.0]),
            covariance=np.eye(2),
            status=TrackStatus.CONFIRMED,
            hits=5,
            misses=0,
            time=10.0,
        )

        with TrackDatabaseManager(temp_db) as db:
            db.open(mode="w")
            db.store_from_track(track)

            stored = db.get_track("trk_42")
            assert stored["state_dim"] == 2

    # === Edge Cases ===

    def test_nonexistent_track_raises(self, temp_db):
        """Test accessing nonexistent track raises KeyError."""
        with TrackDatabaseManager(temp_db) as db:
            db.open(mode="w")
            with pytest.raises(KeyError, match="not found"):
                db.get_track("nonexistent")

    def test_duplicate_detection_id_raises(self, temp_db):
        """Test storing duplicate detection ID raises."""
        with TrackDatabaseManager(temp_db) as db:
            db.open(mode="w")
            db.store_detection("d1", np.array([1.0]), "s", 0.0)
            with pytest.raises(Exception):
                db.store_detection("d1", np.array([2.0]), "s", 1.0)


class TestReadModeDoesNotCreateTheDatabase:
    """``mode='r'`` must not conjure the file it is meant to read.

    ``open`` called ``sqlite3.connect`` unconditionally, so opening a
    mistyped path for reading produced an empty database and the first query
    then failed with ``no such table: detections`` -- reporting a missing
    table rather than the missing file the caller actually had. gh-21 fixed
    exactly this for ``SQLStorage``; ``TrackDatabaseManager`` was not given
    the same treatment at the time.
    """

    def test_read_mode_raises_and_creates_nothing(self, tmp_path):
        missing = tmp_path / "absent.db"

        with pytest.raises(FileNotFoundError, match="No such database"):
            TrackDatabaseManager(str(missing)).open(mode="r")

        assert not missing.exists()

    def test_write_mode_still_creates(self, tmp_path):
        created = tmp_path / "made.db"
        manager = TrackDatabaseManager(str(created))
        manager.open(mode="w")
        manager.close()
        assert created.exists()

    def test_read_mode_opens_an_existing_database(self, tmp_path):
        existing = tmp_path / "there.db"
        writer = TrackDatabaseManager(str(existing))
        writer.open(mode="w")
        writer.close()

        reader = TrackDatabaseManager(str(existing))
        reader.open(mode="r")
        reader.close()

    def test_an_invalid_mode_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="mode must be"):
            TrackDatabaseManager(str(tmp_path / "x.db")).open(mode="rw")
