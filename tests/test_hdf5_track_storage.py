"""Tests for TrackHDF5Storage track archival and time-series queries."""

import tempfile

import numpy as np
import pytest

try:
    import h5py  # noqa: F401

    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


@pytest.mark.skipif(not HAS_H5PY, reason="h5py not installed")
class TestTrackHDF5Storage:
    """Tests for TrackHDF5Storage."""

    @pytest.fixture
    def temp_h5(self):
        """Create temporary HDF5 file."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            yield f.name

    @pytest.fixture
    def sample_track_data(self):
        """Generate sample track data."""
        n = 50
        state_dim = 4
        np.random.seed(42)
        states = np.cumsum(np.random.randn(n, state_dim) * 0.1, axis=0)
        covariances = np.array([np.eye(state_dim) * (1 + i * 0.01) for i in range(n)])
        timestamps = np.linspace(0.0, 10.0, n)
        return states, covariances, timestamps

    @pytest.fixture
    def populated_h5(self, temp_h5, sample_track_data):
        """Create HDF5 file with sample scenario."""
        from pytcl.io.hdf5_track_storage import TrackHDF5Storage

        states, covs, times = sample_track_data

        with TrackHDF5Storage(temp_h5) as store:
            store.open(mode="w")
            store.store_tracking_scenario(
                "scenario_1",
                tracks={
                    "trk_001": {
                        "states": states,
                        "covariances": covs,
                        "timestamps": times,
                        "metadata": {"status": "confirmed"},
                    },
                    "trk_002": {
                        "states": states + 10.0,
                        "covariances": covs,
                        "timestamps": times,
                        "metadata": {"status": "tentative"},
                    },
                },
                detections={
                    "det_001": {
                        "measurement": np.array([1.0, 2.0]),
                        "timestamp": 0.0,
                        "sensor_id": "radar",
                    },
                    "det_002": {
                        "measurement": np.array([3.0, 4.0]),
                        "timestamp": 1.0,
                        "sensor_id": "lidar",
                    },
                },
                metadata={"description": "test scenario"},
            )

        yield temp_h5

    # === Context Manager ===

    def test_context_manager(self, temp_h5):
        """Test context manager opens and closes properly."""
        from pytcl.io.hdf5_track_storage import TrackHDF5Storage

        with TrackHDF5Storage(temp_h5) as store:
            store.open(mode="w")
            assert store._file is not None

        assert store._file is None

    def test_open_close(self, temp_h5):
        """Test explicit open and close."""
        from pytcl.io.hdf5_track_storage import TrackHDF5Storage

        store = TrackHDF5Storage(temp_h5)
        store.open(mode="w")
        assert store._file is not None
        store.close()
        assert store._file is None

    # === Single Track ===

    def test_store_retrieve_track(self, temp_h5, sample_track_data):
        """Test storing and retrieving a track."""
        from pytcl.io.hdf5_track_storage import TrackHDF5Storage

        states, covs, times = sample_track_data

        with TrackHDF5Storage(temp_h5) as store:
            store.open(mode="w")
            store.store_track("trk_001", states, covs, times)

        with TrackHDF5Storage(temp_h5) as store:
            store.open(mode="r")
            track = store.retrieve_track("trk_001")

        np.testing.assert_array_almost_equal(track["states"], states)
        np.testing.assert_array_almost_equal(track["covariances"], covs)
        np.testing.assert_array_almost_equal(track["timestamps"], times)

    def test_store_track_with_metadata(self, temp_h5, sample_track_data):
        """Test storing a track with metadata."""
        from pytcl.io.hdf5_track_storage import TrackHDF5Storage

        states, covs, times = sample_track_data

        with TrackHDF5Storage(temp_h5) as store:
            store.open(mode="w")
            store.store_track(
                "trk_001",
                states,
                covs,
                times,
                metadata={"status": "confirmed", "hits": 10},
            )

        with TrackHDF5Storage(temp_h5) as store:
            store.open(mode="r")
            track = store.retrieve_track("trk_001")
            assert track["metadata"]["status"] == "confirmed"
            assert track["metadata"]["hits"] == 10

    def test_store_track_with_residuals(self, temp_h5, sample_track_data):
        """Test storing a track with residuals."""
        from pytcl.io.hdf5_track_storage import TrackHDF5Storage

        states, covs, times = sample_track_data
        residuals = np.random.randn(len(times), 2)

        with TrackHDF5Storage(temp_h5) as store:
            store.open(mode="w")
            store.store_track("trk_001", states, covs, times, residuals=residuals)

        with TrackHDF5Storage(temp_h5) as store:
            store.open(mode="r")
            track = store.retrieve_track("trk_001")
            assert track["residuals"] is not None
            np.testing.assert_array_almost_equal(track["residuals"], residuals)

    def test_append_track_state(self, temp_h5, sample_track_data):
        """Test appending states to existing track."""
        from pytcl.io.hdf5_track_storage import TrackHDF5Storage

        states, covs, times = sample_track_data

        with TrackHDF5Storage(temp_h5) as store:
            store.open(mode="w")
            store.store_track("trk_001", states, covs, times)
            store.append_track_state(
                "trk_001",
                np.array([99.0, 99.0, 99.0, 99.0]),
                np.eye(4) * 2.0,
                11.0,
            )

        with TrackHDF5Storage(temp_h5) as store:
            store.open(mode="r")
            track = store.retrieve_track("trk_001")
            assert len(track["timestamps"]) == len(times) + 1
            assert track["timestamps"][-1] == 11.0

    # === Single Detection ===

    def test_store_retrieve_detection(self, temp_h5):
        """Test storing and retrieving a detection."""
        from pytcl.io.hdf5_track_storage import TrackHDF5Storage

        with TrackHDF5Storage(temp_h5) as store:
            store.open(mode="w")
            store.store_detection("det_001", np.array([1.0, 2.0]), 5.0, "radar")

        with TrackHDF5Storage(temp_h5) as store:
            store.open(mode="r")
            det = store.retrieve_detection("det_001")
            np.testing.assert_array_equal(det["measurement"], [1.0, 2.0])
            assert det["timestamp"] == 5.0
            assert det["sensor_id"] == "radar"

    def test_store_detection_with_covariance(self, temp_h5):
        """Test storing detection with covariance."""
        from pytcl.io.hdf5_track_storage import TrackHDF5Storage

        cov = np.array([[1.0, 0.1], [0.1, 1.0]])

        with TrackHDF5Storage(temp_h5) as store:
            store.open(mode="w")
            store.store_detection(
                "det_001",
                np.array([1.0, 2.0]),
                5.0,
                "radar",
                covariance=cov,
            )

        with TrackHDF5Storage(temp_h5) as store:
            store.open(mode="r")
            det = store.retrieve_detection("det_001")
            np.testing.assert_array_almost_equal(det["covariance"], cov)

    # === Time-Series Queries ===

    def test_get_track_trajectory_full(self, populated_h5, sample_track_data):
        """Test getting full trajectory."""
        from pytcl.io.hdf5_track_storage import TrackHDF5Storage

        states, _, times = sample_track_data

        with TrackHDF5Storage(populated_h5) as store:
            store.open(mode="r")
            traj = store.get_track_trajectory("trk_001", scenario_id="scenario_1")
            assert len(traj["timestamps"]) == len(times)

    def test_get_track_trajectory_time_range(self, populated_h5):
        """Test getting trajectory within time range."""
        from pytcl.io.hdf5_track_storage import TrackHDF5Storage

        with TrackHDF5Storage(populated_h5) as store:
            store.open(mode="r")
            traj = store.get_track_trajectory(
                "trk_001",
                start_time=2.0,
                end_time=5.0,
                scenario_id="scenario_1",
            )
            assert all(2.0 <= t <= 5.0 for t in traj["timestamps"])
            assert len(traj["timestamps"]) > 0

    def test_get_state_at_time_nearest(self, populated_h5):
        """Test getting nearest state at a time."""
        from pytcl.io.hdf5_track_storage import TrackHDF5Storage

        with TrackHDF5Storage(populated_h5) as store:
            store.open(mode="r")
            result = store.get_state_at_time("trk_001", 5.0, scenario_id="scenario_1")
            assert result["state"].shape == (4,)
            assert abs(result["timestamp"] - 5.0) < 0.3

    def test_get_state_at_time_interpolated(self, populated_h5):
        """Test interpolated state retrieval."""
        from pytcl.io.hdf5_track_storage import TrackHDF5Storage

        with TrackHDF5Storage(populated_h5) as store:
            store.open(mode="r")
            result = store.get_state_at_time(
                "trk_001", 5.0, interpolate=True, scenario_id="scenario_1"
            )
            assert result["state"].shape == (4,)
            assert result["timestamp"] == 5.0

    # === Spatial Queries ===

    def test_get_tracks_in_region(self, populated_h5):
        """Test finding tracks in a spatial region."""
        from pytcl.io.hdf5_track_storage import TrackHDF5Storage

        with TrackHDF5Storage(populated_h5) as store:
            store.open(mode="r")
            # trk_002 has states offset by +10, so a large bbox should find both
            tracks = store.get_tracks_in_region(
                [-100, -100, 100, 100],
                state_indices=(0, 2),
                scenario_id="scenario_1",
            )
            assert len(tracks) == 2

    def test_get_tracks_in_region_filtered(self, populated_h5):
        """Test spatial query excludes tracks outside region."""
        from pytcl.io.hdf5_track_storage import TrackHDF5Storage

        with TrackHDF5Storage(populated_h5) as store:
            store.open(mode="r")
            # Use a region that only includes trk_002 (offset by +10)
            tracks = store.get_tracks_in_region(
                [8, 8, 15, 15],
                state_indices=(0, 2),
                scenario_id="scenario_1",
            )
            assert "trk_002" in tracks

    # === Scenario Operations ===

    def test_store_retrieve_scenario(self, populated_h5):
        """Test full scenario round-trip."""
        from pytcl.io.hdf5_track_storage import TrackHDF5Storage

        with TrackHDF5Storage(populated_h5) as store:
            store.open(mode="r")
            scenario = store.retrieve_tracking_scenario("scenario_1")

            assert len(scenario["tracks"]) == 2
            assert len(scenario["detections"]) == 2
            assert "trk_001" in scenario["tracks"]
            assert "det_001" in scenario["detections"]

    def test_list_scenarios(self, populated_h5):
        """Test listing scenarios."""
        from pytcl.io.hdf5_track_storage import TrackHDF5Storage

        with TrackHDF5Storage(populated_h5) as store:
            store.open(mode="r")
            scenarios = store.list_scenarios()
            assert "scenario_1" in scenarios

    def test_list_tracks_in_scenario(self, populated_h5):
        """Test listing tracks within a scenario."""
        from pytcl.io.hdf5_track_storage import TrackHDF5Storage

        with TrackHDF5Storage(populated_h5) as store:
            store.open(mode="r")
            tracks = store.list_tracks(scenario_id="scenario_1")
            assert set(tracks) == {"trk_001", "trk_002"}

    def test_compare_scenarios(self, temp_h5, sample_track_data):
        """Test scenario comparison."""
        from pytcl.io.hdf5_track_storage import TrackHDF5Storage

        states, covs, times = sample_track_data

        with TrackHDF5Storage(temp_h5) as store:
            store.open(mode="w")
            store.store_tracking_scenario(
                "s1",
                {
                    "trk_001": {
                        "states": states,
                        "covariances": covs,
                        "timestamps": times,
                    }
                },
            )
            store.store_tracking_scenario(
                "s2",
                {
                    "trk_001": {
                        "states": states + 0.01,
                        "covariances": covs,
                        "timestamps": times,
                    },
                    "trk_003": {
                        "states": states,
                        "covariances": covs,
                        "timestamps": times,
                    },
                },
            )

        with TrackHDF5Storage(temp_h5) as store:
            store.open(mode="r")
            comp = store.compare_scenarios("s1", "s2")

            assert "trk_001" in comp["common_tracks"]
            assert "trk_003" in comp["unique_to_2"]
            assert comp["state_differences"]["trk_001"] > 0

    # === Compression & Chunking ===

    def test_compression_ratio(self, temp_h5):
        """Test that compression reduces file size."""
        import os

        from pytcl.io.hdf5_track_storage import TrackHDF5Storage

        n = 5000
        states = np.zeros((n, 6))  # Highly compressible
        covs = np.array([np.eye(6) for _ in range(n)])
        times = np.arange(n, dtype=np.float64)

        with TrackHDF5Storage(
            temp_h5, compression="gzip", compression_level=4
        ) as store:
            store.open(mode="w")
            store.store_track("trk_big", states, covs, times)

        file_size = os.path.getsize(temp_h5)
        uncompressed_size = states.nbytes + covs.nbytes + times.nbytes
        ratio = uncompressed_size / file_size
        assert ratio > 2.0  # At least 2x compression for zeros

    def test_chunked_storage(self, temp_h5):
        """Test that datasets are created with chunks."""
        from pytcl.io.hdf5_track_storage import TrackHDF5Storage

        n = 2000
        states = np.random.randn(n, 4)
        covs = np.array([np.eye(4) for _ in range(n)])
        times = np.arange(n, dtype=np.float64)

        with TrackHDF5Storage(temp_h5, chunk_size=500) as store:
            store.open(mode="w")
            store.store_track("trk_001", states, covs, times)

        with h5py.File(temp_h5, "r") as f:
            ds = f["tracks/trk_001/state_history"]
            assert ds.chunks is not None
            assert ds.chunks[0] == 500

    # === Export/Import ===

    def test_export_to_sql(self, populated_h5):
        """Test exporting HDF5 data to SQL."""
        from pytcl.io.hdf5_track_storage import TrackHDF5Storage
        from pytcl.io.track_database import TrackDatabaseManager

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        with TrackHDF5Storage(populated_h5) as h5_store:
            h5_store.open(mode="r")
            with TrackDatabaseManager(db_path) as db:
                db.open(mode="w")
                h5_store.export_to_sql(db, scenario_id="scenario_1")

                tracks = db.retrieve_all_tracks()
                assert len(tracks) == 2

                dets = db.retrieve_all_detections()
                assert len(dets) == 2

    def test_import_from_sql(self, temp_h5):
        """Test importing SQL data into HDF5."""
        from pytcl.io.hdf5_track_storage import TrackHDF5Storage
        from pytcl.io.track_database import TrackDatabaseManager

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        # Populate SQL
        with TrackDatabaseManager(db_path) as db:
            db.open(mode="w")
            db.store_detection("det_001", np.array([1.0, 2.0]), "radar", 0.0)
            db.initiate_track("trk_001", np.array([1.0, 2.0]), np.eye(2), 0.0)
            db.update_track_state("trk_001", np.array([1.1, 2.1]), np.eye(2) * 0.9, 1.0)

        # Import into HDF5
        with TrackDatabaseManager(db_path) as db:
            db.open(mode="r")
            with TrackHDF5Storage(temp_h5) as h5_store:
                h5_store.open(mode="w")
                h5_store.import_from_sql(db, "imported")

        # Verify
        with TrackHDF5Storage(temp_h5) as h5_store:
            h5_store.open(mode="r")
            scenarios = h5_store.list_scenarios()
            assert "imported" in scenarios

            tracks = h5_store.list_tracks(scenario_id="imported")
            assert "trk_001" in tracks

            track = h5_store.retrieve_track("trk_001", scenario_id="imported")
            assert len(track["timestamps"]) == 2
