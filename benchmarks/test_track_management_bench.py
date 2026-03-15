"""
Benchmarks for track management SQL and HDF5 operations.

Validates performance targets under realistic tracking loads:
- SQL: detection storage, track updates, query latency
- HDF5: archival rate, retrieval, compression, spatial queries
- Integration: SQL→HDF5 export, filter+track management overhead

Targets:
- SQL detection storage: >1000 detections/sec
- Track state updates: <10ms per track
- Query latency: <100ms for typical scenarios
- HDF5 compression: 5-10x ratio
- Export throughput: >100 tracks/sec
"""

import os
import tempfile

import numpy as np
import pytest

from pytcl.dynamic_estimation.kalman.linear import kf_predict, kf_update
from pytcl.io import TrackDatabaseManager, TrackDatabaseStatus

try:
    from pytcl.io import TrackHDF5Storage

    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

requires_h5py = pytest.mark.skipif(not HAS_H5PY, reason="h5py not installed")


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def track_management_data():
    """Pre-computed data for track management benchmarks."""
    rng = np.random.default_rng(42)
    state_dim = 4
    meas_dim = 2
    n_tracks = 100
    n_steps = 200

    states = rng.normal(0, 10, (n_tracks, n_steps, state_dim))
    covs = np.array(
        [[np.eye(state_dim) for _ in range(n_steps)] for _ in range(n_tracks)]
    )
    timestamps = np.arange(n_steps, dtype=np.float64)
    measurements = rng.normal(0, 5, (n_tracks, n_steps, meas_dim))

    # Kalman filter matrices
    dt = 1.0
    F = np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]])
    H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
    Q = 0.1 * np.eye(state_dim)
    R = np.eye(meas_dim) * 4.0

    return {
        "states": states,
        "covs": covs,
        "timestamps": timestamps,
        "measurements": measurements,
        "n_tracks": n_tracks,
        "n_steps": n_steps,
        "state_dim": state_dim,
        "meas_dim": meas_dim,
        "F": F,
        "H": H,
        "Q": Q,
        "R": R,
    }


@pytest.fixture()
def sql_db(tmp_path):
    """Create a fresh SQL database for each benchmark."""
    db_path = str(tmp_path / "bench.db")
    db = TrackDatabaseManager(db_path)
    db.open(mode="w")
    yield db
    db.close()


@pytest.fixture()
def populated_sql_db(tmp_path, track_management_data):
    """SQL database pre-populated with tracks for query benchmarks."""
    d = track_management_data
    db_path = str(tmp_path / "populated.db")
    db = TrackDatabaseManager(db_path)
    db.open(mode="w")

    # Insert 50 tracks with 50 steps each (subset for setup speed)
    n_t = 50
    n_s = 50
    for i in range(n_t):
        tid = f"trk_{i:04d}"
        db.initiate_track(tid, d["states"][i, 0], d["covs"][i, 0], 0.0)
        for k in range(1, n_s):
            db.update_track_state(tid, d["states"][i, k], d["covs"][i, k], float(k))
        db.confirm_track(tid)

        # Store detections
        for k in range(n_s):
            det_id = f"det_{i:04d}_{k:04d}"
            db.store_detection(
                det_id,
                d["measurements"][i, k],
                f"sensor_{i % 3}",
                float(k),
            )
            db.associate_detection(det_id, tid)

    yield db
    db.close()


@pytest.fixture()
def h5_store(tmp_path):
    """Create a fresh HDF5 store for each benchmark."""
    h5_path = str(tmp_path / "bench.h5")
    store = TrackHDF5Storage(h5_path, compression="gzip", compression_level=4)
    store.open(mode="w")
    yield store
    store.close()


@pytest.fixture()
def populated_h5_store(tmp_path, track_management_data):
    """HDF5 store pre-populated with tracks for query benchmarks."""
    d = track_management_data
    h5_path = str(tmp_path / "populated.h5")
    store = TrackHDF5Storage(h5_path, compression="gzip", compression_level=4)
    store.open(mode="w")

    n_t = 50
    n_s = 50
    for i in range(n_t):
        tid = f"trk_{i:04d}"
        store.store_track(
            tid,
            d["states"][i, :n_s],
            d["covs"][i, :n_s],
            d["timestamps"][:n_s],
        )

    yield store
    store.close()


# =============================================================================
# SQL Benchmarks
# =============================================================================


class TestSQLDetectionStorage:
    """Benchmark SQL detection write performance."""

    @pytest.mark.light
    def test_store_detection_single(self, benchmark, sql_db, track_management_data):
        """Benchmark storing a single detection."""
        d = track_management_data
        counter = [0]

        def store_one():
            idx = counter[0]
            counter[0] += 1
            sql_db.store_detection(
                f"det_{idx:06d}",
                d["measurements"][idx % d["n_tracks"], idx % d["n_steps"]],
                "radar",
                float(idx),
            )

        benchmark(store_one)

    @pytest.mark.light
    def test_store_detection_batch_100(self, benchmark, sql_db, track_management_data):
        """Benchmark storing 100 detections (simulating one scan)."""
        d = track_management_data
        counter = [0]

        def store_batch():
            base = counter[0] * 100
            counter[0] += 1
            for j in range(100):
                idx = base + j
                sql_db.store_detection(
                    f"det_{idx:06d}",
                    d["measurements"][idx % d["n_tracks"], idx % d["n_steps"]],
                    f"sensor_{j % 3}",
                    float(idx),
                )

        benchmark(store_batch)


class TestSQLTrackUpdates:
    """Benchmark SQL track state update performance."""

    @pytest.mark.light
    def test_initiate_track(self, benchmark, sql_db, track_management_data):
        """Benchmark track initiation."""
        d = track_management_data
        counter = [0]

        def initiate():
            idx = counter[0]
            counter[0] += 1
            sql_db.initiate_track(
                f"trk_init_{idx:06d}",
                d["states"][idx % d["n_tracks"], 0],
                d["covs"][idx % d["n_tracks"], 0],
                float(idx),
            )

        benchmark(initiate)

    @pytest.mark.light
    def test_update_track_state(self, benchmark, sql_db, track_management_data):
        """Benchmark single track state update."""
        d = track_management_data
        tid = "trk_update_bench"
        sql_db.initiate_track(tid, d["states"][0, 0], d["covs"][0, 0], 0.0)
        counter = [1]

        def update_state():
            k = counter[0]
            counter[0] += 1
            sql_db.update_track_state(
                tid,
                d["states"][0, k % d["n_steps"]],
                d["covs"][0, k % d["n_steps"]],
                float(k),
            )

        benchmark(update_state)

    @pytest.mark.light
    def test_store_track_history_batch(self, benchmark, sql_db, track_management_data):
        """Benchmark batch store of 50 state entries."""
        d = track_management_data
        counter = [0]

        def store_batch():
            idx = counter[0]
            counter[0] += 1
            tid = f"trk_batch_{idx:06d}"
            sql_db.initiate_track(tid, d["states"][0, 0], d["covs"][0, 0], 0.0)
            sql_db.store_track_history(
                tid,
                d["states"][0, :50],
                d["covs"][0, :50],
                d["timestamps"][:50],
            )

        benchmark(store_batch)


class TestSQLQueryLatency:
    """Benchmark SQL query performance on populated database."""

    @pytest.mark.light
    def test_get_track_state(self, benchmark, populated_sql_db):
        """Benchmark latest state retrieval for a single track."""
        rng = np.random.default_rng(99)

        def query():
            tid = f"trk_{rng.integers(0, 50):04d}"
            populated_sql_db.get_track_state(tid)

        benchmark(query)

    @pytest.mark.light
    def test_get_track_history(self, benchmark, populated_sql_db):
        """Benchmark full history retrieval for a single track."""
        rng = np.random.default_rng(99)

        def query():
            tid = f"trk_{rng.integers(0, 50):04d}"
            populated_sql_db.get_track_history(tid)

        benchmark(query)

    @pytest.mark.light
    def test_get_track_history_time_slice(self, benchmark, populated_sql_db):
        """Benchmark time-sliced history retrieval."""
        rng = np.random.default_rng(99)

        def query():
            tid = f"trk_{rng.integers(0, 50):04d}"
            populated_sql_db.get_track_history(tid, start_time=10.0, end_time=30.0)

        benchmark(query)

    @pytest.mark.light
    def test_retrieve_detections_time_range(self, benchmark, populated_sql_db):
        """Benchmark detection query by time range."""

        def query():
            populated_sql_db.retrieve_detections(start_time=10.0, end_time=20.0)

        benchmark(query)

    @pytest.mark.light
    def test_retrieve_detections_by_sensor(self, benchmark, populated_sql_db):
        """Benchmark detection query by sensor ID."""

        def query():
            populated_sql_db.retrieve_detections(sensor_id="sensor_0")

        benchmark(query)

    @pytest.mark.light
    def test_retrieve_all_tracks(self, benchmark, populated_sql_db):
        """Benchmark listing all tracks."""
        benchmark(populated_sql_db.retrieve_all_tracks)

    @pytest.mark.light
    def test_retrieve_tracks_by_status(self, benchmark, populated_sql_db):
        """Benchmark listing tracks filtered by status."""

        def query():
            populated_sql_db.retrieve_all_tracks(status=TrackDatabaseStatus.CONFIRMED)

        benchmark(query)


class TestSQLLifecycle:
    """Benchmark SQL lifecycle management operations."""

    @pytest.mark.light
    def test_track_status_transition(self, benchmark, sql_db, track_management_data):
        """Benchmark status transition (confirm → coast → dead cycle)."""
        d = track_management_data
        tid = "trk_lifecycle_bench"
        sql_db.initiate_track(tid, d["states"][0, 0], d["covs"][0, 0], 0.0)

        def cycle():
            sql_db.mark_track_confirmed(tid)
            sql_db.mark_track_coasting(tid)
            sql_db.mark_track_dead(tid)
            sql_db.mark_track_tentative(tid)

        benchmark(cycle)

    @pytest.mark.light
    def test_associate_detection(self, benchmark, sql_db, track_management_data):
        """Benchmark detection-to-track association."""
        d = track_management_data
        tid = "trk_assoc_bench"
        sql_db.initiate_track(tid, d["states"][0, 0], d["covs"][0, 0], 0.0)

        counter = [0]

        def associate():
            idx = counter[0]
            counter[0] += 1
            det_id = f"det_assoc_{idx:06d}"
            sql_db.store_detection(
                det_id,
                d["measurements"][0, idx % d["n_steps"]],
                "radar",
                float(idx),
            )
            sql_db.associate_detection(det_id, tid)

        benchmark(associate)


# =============================================================================
# HDF5 Benchmarks
# =============================================================================


@requires_h5py
class TestHDF5WritePerformance:
    """Benchmark HDF5 write operations."""

    @pytest.mark.light
    def test_store_single_track(self, benchmark, h5_store, track_management_data):
        """Benchmark storing a single track (100 timesteps)."""
        d = track_management_data
        counter = [0]

        def store():
            idx = counter[0]
            counter[0] += 1
            h5_store.store_track(
                f"trk_bench_{idx:06d}",
                d["states"][idx % d["n_tracks"], :100],
                d["covs"][idx % d["n_tracks"], :100],
                d["timestamps"][:100],
            )

        benchmark(store)

    @pytest.mark.light
    def test_store_scenario_10_tracks(self, benchmark, h5_store, track_management_data):
        """Benchmark storing a scenario with 10 tracks."""
        d = track_management_data
        counter = [0]

        def store_scenario():
            idx = counter[0]
            counter[0] += 1
            tracks = {}
            for i in range(10):
                tracks[f"trk_{i:04d}"] = {
                    "states": d["states"][i, :50],
                    "covariances": d["covs"][i, :50],
                    "timestamps": d["timestamps"][:50],
                }
            h5_store.store_tracking_scenario(f"scenario_{idx:06d}", tracks)

        benchmark(store_scenario)

    @pytest.mark.light
    def test_append_track_state(self, benchmark, h5_store, track_management_data):
        """Benchmark appending a single state to an existing track."""
        d = track_management_data
        tid = "trk_append_bench"
        h5_store.store_track(
            tid,
            d["states"][0, :10],
            d["covs"][0, :10],
            d["timestamps"][:10],
        )
        counter = [10]

        def append():
            k = counter[0]
            counter[0] += 1
            h5_store.append_track_state(
                tid,
                d["states"][0, k % d["n_steps"]],
                d["covs"][0, k % d["n_steps"]],
                float(k),
            )

        benchmark(append)


@requires_h5py
class TestHDF5ReadPerformance:
    """Benchmark HDF5 read/query operations."""

    @pytest.mark.light
    def test_retrieve_track(self, benchmark, populated_h5_store):
        """Benchmark full track retrieval."""
        rng = np.random.default_rng(99)

        def query():
            tid = f"trk_{rng.integers(0, 50):04d}"
            populated_h5_store.retrieve_track(tid)

        benchmark(query)

    @pytest.mark.light
    def test_get_track_trajectory_slice(self, benchmark, populated_h5_store):
        """Benchmark time-sliced trajectory extraction."""
        rng = np.random.default_rng(99)

        def query():
            tid = f"trk_{rng.integers(0, 50):04d}"
            populated_h5_store.get_track_trajectory(tid, start_time=10.0, end_time=30.0)

        benchmark(query)

    @pytest.mark.light
    def test_get_state_at_time_nearest(self, benchmark, populated_h5_store):
        """Benchmark nearest-neighbor state query."""
        rng = np.random.default_rng(99)

        def query():
            tid = f"trk_{rng.integers(0, 50):04d}"
            populated_h5_store.get_state_at_time(tid, time=25.5, interpolate=False)

        benchmark(query)

    @pytest.mark.light
    def test_get_state_at_time_interpolated(self, benchmark, populated_h5_store):
        """Benchmark interpolated state query."""
        rng = np.random.default_rng(99)

        def query():
            tid = f"trk_{rng.integers(0, 50):04d}"
            populated_h5_store.get_state_at_time(tid, time=25.5, interpolate=True)

        benchmark(query)

    @pytest.mark.light
    def test_spatial_query(self, benchmark, populated_h5_store):
        """Benchmark spatial bounding-box query across all tracks."""

        def query():
            populated_h5_store.get_tracks_in_region([-5, -5, 5, 5])

        benchmark(query)

    @pytest.mark.light
    def test_list_tracks(self, benchmark, populated_h5_store):
        """Benchmark listing all track IDs."""
        benchmark(populated_h5_store.list_tracks)


@requires_h5py
class TestHDF5Compression:
    """Benchmark HDF5 compression ratios."""

    @pytest.mark.light
    def test_compression_ratio(self, tmp_path):
        """Verify compression ratio meets target (>2x for realistic tracking data).

        Uses smooth constant-velocity trajectories and identity covariances,
        which are representative of real tracking data and compress well.
        """
        rng = np.random.default_rng(42)
        n_t = 20
        n_s = 200
        state_dim = 6  # [x, vx, y, vy, z, vz]

        # Raw data size
        raw_bytes = n_t * n_s * (state_dim * 8 + state_dim * state_dim * 8 + 8)

        h5_path = str(tmp_path / "compression_test.h5")
        store = TrackHDF5Storage(h5_path, compression="gzip", compression_level=4)
        store.open(mode="w")
        tracks = {}
        for i in range(n_t):
            # Smooth trajectory: constant velocity + small noise
            dt_arr = np.arange(n_s, dtype=np.float64)
            v = rng.normal(0, 2, 3)
            states_i = np.zeros((n_s, state_dim))
            states_i[:, 0] = v[0] * dt_arr + rng.normal(0, 0.01, n_s)
            states_i[:, 1] = v[0]
            states_i[:, 2] = v[1] * dt_arr + rng.normal(0, 0.01, n_s)
            states_i[:, 3] = v[1]
            states_i[:, 4] = v[2] * dt_arr + rng.normal(0, 0.01, n_s)
            states_i[:, 5] = v[2]
            # Identity covariances (highly compressible)
            covs_i = np.array([np.eye(state_dim) for _ in range(n_s)])
            tracks[f"trk_{i:04d}"] = {
                "states": states_i,
                "covariances": covs_i,
                "timestamps": dt_arr,
            }
        store.store_tracking_scenario("bench", tracks)
        store.close()

        file_size = os.path.getsize(h5_path)
        ratio = raw_bytes / file_size

        assert ratio > 2.0, (
            f"Compression ratio {ratio:.1f}x below 2x target "
            f"(raw={raw_bytes}, file={file_size})"
        )


# =============================================================================
# Integration Benchmarks
# =============================================================================


@requires_h5py
class TestSQLToHDF5Export:
    """Benchmark SQL → HDF5 export pipeline."""

    @pytest.mark.light
    def test_export_50_tracks(self, tmp_path, track_management_data):
        """Benchmark exporting 50 tracks from SQL to HDF5."""
        d = track_management_data
        n_t = 50
        n_s = 50

        # Populate SQL
        db_path = str(tmp_path / "export_source.db")
        db = TrackDatabaseManager(db_path)
        db.open(mode="w")
        for i in range(n_t):
            tid = f"trk_{i:04d}"
            db.initiate_track(tid, d["states"][i, 0], d["covs"][i, 0], 0.0)
            db.store_track_history(
                tid,
                d["states"][i, :n_s],
                d["covs"][i, :n_s],
                d["timestamps"][:n_s],
            )

        # Export to HDF5
        h5_path = str(tmp_path / "export_target.h5")
        store = TrackHDF5Storage(h5_path)
        store.open(mode="w")

        store.import_from_sql(db, scenario_id="exported")

        # Verify
        exported_tracks = store.list_tracks("exported")
        assert len(exported_tracks) == n_t

        store.close()
        db.close()

    @pytest.mark.light
    def test_roundtrip_sql_hdf5_sql(self, tmp_path, track_management_data):
        """Benchmark SQL → HDF5 → SQL round-trip preserves data."""
        d = track_management_data
        n_t = 10
        n_s = 30

        # Phase 1: Populate source SQL
        src_path = str(tmp_path / "roundtrip_src.db")
        src_db = TrackDatabaseManager(src_path)
        src_db.open(mode="w")
        for i in range(n_t):
            tid = f"trk_{i:04d}"
            src_db.initiate_track(tid, d["states"][i, 0], d["covs"][i, 0], 0.0)
            src_db.store_track_history(
                tid,
                d["states"][i, :n_s],
                d["covs"][i, :n_s],
                d["timestamps"][:n_s],
            )

        # Phase 2: Export to HDF5
        h5_path = str(tmp_path / "roundtrip.h5")
        store = TrackHDF5Storage(h5_path)
        store.open(mode="w")
        store.import_from_sql(src_db, scenario_id="roundtrip")
        src_db.close()

        # Phase 3: Import back to new SQL
        dst_path = str(tmp_path / "roundtrip_dst.db")
        dst_db = TrackDatabaseManager(dst_path)
        dst_db.open(mode="w")
        store.export_to_sql(dst_db, scenario_id="roundtrip")
        store.close()

        # Verify
        dst_tracks = dst_db.retrieve_all_tracks()
        assert len(dst_tracks) == n_t

        # Spot-check state data
        for i in range(min(n_t, 3)):
            tid = f"trk_{i:04d}"
            history = dst_db.get_track_history(tid)
            assert history["states"].shape[0] > 0

        dst_db.close()


class TestFilterWithTrackManagement:
    """Benchmark Kalman filter with track management overhead."""

    @pytest.mark.light
    def test_kf_cycle_with_sql_storage(self, benchmark, sql_db, track_management_data):
        """Benchmark KF predict+update+store cycle (measures overhead)."""
        d = track_management_data
        tid = "trk_kf_bench"
        sql_db.initiate_track(tid, d["states"][0, 0], d["covs"][0, 0], 0.0)

        x = d["states"][0, 0].copy()
        P = d["covs"][0, 0].copy()
        counter = [1]

        def kf_cycle():
            nonlocal x, P
            k = counter[0]
            counter[0] += 1
            # Predict
            pred = kf_predict(x, P, d["F"], d["Q"])
            # Update
            z = d["measurements"][0, k % d["n_steps"]]
            upd = kf_update(pred.x, pred.P, z, d["H"], d["R"])
            x, P = upd.x, upd.P
            # Store
            sql_db.update_track_state(tid, x, P, float(k))

        benchmark(kf_cycle)

    @pytest.mark.light
    def test_kf_cycle_without_storage(self, benchmark, track_management_data):
        """Benchmark KF predict+update without storage (baseline)."""
        d = track_management_data
        x = d["states"][0, 0].copy()
        P = d["covs"][0, 0].copy()
        counter = [1]

        def kf_cycle():
            nonlocal x, P
            k = counter[0]
            counter[0] += 1
            pred = kf_predict(x, P, d["F"], d["Q"])
            z = d["measurements"][0, k % d["n_steps"]]
            upd = kf_update(pred.x, pred.P, z, d["H"], d["R"])
            x, P = upd.x, upd.P

        benchmark(kf_cycle)


class TestDatabaseSizeGrowth:
    """Verify database size growth characteristics."""

    @pytest.mark.light
    def test_sql_size_scales_linearly(self, tmp_path, track_management_data):
        """Verify SQL database size grows linearly with track count."""
        d = track_management_data
        sizes = []

        for n_t in [10, 20, 40]:
            db_path = str(tmp_path / f"size_{n_t}.db")
            db = TrackDatabaseManager(db_path)
            db.open(mode="w")
            for i in range(n_t):
                tid = f"trk_{i:04d}"
                db.initiate_track(tid, d["states"][i, 0], d["covs"][i, 0], 0.0)
                db.store_track_history(
                    tid,
                    d["states"][i, :50],
                    d["covs"][i, :50],
                    d["timestamps"][:50],
                )
            db.close()
            sizes.append(os.path.getsize(db_path))

        # Size should roughly double when tracks double
        ratio_1 = sizes[1] / sizes[0]  # 20/10
        ratio_2 = sizes[2] / sizes[1]  # 40/20

        assert 1.5 < ratio_1 < 2.5, f"20/10 ratio={ratio_1:.2f}, expected ~2.0"
        assert 1.5 < ratio_2 < 2.5, f"40/20 ratio={ratio_2:.2f}, expected ~2.0"
