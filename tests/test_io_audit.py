"""Correctness audit tests for pytcl.io.

Behavioral verification of every public function/method in pytcl/io/
against documented contracts, using round-trips and plain-Python ground
truth (list comprehensions, hand-computed interpolation, brute-force
region tests). No external numerical reference is required.
"""

import os
import pickle
from collections import namedtuple

import numpy as np
import pytest

from pytcl.io import (
    HDF5Storage,
    MigrationHelper,
    SQLStorage,
    TrackDatabaseManager,
    TrackDatabaseStatus,
    TrackHDF5Storage,
)
from pytcl.io.compat import (
    EKFTrackAdapter,
    IMMTrackAdapter,
    KalmanTrackAdapter,
    ParticleFilterTrackAdapter,
    TrackerDatabaseAdapter,
    UKFTrackAdapter,
    store_filter_result,
)
from pytcl.io.migration import AnalysisResult
from pytcl.io.storage import StorageBackend

pytest.importorskip("h5py")

# =============================================================================
# StorageBackend interface
# =============================================================================


class TestStorageBackendInterface:
    def test_is_abstract(self):
        with pytest.raises(TypeError):
            StorageBackend()

    def test_concrete_backends_implement_interface(self):
        abstract = {
            name
            for name in dir(StorageBackend)
            if getattr(getattr(StorageBackend, name), "__isabstractmethod__", False)
        }
        assert abstract == {
            "open",
            "close",
            "__enter__",
            "__exit__",
            "store_array",
            "retrieve_array",
            "store_scalar",
            "retrieve_scalar",
            "store_group",
            "list_keys",
            "get_metadata",
            "delete",
            "flush",
        }
        for cls in (HDF5Storage, SQLStorage):
            assert issubclass(cls, StorageBackend)
            for name in abstract:
                assert getattr(cls, name) is not getattr(StorageBackend, name)


# =============================================================================
# HDF5Storage
# =============================================================================


@pytest.fixture
def h5_store(tmp_path):
    store = HDF5Storage()
    store.open(str(tmp_path / "audit.h5"), mode="w")
    yield store
    store.close()


class TestHDF5StorageRoundTrip:
    @pytest.mark.parametrize(
        "arr",
        [
            np.arange(12, dtype=np.float64).reshape(3, 4),
            np.arange(6, dtype=np.float32),
            np.arange(8, dtype=np.int32).reshape(2, 2, 2),
            np.array([0, 255, 128], dtype=np.uint8),
            np.array([True, False, True]),
            np.array([1 + 2j, 3 - 4j], dtype=np.complex128),
            np.array([np.nan, np.inf, -np.inf, 0.0, -0.0]),
            np.empty((0, 3), dtype=np.float64),
            np.float64(3.5) * np.ones((500, 40)),
        ],
        ids=[
            "f8_2d",
            "f4_1d",
            "i4_3d",
            "u1",
            "bool",
            "c16",
            "nan_inf",
            "empty",
            "large",
        ],
    )
    def test_array_bit_identical(self, h5_store, arr):
        h5_store.store_array("data", arr)
        out = h5_store.retrieve_array("data")
        assert out.dtype == arr.dtype
        assert out.shape == arr.shape
        np.testing.assert_array_equal(out, arr)
        h5_store.delete("data")

    @pytest.mark.parametrize(
        "value",
        [42, -7, 3.14159, float("inf"), True, False, "hello", "ünïcødé ✓", ""],
        ids=["int", "neg_int", "float", "inf", "true", "false", "str", "unicode", ""],
    )
    def test_scalar_round_trip(self, h5_store, value):
        h5_store.store_scalar("s", value)
        out = h5_store.retrieve_scalar("s")
        assert out == value
        assert isinstance(out, type(value))
        h5_store.delete("s")

    def test_scalar_nan(self, h5_store):
        h5_store.store_scalar("s", float("nan"))
        assert np.isnan(h5_store.retrieve_scalar("s"))

    def test_metadata_round_trip(self, h5_store):
        meta = {"units": "m", "count": 3, "scale": 1.5, "valid": True}
        h5_store.store_array("d", np.zeros(2), metadata=meta)
        out = h5_store.get_metadata("d")
        for k, v in meta.items():
            assert out[k] == v

    def test_string_metadata_not_json_mangled(self, h5_store):
        """Regression: strings that parse as JSON must round-trip as strings."""
        meta = {"note": "123", "flag": "true", "obj": "[1, 2]", "name": "hello"}
        h5_store.store_array("d", np.zeros(2), metadata=meta)
        out = h5_store.get_metadata("d")
        assert out == meta

    def test_nonserializable_metadata_stored_as_str(self, h5_store):
        h5_store.store_array("d", np.zeros(2), metadata={"cfg": {"a": 1}})
        assert h5_store.get_metadata("d")["cfg"] == str({"a": 1})


class TestHDF5StorageContract:
    def test_nested_paths_and_list_keys(self, h5_store):
        h5_store.store_array("gravity/egm96/coeffs", np.ones(4))
        h5_store.store_scalar("gravity/version", 2)
        assert sorted(h5_store.list_keys("gravity")) == ["egm96", "version"]
        assert "gravity" in h5_store.list_keys("/")

    def test_store_group_with_metadata(self, h5_store):
        h5_store.store_group("mission", metadata={"name": "test", "n": 5})
        meta = h5_store.get_metadata("mission")
        assert meta["name"] == "test"
        assert meta["n"] == 5
        # Idempotent re-creation
        h5_store.store_group("mission", metadata={"extra": 1})
        assert h5_store.get_metadata("mission")["extra"] == 1

    def test_delete(self, h5_store):
        h5_store.store_array("d", np.zeros(3))
        h5_store.delete("d")
        with pytest.raises(KeyError):
            h5_store.retrieve_array("d")
        h5_store.delete("d")  # deleting a missing name is a no-op

    def test_missing_keys_raise_keyerror(self, h5_store):
        with pytest.raises(KeyError):
            h5_store.retrieve_array("nope")
        with pytest.raises(KeyError):
            h5_store.retrieve_scalar("nope")
        with pytest.raises(KeyError):
            h5_store.get_metadata("nope")
        with pytest.raises(KeyError):
            h5_store.list_keys("nope")

    def test_not_open_raises_runtimeerror(self):
        store = HDF5Storage()
        for call in [
            lambda: store.store_array("a", np.zeros(2)),
            lambda: store.retrieve_array("a"),
            lambda: store.store_scalar("a", 1),
            lambda: store.retrieve_scalar("a"),
            lambda: store.store_group("g"),
            lambda: store.list_keys(),
            lambda: store.get_metadata("a"),
            lambda: store.delete("a"),
        ]:
            with pytest.raises(RuntimeError):
                call()

    def test_double_close_and_flush(self, tmp_path):
        store = HDF5Storage()
        store.flush()  # no-op when never opened
        store.open(str(tmp_path / "f.h5"), mode="w")
        store.store_array("a", np.ones(2))
        store.flush()
        store.close()
        store.close()  # double close is safe

    def test_reopen_append_vs_write(self, tmp_path):
        path = str(tmp_path / "modes.h5")
        store = HDF5Storage()
        store.open(path, mode="w")
        store.store_array("keep", np.arange(3))
        store.close()

        # 'a' preserves existing data
        store.open(path, mode="a")
        np.testing.assert_array_equal(store.retrieve_array("keep"), np.arange(3))
        store.store_array("more", np.ones(2))
        store.close()

        # 'r' can read both
        store.open(path, mode="r")
        assert sorted(store.list_keys("/")) == ["keep", "more"]
        store.close()

        # 'w' truncates (h5py semantics)
        store.open(path, mode="w")
        assert store.list_keys("/") == []
        store.close()

    def test_context_manager(self, tmp_path):
        with HDF5Storage() as store:
            store.open(str(tmp_path / "cm.h5"), mode="w")
            store.store_scalar("x", 1)
        assert store._file is None


# =============================================================================
# SQLStorage
# =============================================================================


@pytest.fixture
def sql_store(tmp_path):
    store = SQLStorage()
    store.open(str(tmp_path / "audit.db"), mode="w")
    yield store
    store.close()


class TestSQLStorageRoundTrip:
    @pytest.mark.parametrize(
        "arr",
        [
            np.arange(12, dtype=np.float64).reshape(3, 4),
            np.arange(6, dtype=np.float32),
            np.arange(8, dtype=np.int64).reshape(2, 4),
            np.array([True, False, True]),
            np.array([1 + 2j, 3 - 4j], dtype=np.complex128),
            np.array([np.nan, np.inf, -np.inf, 0.0]),
            np.empty((0, 3), dtype=np.float64),
            np.arange(20000, dtype=np.float64).reshape(100, 200),
        ],
        ids=["f8_2d", "f4_1d", "i8_2d", "bool", "c16", "nan_inf", "empty", "large"],
    )
    def test_array_bit_identical(self, sql_store, arr):
        sql_store.store_array("data", arr)
        out = sql_store.retrieve_array("data")
        assert out.dtype == arr.dtype
        assert out.shape == arr.shape
        np.testing.assert_array_equal(out, arr)

    def test_array_overwrite_replaces(self, sql_store):
        sql_store.store_array("d", np.zeros(3))
        sql_store.store_array("d", np.ones((2, 2)))
        np.testing.assert_array_equal(sql_store.retrieve_array("d"), np.ones((2, 2)))

    @pytest.mark.parametrize(
        "value",
        [42, 3.14, True, False, "hello", "ünïcødé ✓", "", 'it\'s "quoted"'],
        ids=["int", "float", "true", "false", "str", "unicode", "empty", "quotes"],
    )
    def test_scalar_round_trip(self, sql_store, value):
        sql_store.store_scalar("s", value)
        out = sql_store.retrieve_scalar("s")
        assert out == value
        assert isinstance(out, type(value))

    def test_metadata_round_trip(self, sql_store):
        meta = {"units": "rad", "n": 2, "nested": {"a": [1, 2]}, "note": "123"}
        sql_store.store_array("d", np.zeros(2), metadata=meta)
        assert sql_store.get_metadata("d") == meta

        sql_store.store_scalar("s", 1, metadata={"k": "v"})
        assert sql_store.get_metadata("s") == {"k": "v"}

    def test_injection_shaped_keys(self, sql_store):
        name = "x'; DROP TABLE _pytcl_arrays;--"
        sql_store.store_array(name, np.arange(3))
        np.testing.assert_array_equal(sql_store.retrieve_array(name), np.arange(3))
        sql_store.store_scalar(name, "v'--")
        assert sql_store.retrieve_scalar(name) == "v'--"


class TestSQLStorageContract:
    def test_list_keys_all_and_prefix(self, sql_store):
        sql_store.store_array("a/one", np.zeros(1))
        sql_store.store_array("a/two", np.zeros(1))
        sql_store.store_array("b/one", np.zeros(1))
        sql_store.store_scalar("a/scalar", 5)
        keys = sql_store.list_keys("/")
        assert {"a/one", "a/two", "b/one", "a/scalar"} <= set(keys)
        assert sorted(sql_store.list_keys("a/")) == ["a/one", "a/scalar", "a/two"]
        assert sql_store.list_keys("b/") == ["b/one"]

    def test_store_group_marker(self, sql_store):
        sql_store.store_group("mission", metadata={"name": "m1"})
        assert "_group:mission" in sql_store.list_keys("/")

    def test_delete_both_tables(self, sql_store):
        sql_store.store_array("x", np.zeros(2))
        sql_store.store_scalar("x", 7)
        sql_store.delete("x")
        with pytest.raises(KeyError):
            sql_store.retrieve_array("x")
        with pytest.raises(KeyError):
            sql_store.retrieve_scalar("x")
        with pytest.raises(KeyError):
            sql_store.get_metadata("x")

    def test_missing_keys_raise_keyerror(self, sql_store):
        with pytest.raises(KeyError):
            sql_store.retrieve_array("nope")
        with pytest.raises(KeyError):
            sql_store.retrieve_scalar("nope")
        with pytest.raises(KeyError):
            sql_store.get_metadata("nope")

    def test_not_open_raises_runtimeerror(self):
        store = SQLStorage()
        for call in [
            lambda: store.store_array("a", np.zeros(2)),
            lambda: store.retrieve_array("a"),
            lambda: store.store_scalar("a", 1),
            lambda: store.retrieve_scalar("a"),
            lambda: store.store_group("g"),
            lambda: store.list_keys(),
            lambda: store.get_metadata("a"),
            lambda: store.delete("a"),
        ]:
            with pytest.raises(RuntimeError):
                call()

    def test_reopen_cycles_preserve_data(self, tmp_path):
        path = str(tmp_path / "cycles.db")
        for i in range(3):
            store = SQLStorage()
            store.open(path, mode="a")
            store.store_scalar(f"k{i}", i)
            store.close()
            store.close()  # double close is safe
        store = SQLStorage()
        store.open(path, mode="r")
        assert [store.retrieve_scalar(f"k{i}") for i in range(3)] == [0, 1, 2]
        store.close()

    def test_write_mode_does_not_truncate(self, tmp_path):
        # SQLStorage documents 'w' as treated like 'a'
        path = str(tmp_path / "w.db")
        store = SQLStorage()
        store.open(path, mode="w")
        store.store_scalar("keep", 1)
        store.close()
        store.open(path, mode="w")
        assert store.retrieve_scalar("keep") == 1
        store.close()

    def test_flush_and_context_manager(self, tmp_path):
        store = SQLStorage()
        store.flush()  # no-op when never opened
        with store:
            store.open(str(tmp_path / "cm.db"), mode="w")
            store.store_scalar("x", 1)
            store.flush()
        assert store._connection is None


# =============================================================================
# TrackDatabaseManager
# =============================================================================


@pytest.fixture
def db(tmp_path):
    mgr = TrackDatabaseManager(str(tmp_path / "tracks.db"))
    mgr.open(mode="w")
    yield mgr
    mgr.close()


def _populate_detections(db):
    """Store a grid of detections and return the ground-truth list."""
    truth = []
    for i in range(20):
        det = {
            "detection_id": f"det_{i:03d}",
            "timestamp": float(i % 7),
            "sensor_id": "radar" if i % 2 == 0 else "eo",
        }
        db.store_detection(
            det["detection_id"],
            np.array([float(i), float(i) + 0.5]),
            det["sensor_id"],
            det["timestamp"],
        )
        truth.append(det)
    return truth


class TestDetectionManagement:
    def test_store_retrieve_round_trip(self, db):
        meas = np.array([1.5, -2.5, np.nan])
        cov = np.array([[1.0, 0.1, 0], [0.1, 2.0, 0], [0, 0, 3.0]])
        meta = {"snr": np.float64(12.5), "n": np.int64(3), "arr": np.array([1.0, 2.0])}
        db.store_detection("d1", meas, "sensör_1", 5.0, covariance=cov, metadata=meta)
        out = db.retrieve_detection("d1")
        np.testing.assert_array_equal(out["measurement"], meas)
        np.testing.assert_array_equal(out["covariance"], cov)
        assert out["sensor_id"] == "sensör_1"
        assert out["timestamp"] == 5.0
        assert out["association_status"] == "unassociated"
        assert out["associated_track_id"] is None
        assert out["metadata"] == {"snr": 12.5, "n": 3, "arr": [1.0, 2.0]}

    def test_detection_without_covariance(self, db):
        db.store_detection("d1", np.array([1.0]), "s", 0.0)
        assert db.retrieve_detection("d1")["covariance"] is None

    def test_retrieve_detection_missing_raises(self, db):
        with pytest.raises(KeyError):
            db.retrieve_detection("nope")

    def test_retrieve_detections_filters_vs_ground_truth(self, db):
        truth = _populate_detections(db)

        def expect(**kw):
            rows = [
                d
                for d in truth
                if (kw.get("start") is None or d["timestamp"] >= kw["start"])
                and (kw.get("end") is None or d["timestamp"] <= kw["end"])
                and (kw.get("sensor") is None or d["sensor_id"] == kw["sensor"])
            ]
            return sorted(d["detection_id"] for d in rows)

        cases = [
            dict(start=None, end=None, sensor=None),
            dict(start=2.0, end=None, sensor=None),
            dict(start=None, end=4.0, sensor=None),
            dict(start=2.0, end=4.0, sensor=None),
            dict(start=None, end=None, sensor="radar"),
            dict(start=1.0, end=5.0, sensor="eo"),
            dict(start=100.0, end=None, sensor=None),
        ]
        for kw in cases:
            got = db.retrieve_detections(
                start_time=kw["start"], end_time=kw["end"], sensor_id=kw["sensor"]
            )
            assert sorted(d["detection_id"] for d in got) == expect(**kw), kw
            # Ordered by timestamp
            ts = [d["timestamp"] for d in got]
            assert ts == sorted(ts)

    def test_retrieve_detections_limit(self, db):
        _populate_detections(db)
        got = db.retrieve_detections(limit=5)
        assert len(got) == 5
        assert [d["detection_id"] for d in db.retrieve_all_detections()] == [
            d["detection_id"] for d in db.retrieve_detections()
        ]

    def test_associate_detection(self, db):
        db.store_detection("d1", np.array([1.0]), "s", 0.0)
        db.initiate_track("t1", np.zeros(2), np.eye(2), 0.0)
        db.associate_detection("d1", "t1", confidence=0.75)
        out = db.retrieve_detection("d1")
        assert out["association_status"] == "associated"
        assert out["associated_track_id"] == "t1"
        assert out["association_confidence"] == 0.75
        assert (
            db.retrieve_detections(association_status="associated")[0]["detection_id"]
            == "d1"
        )
        with pytest.raises(KeyError):
            db.associate_detection("nope", "t1")


class TestTrackInitiationAndState:
    def test_initiate_track_round_trip(self, db):
        x0 = np.array([1.0, 2.0, -3.0, np.inf])
        p0 = np.diag([1.0, 2.0, 3.0, 4.0])
        db.initiate_track("t1", x0, p0, 10.0, metadata={"src": "radar"})
        info = db.get_track("t1")
        assert info["status"] == "tentative"
        assert info["birth_time"] == 10.0
        assert info["last_update_time"] == 10.0
        assert info["state_dim"] == 4
        assert info["hits"] == 0 and info["misses"] == 0
        assert info["metadata"] == {"src": "radar"}

        st = db.get_track_state("t1")
        np.testing.assert_array_equal(st["state"], x0)
        np.testing.assert_array_equal(st["covariance"], p0)
        assert st["timestamp"] == 10.0
        assert st["status"] == TrackDatabaseStatus.TENTATIVE

    def test_get_track_missing_raises(self, db):
        with pytest.raises(KeyError):
            db.get_track("nope")
        with pytest.raises(KeyError):
            db.get_track_state("nope")
        with pytest.raises(KeyError):
            db.get_track_history("nope")

    def test_update_counters(self, db):
        db.initiate_track("t1", np.zeros(2), np.eye(2), 0.0)
        db.update_track_state("t1", np.ones(2), np.eye(2), 1.0, update_type="update")
        info = db.get_track("t1")
        assert (info["hits"], info["misses"], info["total_misses"]) == (1, 0, 0)

        db.update_track_state(
            "t1", np.ones(2), np.eye(2), 2.0, update_type="prediction"
        )
        db.update_track_state(
            "t1", np.ones(2), np.eye(2), 3.0, update_type="prediction"
        )
        info = db.get_track("t1")
        assert (info["hits"], info["misses"], info["total_misses"]) == (1, 2, 2)

        db.update_track_state("t1", np.ones(2), np.eye(2), 4.0, update_type="update")
        info = db.get_track("t1")
        assert (info["hits"], info["misses"], info["total_misses"]) == (2, 0, 2)

        db.update_track_state("t1", np.ones(2), np.eye(2), 5.0, update_type="smoothed")
        info = db.get_track("t1")
        assert (info["hits"], info["misses"], info["total_misses"]) == (2, 0, 2)
        assert info["last_update_time"] == 5.0

    def test_history_ordering_and_windowing(self, db):
        db.initiate_track("t1", np.array([0.0, 0.0]), np.eye(2), 0.0)
        # Insert out of chronological order
        times = [3.0, 1.0, 4.0, 2.0, 5.0]
        for t in times:
            db.update_track_state("t1", np.array([t, -t]), np.eye(2) * t, t)

        all_times = sorted([0.0] + times)
        hist = db.get_track_history("t1")
        np.testing.assert_array_equal(hist["timestamps"], all_times)
        # states aligned with sorted timestamps
        np.testing.assert_array_equal(hist["states"][:, 0], all_times)
        assert hist["covariances"].shape == (6, 2, 2)
        assert hist["residuals"] is None

        # Inclusive window vs ground truth
        expected = [t for t in all_times if 1.0 <= t <= 4.0]
        win = db.get_track_history("t1", start_time=1.0, end_time=4.0)
        np.testing.assert_array_equal(win["timestamps"], expected)

        with pytest.raises(KeyError):
            db.get_track_history("t1", start_time=100.0)

    def test_residual_round_trip(self, db):
        db.initiate_track("t1", np.zeros(2), np.eye(2), 0.0)
        res = np.array([0.5, -0.5])
        db.update_track_state("t1", np.ones(2), np.eye(2), 1.0, residual=res)
        # History starting after the residual-free initiate row
        hist = db.get_track_history("t1", start_time=0.5)
        np.testing.assert_array_equal(hist["residuals"], [res])

    def test_store_track_history_batch(self, db):
        rng = np.random.default_rng(0)
        n, dim = 8, 3
        states = rng.normal(size=(n, dim))
        covs = np.array([np.eye(dim) * (i + 1) for i in range(n)])
        ts = np.arange(n, dtype=float)
        residuals = rng.normal(size=(n, 2))

        db.initiate_track("t1", states[0], covs[0], 0.0)
        db.store_track_history("t1", states[1:], covs[1:], ts[1:], residuals[1:])

        hist = db.get_track_history("t1", start_time=0.5)
        np.testing.assert_array_equal(hist["states"], states[1:])
        np.testing.assert_array_equal(hist["covariances"], covs[1:])
        np.testing.assert_array_equal(hist["timestamps"], ts[1:])
        np.testing.assert_array_equal(hist["residuals"], residuals[1:])

        st = db.get_track_state("t1")
        np.testing.assert_array_equal(st["state"], states[-1])

    def test_get_initiation_queue(self, db):
        for i in range(6):
            db.store_detection(f"d{i}", np.array([float(i)]), "s", float(i))
        db.initiate_track("t1", np.zeros(2), np.eye(2), 0.0)
        db.associate_detection("d2", "t1")

        queue = db.get_initiation_queue()
        assert sorted(d["detection_id"] for d in queue) == [
            "d0",
            "d1",
            "d3",
            "d4",
            "d5",
        ]
        # max_age: newest is t=5, cutoff = 5 - 2 = 3 (inclusive)
        queue = db.get_initiation_queue(max_age=2.0)
        assert sorted(d["detection_id"] for d in queue) == ["d3", "d4", "d5"]

    def test_get_initiation_queue_empty_db(self, db):
        assert db.get_initiation_queue(max_age=1.0) == []
        assert db.get_initiation_queue() == []

    def test_retrieve_all_tracks_status_filter(self, db):
        for tid in ("a", "b", "c"):
            db.initiate_track(tid, np.zeros(2), np.eye(2), 0.0)
        db.confirm_track("b")
        db.mark_track_dead("c")

        assert len(db.retrieve_all_tracks()) == 3
        tentative = db.retrieve_all_tracks(TrackDatabaseStatus.TENTATIVE)
        assert [t["track_id"] for t in tentative] == ["a"]
        confirmed = db.retrieve_all_tracks(TrackDatabaseStatus.CONFIRMED)
        assert [t["track_id"] for t in confirmed] == ["b"]


class TestTrackLifecycle:
    def test_status_transitions(self, db):
        db.initiate_track("t1", np.zeros(2), np.eye(2), 0.0)
        for method, status in [
            (db.confirm_track, "confirmed"),
            (db.mark_track_coasting, "coasting"),
            (db.mark_track_tentative, "tentative"),
            (db.mark_track_confirmed, "confirmed"),
            (db.mark_track_dead, "dead"),
        ]:
            method("t1")
            assert db.get_track("t1")["status"] == status

    def test_prune_old_detections_exact(self, db):
        # newest ts = 10; threshold 4 -> cutoff 6; only unassociated ts < 6 go
        for i, ts in enumerate([0.0, 3.0, 5.0, 6.0, 8.0, 10.0]):
            db.store_detection(f"d{i}", np.array([1.0]), "s", ts)
        db.initiate_track("t1", np.zeros(2), np.eye(2), 0.0)
        db.associate_detection("d1", "t1")  # associated old det survives

        removed = db.prune_old_detections(4.0)
        assert removed == 2  # d0 (0.0) and d2 (5.0)
        survivors = sorted(d["detection_id"] for d in db.retrieve_all_detections())
        assert survivors == ["d1", "d3", "d4", "d5"]

    def test_prune_old_detections_empty(self, db):
        assert db.prune_old_detections(1.0) == 0

    def test_prune_dead_tracks_exact(self, db):
        # newest last_update_time = 10; threshold 4 -> cutoff 6
        db.initiate_track("dead_old", np.zeros(2), np.eye(2), 0.0)
        db.initiate_track("dead_new", np.zeros(2), np.eye(2), 8.0)
        db.initiate_track("alive_old", np.zeros(2), np.eye(2), 1.0)
        db.initiate_track("alive_new", np.zeros(2), np.eye(2), 10.0)
        db.mark_track_dead("dead_old")
        db.mark_track_dead("dead_new")
        db.store_detection("d0", np.array([1.0]), "s", 0.0)
        db.associate_detection("d0", "dead_old")

        removed = db.prune_dead_tracks(4.0)
        assert removed == 1
        remaining = sorted(t["track_id"] for t in db.retrieve_all_tracks())
        assert remaining == ["alive_new", "alive_old", "dead_new"]
        with pytest.raises(KeyError):
            db.get_track("dead_old")
        with pytest.raises(KeyError):
            db.get_track_history("dead_old")
        # association rows for the pruned track are removed
        db._cursor.execute(
            "SELECT COUNT(*) FROM track_associations WHERE track_id = 'dead_old'"
        )
        assert db._cursor.fetchone()[0] == 0

    def test_prune_dead_tracks_empty(self, db):
        assert db.prune_dead_tracks(1.0) == 0

    def test_merge_tracks(self, db):
        db.initiate_track("keep", np.array([0.0, 0.0]), np.eye(2), 0.0)
        db.initiate_track("gone", np.array([9.0, 9.0]), np.eye(2), 0.5)
        db.update_track_state("keep", np.array([1.0, 1.0]), np.eye(2), 1.0)
        db.update_track_state("gone", np.array([8.0, 8.0]), np.eye(2), 1.5)
        db.update_track_state(
            "gone", np.array([7.0, 7.0]), np.eye(2), 2.5, update_type="prediction"
        )
        db.store_detection("d1", np.array([1.0]), "s", 1.5)
        db.associate_detection("d1", "gone")

        keep_before = db.get_track("keep")
        gone_before = db.get_track("gone")

        db.merge_tracks("keep", "gone")

        # Histories combined: 2 (keep) + 3 (gone)
        hist = db.get_track_history("keep")
        assert len(hist["timestamps"]) == 5
        np.testing.assert_array_equal(hist["timestamps"], [0.0, 0.5, 1.0, 1.5, 2.5])
        # Detections re-associated
        assert db.retrieve_detection("d1")["associated_track_id"] == "keep"
        # Merged marked DEAD; no history left under old id
        assert db.get_track("gone")["status"] == "dead"
        with pytest.raises(KeyError):
            db.get_track_history("gone")
        # Counters summed
        keep_after = db.get_track("keep")
        assert keep_after["hits"] == keep_before["hits"] + gone_before["hits"]
        assert (
            keep_after["total_misses"]
            == keep_before["total_misses"] + gone_before["total_misses"]
        )


class TestTrackDatabaseRobustness:
    def test_injection_shaped_track_ids(self, db):
        tid = "trk'; DROP TABLE tracks;--"
        did = 'det" OR "1"="1'
        db.initiate_track(tid, np.array([1.0, 2.0]), np.eye(2), 0.0)
        db.store_detection(did, np.array([3.0]), "s'ensor", 0.0)
        db.associate_detection(did, tid)
        db.update_track_state(tid, np.array([1.5, 2.5]), np.eye(2), 1.0)

        st = db.get_track_state(tid)
        np.testing.assert_array_equal(st["state"], [1.5, 2.5])
        assert db.retrieve_detection(did)["associated_track_id"] == tid
        db.mark_track_dead(tid)
        assert db.get_track(tid)["status"] == "dead"

    def test_not_open_raises_runtimeerror(self, tmp_path):
        mgr = TrackDatabaseManager(str(tmp_path / "x.db"))
        for call in [
            lambda: mgr.store_detection("d", np.zeros(1), "s", 0.0),
            lambda: mgr.retrieve_detections(),
            lambda: mgr.retrieve_detection("d"),
            lambda: mgr.associate_detection("d", "t"),
            lambda: mgr.initiate_track("t", np.zeros(2), np.eye(2), 0.0),
            lambda: mgr.get_initiation_queue(),
            lambda: mgr.update_track_state("t", np.zeros(2), np.eye(2), 0.0),
            lambda: mgr.get_track_state("t"),
            lambda: mgr.get_track_history("t"),
            lambda: mgr.get_track("t"),
            lambda: mgr.retrieve_all_tracks(),
            lambda: mgr.prune_old_detections(1.0),
            lambda: mgr.prune_dead_tracks(1.0),
            lambda: mgr.merge_tracks("a", "b"),
            lambda: mgr.confirm_track("t"),
            lambda: mgr.store_from_track(
                _TrackTuple(1, np.zeros(2), np.eye(2), None, 0, 0, 0.0)
            ),
        ]:
            with pytest.raises(RuntimeError):
                call()

    def test_open_w_does_not_truncate(self, tmp_path):
        """Documented: 'w' is write/create, existing data is preserved."""
        path = str(tmp_path / "persist.db")
        mgr = TrackDatabaseManager(path)
        mgr.open(mode="w")
        mgr.initiate_track("t1", np.zeros(2), np.eye(2), 0.0)
        mgr.close()
        mgr.close()  # double close is safe

        mgr.open(mode="w")
        assert mgr.get_track("t1")["track_id"] == "t1"
        mgr.close()

        mgr.open(mode="a")
        assert mgr.get_track("t1")["track_id"] == "t1"
        mgr.close()

    def test_context_manager(self, tmp_path):
        with TrackDatabaseManager(str(tmp_path / "cm.db")) as mgr:
            mgr.open(mode="w")
            mgr.initiate_track("t1", np.zeros(2), np.eye(2), 0.0)
        assert mgr._connection is None


_TrackTuple = namedtuple("_TrackTuple", "id state covariance status hits misses time")


class TestTrackConversion:
    def test_track_to_pytcl(self, db):
        from pytcl.trackers.multi_target import TrackStatus

        x = np.array([1.0, 2.0, 3.0, 4.0])
        p = np.eye(4) * 2.0
        db.initiate_track("t1", x, p, 5.0)
        db.confirm_track("t1")
        trk = db.track_to_pytcl("t1")
        np.testing.assert_array_equal(trk.state, x)
        np.testing.assert_array_equal(trk.covariance, p)
        assert trk.status == TrackStatus.CONFIRMED
        assert trk.time == 5.0

        db.mark_track_coasting("t1")
        assert db.track_to_pytcl("t1").status == TrackStatus.CONFIRMED
        db.mark_track_dead("t1")
        assert db.track_to_pytcl("t1").status == TrackStatus.DELETED
        db.mark_track_tentative("t1")
        assert db.track_to_pytcl("t1").status == TrackStatus.TENTATIVE

    def test_tracks_to_tracklist(self, db):
        db.initiate_track("a", np.zeros(2), np.eye(2), 0.0)
        db.initiate_track("b", np.ones(2), np.eye(2), 0.0)
        db.confirm_track("b")
        tl = db.tracks_to_tracklist()
        assert len(list(tl)) == 2
        tl_confirmed = db.tracks_to_tracklist(TrackDatabaseStatus.CONFIRMED)
        assert len(list(tl_confirmed)) == 1

    def test_store_from_track_and_tracklist(self, db):
        t1 = _TrackTuple(7, np.array([1.0, 2.0]), np.eye(2), None, 0, 0, 3.0)
        db.store_from_track(t1)
        st = db.get_track_state("trk_7")
        np.testing.assert_array_equal(st["state"], [1.0, 2.0])
        assert st["timestamp"] == 3.0
        assert db.get_track("trk_7")["status"] == "tentative"

        # Existing track: second store updates rather than re-initiates
        t1b = t1._replace(state=np.array([5.0, 6.0]), time=4.0)
        db.store_from_track(t1b)
        st = db.get_track_state("trk_7")
        np.testing.assert_array_equal(st["state"], [5.0, 6.0])
        assert db.get_track("trk_7")["hits"] == 1

        # Timestamp override
        db.store_from_track(t1b._replace(time=99.0), timestamp=10.0)
        assert db.get_track_state("trk_7")["timestamp"] == 10.0

        db.store_from_tracklist(
            [
                _TrackTuple(8, np.zeros(2), np.eye(2), None, 0, 0, 0.0),
                _TrackTuple(9, np.ones(2), np.eye(2), None, 0, 0, 0.0),
            ]
        )
        assert db.get_track("trk_8")["track_id"] == "trk_8"
        assert db.get_track("trk_9")["track_id"] == "trk_9"


# =============================================================================
# TrackHDF5Storage
# =============================================================================


@pytest.fixture
def h5_tracks(tmp_path):
    store = TrackHDF5Storage(str(tmp_path / "tracks.h5"))
    store.open(mode="w")
    yield store
    store.close()


def _make_track(n=10, dim=4, seed=0, meas_dim=2):
    rng = np.random.default_rng(seed)
    states = rng.normal(size=(n, dim))
    covs = np.array([np.eye(dim) * (i + 1.0) for i in range(n)])
    ts = np.arange(n, dtype=float)
    residuals = rng.normal(size=(n, meas_dim))
    return states, covs, ts, residuals


class TestTrackHDF5RoundTrip:
    def test_store_retrieve_bit_identical(self, h5_tracks):
        states, covs, ts, residuals = _make_track()
        states[0, 0] = np.nan
        states[1, 1] = np.inf
        meta = {"status": "confirmed", "birth_time": 0.0, "n": 3, "ok": True}
        h5_tracks.store_track(
            "t1", states, covs, ts, metadata=meta, residuals=residuals
        )
        out = h5_tracks.retrieve_track("t1")
        np.testing.assert_array_equal(out["states"], states)
        np.testing.assert_array_equal(out["covariances"], covs)
        np.testing.assert_array_equal(out["timestamps"], ts)
        np.testing.assert_array_equal(out["residuals"], residuals)
        for k, v in meta.items():
            assert out["metadata"][k] == v

    def test_retrieve_without_residuals(self, h5_tracks):
        states, covs, ts, _ = _make_track(n=3)
        h5_tracks.store_track("t1", states, covs, ts)
        assert h5_tracks.retrieve_track("t1")["residuals"] is None

    def test_compression_large_track(self, h5_tracks):
        states, covs, ts, _ = _make_track(n=2500, dim=4)
        h5_tracks.store_track("big", states, covs, ts)
        out = h5_tracks.retrieve_track("big")
        np.testing.assert_array_equal(out["states"], states)
        np.testing.assert_array_equal(out["covariances"], covs)

    def test_append_track_state(self, h5_tracks):
        states, covs, ts, residuals = _make_track(n=4)
        h5_tracks.store_track("t1", states, covs, ts, residuals=residuals)
        new_s = np.array([9.0, 8.0, 7.0, 6.0])
        new_c = np.eye(4) * 99.0
        new_r = np.array([0.1, 0.2])
        h5_tracks.append_track_state("t1", new_s, new_c, 4.0, residual=new_r)
        out = h5_tracks.retrieve_track("t1")
        np.testing.assert_array_equal(out["states"], np.vstack([states, [new_s]]))
        np.testing.assert_array_equal(out["covariances"][-1], new_c)
        np.testing.assert_array_equal(out["timestamps"], np.arange(5.0))
        np.testing.assert_array_equal(out["residuals"][-1], new_r)

    def test_detection_round_trip(self, h5_tracks):
        z = np.array([1.0, np.nan, -3.5])
        cov = np.eye(3) * 0.5
        h5_tracks.store_detection(
            "d1", z, 2.5, "sensör", covariance=cov, metadata={"snr": 10.0}
        )
        out = h5_tracks.retrieve_detection("d1")
        np.testing.assert_array_equal(out["measurement"], z)
        np.testing.assert_array_equal(out["covariance"], cov)
        assert out["timestamp"] == 2.5
        assert out["sensor_id"] == "sensör"
        assert out["metadata"]["snr"] == 10.0
        assert "timestamp" not in out["metadata"]

        h5_tracks.store_detection("d2", z, 0.0, "s")
        assert h5_tracks.retrieve_detection("d2")["covariance"] is None

    def test_missing_ids_raise_keyerror(self, h5_tracks):
        with pytest.raises(KeyError):
            h5_tracks.retrieve_track("nope")
        with pytest.raises(KeyError):
            h5_tracks.retrieve_detection("nope")
        with pytest.raises(KeyError):
            h5_tracks.get_track_trajectory("nope")
        with pytest.raises(KeyError):
            h5_tracks.get_state_at_time("nope", 0.0)
        with pytest.raises(KeyError):
            h5_tracks.retrieve_tracking_scenario("nope")


class TestTrackHDF5Queries:
    def test_trajectory_windowing_vs_brute_force(self, h5_tracks):
        states, covs, ts, _ = _make_track(n=20)
        h5_tracks.store_track("t1", states, covs, ts)

        for lo, hi in [(None, None), (5.0, None), (None, 12.0), (3.0, 7.0), (3.5, 6.5)]:
            got = h5_tracks.get_track_trajectory("t1", start_time=lo, end_time=hi)
            mask = np.ones(len(ts), dtype=bool)
            if lo is not None:
                mask &= ts >= lo
            if hi is not None:
                mask &= ts <= hi
            np.testing.assert_array_equal(got["timestamps"], ts[mask])
            np.testing.assert_array_equal(got["states"], states[mask])
            np.testing.assert_array_equal(got["covariances"], covs[mask])

        empty = h5_tracks.get_track_trajectory("t1", start_time=100.0)
        assert len(empty["timestamps"]) == 0
        assert len(empty["states"]) == 0

    def test_state_at_time_nearest(self, h5_tracks):
        states, covs, ts, _ = _make_track(n=5)
        h5_tracks.store_track("t1", states, covs, ts)
        for q in [-1.0, 0.4, 1.6, 2.0, 10.0]:
            got = h5_tracks.get_state_at_time("t1", q)
            idx = int(np.argmin(np.abs(ts - q)))
            np.testing.assert_array_equal(got["state"], states[idx])
            np.testing.assert_array_equal(got["covariance"], covs[idx])
            assert got["timestamp"] == ts[idx]

    def test_state_at_time_interpolation_hand_computed(self, h5_tracks):
        states = np.array([[0.0, 10.0], [4.0, 20.0], [8.0, 40.0]])
        covs = np.array([np.eye(2) * 1.0, np.eye(2) * 3.0, np.eye(2) * 5.0])
        ts = np.array([0.0, 2.0, 4.0])
        h5_tracks.store_track("t1", states, covs, ts)

        # q=0.5: alpha=0.25 between rows 0 and 1
        got = h5_tracks.get_state_at_time("t1", 0.5, interpolate=True)
        np.testing.assert_allclose(got["state"], [1.0, 12.5])
        np.testing.assert_allclose(got["covariance"], np.eye(2) * 1.5)
        assert got["timestamp"] == 0.5

        # q=3.0: alpha=0.5 between rows 1 and 2
        got = h5_tracks.get_state_at_time("t1", 3.0, interpolate=True)
        np.testing.assert_allclose(got["state"], [6.0, 30.0])
        np.testing.assert_allclose(got["covariance"], np.eye(2) * 4.0)

        # Exact node returns node values
        got = h5_tracks.get_state_at_time("t1", 2.0, interpolate=True)
        np.testing.assert_allclose(got["state"], states[1])

        # Clamped outside range
        got = h5_tracks.get_state_at_time("t1", -5.0, interpolate=True)
        np.testing.assert_array_equal(got["state"], states[0])
        assert got["timestamp"] == 0.0
        got = h5_tracks.get_state_at_time("t1", 99.0, interpolate=True)
        np.testing.assert_array_equal(got["state"], states[-1])
        assert got["timestamp"] == 4.0

    def test_tracks_in_region_vs_brute_force(self, h5_tracks):
        rng = np.random.default_rng(42)
        tracks = {}
        for i in range(8):
            n = 15
            states = rng.uniform(-10, 10, size=(n, 4))
            covs = np.array([np.eye(4)] * n)
            ts = np.arange(n, dtype=float)
            tid = f"t{i}"
            tracks[tid] = (states, ts)
            h5_tracks.store_track(tid, states, covs, ts)

        cases = [
            dict(bbox=[-5, -5, 5, 5], time_range=None, idx=(0, 2)),
            dict(bbox=[0, 0, 10, 10], time_range=[3.0, 8.0], idx=(0, 2)),
            dict(bbox=[-2, -2, 2, 2], time_range=[0.0, 5.0], idx=(1, 3)),
            dict(bbox=[100, 100, 110, 110], time_range=None, idx=(0, 2)),
        ]
        for case in cases:
            got = sorted(
                h5_tracks.get_tracks_in_region(
                    case["bbox"],
                    time_range=case["time_range"],
                    state_indices=case["idx"],
                )
            )
            x0, y0, x1, y1 = case["bbox"]
            ix, iy = case["idx"]
            expected = []
            for tid, (states, ts) in tracks.items():
                sel = states
                if case["time_range"] is not None:
                    m = (ts >= case["time_range"][0]) & (ts <= case["time_range"][1])
                    sel = states[m]
                inside = (
                    (sel[:, ix] >= x0)
                    & (sel[:, ix] <= x1)
                    & (sel[:, iy] >= y0)
                    & (sel[:, iy] <= y1)
                )
                if inside.any():
                    expected.append(tid)
            assert got == sorted(expected), case


class TestTrackHDF5Scenarios:
    def _scenario(self, h5_tracks, sid="s1", seed=1):
        s1, c1, t1, r1 = _make_track(n=6, seed=seed)
        s2, c2, t2, _ = _make_track(n=4, seed=seed + 1)
        tracks = {
            "ta": {
                "states": s1,
                "covariances": c1,
                "timestamps": t1,
                "residuals": r1,
            },
            "tb": {"states": s2, "covariances": c2, "timestamps": t2},
        }
        detections = {
            "d1": {
                "measurement": np.array([1.0, 2.0]),
                "timestamp": 0.5,
                "sensor_id": "radar",
                "covariance": np.eye(2),
            },
            "d2": {"measurement": np.array([3.0]), "timestamp": 1.5},
        }
        h5_tracks.store_tracking_scenario(
            sid, tracks, detections, metadata={"name": "audit"}
        )
        return tracks, detections

    def test_scenario_round_trip(self, h5_tracks):
        tracks, detections = self._scenario(h5_tracks)
        out = h5_tracks.retrieve_tracking_scenario("s1")

        assert sorted(out["tracks"]) == ["ta", "tb"]
        np.testing.assert_array_equal(
            out["tracks"]["ta"]["states"], tracks["ta"]["states"]
        )
        np.testing.assert_array_equal(
            out["tracks"]["ta"]["residuals"], tracks["ta"]["residuals"]
        )
        assert out["tracks"]["tb"]["residuals"] is None
        np.testing.assert_array_equal(
            out["detections"]["d1"]["measurement"], [1.0, 2.0]
        )
        assert out["detections"]["d2"]["sensor_id"] == "unknown"
        assert out["metadata"]["name"] == "audit"
        assert out["metadata"]["n_tracks"] == 2
        assert out["metadata"]["n_detections"] == 2

    def test_scenario_isolation_and_listing(self, h5_tracks):
        self._scenario(h5_tracks, sid="s1")
        self._scenario(h5_tracks, sid="s2", seed=5)
        states, covs, ts, _ = _make_track(n=3)
        h5_tracks.store_track("standalone", states, covs, ts)
        h5_tracks.store_detection("sd", np.array([1.0]), 0.0, "s")

        assert sorted(h5_tracks.list_scenarios()) == ["s1", "s2"]
        assert h5_tracks.list_tracks() == ["standalone"]
        assert sorted(h5_tracks.list_tracks("s1")) == ["ta", "tb"]
        assert h5_tracks.list_detections() == ["sd"]
        assert sorted(h5_tracks.list_detections("s1")) == ["d1", "d2"]
        # Scenario-scoped retrieval does not see standalone data
        with pytest.raises(KeyError):
            h5_tracks.retrieve_track("standalone", scenario_id="s1")

    def test_listing_empty(self, h5_tracks):
        assert h5_tracks.list_scenarios() == []
        assert h5_tracks.list_tracks() == []
        assert h5_tracks.list_detections() == []
        assert h5_tracks.list_tracks("nope") == []
        assert h5_tracks.list_detections("nope") == []

    def test_compare_scenarios(self, h5_tracks):
        base_s, covs, ts, _ = _make_track(n=5, dim=2)
        shifted = base_s + 2.0
        h5_tracks.store_tracking_scenario(
            "s1",
            {
                "common": {"states": base_s, "covariances": covs, "timestamps": ts},
                "only1": {"states": base_s, "covariances": covs, "timestamps": ts},
            },
        )
        h5_tracks.store_tracking_scenario(
            "s2",
            {
                "common": {"states": shifted, "covariances": covs, "timestamps": ts},
                "only2": {"states": base_s, "covariances": covs, "timestamps": ts},
            },
        )
        cmp_out = h5_tracks.compare_scenarios("s1", "s2")
        assert cmp_out["common_tracks"] == ["common"]
        assert cmp_out["unique_to_1"] == ["only1"]
        assert cmp_out["unique_to_2"] == ["only2"]
        # RMSE of a constant offset of 2.0 is exactly 2.0
        assert cmp_out["state_differences"]["common"] == pytest.approx(2.0)


class TestTrackHDF5SQLBridge:
    def test_export_to_sql_round_trip(self, h5_tracks, tmp_path):
        states, covs, ts, residuals = _make_track(n=6)
        h5_tracks.store_track(
            "t1",
            states,
            covs,
            ts,
            metadata={"status": "confirmed"},
            residuals=residuals,
        )
        h5_tracks.store_detection(
            "d1", np.array([1.0, 2.0]), 0.5, "radar", covariance=np.eye(2)
        )

        db = TrackDatabaseManager(str(tmp_path / "export.db"))
        db.open(mode="w")
        h5_tracks.export_to_sql(db)

        hist = db.get_track_history("t1")
        np.testing.assert_array_equal(hist["states"], states)
        np.testing.assert_array_equal(hist["covariances"], covs)
        np.testing.assert_array_equal(hist["timestamps"], ts)

        det = db.retrieve_detection("d1")
        np.testing.assert_array_equal(det["measurement"], [1.0, 2.0])
        assert det["sensor_id"] == "radar"
        assert det["timestamp"] == 0.5
        db.close()

    def test_export_to_sql_residual_alignment(self, h5_tracks, tmp_path):
        """Regression: residual for timestamp i must be residuals[i]."""
        n = 5
        states = np.zeros((n, 2))
        covs = np.array([np.eye(2)] * n)
        ts = np.arange(n, dtype=float)
        residuals = np.array([[10.0 + i, 20.0 + i] for i in range(n)])
        h5_tracks.store_track("t1", states, covs, ts, residuals=residuals)

        db = TrackDatabaseManager(str(tmp_path / "res.db"))
        db.open(mode="w")
        h5_tracks.export_to_sql(db)
        db._cursor.execute(
            "SELECT timestamp, residual FROM track_states "
            "WHERE track_id = 't1' ORDER BY timestamp"
        )
        rows = db._cursor.fetchall()
        assert rows[0][1] is None  # initiate row has no residual
        for t, blob in rows[1:]:
            np.testing.assert_array_equal(
                np.frombuffer(blob, dtype=np.float64), residuals[int(t)]
            )
        db.close()

    def test_import_from_sql_round_trip(self, h5_tracks, tmp_path):
        db = TrackDatabaseManager(str(tmp_path / "import.db"))
        db.open(mode="w")
        states, covs, ts, _ = _make_track(n=5)
        db.initiate_track("t1", states[0], covs[0], ts[0])
        db.store_track_history("t1", states[1:], covs[1:], ts[1:])
        db.confirm_track("t1")
        db.store_detection("d1", np.array([7.0, 8.0]), "eo", 2.0)

        h5_tracks.import_from_sql(db, scenario_id="imported")
        out = h5_tracks.retrieve_tracking_scenario("imported")
        np.testing.assert_array_equal(out["tracks"]["t1"]["states"], states)
        np.testing.assert_array_equal(out["tracks"]["t1"]["timestamps"], ts)
        assert out["tracks"]["t1"]["metadata"]["status"] == "confirmed"
        np.testing.assert_array_equal(
            out["detections"]["d1"]["measurement"], [7.0, 8.0]
        )
        db.close()


class TestTrackHDF5Robustness:
    def test_not_open_raises_runtimeerror(self, tmp_path):
        store = TrackHDF5Storage(str(tmp_path / "x.h5"))
        states, covs, ts, _ = _make_track(n=2)
        for call in [
            lambda: store.store_track("t", states, covs, ts),
            lambda: store.retrieve_track("t"),
            lambda: store.append_track_state("t", states[0], covs[0], 0.0),
            lambda: store.store_detection("d", np.zeros(2), 0.0, "s"),
            lambda: store.retrieve_detection("d"),
            lambda: store.get_track_trajectory("t"),
            lambda: store.get_state_at_time("t", 0.0),
            lambda: store.get_tracks_in_region([0, 0, 1, 1]),
            lambda: store.store_tracking_scenario("s", {}),
            lambda: store.retrieve_tracking_scenario("s"),
            lambda: store.list_scenarios(),
            lambda: store.list_tracks(),
            lambda: store.list_detections(),
            lambda: store.compare_scenarios("a", "b"),
            lambda: store.export_to_sql(None),
            lambda: store.import_from_sql(None, "s"),
        ]:
            with pytest.raises(RuntimeError):
                call()

    def test_reopen_append_preserves(self, tmp_path):
        path = str(tmp_path / "modes.h5")
        states, covs, ts, _ = _make_track(n=3)
        store = TrackHDF5Storage(path)
        store.open(mode="w")
        store.store_track("t1", states, covs, ts)
        store.flush()
        store.close()
        store.close()  # double close safe

        store.open(mode="a")
        np.testing.assert_array_equal(store.retrieve_track("t1")["states"], states)
        store.close()

        with TrackHDF5Storage(path) as ro:
            ro.open(mode="r")
            assert ro.list_tracks() == ["t1"]


# =============================================================================
# Compat adapters
# =============================================================================


def _cv_model():
    dt = 1.0
    F = np.array(
        [[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]], dtype=float
    )
    H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]], dtype=float)
    Q = 0.1 * np.eye(4)
    R = 4.0 * np.eye(2)
    return F, H, Q, R


class TestKalmanTrackAdapter:
    def test_matches_manual_kf_ground_truth(self, db):
        from pytcl.dynamic_estimation.kalman.linear import kf_predict, kf_update

        F, H, Q, R = _cv_model()
        x0 = np.array([0.0, 1.0, 0.0, -1.0])
        P0 = np.eye(4) * 10.0
        rng = np.random.default_rng(3)
        measurements = [rng.normal(size=2) + k for k in range(1, 5)]

        adapter = KalmanTrackAdapter(db, "trk_kf", F, H, Q, R)
        assert adapter.track_id == "trk_kf"
        assert adapter.state is None and adapter.covariance is None
        adapter.initialize(x0, P0, timestamp=0.0, metadata={"src": "audit"})

        # Ground truth loop
        x, P = x0.copy(), P0.copy()
        expected = [(x.copy(), P.copy())]
        expected_residuals = []
        for z in measurements:
            pred = kf_predict(x, P, F, Q)
            expected.append((pred.x, pred.P))
            upd = kf_update(pred.x, pred.P, z, H, R)
            expected_residuals.append(getattr(upd, "y", None))
            x, P = upd.x, upd.P
            expected.append((x.copy(), P.copy()))

        for k, z in enumerate(measurements):
            adapter.predict_update(z, timestamp=float(k + 1))

        np.testing.assert_allclose(adapter.state, x, rtol=1e-12)
        np.testing.assert_allclose(adapter.covariance, P, rtol=1e-12)

        # DB latest state matches
        st = db.get_track_state("trk_kf")
        np.testing.assert_allclose(st["state"], x, rtol=1e-12)
        np.testing.assert_allclose(st["covariance"], P, rtol=1e-12)

        # Full history: 1 initiate + 2 rows per step (prediction + update)
        hist = db.get_track_history("trk_kf")
        assert len(hist["timestamps"]) == 1 + 2 * len(measurements)

        # Detections stored and associated
        dets = db.retrieve_detections(association_status="associated")
        assert len(dets) == len(measurements)
        for d in dets:
            assert d["associated_track_id"] == "trk_kf"

        # Counters: each step is prediction (miss) + update (hit, resets misses)
        info = db.get_track("trk_kf")
        assert info["hits"] == len(measurements)
        assert info["misses"] == 0
        assert info["total_misses"] == len(measurements)

    def test_stored_residual_matches_kf_update(self, db):
        from pytcl.dynamic_estimation.kalman.linear import kf_predict, kf_update

        F, H, Q, R = _cv_model()
        x0, P0 = np.zeros(4), np.eye(4)
        z = np.array([1.0, -1.0])

        adapter = KalmanTrackAdapter(db, "t", F, H, Q, R)
        adapter.initialize(x0, P0, 0.0)
        adapter.predict(1.0)
        pred = kf_predict(x0, P0, F, Q)
        upd = kf_update(pred.x, pred.P, z, H, R)
        adapter.update(z, 1.0, detection_id="mydet", sensor_id="radar")

        if hasattr(upd, "y"):
            db._cursor.execute(
                "SELECT residual FROM track_states WHERE track_id = 't' "
                "AND update_type = 'update' AND timestamp = 1.0"
            )
            stored = np.frombuffer(db._cursor.fetchone()[0], dtype=np.float64)
            np.testing.assert_allclose(stored, upd.y, rtol=1e-12)
        det = db.retrieve_detection("mydet")
        np.testing.assert_array_equal(det["measurement"], z)
        assert det["sensor_id"] == "radar"
        np.testing.assert_array_equal(det["covariance"], R)

    def test_uninitialized_raises(self, db):
        F, H, Q, R = _cv_model()
        adapter = KalmanTrackAdapter(db, "t", F, H, Q, R)
        with pytest.raises(RuntimeError):
            adapter.predict(0.0)
        with pytest.raises(RuntimeError):
            adapter.update(np.zeros(2), 0.0)


class TestEKFUKFAdapters:
    def _funcs(self):
        F, H, Q, R = _cv_model()

        def f(x):
            return F @ x

        def F_func(x):
            return F

        def h(x):
            return H @ x

        def H_func(x):
            return H

        return f, F_func, h, H_func, Q, R

    def test_ekf_matches_direct_calls(self, db):
        from pytcl.dynamic_estimation.kalman.extended import ekf_predict, ekf_update

        f, F_func, h, H_func, Q, R = self._funcs()
        x0, P0 = np.array([1.0, 0.5, -1.0, 0.2]), np.eye(4) * 5.0
        z = np.array([1.3, -0.9])

        adapter = EKFTrackAdapter(db, "ekf", f, F_func, h, H_func, Q, R)
        assert adapter.state is None and adapter.covariance is None
        adapter.initialize(x0, P0, 0.0)
        adapter.predict(1.0)
        adapter.update(z, 1.0)

        pred = ekf_predict(x0, P0, f, F_func(x0), Q)
        upd = ekf_update(pred.x, pred.P, z, h, H_func(pred.x), R)
        np.testing.assert_allclose(adapter.state, upd.x, rtol=1e-12)
        np.testing.assert_allclose(adapter.covariance, upd.P, rtol=1e-12)

        st = db.get_track_state("ekf")
        np.testing.assert_allclose(st["state"], upd.x, rtol=1e-12)

        bare = EKFTrackAdapter(db, "x", f, F_func, h, H_func, Q, R)
        with pytest.raises(RuntimeError):
            bare.predict(0.0)
        with pytest.raises(RuntimeError):
            bare.update(z, 0.0)

    def test_ukf_matches_direct_calls(self, db):
        from pytcl.dynamic_estimation.kalman.unscented import ukf_predict, ukf_update

        f, _, h, _, Q, R = self._funcs()
        x0, P0 = np.array([1.0, 0.5, -1.0, 0.2]), np.eye(4) * 5.0
        z = np.array([1.3, -0.9])

        adapter = UKFTrackAdapter(db, "ukf", f, h, Q, R, alpha=0.5, beta=2.0, kappa=1.0)
        adapter.initialize(x0, P0, 0.0)
        adapter.predict(1.0)
        adapter.update(z, 1.0)

        pred = ukf_predict(x0, P0, f, Q, 0.5, 2.0, 1.0)
        upd = ukf_update(pred.x, pred.P, z, h, R, 0.5, 2.0, 1.0)
        np.testing.assert_allclose(adapter.state, upd.x, rtol=1e-10)
        np.testing.assert_allclose(adapter.covariance, upd.P, rtol=1e-10)

        st = db.get_track_state("ukf")
        np.testing.assert_allclose(st["state"], upd.x, rtol=1e-10)

        bare = UKFTrackAdapter(db, "x", f, h, Q, R)
        with pytest.raises(RuntimeError):
            bare.predict(0.0)
        with pytest.raises(RuntimeError):
            bare.update(z, 0.0)


class _FakeIMM:
    def __init__(self):
        self.x = None
        self.P = None
        self.mode_probs = np.array([0.6, 0.4])

    def initialize(self, x, P, mode_probs):
        self.x, self.P = x.copy(), P.copy()

    def predict(self):
        self.x = self.x + 1.0
        self.P = self.P * 2.0

    def update(self, z):
        self.x = (self.x + z) / 2.0
        self.P = self.P * 0.5


class TestIMMTrackAdapter:
    def test_persists_imm_outputs(self, db):
        imm = _FakeIMM()
        adapter = IMMTrackAdapter(db, "imm", imm)
        with pytest.raises(RuntimeError):
            adapter.predict(0.0)
        with pytest.raises(RuntimeError):
            adapter.update(np.zeros(2), 0.0)

        x0, P0 = np.array([1.0, 2.0]), np.eye(2)
        adapter.initialize(x0, P0, 0.0)
        adapter.predict(1.0)
        np.testing.assert_array_equal(db.get_track_state("imm")["state"], x0 + 1.0)

        z = np.array([4.0, 5.0])
        adapter.update(z, 1.0)
        np.testing.assert_array_equal(
            db.get_track_state("imm")["state"], (x0 + 1.0 + z) / 2.0
        )
        np.testing.assert_array_equal(adapter.mode_probs, [0.6, 0.4])


class TestParticleFilterTrackAdapter:
    def test_stores_particle_mean(self, db):
        def f(x):
            return x

        def h(x):
            return x[:1]

        Q = 0.01 * np.eye(2)
        R = np.array([[0.5]])
        adapter = ParticleFilterTrackAdapter(db, "pf", f, h, Q, R, n_particles=100)
        with pytest.raises(RuntimeError):
            adapter.predict_update(np.zeros(1), 0.0)

        x0, P0 = np.array([1.0, -1.0]), np.eye(2) * 0.1
        adapter.initialize(x0, P0, 0.0)
        assert adapter.particles.shape == (100, 2)
        np.testing.assert_array_equal(db.get_track_state("pf")["state"], x0)

        adapter.predict_update(np.array([1.2]), 1.0)
        # After resampling weights are uniform: stored state == particle mean
        st = db.get_track_state("pf")
        np.testing.assert_allclose(
            st["state"], adapter.particles.mean(axis=0), rtol=1e-10
        )
        np.testing.assert_allclose(adapter.weights, np.full(100, 0.01))
        assert st["covariance"].shape == (2, 2)


class _FakeTracker:
    """Deterministic tracker double for TrackerDatabaseAdapter."""

    def __init__(self, script):
        self._script = script
        self._scan = 0

    def process(self, measurements, dt):
        tracks = self._script[self._scan]
        self._scan += 1
        return tracks


class TestTrackerDatabaseAdapter:
    def test_process_scan_syncs_database(self, db):
        from pytcl.trackers.multi_target import TrackStatus

        def mk(i, hits, misses, x):
            return _TrackTuple(
                i,
                np.array([x, 0.0]),
                np.eye(2),
                TrackStatus.TENTATIVE,
                hits,
                misses,
                0.0,
            )

        script = [
            [mk(1, 1, 0, 1.0)],
            [mk(1, 2, 0, 1.5), mk(2, 1, 0, 9.0)],
            [mk(1, 3, 0, 2.0), mk(2, 1, 5, 9.0)],
        ]
        adapter = TrackerDatabaseAdapter(
            db, _FakeTracker(script), confirm_hits=3, max_misses=5
        )

        out = adapter.process_scan([np.array([1.0, 0.0])], dt=1.0, timestamp=0.0)
        assert len(out) == 1
        assert db.get_track("trk_1")["status"] == "tentative"

        adapter.process_scan([np.array([1.5, 0.0]), np.array([9.0, 0.0])], 1.0, 1.0)
        assert db.get_track("trk_2")["status"] == "tentative"

        adapter.process_scan([np.array([2.0, 0.0])], 1.0, 2.0)
        assert db.get_track("trk_1")["status"] == "confirmed"  # hits >= 3
        assert db.get_track("trk_2")["status"] == "dead"  # misses >= 5

        # States persisted per scan
        np.testing.assert_array_equal(db.get_track_state("trk_1")["state"], [2.0, 0.0])
        hist = db.get_track_history("trk_1")
        np.testing.assert_array_equal(hist["timestamps"], [0.0, 1.0, 2.0])

        # All measurements stored as detections (1 + 2 + 1)
        assert len(db.retrieve_all_detections()) == 4

        tl = adapter.get_track_list()
        assert len(list(tl)) == 2


_ResultXP = namedtuple("_ResultXP", "x P")
_ResultXPy = namedtuple("_ResultXPy", "x P y")
_ResultXS = namedtuple("_ResultXS", "x S")
_ResultX = namedtuple("_ResultX", "x")


class TestStoreFilterResult:
    def test_with_p(self, db):
        db.initiate_track("t", np.zeros(2), np.eye(2), 0.0)
        store_filter_result(db, "t", _ResultXP(np.ones(2), np.eye(2) * 3.0), 1.0)
        st = db.get_track_state("t")
        np.testing.assert_array_equal(st["state"], [1.0, 1.0])
        np.testing.assert_array_equal(st["covariance"], np.eye(2) * 3.0)

    def test_with_cholesky_factor(self, db):
        db.initiate_track("t", np.zeros(2), np.eye(2), 0.0)
        S = np.array([[2.0, 0.0], [1.0, 1.0]])
        store_filter_result(db, "t", _ResultXS(np.ones(2), S), 1.0)
        np.testing.assert_array_equal(db.get_track_state("t")["covariance"], S @ S.T)

    def test_with_residual_and_update_type(self, db):
        db.initiate_track("t", np.zeros(2), np.eye(2), 0.0)
        res = np.array([0.25, -0.75])
        store_filter_result(
            db,
            "t",
            _ResultXPy(np.ones(2), np.eye(2), res),
            1.0,
            update_type="prediction",
        )
        hist = db.get_track_history("t", start_time=0.5)
        np.testing.assert_array_equal(hist["residuals"], [res])
        info = db.get_track("t")
        assert info["misses"] == 1  # prediction increments misses

    def test_missing_p_and_s_raises(self, db):
        db.initiate_track("t", np.zeros(2), np.eye(2), 0.0)
        with pytest.raises(ValueError):
            store_filter_result(db, "t", _ResultX(np.ones(2)), 1.0)


# =============================================================================
# MigrationHelper
# =============================================================================


class TestMigrationAnalysis:
    def test_analyze_source_string(self):
        code = (
            "from pytcl.dynamic_estimation.kalman import kf_predict, kf_update\n"
            "import pickle\n"
            "pred = kf_predict(x, P, F, Q)\n"
            "pickle.dump(tracks, open('t.pkl', 'wb'))\n"
        )
        result = MigrationHelper().analyze_v1_code(code)
        assert isinstance(result, AnalysisResult)
        assert result.filter_types == ["kf"]
        assert result.storage_patterns == ["pickle"]
        assert result.estimated_complexity == "low"
        assert len(result.detected_imports) == 1
        assert any("KalmanTrackAdapter" in r for r in result.recommendations)
        assert any("pickle" in r for r in result.recommendations)
        assert "kf" in repr(result)
        assert "Migration Analysis" in result.summary()

    def test_analyze_file_path(self, tmp_path):
        script = tmp_path / "legacy.py"
        script.write_text("ekf_predict(x, P, f, F, Q)\nnp.save('s.npy', states)\n")
        result = MigrationHelper().analyze_v1_code(str(script))
        assert result.filter_types == ["ekf"]
        assert result.storage_patterns == ["numpy"]

    def test_complexity_scaling(self):
        helper = MigrationHelper()
        multi = helper.analyze_v1_code("MHTTracker()\nkf_predict()")
        assert multi.estimated_complexity == "medium"
        heavy = helper.analyze_v1_code(
            "kf_predict(); ekf_predict(); ukf_predict(); ckf_predict(); hinf_predict()"
        )
        assert heavy.estimated_complexity == "high"

    def test_no_patterns(self):
        result = MigrationHelper().analyze_v1_code("print('hello')")
        assert result.filter_types == []
        assert result.storage_patterns == []
        assert result.estimated_complexity == "low"
        assert len(result.recommendations) >= 1  # still suggests TrackDatabaseManager


class TestMigrationConversion:
    def _legacy_via_pickle(self, tmp_path, with_covs=True, with_times=True):
        rng = np.random.default_rng(9)
        legacy = {}
        for i in range(3):
            n = 5 + i
            entry = {"states": rng.normal(size=(n, 4))}
            entry["covariances"] = (
                np.array([np.eye(4) * (j + 1) for j in range(n)]) if with_covs else None
            )
            entry["timestamps"] = np.linspace(0, 10, n) if with_times else None
            legacy[f"trk_{i}"] = entry
        # Round-trip through an actual pickle file (synthetic legacy format)
        pkl = tmp_path / "legacy.pkl"
        with open(pkl, "wb") as fh:
            pickle.dump(legacy, fh)
        with open(pkl, "rb") as fh:
            return pickle.load(fh)

    def test_convert_legacy_to_sql_round_trip(self, tmp_path):
        legacy = self._legacy_via_pickle(tmp_path)
        out_db = str(tmp_path / "converted.db")
        count = MigrationHelper().convert_legacy_tracks_to_sql(legacy, out_db)
        assert count == 3
        assert os.path.exists(out_db)

        db = TrackDatabaseManager(out_db)
        db.open(mode="r")
        for tid, data in legacy.items():
            hist = db.get_track_history(tid)
            np.testing.assert_array_equal(hist["states"], data["states"])
            np.testing.assert_array_equal(hist["covariances"], data["covariances"])
            np.testing.assert_array_equal(hist["timestamps"], data["timestamps"])
            assert db.get_track(tid)["status"] == "confirmed"
        db.close()

    def test_convert_legacy_defaults(self, tmp_path):
        legacy = self._legacy_via_pickle(tmp_path, with_covs=False, with_times=False)
        out_db = str(tmp_path / "defaults.db")
        MigrationHelper().convert_legacy_tracks_to_sql(legacy, out_db)

        db = TrackDatabaseManager(out_db)
        db.open(mode="r")
        for tid, data in legacy.items():
            n = len(data["states"])
            hist = db.get_track_history(tid)
            np.testing.assert_array_equal(hist["timestamps"], np.arange(n))
            np.testing.assert_array_equal(
                hist["covariances"], np.array([np.eye(4)] * n)
            )
        db.close()

    def test_convert_legacy_to_hdf5_round_trip(self, tmp_path):
        legacy = self._legacy_via_pickle(tmp_path)
        out_h5 = str(tmp_path / "converted.h5")
        count = MigrationHelper().convert_legacy_tracks_to_hdf5(
            legacy, out_h5, scenario_id="mig"
        )
        assert count == 3

        store = TrackHDF5Storage(out_h5)
        store.open(mode="r")
        assert store.list_scenarios() == ["mig"]
        for tid, data in legacy.items():
            out = store.retrieve_track(tid, scenario_id="mig")
            np.testing.assert_array_equal(out["states"], data["states"])
            np.testing.assert_array_equal(out["timestamps"], data["timestamps"])
        store.close()


class TestMigrationTemplates:
    @pytest.mark.parametrize(
        "backend,filter_type",
        [
            ("sql", "kf"),
            ("sql", "ekf"),
            ("sql", "ukf"),
            ("hdf5", "kf"),
            ("both", "kf"),
            ("sql", "particle"),  # falls back to sql/kf
            ("hdf5", "imm"),  # falls back to hdf5/kf
            ("nosuch", "nosuch"),  # falls back to sql/kf
        ],
    )
    def test_templates_are_valid_python(self, backend, filter_type):
        code = MigrationHelper().generate_v2_template(
            backend=backend, filter_type=filter_type, n_targets=2
        )
        assert isinstance(code, str) and code
        compile(code, "<template>", "exec")  # must be syntactically valid

    def test_checklist(self):
        checklist = MigrationHelper.generate_migration_checklist()
        assert "Migration Checklist" in checklist
        assert "convert_legacy_tracks_to_sql" in checklist
