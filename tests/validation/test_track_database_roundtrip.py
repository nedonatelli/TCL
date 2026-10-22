"""Round-trip and schema-version tests for TrackDatabaseManager."""

import ntpath
import os
import sqlite3
import warnings

import numpy as np
import pytest

from pytcl.io.track_database import TrackDatabaseManager
from tests.validation._v2110 import shipped_module

SHAPES = [(3,), (1, 3), (2, 3), (6,), (3, 3)]


@pytest.mark.parametrize("shape", SHAPES)
def test_array_round_trips_at_every_shape(tmp_path, shape):
    db = TrackDatabaseManager(tmp_path / "t.db")
    db.open()
    value = np.arange(np.prod(shape), dtype=float).reshape(shape)
    db.store_detection("d", value, "s", 0.0)
    got = db.retrieve_detection("d")["measurement"]
    assert got.shape == value.shape
    assert np.array_equal(got, value), f"{shape}: {got}"
    db.close()


def test_covariance_round_trips(tmp_path):
    db = TrackDatabaseManager(tmp_path / "t.db")
    db.open()
    P = np.array([[4.0, 1.0], [1.0, 9.0]])
    db.store_detection("d", np.zeros(2), "s", 0.0, covariance=P)
    assert np.array_equal(db.retrieve_detection("d")["covariance"], P)
    db.close()


def test_non_square_covariance_round_trips(tmp_path):
    db = TrackDatabaseManager(tmp_path / "t.db")
    db.open()
    P = np.arange(6, dtype=float).reshape(2, 3)
    db.store_detection("d", np.zeros(2), "s", 0.0, covariance=P)
    assert np.array_equal(db.retrieve_detection("d")["covariance"], P)
    db.close()


def _version(path):
    con = sqlite3.connect(path)
    try:
        return con.execute(
            "SELECT value FROM _pytcl_track_db_metadata WHERE key='schema_version'"
        ).fetchone()[0]
    finally:
        con.close()


def test_new_database_is_schema_2(tmp_path):
    db = TrackDatabaseManager(tmp_path / "t.db")
    db.open()
    db.close()
    assert _version(tmp_path / "t.db") == "2.0"


def _write_v1_database(path):
    """A version-1.0 file written by the shipped v2.11.0 TrackDatabaseManager."""
    meas = np.array([1.0, 2.0, 3.0])
    cov = np.array([[4.0, 1.0, 0.0], [1.0, 9.0, 0.0], [0.0, 0.0, 1.0]])
    old = shipped_module("pytcl/io/track_database.py").TrackDatabaseManager(path)
    old.open("w")
    old.store_detection("vec", meas, "s", 0.0, covariance=cov)
    old.store_detection("two_d", np.arange(6, dtype=float).reshape(2, 3), "s", 1.0)
    old.close()
    return meas, cov


@pytest.mark.parametrize("mode", ["r", "a"])
def test_schema_1_database_reads_under_old_interpretation(tmp_path, mode):
    path = tmp_path / "v1.db"
    meas, cov = _write_v1_database(path)
    db = TrackDatabaseManager(path)
    db.open(mode=mode)
    vec = db.retrieve_detection("vec")
    assert np.array_equal(vec["measurement"], meas)
    assert np.array_equal(vec["covariance"], cov)
    # Truncated at write time under v1: the old read result is [0, 1].
    assert np.array_equal(db.retrieve_detection("two_d")["measurement"], [0.0, 1.0])
    db.close()
    assert _version(path) == "1.0"


def test_writes_to_schema_1_file_stay_legacy_and_one_version(tmp_path):
    path = tmp_path / "v1.db"
    meas, cov = _write_v1_database(path)
    db = TrackDatabaseManager(path)
    db.open(mode="a")
    db.store_detection("new", np.array([7.0, 8.0]), "s", 2.0, covariance=np.eye(2))
    got = db.retrieve_detection("new")
    assert np.array_equal(got["measurement"], [7.0, 8.0])
    assert np.array_equal(got["covariance"], np.eye(2))
    assert np.array_equal(db.retrieve_detection("vec")["measurement"], meas)
    with pytest.raises(ValueError, match="schema 1.0.*schema 2.0"):
        db.store_detection("tw", np.arange(6.0).reshape(2, 3), "s", 3.0)
    with pytest.raises(ValueError, match="schema 1.0"):
        db.store_detection("nc", np.zeros(2), "s", 3.0, covariance=np.eye(3))
    assert [d["detection_id"] for d in db.retrieve_detections()].count("tw") == 0
    db.close()
    assert _version(path) == "1.0"

    con = sqlite3.connect(path)
    blob = con.execute(
        "SELECT measurement FROM detections WHERE detection_id='new'"
    ).fetchone()[0]
    con.close()
    assert blob == np.array([7.0, 8.0]).tobytes()


def test_legacy_valid_write_is_accepted(tmp_path):
    path = tmp_path / "v1.db"
    _write_v1_database(path)
    db = TrackDatabaseManager(path)
    db.open(mode="a")
    db.store_detection("n", np.ones(3), "s", 2.0, covariance=np.eye(3))
    assert np.array_equal(db.retrieve_detection("n")["covariance"], np.eye(3))
    db.close()


@pytest.mark.parametrize("schema", ["new", "v1"])
class TestTrackStateShapes:
    def _db(self, tmp_path, schema):
        path = tmp_path / "t.db"
        if schema == "v1":
            _write_v1_database(path)
        db = TrackDatabaseManager(path)
        db.open(mode="a")
        return db

    def test_valid_state_round_trips(self, tmp_path, schema):
        db = self._db(tmp_path, schema)
        P = np.array([[4.0, 1.0, 0.0], [1.0, 9.0, 0.0], [0.0, 0.0, 1.0]])
        db.initiate_track("t", np.array([1.0, 2.0, 3.0]), P, 0.0)
        got = db.get_track_state("t")
        assert np.array_equal(got["state"], [1.0, 2.0, 3.0])
        assert np.array_equal(got["covariance"], P)
        db.close()

    def test_row_vector_state_raises(self, tmp_path, schema):
        db = self._db(tmp_path, schema)
        with pytest.raises(ValueError, match=r"shape \(n,\).*\(1, 3\)"):
            db.initiate_track("t", np.ones((1, 3)), np.eye(3), 0.0)
        db.close()

    def test_column_state_reads_back_as_v2_11_0_stored_it(self, tmp_path, schema):
        P = np.array([[4.0, 1.0, 0.0], [1.0, 9.0, 0.0], [0.0, 0.0, 1.0]])
        col = np.array([[1.0], [2.0], [3.0]])
        old = shipped_module("pytcl/io/track_database.py").TrackDatabaseManager(
            tmp_path / "old.db"
        )
        old.open("w")
        old.initiate_track("t", col, P, 0.0)
        old.update_track_state("t", 2 * col, 2 * P, 1.0)
        old.store_track_history("t", 3 * col[None], 3 * P[None], [2.0])
        old_rows = old.get_track_history("t")
        old.close()

        db = self._db(tmp_path, schema)
        db.initiate_track("t", col, P, 0.0)
        db.update_track_state("t", 2 * col, 2 * P, 1.0)
        db.store_track_history("t", 3 * col[None], 3 * P[None], [2.0])
        rows = db.get_track_history("t")
        db.close()
        assert rows["states"].shape == old_rows["states"].shape == (3, 3)
        assert np.array_equal(rows["states"], old_rows["states"])
        assert np.array_equal(rows["covariances"], old_rows["covariances"])

    def test_non_square_covariance_raises(self, tmp_path, schema):
        db = self._db(tmp_path, schema)
        with pytest.raises(ValueError, match=r"\(3, 3\).*\(2, 3\)"):
            db.initiate_track("t", np.ones(3), np.ones((2, 3)), 0.0)
        db.initiate_track("t", np.ones(3), np.eye(3), 0.0)
        with pytest.raises(ValueError, match=r"\(3, 3\)"):
            db.update_track_state("t", np.ones(3), np.eye(2), 1.0)
        with pytest.raises(ValueError, match=r"shape \(n,\)"):
            db.update_track_state("t", np.ones((1, 3)), np.eye(3), 1.0)
        with pytest.raises(ValueError, match=r"\(3, 3\)"):
            db.update_track_state("t", np.ones((3, 1)), np.ones(9), 1.0)
        db.close()

    def test_failed_initiate_leaves_no_track(self, tmp_path, schema):
        db = self._db(tmp_path, schema)
        with pytest.raises(ValueError):
            db.initiate_track("t", np.ones(3), np.ones((2, 3)), 0.0)
        db._connection.commit()
        with pytest.raises(KeyError):
            db.get_track("t")
        db.close()


def _seeded(path, mode="w"):
    db = TrackDatabaseManager(path)
    db.open(mode)
    db.store_detection("old", np.zeros(2), "s", 0.0)
    db.initiate_track("trk", np.zeros(2), np.eye(2), 0.0)
    db.close()


def test_read_mode_rejects_writes(tmp_path):
    path = tmp_path / "t.db"
    TrackDatabaseManager(path).open("w")
    db = TrackDatabaseManager(path)
    db.open("r")
    with pytest.raises(PermissionError, match="read-only"):
        db.store_detection("d", np.zeros(2), "s", 0.0)
    assert db.retrieve_all_detections() == []
    db.close()


_WRITERS = {
    "store_detection": lambda db: db.store_detection("d", np.zeros(2), "s", 0.0),
    "associate_detection": lambda db: db.associate_detection("old", "trk"),
    "initiate_track": lambda db: db.initiate_track("t2", np.zeros(2), np.eye(2), 0.0),
    "confirm_track": lambda db: db.confirm_track("trk"),
    "update_track_state": lambda db: db.update_track_state(
        "trk", np.ones(2), np.eye(2), 1.0
    ),
    "store_track_history": lambda db: db.store_track_history(
        "trk", np.ones((1, 2)), np.eye(2)[None], [1.0]
    ),
    "mark_track_tentative": lambda db: db.mark_track_tentative("trk"),
    "mark_track_confirmed": lambda db: db.mark_track_confirmed("trk"),
    "mark_track_coasting": lambda db: db.mark_track_coasting("trk"),
    "mark_track_dead": lambda db: db.mark_track_dead("trk"),
    "prune_old_detections": lambda db: db.prune_old_detections(-1.0),
    "prune_dead_tracks": lambda db: db.prune_dead_tracks(-1.0),
    "merge_tracks": lambda db: db.merge_tracks("trk", "trk2"),
}


@pytest.mark.parametrize("name", sorted(_WRITERS))
def test_every_write_method_is_refused_in_read_mode(tmp_path, name):
    path = tmp_path / "t.db"
    _seeded(path)
    before = path.read_bytes()
    db = TrackDatabaseManager(path)
    db.open("r")
    with pytest.raises(PermissionError, match="read-only"):
        _WRITERS[name](db)
    db.close()
    assert path.read_bytes() == before


def test_write_mode_preserves_rows_and_warns(tmp_path):
    path = tmp_path / "t.db"
    db = TrackDatabaseManager(path)
    db.open("w")
    db.store_detection("old", np.zeros(2), "s", 0.0)
    db.close()
    db2 = TrackDatabaseManager(path)
    with pytest.warns(FutureWarning, match="does not truncate.*v2.12.0"):
        db2.open("w")
    assert [d["detection_id"] for d in db2.retrieve_all_detections()] == ["old"]
    db2.close()


def test_write_mode_on_new_or_empty_file_does_not_warn(tmp_path):
    path = tmp_path / "t.db"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for _ in range(2):
            db = TrackDatabaseManager(path)
            db.open("w")
            db.close()


def test_write_mode_keeps_schema_1_file_version(tmp_path):
    path = tmp_path / "t.db"
    _write_v1_database(path)
    db = TrackDatabaseManager(path)
    with pytest.warns(FutureWarning):
        db.open("w")
    db.close()
    assert _version(path) == "1.0"


def test_append_mode_keeps_rows(tmp_path):
    path = tmp_path / "t.db"
    _seeded(path)
    db = TrackDatabaseManager(path)
    db.open("a")
    assert [d["detection_id"] for d in db.retrieve_all_detections()] == ["old"]
    db.close()


def test_read_mode_missing_file_raises_and_creates_nothing(tmp_path):
    path = tmp_path / "missing.db"
    with pytest.raises(FileNotFoundError):
        TrackDatabaseManager(path).open("r")
    assert not path.exists()


def test_sql_storage_read_mode_rejects_writes(tmp_path):
    from pytcl.io.sql_storage import SQLStorage

    path = str(tmp_path / "s.db")
    with SQLStorage() as s:
        s.open(path, "w")
        s.store_array("a", np.arange(3.0))
    before = (tmp_path / "s.db").read_bytes()
    with SQLStorage() as s:
        s.open(path, "r")
        for write in (
            lambda: s.store_array("b", np.zeros(2)),
            lambda: s.store_scalar("c", 1.0),
            lambda: s.store_group("g"),
            lambda: s.delete("a"),
        ):
            with pytest.raises(PermissionError, match="read-only"):
                write()
        assert np.array_equal(s.retrieve_array("a"), np.arange(3.0))
    assert (tmp_path / "s.db").read_bytes() == before


def test_sql_storage_write_mode_preserves_data_and_warns(tmp_path):
    from pytcl.io.sql_storage import SQLStorage

    path = str(tmp_path / "s.db")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with SQLStorage() as s:
            s.open(path, "w")
            s.store_array("old", np.arange(3.0))
    with SQLStorage() as s:
        with pytest.warns(FutureWarning, match="does not truncate.*v2.12.0"):
            s.open(path, "w")
        assert np.array_equal(s.retrieve_array("old"), np.arange(3.0))


def _confirmed_track(hits=5, misses=1):
    from pytcl.trackers.multi_target import Track, TrackStatus

    return Track(
        id=7,
        state=np.array([1.0, 2.0, 3.0, 4.0]),
        covariance=np.eye(4),
        status=TrackStatus.CONFIRMED,
        hits=hits,
        misses=misses,
        time=3.0,
    )


def _db_with_track(tmp_path):
    track = _confirmed_track()
    db = TrackDatabaseManager(tmp_path / "t.db")
    db.open()
    db.store_from_track(track)
    return db, db.retrieve_all_tracks()[0]


def test_track_row_id_is_stable_across_processes():
    import subprocess
    import sys

    script = (
        "from pytcl.io.track_database import _track_row_id; print(_track_row_id('t-1'))"
    )
    ids = {
        subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
        ).stdout.strip()
        for seed in ("1", "2", "3")
    }
    assert len(ids) == 1, f"row id varies per process: {ids}"
    assert 0 <= int(next(iter(ids))) < 2**31


def test_track_to_pytcl_id_matches_row_id(tmp_path):
    from pytcl.io.track_database import _track_row_id

    db, row = _db_with_track(tmp_path)
    assert db.track_to_pytcl(row["track_id"]).id == _track_row_id(row["track_id"])
    db.close()


def test_store_from_track_preserves_status_and_counts(tmp_path):
    db, row = _db_with_track(tmp_path)
    assert row["status"] == "confirmed"
    assert (row["hits"], row["misses"]) == (5, 1)
    db.close()


def test_store_from_track_restore_preserves_counts_on_existing_row(tmp_path):
    db, row = _db_with_track(tmp_path)
    db.store_from_track(_confirmed_track(hits=9, misses=2)._replace(time=4.0))
    row = db.get_track(row["track_id"])
    assert row["status"] == "confirmed"
    assert (row["hits"], row["misses"]) == (9, 2)
    db.close()


def test_store_from_track_still_refused_in_read_mode(tmp_path):
    db, _ = _db_with_track(tmp_path)
    db.close()
    db = TrackDatabaseManager(tmp_path / "t.db")
    db.open("r")
    with pytest.raises(Exception, match="read"):
        db.store_from_track(_confirmed_track())
    db.close()


def test_associating_an_unknown_track_raises_and_writes_nothing(tmp_path):
    db = TrackDatabaseManager(tmp_path / "t.db")
    db.open()
    db.store_detection("d-1", np.zeros(2), "s", 0.0)
    with pytest.raises(KeyError, match="no such track"):
        db.associate_detection("d-1", "does-not-exist")
    assert db.retrieve_detection("d-1")["association_status"] == "unassociated"
    assert (
        db._cursor.execute("SELECT COUNT(*) FROM track_associations").fetchone()[0] == 0
    )
    db.close()


def test_self_merge_is_rejected_and_changes_nothing(tmp_path):
    db, row = _db_with_track(tmp_path)
    with pytest.raises(ValueError, match="itself"):
        db.merge_tracks(row["track_id"], row["track_id"])
    after = db.get_track(row["track_id"])
    assert after["hits"] == 5
    assert after["status"] == "confirmed"
    db.close()


def test_retrieve_detections_limit_is_bound_and_validated(tmp_path):
    db = TrackDatabaseManager(tmp_path / "t.db")
    db.open()
    for i in range(4):
        db.store_detection(f"d{i}", np.zeros(2), "s", float(i))
    assert len(db.retrieve_detections(limit=2)) == 2
    assert db.retrieve_detections(limit=0) == []
    for bad in ("1; DROP TABLE detections", True, 2.0):
        with pytest.raises(ValueError, match="limit"):
            db.retrieve_detections(limit=bad)
    assert len(db.retrieve_detections()) == 4
    db.close()


@pytest.mark.parametrize("limit", [-1, -5, np.int64(-1)])
def test_negative_limit_means_no_limit_as_in_v2_11_0(tmp_path, limit):
    ids = []
    for module, name in (
        (shipped_module("pytcl/io/track_database.py"), "old.db"),
        (None, "new.db"),
    ):
        cls = TrackDatabaseManager if module is None else module.TrackDatabaseManager
        db = cls(tmp_path / name)
        db.open()
        for i in range(4):
            db.store_detection(f"d{i}", np.zeros(2), "s", float(i))
        ids.append([d["detection_id"] for d in db.retrieve_detections(limit=limit)])
        db.close()
    assert ids[1] == ids[0] == ["d0", "d1", "d2", "d3"]


def _old_and_new(tmp_path, schema="new"):
    old = shipped_module("pytcl/io/track_database.py").TrackDatabaseManager(
        tmp_path / "old.db"
    )
    old.open("w")
    path = tmp_path / "new.db"
    if schema == "v1":
        _write_v1_database(path)
    new = TrackDatabaseManager(path)
    new.open("a")
    return old, new


@pytest.mark.parametrize("schema", ["new", "v1"])
@pytest.mark.parametrize("shape", [(3, 1), (1, 1), (3, 1, 1)])
def test_column_measurement_reads_back_as_v2_11_0_stored_it(tmp_path, schema, shape):
    col = np.arange(1.0, 1.0 + shape[0]).reshape(shape)
    cov = 2 * np.eye(shape[0])
    got = []
    for db in _old_and_new(tmp_path, schema):
        db.store_detection("c", col, "s", 5.0, covariance=cov)
        got.append(db.retrieve_detection("c"))
        db.close()
    old, new = got
    assert new["measurement"].shape == old["measurement"].shape == (shape[0],)
    assert np.array_equal(new["measurement"], old["measurement"])
    assert np.array_equal(new["covariance"], old["covariance"])


def test_compat_adapter_column_x0_stores_as_v2_11_0_did(tmp_path):
    from pytcl.io.compat import KalmanTrackAdapter

    x0 = np.array([[1.0], [2.0], [3.0], [4.0]])
    P0 = np.diag([1.0, 2.0, 3.0, 4.0])
    old, new = _old_and_new(tmp_path)
    old.initiate_track("trk", x0, P0, 0.0)
    with pytest.warns(DeprecationWarning):
        adapter = KalmanTrackAdapter(new, "trk", np.eye(4), np.eye(2, 4), P0, P0)
    adapter.initialize(x0, P0, timestamp=0.0)
    want, got = old.get_track_state("trk"), new.get_track_state("trk")
    old.close()
    new.close()
    assert got["state"].shape == want["state"].shape == (4,)
    assert np.array_equal(got["state"], want["state"])
    assert np.array_equal(got["covariance"], want["covariance"])


def _mht_track(n_hits=4, n_misses=1):
    from pytcl.trackers.hypothesis import MHTTrack, MHTTrackStatus

    return MHTTrack(
        id=3,
        state=np.array([1.0, 2.0]),
        covariance=np.eye(2),
        score=0.5,
        status=MHTTrackStatus.CONFIRMED,
        history=[0, -1],
        parent_id=-1,
        scan_created=0,
        n_hits=n_hits,
        n_misses=n_misses,
    )


def test_store_from_track_accepts_mht_track_and_keeps_its_counts(tmp_path):
    db = TrackDatabaseManager(tmp_path / "t.db")
    db.open()
    db.store_from_track(_mht_track(), timestamp=1.0)
    row = db.get_track("trk_3")
    assert (row["status"], row["hits"], row["misses"]) == ("confirmed", 4, 1)
    db.store_from_track(_mht_track(n_hits=6, n_misses=0), timestamp=2.0)
    row = db.get_track("trk_3")
    assert (row["status"], row["hits"], row["misses"]) == ("confirmed", 6, 0)
    assert np.array_equal(db.get_track_state("trk_3")["state"], [1.0, 2.0])
    db.close()


_COLUMNS = ("status", "birth_time", "last_update_time", "hits", "misses")


def _stored(db, obj, times):
    for t in times:
        db.store_from_track(obj._replace(time=t))
    row = db.get_track(f"trk_{obj.id}")
    return {k: row[k] for k in _COLUMNS}


def test_status_none_track_stores_the_v2_11_0_row(tmp_path):
    from collections import namedtuple

    Bare = namedtuple("Bare", "id state covariance status time")
    obj = Bare(5, np.array([1.0, 2.0]), np.eye(2), None, 1.0)
    old, new = _old_and_new(tmp_path)
    want = _stored(old, obj, (1.0, 2.0, 3.0))
    got = _stored(new, obj, (1.0, 2.0, 3.0))
    old.close()
    new.close()
    assert want == {
        "status": "tentative",
        "birth_time": 1.0,
        "last_update_time": 3.0,
        "hits": 2,
        "misses": 0,
    }
    assert got == want


def test_status_none_with_counts_keeps_v2_11_0_status_and_stores_counts(tmp_path):
    from collections import namedtuple

    Counted = namedtuple("Counted", "id state covariance status hits misses time")
    obj = Counted(5, np.array([1.0, 2.0]), np.eye(2), None, 7, 2, 1.0)
    old, new = _old_and_new(tmp_path)
    want = _stored(old, obj, (1.0, 2.0))
    got = _stored(new, obj, (1.0, 2.0))
    old.close()
    new.close()
    assert got["status"] == want["status"] == "tentative"
    assert (got["hits"], got["misses"]) == (7, 2)


@pytest.mark.parametrize(
    "path, uri",
    [
        (r"C:\Users\a b\t?#%.db", "file:///C:/Users/a%20b/t%3F%23%25.db?mode=ro"),
        (r"Z:\x.db", "file:///Z:/x.db?mode=ro"),
        (r"\\server\share\dir\t.db", "file:////server/share/dir/t.db?mode=ro"),
    ],
)
def test_read_only_uri_for_windows_paths(path, uri):
    from pytcl.io.sql_storage import _read_only_uri

    assert _read_only_uri(path, ntpath) == uri


def test_read_mode_opens_a_path_with_uri_metacharacters(tmp_path):
    # "?" is not a legal Windows file-name character.
    path = tmp_path / ("a b#%.db" if os.name == "nt" else "a b?#%.db")
    db = TrackDatabaseManager(path)
    db.open("w")
    db.store_detection("d", np.array([1.0, 2.0]), "s", 0.0)
    db.close()
    db = TrackDatabaseManager(path)
    db.open("r")
    assert np.array_equal(db.retrieve_detection("d")["measurement"], [1.0, 2.0])
    db.close()
