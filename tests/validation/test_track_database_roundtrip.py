"""Round-trip and schema-version tests for TrackDatabaseManager."""

import sqlite3

import numpy as np
import pytest

from pytcl.io.track_database import TrackDatabaseManager

SHAPES = [(3,), (1, 3), (3, 1), (2, 3), (6,), (3, 3)]


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
    """A version-1.0 file byte-for-byte as the pre-fix code wrote it."""
    meas = np.array([1.0, 2.0, 3.0])
    cov = np.array([[4.0, 1.0, 0.0], [1.0, 9.0, 0.0], [0.0, 0.0, 1.0]])
    twod = np.arange(6, dtype=float).reshape(2, 3)
    con = sqlite3.connect(path)
    con.execute("""CREATE TABLE detections (
        detection_id TEXT PRIMARY KEY,
        timestamp REAL NOT NULL,
        sensor_id TEXT NOT NULL,
        measurement BLOB NOT NULL,
        measurement_dim INTEGER NOT NULL,
        measurement_dtype TEXT NOT NULL DEFAULT 'float64',
        covariance BLOB,
        association_status TEXT DEFAULT 'unassociated',
        associated_track_id TEXT,
        association_confidence REAL DEFAULT 0.0,
        metadata TEXT)""")
    con.execute(
        "CREATE TABLE _pytcl_track_db_metadata "
        "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    con.execute("INSERT INTO _pytcl_track_db_metadata VALUES ('schema_version','1.0')")
    rows = [
        ("vec", 0.0, "s", meas.tobytes(), 3, "float64", cov.tobytes()),
        ("two_d", 1.0, "s", twod.tobytes(), 2, "float64", None),
    ]
    con.executemany(
        "INSERT INTO detections (detection_id, timestamp, sensor_id, measurement,"
        " measurement_dim, measurement_dtype, covariance) VALUES (?,?,?,?,?,?,?)",
        rows,
    )
    con.commit()
    con.close()
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

    def test_non_square_covariance_raises(self, tmp_path, schema):
        db = self._db(tmp_path, schema)
        with pytest.raises(ValueError, match=r"\(3, 3\).*\(2, 3\)"):
            db.initiate_track("t", np.ones(3), np.ones((2, 3)), 0.0)
        db.initiate_track("t", np.ones(3), np.eye(3), 0.0)
        with pytest.raises(ValueError, match=r"\(3, 3\)"):
            db.update_track_state("t", np.ones(3), np.eye(2), 1.0)
        with pytest.raises(ValueError, match=r"shape \(n,\)"):
            db.update_track_state("t", np.ones((3, 1)), np.eye(3), 1.0)
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


def test_write_mode_truncates(tmp_path):
    path = tmp_path / "t.db"
    db = TrackDatabaseManager(path)
    db.open("w")
    db.store_detection("old", np.zeros(2), "s", 0.0)
    db.close()
    db2 = TrackDatabaseManager(path)
    db2.open("w")
    assert db2.retrieve_all_detections() == []
    assert db2.retrieve_all_tracks() == []
    db2.close()


def test_write_mode_over_schema_1_file_yields_fresh_schema_2(tmp_path):
    path = tmp_path / "t.db"
    _write_v1_database(path)
    db = TrackDatabaseManager(path)
    db.open("w")
    assert db.retrieve_all_detections() == []
    db.store_detection("row", np.arange(6.0).reshape(2, 3), "s", 0.0)
    db.close()
    assert _version(path) == "2.0"


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


def test_sql_storage_write_mode_truncates(tmp_path):
    from pytcl.io.sql_storage import SQLStorage

    path = str(tmp_path / "s.db")
    with SQLStorage() as s:
        s.open(path, "w")
        s.store_array("old", np.arange(3.0))
    with SQLStorage() as s:
        s.open(path, "w")
        assert s.list_keys() == []
        with pytest.raises(KeyError):
            s.retrieve_array("old")
