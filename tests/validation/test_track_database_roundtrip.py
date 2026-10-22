"""Round-trip and schema-version tests for TrackDatabaseManager."""

import sqlite3
import warnings

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
    with pytest.warns(UserWarning, match="legacy schema 1.0"):
        db.store_detection("tw", np.arange(6.0).reshape(2, 3), "s", 3.0)
    db.close()
    assert _version(path) == "1.0"

    con = sqlite3.connect(path)
    blob = con.execute(
        "SELECT measurement FROM detections WHERE detection_id='new'"
    ).fetchone()[0]
    con.close()
    assert blob == np.array([7.0, 8.0]).tobytes()


def test_legacy_1d_write_does_not_warn(tmp_path):
    path = tmp_path / "v1.db"
    _write_v1_database(path)
    db = TrackDatabaseManager(path)
    db.open(mode="a")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        db.store_detection("n", np.ones(3), "s", 2.0, covariance=np.eye(3))
    db.close()
