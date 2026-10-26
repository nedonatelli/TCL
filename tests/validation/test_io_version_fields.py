"""Stored schema versions are validated on read, and never rewritten on reopen.

``schema_version=99`` loaded silently from every io backend through v2.11.0;
v2.11.1 warned and loaded it anyway. A version the reader does not know now
raises FormatError and leaves the file untouched; a session newer than this
pytcl keeps its pre-existing FormatError.
"""

import sqlite3
import warnings

import h5py
import msgspec
import numpy as np
import pytest

from pytcl.core.exceptions import DependencyError, FormatError
from pytcl.io import (
    TrackDatabaseManager,
    TrackHDF5Storage,
    load_session_file,
    save_session_file,
)
from pytcl.io.asdf_io import (
    load_states_asdf,
    save_states_asdf,
)
from pytcl.io.session import SESSION_SCHEMA_VERSION
from pytcl.trackers import MultiTargetTracker

_data_skip = (FileNotFoundError, DependencyError)

BACKENDS = ["session", "asdf", "hdf5_track", "track_db"]
UNKNOWN = {"session": 0, "asdf": 99, "hdf5_track": "99.0", "track_db": "99.0"}
CURRENT = {
    "session": SESSION_SCHEMA_VERSION,
    "asdf": 1,
    "hdf5_track": "1.0",
    "track_db": "2.0",
}


def _tracker():
    t = MultiTargetTracker(
        4, 2, np.eye(4), np.eye(2, 4), 0.01 * np.eye(4), 0.1 * np.eye(2)
    )
    t.process([np.array([0.0, 0.0])], dt=1.0)
    return t


def _write_with_schema_version(tmp_path, backend, version):
    if backend == "session":
        path = tmp_path / "s.msgpack"
        save_session_file(_tracker(), path)
        raw = msgspec.msgpack.decode(path.read_bytes())
        raw["schema_version"] = version
        path.write_bytes(msgspec.msgpack.encode(raw))
    elif backend == "asdf":
        asdf = pytest.importorskip("asdf")
        path = tmp_path / "a.asdf"
        save_states_asdf(path, np.array([1.0, 2.0]), np.eye(2))
        with asdf.open(path) as af:
            tree = {"pytcl": dict(af.tree["pytcl"])}
            tree["pytcl"] = {k: np.array(v) for k, v in tree["pytcl"].items()}
        tree["pytcl"]["schema_version"] = version
        asdf.AsdfFile(tree).write_to(path)
    elif backend == "hdf5_track":
        path = tmp_path / "t.h5"
        with TrackHDF5Storage(str(path)) as store:
            store.open("w")
        with h5py.File(path, "a") as f:
            f["_metadata"].attrs["schema_version"] = version
    else:
        path = tmp_path / "t.db"
        with TrackDatabaseManager(str(path)) as db:
            db.open("w")
        con = sqlite3.connect(path)
        con.execute(
            "UPDATE _pytcl_track_db_metadata SET value = ? "
            "WHERE key = 'schema_version'",
            (version,),
        )
        con.commit()
        con.close()
    return path


def _read(backend, path):
    if backend == "session":
        load_session_file(path)
    elif backend == "asdf":
        load_states_asdf(path)
    elif backend == "hdf5_track":
        with TrackHDF5Storage(str(path)) as store:
            store.open("a")
    else:
        with TrackDatabaseManager(str(path)) as db:
            db.open("a")


def _stored_version(backend, path):
    if backend == "session":
        return msgspec.msgpack.decode(path.read_bytes())["schema_version"]
    if backend == "asdf":
        import asdf

        with asdf.open(path) as af:
            return int(af.tree["pytcl"]["schema_version"])
    if backend == "hdf5_track":
        with h5py.File(path, "r") as f:
            return str(f["_metadata"].attrs["schema_version"])
    con = sqlite3.connect(path)
    try:
        return con.execute(
            "SELECT value FROM _pytcl_track_db_metadata WHERE key = 'schema_version'"
        ).fetchone()[0]
    finally:
        con.close()


@pytest.mark.parametrize("backend", BACKENDS)
def test_unknown_schema_version_is_rejected(tmp_path, backend):
    try:
        path = _write_with_schema_version(tmp_path, backend, UNKNOWN[backend])
        with pytest.raises(FormatError, match="schema version") as exc_info:
            _read(backend, path)
    except _data_skip as exc:
        pytest.skip(str(exc))
    msg = str(exc_info.value)
    assert repr(UNKNOWN[backend]) in msg
    assert "reads schema versions" in msg


@pytest.mark.parametrize("backend", BACKENDS)
def test_current_schema_version_does_not_warn(tmp_path, backend):
    try:
        path = _write_with_schema_version(tmp_path, backend, CURRENT[backend])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _read(backend, path)
    except _data_skip as exc:
        pytest.skip(str(exc))


@pytest.mark.parametrize("backend", ["hdf5_track", "track_db"])
def test_legacy_1_0_files_load_without_warning(tmp_path, backend):
    path = _write_with_schema_version(tmp_path, backend, "1.0")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _read(backend, path)


@pytest.mark.parametrize("backend", BACKENDS)
def test_reopening_does_not_rewrite_the_version(tmp_path, backend):
    version = {"session": 1, "asdf": 1, "hdf5_track": "1.0", "track_db": "1.0"}[backend]
    try:
        path = _write_with_schema_version(tmp_path, backend, version)
        _read(backend, path)
        assert _stored_version(backend, path) == version
    except _data_skip as exc:
        pytest.skip(str(exc))


def _store(backend, path):
    if backend == "hdf5_track":
        return TrackHDF5Storage(str(path))
    return TrackDatabaseManager(str(path))


def _handle_of(store):
    if isinstance(store, TrackHDF5Storage):
        return store._file
    return store._connection


@pytest.mark.parametrize("backend", ["hdf5_track", "track_db"])
@pytest.mark.parametrize("mode", ["r", "a"])
def test_rejected_file_is_byte_identical_and_released(tmp_path, backend, mode):
    path = _write_with_schema_version(tmp_path, backend, "99.0")
    before = path.read_bytes()
    store = _store(backend, path)
    with pytest.raises(FormatError, match="schema version"):
        store.open(mode)
    assert path.read_bytes() == before
    assert _handle_of(store) is None
    path.unlink()


def test_rejected_track_db_with_a_different_layout_is_not_migrated(tmp_path):
    # A newer schema need not have this pytcl's tables and indices; v2.11.1
    # created them (CREATE ... IF NOT EXISTS, then commit) before warning.
    path = _write_with_schema_version(tmp_path, "track_db", "99.0")
    con = sqlite3.connect(path)
    con.execute("DROP TABLE track_associations")
    con.execute("DROP INDEX idx_det_time")
    con.commit()
    con.close()
    before = path.read_bytes()
    db = TrackDatabaseManager(str(path))
    with pytest.raises(FormatError, match="schema version"):
        db.open("a")
    assert db._connection is None
    assert path.read_bytes() == before


def test_session_newer_than_supported_still_raises(tmp_path):
    path = _write_with_schema_version(tmp_path, "session", 99)
    with pytest.raises(FormatError, match="newer than this pytcl supports"):
        load_session_file(path)


def _reject_in_mode(backend, tmp_path, mode="a"):
    path = _write_with_schema_version(tmp_path, backend, UNKNOWN[backend])
    with pytest.raises(FormatError, match="schema version"):
        if backend in ("hdf5_track", "track_db"):
            with _store(backend, path) as store:
                store.open(mode)
        else:
            _read(backend, path)


@pytest.mark.parametrize("backend", ["session", "asdf", "hdf5_track", "track_db"])
def test_every_backend_rejects_through_its_public_reader(tmp_path, backend):
    try:
        _reject_in_mode(backend, tmp_path)
    except _data_skip as exc:
        pytest.skip(str(exc))


@pytest.mark.parametrize("mode", ["r", "a"])
def test_track_db_rejects_in_every_reading_mode(tmp_path, mode):
    _reject_in_mode("track_db", tmp_path, mode)


def test_load_session_bytes_rejects_version_0(tmp_path):
    from pytcl.io import load_session

    path = _write_with_schema_version(tmp_path, "session", 0)
    with pytest.raises(FormatError, match="schema version 0"):
        load_session(path.read_bytes())


@pytest.mark.parametrize("stored", ["abc", ""])
def test_unparseable_track_db_version_is_rejected(tmp_path, stored):
    from tests.validation.test_track_database_roundtrip import _write_v1_database

    path = tmp_path / "v1.db"
    _write_v1_database(path)
    con = sqlite3.connect(path)
    con.execute(
        "UPDATE _pytcl_track_db_metadata SET value = ? WHERE key = 'schema_version'",
        (stored,),
    )
    con.commit()
    con.close()
    before = path.read_bytes()
    db = TrackDatabaseManager(path)
    with pytest.raises(FormatError, match="schema version"):
        db.open("a")
    assert db._connection is None
    assert path.read_bytes() == before
    assert _stored_version("track_db", path) == stored
