"""Storage contracts that were unstated, and therefore inconsistent (gh-21).

Six gaps, sharing a cause: the base class did not say what should happen, so
each backend did whatever its underlying library did, and the difference showed
up only when a caller moved between them or hit a path nobody had exercised.

The existing io suite is large -- 134 tests -- and it passed throughout. Two of
those tests are worth looking at, because they show how:

- the residual round-trip asks for ``start_time=0.5`` to skip the
  residual-free initiate row, and
- the HDF5-to-SQL export test reads the raw ``track_states`` table with a
  cursor rather than calling ``get_track_history``.

Both are working around ``get_track_history`` dropping residuals, rather than
asserting it should not. Written that way, the suite grows around a defect and
holds it in place -- so the tests here go through the public accessors on
purpose, and none of them takes a window chosen to avoid the awkward row.
"""

import numpy as np
import pytest

from pytcl.io.sql_storage import SQLStorage
from pytcl.io.track_database import TrackDatabaseManager

h5py = pytest.importorskip("h5py", reason="HDF5Storage needs h5py")

from pytcl.io.hdf5_storage import HDF5Storage  # noqa: E402


@pytest.fixture
def db(tmp_path):
    manager = TrackDatabaseManager(str(tmp_path / "tracks.db"))
    manager.open(mode="w")
    yield manager
    manager.close()


class TestReadModeDoesNotCreate:
    """``open(mode='r')`` on a missing file used to create an empty one."""

    def test_reading_a_missing_database_raises(self, tmp_path):
        missing = tmp_path / "not-there.db"
        store = SQLStorage()

        with pytest.raises(FileNotFoundError, match="No such database"):
            store.open(str(missing), mode="r")

    def test_reading_a_missing_database_leaves_no_file_behind(self, tmp_path):
        """The part a caller would not think to check.

        ``sqlite3.connect`` creates the file whatever the caller intended, so
        a failed read used to leave a stray empty database on disk -- and a
        second attempt would then "succeed" against it.
        """
        missing = tmp_path / "not-there.db"
        store = SQLStorage()

        with pytest.raises(FileNotFoundError):
            store.open(str(missing), mode="r")

        assert not missing.exists(), "a failed read created the database anyway"

    @pytest.mark.parametrize("mode", ["w", "a"])
    def test_writing_modes_still_create(self, tmp_path, mode):
        path = tmp_path / f"new-{mode}.db"
        with SQLStorage() as store:
            store.open(str(path), mode=mode)
            store.store_array("x", np.arange(4.0))
        assert path.exists()

    def test_reading_an_existing_database_works(self, tmp_path):
        path = tmp_path / "real.db"
        with SQLStorage() as store:
            store.open(str(path), mode="w")
            store.store_array("x", np.arange(4.0))

        with SQLStorage() as store:
            store.open(str(path), mode="r")
            np.testing.assert_array_equal(store.retrieve_array("x"), np.arange(4.0))

    def test_a_missing_array_raises_key_error_not_a_sqlite_error(self, tmp_path):
        """The documented failure, which the stray-file bug used to pre-empt.

        Against an accidentally-created empty database the table did not exist
        either, so the caller got ``sqlite3.OperationalError`` about missing
        tables instead of the ``KeyError`` the contract promises.
        """
        path = tmp_path / "real.db"
        with SQLStorage() as store:
            store.open(str(path), mode="w")
            store.store_array("present", np.arange(3.0))

        with SQLStorage() as store:
            store.open(str(path), mode="r")
            with pytest.raises(KeyError):
                store.retrieve_array("absent")

    @pytest.mark.parametrize("mode", ["x", "rb", "", "read"])
    def test_an_unknown_mode_is_rejected(self, tmp_path, mode):
        with pytest.raises(ValueError, match="mode must be"):
            SQLStorage().open(str(tmp_path / "any.db"), mode=mode)


class TestDeadBackendParameterIsGone:
    """``SQLStorage(db_type=...)`` advertised backends that did not exist."""

    def test_the_parameter_is_removed(self):
        with pytest.raises(TypeError):
            SQLStorage(db_type="postgresql://localhost/db")

    def test_the_default_construction_still_works(self, tmp_path):
        with SQLStorage() as store:
            store.open(str(tmp_path / "ok.db"), mode="w")
            store.store_array("x", np.ones(3))
            np.testing.assert_array_equal(store.retrieve_array("x"), np.ones(3))


class TestOverwriteSemanticsAgree:
    """The two backends disagreed and neither documented it."""

    @pytest.fixture(params=["sql", "hdf5"])
    def store(self, request, tmp_path):
        if request.param == "sql":
            backend, path = SQLStorage(), tmp_path / "s.db"
        else:
            backend, path = HDF5Storage(), tmp_path / "s.h5"
        backend.open(str(path), mode="w")
        yield backend
        backend.close()

    def test_storing_twice_replaces(self, store):
        """SQL replaced; HDF5 let h5py raise ValueError on an existing name."""
        store.store_array("data", np.arange(5.0))
        store.store_array("data", np.arange(100.0, 103.0))

        np.testing.assert_array_equal(
            store.retrieve_array("data"), np.arange(100.0, 103.0)
        )

    def test_replacing_can_change_the_shape(self, store):
        """A same-name store of a different shape must not be a partial write."""
        store.store_array("data", np.zeros((4, 3)))
        store.store_array("data", np.ones((2, 7)))
        assert store.retrieve_array("data").shape == (2, 7)

    def test_replacing_replaces_metadata_rather_than_merging(self, store):
        """The base class specifies wholesale replacement of the metadata too.

        Read back through ``get_metadata``, the accessor both backends
        implement, so this checks what a caller can actually see rather than
        what the storage happens to hold.
        """
        store.store_array("data", np.ones(3), metadata={"units": "m", "old": 1})
        store.store_array("data", np.ones(3), metadata={"units": "km"})

        metadata = store.get_metadata("data")

        assert metadata.get("units") == "km"
        assert "old" not in metadata, "metadata from the earlier store survived"

    def test_storing_a_scalar_twice_replaces_on_both_backends(self, store):
        """The gh-21 fix unified store_array and missed store_scalar.

        The identical divergence sat one method down: SQLStorage replaced,
        HDF5Storage let h5py raise ValueError on the existing name. The
        class above could not catch it -- every test in it goes through
        store_array.
        """
        store.store_scalar("count", 1.0)
        store.store_scalar("count", 2.0)
        assert store.retrieve_scalar("count") == 2.0


class TestResidualsStayAlignedWithTimestamps:
    """``get_track_history`` keyed residuals off the first row."""

    @staticmethod
    def _predict_then_update(db, track_id="t1", steps=3):
        """The shape KalmanTrackAdapter produces: a prediction, then an update.

        The initiate row carries no residual either, so row zero never has one
        -- which is exactly what the old check looked at.
        """
        db.initiate_track(track_id, np.zeros(2), np.eye(2), 0.0)
        for step in range(1, steps + 1):
            db.update_track_state(
                track_id,
                np.ones(2),
                np.eye(2),
                2.0 * step - 1.0,
                update_type="prediction",
            )
            db.update_track_state(
                track_id,
                np.ones(2),
                np.eye(2),
                2.0 * step,
                residual=np.array([0.1 * step, 0.2 * step]),
            )

    def test_residuals_are_returned_when_the_first_row_has_none(self, db):
        """The defect, directly. This used to return None."""
        self._predict_then_update(db)
        history = db.get_track_history("t1")

        assert history["residuals"] is not None, (
            "residuals dropped because the first row is an initiation (gh-21)"
        )

    def test_residuals_are_row_aligned_with_timestamps(self, db):
        """Mixed rows used to give a shorter array, silently misaligned.

        Every residual then belonged to the wrong timestamp, with nothing to
        indicate it.
        """
        self._predict_then_update(db)
        history = db.get_track_history("t1")

        assert history["residuals"].shape[0] == len(history["timestamps"])

    def test_rows_without_a_residual_are_nan(self, db):
        """NaN marks the gaps, so a caller can tell which rows carry one."""
        self._predict_then_update(db, steps=3)
        history = db.get_track_history("t1")
        residuals = history["residuals"]

        missing = np.isnan(residuals).any(axis=1)
        # 1 initiation + 3 predictions carry none; 3 updates do.
        assert missing.sum() == 4
        assert (~missing).sum() == 3

    def test_the_residuals_land_on_the_right_timestamps(self, db):
        """Alignment is only worth anything if the values are in the right rows."""
        self._predict_then_update(db, steps=3)
        history = db.get_track_history("t1")

        for index, timestamp in enumerate(history["timestamps"]):
            residual = history["residuals"][index]
            if timestamp in (2.0, 4.0, 6.0):  # the update rows
                step = timestamp / 2.0
                np.testing.assert_allclose(residual, [0.1 * step, 0.2 * step])
            else:
                assert np.all(np.isnan(residual))

    def test_a_history_with_no_residuals_at_all_reports_none(self, db):
        """The one case that should still be None."""
        db.initiate_track("t1", np.zeros(2), np.eye(2), 0.0)
        for step in range(1, 4):
            db.update_track_state("t1", np.ones(2), np.eye(2), float(step))

        assert db.get_track_history("t1")["residuals"] is None

    def test_a_window_beginning_with_a_prediction_keeps_its_residuals(self, db):
        """No window should need choosing to avoid the residual-free row."""
        self._predict_then_update(db, steps=3)
        window = db.get_track_history("t1", start_time=3.0)

        assert window["residuals"] is not None
        assert window["residuals"].shape[0] == len(window["timestamps"])
        assert not np.all(np.isnan(window["residuals"]))


class TestUnknownTrackIdsAreRejected:
    """Writes against a nonexistent track used to succeed silently."""

    def test_updating_an_unknown_track_raises(self, db):
        with pytest.raises(KeyError, match="No track"):
            db.update_track_state("ghost", np.ones(2), np.eye(2), 1.0)

    def test_the_rejected_update_leaves_no_orphan_rows(self, db):
        """The reason it mattered.

        The state row was inserted and the tracks-table update matched nothing,
        so the history existed while the track did not -- and the state was
        still retrievable by id, so a typo produced something that looked like
        a track in every respect but the one that counts.
        """
        with pytest.raises(KeyError):
            db.update_track_state("ghost", np.ones(2), np.eye(2), 1.0)

        db._cursor.execute(
            "SELECT COUNT(*) FROM track_states WHERE track_id = ?", ("ghost",)
        )
        assert db._cursor.fetchone()[0] == 0

    def test_a_known_track_still_updates(self, db):
        db.initiate_track("real", np.zeros(2), np.eye(2), 0.0)
        db.update_track_state("real", np.ones(2), np.eye(2), 1.0)
        assert len(db.get_track_history("real")["timestamps"]) == 2

    @pytest.mark.parametrize("missing", ["keep", "merge"])
    def test_merging_an_unknown_track_raises(self, db, missing):
        db.initiate_track("real", np.zeros(2), np.eye(2), 0.0)
        pair = ("ghost", "real") if missing == "keep" else ("real", "ghost")

        with pytest.raises(KeyError, match="No track"):
            db.merge_tracks(*pair)


class TestMergeCombinesTrackLevelFields:
    """``merge_tracks`` moved the history but left the track record behind."""

    @staticmethod
    def _two_tracks(db):
        db.initiate_track("keep", np.zeros(2), np.eye(2), 0.0, metadata={"sensor": "a"})
        db.initiate_track(
            "gone", np.zeros(2), np.eye(2), 5.0, metadata={"sensor": "b", "extra": 7}
        )
        db.update_track_state("keep", np.ones(2), np.eye(2), 1.0)
        db.update_track_state("gone", np.ones(2), np.eye(2), 9.0)

    def test_the_kept_track_takes_the_later_update_time(self, db):
        """The consequence: a merge bringing in newer states used to leave the
        kept track looking stale, so staleness-based pruning would delete the
        track that had just been reinforced."""
        self._two_tracks(db)
        before = db.get_track("keep")["last_update_time"]

        db.merge_tracks("keep", "gone")

        assert before == 1.0
        assert db.get_track("keep")["last_update_time"] == 9.0

    def test_the_kept_track_takes_the_earlier_birth_time(self, db):
        """The merged lifetime is the union of the two."""
        db.initiate_track("keep", np.zeros(2), np.eye(2), 10.0)
        db.initiate_track("gone", np.zeros(2), np.eye(2), 2.0)

        db.merge_tracks("keep", "gone")

        assert db.get_track("keep")["birth_time"] == 2.0

    def test_metadata_is_folded_in(self, db):
        self._two_tracks(db)
        db.merge_tracks("keep", "gone")

        metadata = db.get_track("keep")["metadata"]
        assert metadata["extra"] == 7, "keys from the merged track were lost"

    def test_the_kept_tracks_own_metadata_wins(self, db):
        """Merging is additive; it must not overwrite what the survivor knows."""
        self._two_tracks(db)
        db.merge_tracks("keep", "gone")

        assert db.get_track("keep")["metadata"]["sensor"] == "a"

    def test_the_history_still_moves_across(self, db):
        """What already worked, kept under test alongside what did not."""
        self._two_tracks(db)
        db.merge_tracks("keep", "gone")

        timestamps = db.get_track_history("keep")["timestamps"]
        assert 9.0 in timestamps

    def test_the_merged_track_is_marked_dead(self, db):
        self._two_tracks(db)
        db.merge_tracks("keep", "gone")

        assert db.get_track("gone")["status"] == "dead"
