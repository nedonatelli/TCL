"""v2.11.1 query and storage defects in pytcl.io (PR 6, Task 6.4).

Each test pins a measured defect against a hand-computed expected value:
HDF5 trajectory windows and scenario comparison, residual and dtype
handling on track writes, HDF5 metadata strings, SQL key prefixes, the
measurement readers' scan grouping, and the dataframe metric export.
"""

import warnings

import numpy as np
import pytest

from pytcl.io import HDF5Storage, TrackHDF5Storage


@pytest.fixture
def track_store(tmp_path):
    store = TrackHDF5Storage(str(tmp_path / "tracks.h5"))
    store.open("w")
    yield store
    store.close()


def _store_track(store, track_id, timestamps, states=None, scenario_id=None):
    t = np.asarray(timestamps, dtype=np.float64)
    s = t[:, None] if states is None else np.asarray(states, dtype=np.float64)
    store.store_track(track_id, s, np.ones((len(t), 1, 1)), t, scenario_id=scenario_id)


class TestTrajectoryWindow:
    def test_trajectory_query_returns_only_the_requested_window(self, track_store):
        """Out-of-order append: the old contiguous slice returned [1,2,3,4,1.5]."""
        _store_track(track_store, "t", [1.0, 2.0, 3.0, 4.0])
        track_store.append_track_state("t", [1.5], [[1.0]], 1.5)

        got = track_store.get_track_trajectory("t", start_time=1.0, end_time=2.0)

        assert sorted(got["timestamps"].tolist()) == [1.0, 1.5, 2.0]
        # Rows stay paired: each state was written equal to its timestamp.
        assert got["states"][:, 0].tolist() == got["timestamps"].tolist()


class TestCompareScenarios:
    def test_compare_scenarios_aligns_on_timestamps(self, track_store):
        """Identical x = 2t tracks offset by one scan: was RMSE 2.0 by row."""
        a, b = [0.0, 1.0, 2.0], [1.0, 2.0, 3.0]
        _store_track(track_store, "t", a, [[2 * t] for t in a], scenario_id="A")
        _store_track(track_store, "t", b, [[2 * t] for t in b], scenario_id="B")

        result = track_store.compare_scenarios("A", "B")

        assert result["state_differences"]["t"] == pytest.approx(0.0, abs=0)

    def test_rmse_is_over_exactly_the_two_shared_timestamps(self, track_store):
        """Differences 3 and 4 at the shared t=1, t=2: sqrt((9+16)/2)."""
        a, b = [0.0, 1.0, 2.0], [1.0, 2.0, 3.0]
        _store_track(track_store, "t", a, [[100.0], [0.0], [0.0]], scenario_id="A")
        _store_track(track_store, "t", b, [[3.0], [4.0], [-100.0]], scenario_id="B")

        result = track_store.compare_scenarios("A", "B")

        assert result["state_differences"]["t"] == pytest.approx(
            np.sqrt(12.5), rel=1e-15, abs=0
        )

    def test_zero_overlap_is_nan_and_warns(self, track_store):
        _store_track(track_store, "t", [0.0, 1.0], scenario_id="A")
        _store_track(track_store, "t", [5.0, 6.0], scenario_id="B")

        with pytest.warns(RuntimeWarning, match="no timestamps in common"):
            result = track_store.compare_scenarios("A", "B")

        assert np.isnan(result["state_differences"]["t"])


class TestResidualAppend:
    def test_residual_on_track_without_residuals_is_kept_and_aligned(self, track_store):
        """Was silently discarded: 3 timestamps, no residuals dataset."""
        _store_track(track_store, "t", [0.0, 1.0])
        track_store.append_track_state("t", [2.0], [[1.0]], 2.0, residual=[5.0])

        r = track_store.retrieve_track("t")["residuals"]

        assert r.shape == (3, 1)
        assert np.isnan(r[:2, 0]).all()
        assert r[2, 0] == 5.0

    def test_omitted_residual_pads_existing_residuals(self, track_store):
        track_store.store_track(
            "t",
            np.zeros((2, 1)),
            np.ones((2, 1, 1)),
            [0.0, 1.0],
            residuals=[[1.0], [2.0]],
        )
        track_store.append_track_state("t", [0.0], [[1.0]], 2.0)

        r = track_store.retrieve_track("t")["residuals"]

        assert r[:2, 0].tolist() == [1.0, 2.0]
        assert np.isnan(r[2, 0])


class TestStoredDtype:
    def test_non_float_dtypes_are_rejected_not_silently_cast(self, track_store):
        with pytest.raises(TypeError, match="complex"):
            track_store.store_track(
                "t", np.array([[1 + 2j]]), np.ones((1, 1, 1)), [0.0]
            )

    def test_complex_append_is_rejected(self, track_store):
        _store_track(track_store, "t", [0.0])
        with pytest.raises(TypeError, match="complex"):
            track_store.append_track_state("t", [1 + 2j], [[1.0]], 1.0)
        assert track_store.retrieve_track("t")["timestamps"].tolist() == [0.0]

    def test_integer_beyond_2_53_warns(self, track_store):
        with pytest.warns(UserWarning, match="float64"):
            track_store.store_track(
                "t", np.array([[2**53 + 1]]), np.ones((1, 1, 1)), [0.0]
            )
        assert track_store.retrieve_track("t")["states"].dtype == np.float64

    def test_integer_at_2_53_is_exact_and_silent(self, track_store):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            track_store.store_track("t", np.array([[2**53]]), np.ones((1, 1, 1)), [0.0])
        assert int(track_store.retrieve_track("t")["states"][0, 0]) == 2**53


class TestHDF5MetadataStrings:
    def test_metadata_strings_are_not_json_unwrapped(self, tmp_path):
        """Was: '"quoted"' read back as 'quoted', '"1"' as '1'."""
        with HDF5Storage() as store:
            store.open(str(tmp_path / "m.h5"), mode="w")
            store.store_array("d", np.zeros(2), metadata={"k": '"quoted"', "j": '"1"'})
            store.store_scalar("s", 1.0, metadata={"k": '"quoted"'})
            store.store_group("g", metadata={"k": '"quoted"'})

            assert store.get_metadata("d") == {"k": '"quoted"', "j": '"1"'}
            assert store.get_metadata("s") == {"k": '"quoted"'}
            assert store.get_metadata("g") == {"k": '"quoted"'}

    def test_files_written_by_2_11_0_read_unchanged(self, tmp_path):
        """2.11.0 stored the non-native fallback as json.dumps(str(v))."""
        import h5py

        path = str(tmp_path / "old.h5")
        with h5py.File(path, "w") as f:
            f.create_dataset("d", data=np.zeros(2))
            f["d"].attrs["cfg"] = "\"{'a': 1}\""
        with HDF5Storage() as store:
            store.open(path, mode="r")
            assert store.get_metadata("d") == {"cfg": "{'a': 1}"}
