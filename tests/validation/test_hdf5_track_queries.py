"""v2.11.1 query and storage defects in pytcl.io (PR 6, Task 6.4).

Each test pins a measured defect against a hand-computed expected value:
HDF5 trajectory windows and scenario comparison, residual and dtype
handling on track writes, HDF5 metadata strings, SQL key prefixes, the
measurement readers' scan grouping, and the dataframe metric export.
"""

import warnings

import numpy as np
import pytest

from pytcl.io import HDF5Storage, SQLStorage, TrackHDF5Storage


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

        assert result["state_differences"]["t"] == np.sqrt(12.5)

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

    @pytest.mark.parametrize("has_residuals", [True, False])
    def test_nan_pad_on_integer_storage_raises_and_leaves_track_intact(
        self, tmp_path, has_residuals
    ):
        """Was: the NaN pad was stored as 0 (arm64) / INT_MIN (x86)."""
        store = TrackHDF5Storage(str(tmp_path / "i.h5"), dtype="int64")
        store.open("w")
        try:
            store.store_track(
                "t",
                np.zeros((2, 1), dtype=np.int64),
                np.ones((2, 1, 1), dtype=np.int64),
                [0.0, 1.0],
                residuals=[[1], [2]] if has_residuals else None,
            )
            residual = None if has_residuals else [5]
            with pytest.raises(TypeError, match="int64"):
                store.append_track_state("t", [0], [[1]], 2.0, residual=residual)

            got = store.retrieve_track("t")
            assert got["timestamps"].tolist() == [0.0, 1.0]
            assert got["states"].shape == (2, 1)
            if has_residuals:
                assert got["residuals"][:, 0].tolist() == [1, 2]
            else:
                assert got["residuals"] is None
        finally:
            store.close()


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


class TestSQLKeyPrefix:
    def test_like_wildcards_in_keys_are_escaped(self, tmp_path):
        """Was: list_keys("a_b") -> ['aXb/y', 'a_b/x', 'a_bc/z']."""
        with SQLStorage() as store:
            store.open(str(tmp_path / "s.db"), mode="w")
            for key in ("a_b/x", "aXb/y", "a_bc/z", "a%b/w"):
                store.store_array(key, np.zeros(1))

            assert store.list_keys("a_b") == ["a_b/x"]
            assert store.list_keys("a_b/") == ["a_b/x"]
            assert store.list_keys("a%b") == ["a%b/w"]

    def test_group_match_is_case_sensitive(self, tmp_path):
        """Was: "a_b" and "A_B" both gave ['A_B/y', 'a_b/x'] (LIKE folds case)."""
        with SQLStorage() as store:
            store.open(str(tmp_path / "s.db"), mode="w")
            for key in ("a_b/x", "A_B/y", "Mission/t"):
                store.store_array(key, np.zeros(1))

            assert store.list_keys("a_b") == ["a_b/x"]
            assert store.list_keys("A_B") == ["A_B/y"]
            assert store.list_keys("mission") == []

    def test_group_does_not_list_the_bare_key_of_the_same_name(self, tmp_path):
        """Was: list_keys("a") returned the bare key "a" alongside "a/x"."""
        with SQLStorage() as store:
            store.open(str(tmp_path / "s.db"), mode="w")
            store.store_array("a", np.zeros(1))
            store.store_array("a/x", np.zeros(1))

            assert store.list_keys("a") == ["a/x"]


class TestReaderScanGrouping:
    def test_int64_times_above_2_53_do_not_merge_silently(self, tmp_path):
        """Was: one scan [[1.0], [2.0]] at 9007199254740992.0."""
        pl = pytest.importorskip("polars")
        from pytcl.io.readers import read_measurements_parquet

        path = str(tmp_path / "m.parquet")
        pl.DataFrame({"t": [2**53, 2**53 + 1], "x": [1.0, 2.0]}).write_parquet(path)

        with pytest.raises(ValueError, match="float64"):
            read_measurements_parquet(path, time_column="t", measurement_columns=["x"])

    def test_distinct_int64_times_group_by_exact_value(self, tmp_path):
        pl = pytest.importorskip("polars")
        from pytcl.io.readers import read_measurements_parquet

        path = str(tmp_path / "m.parquet")
        big = 2**53 + 2
        pl.DataFrame({"t": [big, 2**53, big], "x": [1.0, 2.0, 3.0]}).write_parquet(path)

        ms = read_measurements_parquet(path, time_column="t", measurement_columns=["x"])

        assert ms.times.tolist() == [float(2**53), float(big)]
        assert [s[:, 0].tolist() for s in ms.scans] == [[2.0], [1.0, 3.0]]

    def test_nan_time_raises_value_error(self, tmp_path):
        """Was: KeyError: nan."""
        pl = pytest.importorskip("polars")
        from pytcl.io.readers import read_measurements_parquet

        path = str(tmp_path / "m.parquet")
        pl.DataFrame({"t": [float("nan"), 1.0], "x": [1.0, 2.0]}).write_parquet(path)

        with pytest.raises(ValueError, match="NaN"):
            read_measurements_parquet(path, time_column="t", measurement_columns=["x"])


class TestMetricsExport:
    def test_bool_series_warns(self):
        pytest.importorskip("polars")
        from pytcl.io.dataframes import metrics_to_polars

        with pytest.warns(UserWarning, match="bool"):
            df = metrics_to_polars([0.0, 1.0], flag=np.array([True, False]))
        assert df["flag"].to_list() == [1.0, 0.0]

    def test_integer_beyond_2_53_warns(self):
        """Was: 2**53+1 exported silently as 9007199254740992.0."""
        pytest.importorskip("polars")
        from pytcl.io.dataframes import metrics_to_polars

        with pytest.warns(UserWarning, match="2\\*\\*53"):
            metrics_to_polars([0.0, 1.0], big=np.array([2**53 + 1, 0]))

    def test_exact_integers_export_silently(self):
        pytest.importorskip("polars")
        from pytcl.io.dataframes import metrics_to_polars

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            df = metrics_to_polars([0, 1], n=np.array([2**53, -(2**53)]))
        assert df["n"].to_list() == [float(2**53), -float(2**53)]
