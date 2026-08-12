"""polars DataFrame accessors: schema, bitwise fidelity, and the DependencyError guard."""

import numpy as np
import pytest

pl = pytest.importorskip("polars")

from pytcl.io.dataframes import (  # noqa: E402
    explode_state_columns,
    metrics_to_polars,
    tracks_to_polars,
)


def _history():
    # Same builder as tests/unit/test_io_serialize.py::_history (test helper, duplicated
    # per the Task 2 brief rather than importing across test modules).
    from pytcl.trackers import MultiTargetTracker

    rng = np.random.default_rng(3)
    tracker = MultiTargetTracker(
        state_dim=4,
        meas_dim=2,
        F=np.array(
            [[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]], dtype=float
        ),
        H=np.array([[1.0, 0, 0, 0], [0, 0, 1.0, 0]]),
        Q=np.eye(4) * 0.01,
        R=np.eye(2) * 1.0,
        confirm_hits=1,
    )
    history, times = [], []
    for k in range(6):
        z = [np.array([k + rng.normal(0, 0.3), k + rng.normal(0, 0.3)])]
        history.append(tracker.process(z, dt=1.0))
        times.append(float(k))
    return history, times


class TestTracksToPolars:
    def test_long_schema(self):
        history, times = _history()
        df = tracks_to_polars(history, times)
        assert df.columns == ["track_id", "t", "status", "state", "covariance"]
        assert df["t"].dtype == pl.Float64
        n_rows = sum(len(scan) for scan in history)
        assert df.height == n_rows
        first = df.row(0, named=True)
        dim = len(first["state"])
        assert len(first["covariance"]) == dim * dim

    def test_values_match_source_bitwise(self):
        history, times = _history()
        df = tracks_to_polars(history, times)
        tr = history[-1][0]
        row = df.filter((pl.col("t") == times[-1]) & (pl.col("track_id") == tr.id)).row(
            0, named=True
        )
        assert (
            np.asarray(row["state"]).tobytes()
            == np.asarray(tr.state, dtype=np.float64).tobytes()
        )

    def test_explode_layout(self):
        history, times = _history()
        df = explode_state_columns(
            tracks_to_polars(history, times), ["x", "vx", "y", "vy"]
        )
        assert {"x", "vx", "y", "vy"}.issubset(df.columns)
        row = df.row(0, named=True)
        assert row["x"] == row["state"][0] and row["vy"] == row["state"][3]

    def test_explode_wrong_layout_raises(self):
        history, times = _history()
        with pytest.raises(ValueError, match="layout"):
            explode_state_columns(tracks_to_polars(history, times), ["x", "y"])

    def test_metrics_table_and_parquet_round_trip(self, tmp_path):
        t = np.arange(5.0)
        ospa = np.array([1.0, 0.5, 0.25, 0.2, 0.1])
        df = metrics_to_polars(t, ospa=ospa)
        assert df.columns == ["t", "ospa"]
        p = tmp_path / "m.parquet"
        df.write_parquet(p)
        assert pl.read_parquet(p)["ospa"].to_list() == ospa.tolist()

    def test_dependency_error_without_polars(self, monkeypatch):
        import pytcl.io.dataframes as mod

        monkeypatch.setattr(mod, "_import_polars", mod._raise_missing)
        from pytcl.core.exceptions import DependencyError

        with pytest.raises(DependencyError, match="dataframe"):
            tracks_to_polars([], [])

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="length"):
            tracks_to_polars([[]], [0.0, 1.0])

    def test_empty_history_round_trips(self):
        df = tracks_to_polars([], [])
        assert df.height == 0
        assert df.columns == ["track_id", "t", "status", "state", "covariance"]
