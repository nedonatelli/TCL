"""CSV/Parquet measurement readers: grouping, column mapping, and the
DependencyError guard.
"""

import numpy as np
import pytest

from pytcl.io.readers import (
    MeasurementSet,
    read_measurements_csv,
    read_measurements_parquet,
)

pl = pytest.importorskip("polars")


def _write_csv(path, rows, header):
    lines = [",".join(header)]
    lines.extend(",".join(str(v) for v in row) for row in rows)
    path.write_text("\n".join(lines) + "\n")


# Shared fixture data: three rows, two of them sharing timestamp 0.0, out of
# ascending order on disk so ordering is actually exercised.
_HEADER = ["t", "x", "y", "aircraft"]
_ROWS = [
    (1.0, 5.0, 6.0, "B2"),
    (0.0, 1.0, 2.0, "A1"),
    (0.0, 3.0, 4.0, "A2"),
]


class TestReadMeasurementsCsv:
    def test_grouping_and_ordering(self, tmp_path):
        path = tmp_path / "meas.csv"
        _write_csv(path, _ROWS, _HEADER)
        ms = read_measurements_csv(
            path, time_column="t", measurement_columns=["x", "y"]
        )
        assert isinstance(ms, MeasurementSet)
        assert ms.times.tolist() == [0.0, 1.0]
        assert ms.scans[0].shape == (2, 2)
        assert ms.scans[1].shape == (1, 2)
        assert ms.scans[0].tolist() == [[1.0, 2.0], [3.0, 4.0]]
        assert ms.scans[1].tolist() == [[5.0, 6.0]]

    def test_times_ascending_and_unique(self, tmp_path):
        path = tmp_path / "meas.csv"
        _write_csv(path, _ROWS, _HEADER)
        ms = read_measurements_csv(
            path, time_column="t", measurement_columns=["x", "y"]
        )
        assert np.all(np.diff(ms.times) > 0)
        assert len(ms.times) == len(set(ms.times.tolist()))

    def test_dtype_float64(self, tmp_path):
        path = tmp_path / "meas.csv"
        _write_csv(path, _ROWS, _HEADER)
        ms = read_measurements_csv(
            path, time_column="t", measurement_columns=["x", "y"]
        )
        assert ms.times.dtype == np.float64
        assert all(scan.dtype == np.float64 for scan in ms.scans)

    def test_ids_threaded_when_requested(self, tmp_path):
        path = tmp_path / "meas.csv"
        _write_csv(path, _ROWS, _HEADER)
        ms = read_measurements_csv(
            path,
            time_column="t",
            measurement_columns=["x", "y"],
            id_column="aircraft",
        )
        assert ms.ids is not None
        assert ms.ids[0].tolist() == ["A1", "A2"]
        assert ms.ids[1].tolist() == ["B2"]

    def test_ids_none_when_not_requested(self, tmp_path):
        path = tmp_path / "meas.csv"
        _write_csv(path, _ROWS, _HEADER)
        ms = read_measurements_csv(
            path, time_column="t", measurement_columns=["x", "y"]
        )
        assert ms.ids is None

    def test_missing_measurement_column_raises(self, tmp_path):
        path = tmp_path / "meas.csv"
        _write_csv(path, _ROWS, _HEADER)
        with pytest.raises(ValueError, match="z") as exc_info:
            read_measurements_csv(path, time_column="t", measurement_columns=["x", "z"])
        assert "t" in str(exc_info.value) and "x" in str(exc_info.value)

    def test_missing_time_column_raises(self, tmp_path):
        path = tmp_path / "meas.csv"
        _write_csv(path, _ROWS, _HEADER)
        with pytest.raises(ValueError, match="available columns"):
            read_measurements_csv(path, time_column="bogus", measurement_columns=["x"])

    def test_missing_id_column_raises(self, tmp_path):
        path = tmp_path / "meas.csv"
        _write_csv(path, _ROWS, _HEADER)
        with pytest.raises(ValueError, match="callsign"):
            read_measurements_csv(
                path,
                time_column="t",
                measurement_columns=["x", "y"],
                id_column="callsign",
            )


class TestReadMeasurementsParquet:
    def _write_parquet(self, path):
        df = pl.DataFrame(
            {
                "t": [row[0] for row in _ROWS],
                "x": [row[1] for row in _ROWS],
                "y": [row[2] for row in _ROWS],
                "aircraft": [row[3] for row in _ROWS],
            }
        )
        df.write_parquet(path)
        return path

    def test_grouping_and_ordering(self, tmp_path):
        path = self._write_parquet(tmp_path / "meas.parquet")
        ms = read_measurements_parquet(
            path, time_column="t", measurement_columns=["x", "y"]
        )
        assert ms.times.tolist() == [0.0, 1.0]
        assert ms.scans[0].tolist() == [[1.0, 2.0], [3.0, 4.0]]
        assert ms.scans[1].tolist() == [[5.0, 6.0]]

    def test_dtype_float64(self, tmp_path):
        path = self._write_parquet(tmp_path / "meas.parquet")
        ms = read_measurements_parquet(
            path, time_column="t", measurement_columns=["x", "y"]
        )
        assert ms.times.dtype == np.float64
        assert all(scan.dtype == np.float64 for scan in ms.scans)

    def test_ids_threaded_when_requested(self, tmp_path):
        path = self._write_parquet(tmp_path / "meas.parquet")
        ms = read_measurements_parquet(
            path,
            time_column="t",
            measurement_columns=["x", "y"],
            id_column="aircraft",
        )
        assert ms.ids is not None
        assert ms.ids[0].tolist() == ["A1", "A2"]
        assert ms.ids[1].tolist() == ["B2"]

    def test_missing_column_raises(self, tmp_path):
        path = self._write_parquet(tmp_path / "meas.parquet")
        with pytest.raises(ValueError, match="available columns"):
            read_measurements_parquet(
                path, time_column="t", measurement_columns=["x", "bogus"]
            )

    def test_csv_and_parquet_agree_bitwise(self, tmp_path):
        csv_path = tmp_path / "meas.csv"
        _write_csv(csv_path, _ROWS, _HEADER)
        parquet_path = self._write_parquet(tmp_path / "meas.parquet")

        ms_csv = read_measurements_csv(
            csv_path, time_column="t", measurement_columns=["x", "y"]
        )
        ms_parquet = read_measurements_parquet(
            parquet_path, time_column="t", measurement_columns=["x", "y"]
        )
        assert ms_csv.times.tobytes() == ms_parquet.times.tobytes()
        for a, b in zip(ms_csv.scans, ms_parquet.scans):
            assert a.tobytes() == b.tobytes()

    def test_dependency_error_without_polars(self, monkeypatch, tmp_path):
        import pytcl.io.readers as mod

        monkeypatch.setattr(mod, "_import_polars", mod._raise_missing)
        from pytcl.core.exceptions import DependencyError

        with pytest.raises(DependencyError, match="dataframe"):
            read_measurements_parquet(
                tmp_path / "does_not_matter.parquet",
                time_column="t",
                measurement_columns=["x"],
            )
