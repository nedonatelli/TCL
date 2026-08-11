"""CSV and Parquet readers for tabular measurement data.

Both readers turn a flat table (one row per measurement report) into a
:class:`MeasurementSet`: measurements grouped into scans by the exact value
of a timestamp column, ascending. This is the ingest side of the pipeline
whose export side is :mod:`pytcl.io.dataframes` -- a table written by
``metrics_to_polars`` (or any tool) with a time column and one column per
measurement component reads back through these functions unchanged.

`read_measurements_csv` uses only the standard library and has no optional
dependency. `read_measurements_parquet` requires polars (the ``dataframe``
extra); it is imported lazily, mirroring the guard in
:mod:`pytcl.io.dataframes` (`_import_polars`/`_dependency_error`), so
importing this module never requires polars to be installed.
"""

from __future__ import annotations

import csv
from os import PathLike
from typing import Any, Mapping, NamedTuple, Sequence

import numpy as np
from numpy.typing import NDArray

from pytcl.core.exceptions import DependencyError
from pytcl.core.optional_deps import DISTRIBUTION_NAME

__all__ = [
    "MeasurementSet",
    "read_measurements_csv",
    "read_measurements_parquet",
]


class MeasurementSet(NamedTuple):
    """Measurements grouped into scans by exact timestamp.

    Attributes
    ----------
    times : ndarray, shape (n_scans,)
        Unique scan timestamps, strictly ascending, float64.
    scans : list of ndarray
        ``scans[k]`` holds the ``(n_k, n_cols)`` float64 measurement rows
        recorded at ``times[k]``, columns in `measurement_columns` order.
    ids : list of ndarray, or None
        ``ids[k]`` holds one identifier per row of ``scans[k]``, when
        `id_column` was given to the reader; ``None`` otherwise.
    """

    times: NDArray[np.float64]
    scans: list[NDArray[np.float64]]
    ids: list[NDArray[Any]] | None


def _dependency_error() -> DependencyError:
    """Build the DependencyError raised when polars is unavailable.

    Mirrors ``pytcl.io.dataframes._dependency_error``'s guard pattern; kept
    local (rather than imported) so this module's error message can name
    the reader feature specifically.
    """
    return DependencyError(
        "polars is required to read Parquet measurement files.",
        package="polars",
        feature="dataframe import of measurement results",
        install_command=f"pip install {DISTRIBUTION_NAME}[dataframe]",
    )


def _import_polars() -> Any:
    """Import and return the ``polars`` module, or raise `DependencyError`.

    Mirrors ``pytcl.io.dataframes._import_polars``.

    Returns
    -------
    module
        The imported ``polars`` module.

    Raises
    ------
    DependencyError
        If polars is not installed.
    """
    try:
        import polars as pl
    except ImportError as e:
        raise _dependency_error() from e
    return pl


def _raise_missing() -> Any:
    """Unconditionally raise `DependencyError`.

    Same signature as `_import_polars`; tests monkeypatch `_import_polars`
    to this function to simulate polars being absent without actually
    uninstalling it.
    """
    raise _dependency_error()


def _check_columns(
    available: Sequence[str],
    time_column: str,
    measurement_columns: Sequence[str],
    id_column: str | None,
) -> None:
    """Raise ValueError listing `available` if any requested column is missing."""
    needed = [time_column, *measurement_columns]
    if id_column is not None:
        needed.append(id_column)
    missing = [c for c in needed if c not in available]
    if missing:
        raise ValueError(
            f"column(s) {missing} not found; available columns: {list(available)}"
        )


def _rows_to_measurement_set(
    rows: Sequence[Mapping[str, Any]],
    available: Sequence[str],
    *,
    time_column: str,
    measurement_columns: Sequence[str],
    id_column: str | None,
) -> MeasurementSet:
    """Group row mappings into a `MeasurementSet` by exact time-column value.

    Shared by `read_measurements_csv` and `read_measurements_parquet`; each
    reader is responsible only for producing `rows` as a sequence of
    ``{column_name: value}`` mappings and `available` as the table's column
    names.
    """
    _check_columns(available, time_column, measurement_columns, id_column)

    groups: dict[float, list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(float(row[time_column]), []).append(row)

    times = np.array(sorted(groups), dtype=np.float64)
    scans: list[NDArray[np.float64]] = []
    ids: list[NDArray[Any]] | None = [] if id_column is not None else None
    for t in times:
        group_rows = groups[float(t)]
        scans.append(
            np.array(
                [[float(row[c]) for c in measurement_columns] for row in group_rows],
                dtype=np.float64,
            )
        )
        if ids is not None:
            ids.append(np.array([row[id_column] for row in group_rows]))

    return MeasurementSet(times=times, scans=scans, ids=ids)


def read_measurements_csv(
    path: str | PathLike[str],
    *,
    time_column: str,
    measurement_columns: Sequence[str],
    id_column: str | None = None,
) -> MeasurementSet:
    """Read measurement reports from a CSV file into a `MeasurementSet`.

    Parameters
    ----------
    path : str or path-like
        CSV file to read; must have a header row.
    time_column : str
        Name of the column holding each row's scan timestamp. Rows with the
        exact same value (compared as parsed ``float``) are grouped into
        the same scan.
    measurement_columns : sequence of str
        Names of the columns to stack into each scan's measurement matrix,
        in order.
    id_column : str, optional
        Name of a column holding a per-row identifier. When given, the
        returned `MeasurementSet.ids` carries one array per scan; when
        omitted, `MeasurementSet.ids` is ``None``.

    Returns
    -------
    MeasurementSet
        Scans ordered by ascending unique timestamp.

    Raises
    ------
    ValueError
        If `time_column`, any of `measurement_columns`, or `id_column` is
        not a column in the file; the message lists the available columns.

    Examples
    --------
    >>> import tempfile, os
    >>> from pytcl.io.readers import read_measurements_csv
    >>> tmpdir = tempfile.TemporaryDirectory()
    >>> path = os.path.join(tmpdir.name, "meas.csv")
    >>> _ = open(path, "w").write(
    ...     "t,x,y\\n0.0,1.0,2.0\\n0.0,3.0,4.0\\n1.0,5.0,6.0\\n"
    ... )
    >>> ms = read_measurements_csv(path, time_column="t",
    ...                             measurement_columns=["x", "y"])
    >>> ms.times.tolist()
    [0.0, 1.0]
    >>> ms.scans[0].tolist()
    [[1.0, 2.0], [3.0, 4.0]]
    >>> ms.ids is None
    True
    >>> tmpdir.cleanup()
    """
    measurement_columns = list(measurement_columns)
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        available = reader.fieldnames or []
        rows = list(reader)

    return _rows_to_measurement_set(
        rows,
        available,
        time_column=time_column,
        measurement_columns=measurement_columns,
        id_column=id_column,
    )


def read_measurements_parquet(
    path: str | PathLike[str],
    *,
    time_column: str,
    measurement_columns: Sequence[str],
    id_column: str | None = None,
) -> MeasurementSet:
    """Read measurement reports from a Parquet file into a `MeasurementSet`.

    Same grouping and column-mapping contract as `read_measurements_csv`;
    see that function's docstring. Column dtypes are read natively via
    polars rather than parsed from text.

    Parameters
    ----------
    path : str or path-like
        Parquet file to read.
    time_column : str
        Name of the column holding each row's scan timestamp. Rows with the
        exact same value are grouped into the same scan.
    measurement_columns : sequence of str
        Names of the columns to stack into each scan's measurement matrix,
        in order.
    id_column : str, optional
        Name of a column holding a per-row identifier.

    Returns
    -------
    MeasurementSet
        Scans ordered by ascending unique timestamp.

    Raises
    ------
    ValueError
        If `time_column`, any of `measurement_columns`, or `id_column` is
        not a column in the file; the message lists the available columns.
    DependencyError
        If polars is not installed.

    Examples
    --------
    >>> import tempfile, os
    >>> import polars as pl
    >>> from pytcl.io.readers import read_measurements_parquet
    >>> tmpdir = tempfile.TemporaryDirectory()
    >>> path = os.path.join(tmpdir.name, "meas.parquet")
    >>> pl.DataFrame(
    ...     {"t": [0.0, 0.0, 1.0], "x": [1.0, 3.0, 5.0], "y": [2.0, 4.0, 6.0]}
    ... ).write_parquet(path)
    >>> ms = read_measurements_parquet(path, time_column="t",
    ...                                 measurement_columns=["x", "y"])
    >>> ms.times.tolist()
    [0.0, 1.0]
    >>> ms.scans[0].tolist()
    [[1.0, 2.0], [3.0, 4.0]]
    >>> tmpdir.cleanup()
    """
    pl = _import_polars()
    measurement_columns = list(measurement_columns)
    df = pl.read_parquet(path)
    rows = df.rows(named=True)

    return _rows_to_measurement_set(
        rows,
        df.columns,
        time_column=time_column,
        measurement_columns=measurement_columns,
        id_column=id_column,
    )
