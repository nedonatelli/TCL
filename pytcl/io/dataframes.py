"""polars DataFrame accessors for track histories and scalar metrics.

polars is an optional dependency (the ``dataframe`` extra). It is imported
lazily inside `_import_polars`, so importing this module never requires
polars to be installed; calling any of the public functions without it
raises :class:`~pytcl.core.exceptions.DependencyError`.

None of the public signatures below mention a polars type by name (the
boundary rule for optional heavy dependencies) — return values are
annotated ``Any`` and documented as ``polars.DataFrame`` in each docstring.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from pytcl.core.exceptions import DependencyError
from pytcl.core.optional_deps import DISTRIBUTION_NAME

__all__ = [
    "tracks_to_polars",
    "explode_state_columns",
    "metrics_to_polars",
]


def _dependency_error() -> DependencyError:
    """Build the DependencyError raised when polars is unavailable."""
    return DependencyError(
        "polars is required for dataframe export/import of track results.",
        package="polars",
        feature="dataframe export/import",
        install_command=f"pip install {DISTRIBUTION_NAME}[dataframe]",
    )


def _import_polars() -> Any:
    """Import and return the ``polars`` module, or raise `DependencyError`.

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


def tracks_to_polars(history: Sequence[Sequence[Any]], times: Sequence[float]) -> Any:
    """Flatten a per-scan track history into a long-format polars DataFrame.

    Parameters
    ----------
    history : sequence of sequence of Track-like
        Per-scan lists of objects exposing ``id``, ``state``, ``covariance``,
        and ``status`` (a ``TrackStatus`` enum or plain ``str``) — the same
        shape consumed by :func:`pytcl.io.serialize.encode_tracks`.
    times : sequence of float
        Timestamp for each scan in `history`; ``len(times) == len(history)``.

    Returns
    -------
    polars.DataFrame
        One row per (scan, track) pair, columns ``track_id`` (Int64), ``t``
        (Float64), ``status`` (String), ``state`` (List[Float64]),
        ``covariance`` (List[Float64], row-major flattened, length
        ``len(state) ** 2``).

    Raises
    ------
    DependencyError
        If polars is not installed.

    Examples
    --------
    >>> import numpy as np
    >>> from pytcl.trackers import Track, TrackStatus
    >>> track = Track(id=1, state=np.array([1.0, 2.0]),
    ...                covariance=np.eye(2), status=TrackStatus.CONFIRMED,
    ...                hits=1, misses=0, time=0.0)
    >>> df = tracks_to_polars([[track]], [0.0])
    >>> df.columns
    ['track_id', 't', 'status', 'state', 'covariance']
    >>> df.height
    1
    """
    pl = _import_polars()

    track_ids: list[int] = []
    ts: list[float] = []
    statuses: list[str] = []
    states: list[list[float]] = []
    covariances: list[list[float]] = []
    for scan, t in zip(history, times):
        for tr in scan:
            track_ids.append(int(tr.id))
            ts.append(float(t))
            statuses.append(getattr(tr.status, "value", tr.status))
            states.append(np.asarray(tr.state, dtype=np.float64).tolist())
            covariances.append(
                np.asarray(tr.covariance, dtype=np.float64).ravel().tolist()
            )

    return pl.DataFrame(
        {
            "track_id": track_ids,
            "t": ts,
            "status": statuses,
            "state": states,
            "covariance": covariances,
        },
        schema={
            "track_id": pl.Int64,
            "t": pl.Float64,
            "status": pl.String,
            "state": pl.List(pl.Float64),
            "covariance": pl.List(pl.Float64),
        },
    )


def explode_state_columns(df: Any, layout: Sequence[str]) -> Any:
    """Add one Float64 column per state-vector component.

    Parameters
    ----------
    df : polars.DataFrame
        A DataFrame with a ``state`` column of type List[Float64] (as
        produced by `tracks_to_polars`).
    layout : sequence of str
        Column name for each state-vector component, in order; its length
        must equal the dimension of the vectors in ``state``.

    Returns
    -------
    polars.DataFrame
        `df` with one extra Float64 column per `layout` entry, appended via
        ``df.with_columns``.

    Raises
    ------
    ValueError
        If `layout`'s length does not match the state dimension.
    DependencyError
        If polars is not installed.

    Examples
    --------
    >>> import numpy as np
    >>> from pytcl.trackers import Track, TrackStatus
    >>> track = Track(id=1, state=np.array([1.0, 2.0]),
    ...                covariance=np.eye(2), status=TrackStatus.CONFIRMED,
    ...                hits=1, misses=0, time=0.0)
    >>> df = tracks_to_polars([[track]], [0.0])
    >>> wide = explode_state_columns(df, ["x", "vx"])
    >>> wide.row(0, named=True)["x"]
    1.0
    """
    pl = _import_polars()

    layout = list(layout)
    if df.height > 0:
        dim = len(df["state"][0])
        if len(layout) != dim:
            raise ValueError(
                f"layout length ({len(layout)}) must equal the state dimension ({dim})"
            )

    return df.with_columns(
        [pl.col("state").list.get(i).alias(name) for i, name in enumerate(layout)]
    )


def metrics_to_polars(times: Sequence[float], **series: Any) -> Any:
    """Assemble scalar-per-scan metric series into a flat polars DataFrame.

    Parameters
    ----------
    times : sequence of float
        Timestamp for each scan; becomes the ``t`` column.
    **series : array_like
        Named 1-D metric series (e.g. ``ospa=ospa_values``), each of length
        ``len(times)``; each becomes a Float64 column of the same name.

    Returns
    -------
    polars.DataFrame
        Columns ``["t", *series]`` in the order `times` then `series` were
        given, all Float64.

    Raises
    ------
    ValueError
        If a series is not 1-D or its length does not match `times`.
    DependencyError
        If polars is not installed.

    Examples
    --------
    >>> import numpy as np
    >>> df = metrics_to_polars(np.arange(3.0), ospa=np.array([1.0, 0.5, 0.1]))
    >>> df.columns
    ['t', 'ospa']
    >>> df["ospa"].to_list()
    [1.0, 0.5, 0.1]
    """
    pl = _import_polars()

    t_arr = np.asarray(times, dtype=np.float64)
    data: dict[str, list[float]] = {"t": t_arr.tolist()}
    schema: dict[str, Any] = {"t": pl.Float64}
    for name, values in series.items():
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim != 1:
            raise ValueError(f"series {name!r} must be 1-D, got shape {arr.shape}")
        if len(arr) != len(t_arr):
            raise ValueError(
                f"series {name!r} has length {len(arr)}, expected "
                f"{len(t_arr)} (len(times))"
            )
        data[name] = arr.tolist()
        schema[name] = pl.Float64

    return pl.DataFrame(data, schema=schema)
