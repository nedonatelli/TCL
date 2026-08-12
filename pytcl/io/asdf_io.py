"""ASDF export/import for pytcl track histories and single filter states.

ASDF is an optional dependency (the ``asdf`` extra). It is imported lazily
inside `_import_asdf`, so importing this module never requires asdf to be
installed; calling any of the public functions without it raises
:class:`~pytcl.core.exceptions.DependencyError`.

Track histories are flattened into parallel arrays under a
``pytcl/tracks`` tree, one row per (scan, track) record -- the same
flattening `pytcl.io.dataframes.tracks_to_polars` uses, but written as an
ASDF/ndarray tree rather than a DataFrame. States must be the same
dimension across every record in a single history (`save_tracks_asdf`
raises ``ValueError`` otherwise); `save_states_asdf` stores a single
state/covariance pair the same way, without the track bookkeeping.

None of the public signatures below mention an asdf type by name (the
boundary rule for optional heavy dependencies established in
`pytcl.io.dataframes`) -- ``path`` is a plain path, and results are plain
tuples of Python/numpy objects.
"""

from __future__ import annotations

from os import PathLike
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from pytcl.core.exceptions import DependencyError
from pytcl.core.optional_deps import DISTRIBUTION_NAME
from pytcl.io.serialize import SimpleTrack

__all__ = [
    "save_tracks_asdf",
    "load_tracks_asdf",
    "save_states_asdf",
    "load_states_asdf",
]

_SCHEMA_VERSION = 1


def _dependency_error() -> DependencyError:
    """Build the DependencyError raised when asdf is unavailable."""
    return DependencyError(
        "asdf is required for ASDF export/import of track results.",
        package="asdf",
        feature="ASDF export/import",
        install_command=f"pip install {DISTRIBUTION_NAME}[asdf]",
    )


def _import_asdf() -> Any:
    """Import and return the ``asdf`` module, or raise `DependencyError`.

    Returns
    -------
    module
        The imported ``asdf`` module.

    Raises
    ------
    DependencyError
        If asdf is not installed.
    """
    try:
        import asdf
    except ImportError as e:
        raise _dependency_error() from e
    return asdf


def _raise_missing() -> Any:
    """Unconditionally raise `DependencyError`.

    Same signature as `_import_asdf`; tests monkeypatch `_import_asdf` to
    this function to simulate asdf being absent without actually
    uninstalling it.
    """
    raise _dependency_error()


def save_tracks_asdf(
    path: str | PathLike[str],
    history: Sequence[Sequence[Any]],
    times: Sequence[float],
) -> None:
    """Write a per-scan track history to an ASDF file.

    Parameters
    ----------
    path : str or path-like
        Destination ASDF file; overwritten if it exists.
    history : sequence of sequence of Track-like
        Per-scan lists of objects exposing ``id``, ``state``, ``covariance``,
        and ``status`` (a ``TrackStatus`` enum or plain ``str``) -- the same
        shape consumed by :func:`pytcl.io.serialize.encode_tracks`.
    times : sequence of float
        Timestamp for each scan in `history`; ``len(times) == len(history)``.

    Raises
    ------
    ValueError
        If the state vectors in `history` are not all the same dimension.
    DependencyError
        If asdf is not installed.

    Examples
    --------
    >>> import tempfile, os
    >>> import numpy as np
    >>> from pytcl.trackers import Track, TrackStatus
    >>> track = Track(id=1, state=np.array([1.0, 2.0]),
    ...                covariance=np.eye(2), status=TrackStatus.CONFIRMED,
    ...                hits=1, misses=0, time=0.0)
    >>> tmpdir = tempfile.TemporaryDirectory()
    >>> path = os.path.join(tmpdir.name, "tracks.asdf")
    >>> save_tracks_asdf(path, [[track]], [0.0])
    >>> times, history = load_tracks_asdf(path)
    >>> times
    [0.0]
    >>> tmpdir.cleanup()
    """
    asdf = _import_asdf()

    track_ids: list[int] = []
    scan_indices: list[int] = []
    statuses: list[str] = []
    states: list[NDArray[np.float64]] = []
    covariances: list[NDArray[np.float64]] = []
    dim: int | None = None
    for scan_index, (scan, t) in enumerate(zip(history, times)):
        for tr in scan:
            state = np.asarray(tr.state, dtype=np.float64)
            cov = np.asarray(tr.covariance, dtype=np.float64)
            if dim is None:
                dim = state.shape[0]
            elif state.shape[0] != dim:
                raise ValueError(
                    f"state dim {state.shape[0]} at scan {scan_index} "
                    f"(track {tr.id}) does not match earlier dim {dim}; "
                    "states must be uniform-dim within one history"
                )
            track_ids.append(int(tr.id))
            scan_indices.append(scan_index)
            statuses.append(getattr(tr.status, "value", tr.status))
            states.append(state)
            covariances.append(cov)

    if dim is None:
        dim = 0
    states_arr = np.asarray(states, dtype=np.float64).reshape(len(states), dim)
    covariances_arr = np.asarray(covariances, dtype=np.float64).reshape(
        len(covariances), dim, dim
    )

    tree = {
        "pytcl": {
            "schema_version": _SCHEMA_VERSION,
            "times": np.asarray(times, dtype=np.float64),
            "tracks": {
                "track_id": np.asarray(track_ids, dtype=np.int64),
                "scan_index": np.asarray(scan_indices, dtype=np.int64),
                "status": statuses,
                "states": states_arr,
                "covariances": covariances_arr,
            },
        }
    }
    af = asdf.AsdfFile(tree)
    af.write_to(path)


def load_tracks_asdf(
    path: str | PathLike[str],
) -> tuple[list[float], list[list[SimpleTrack]]]:
    """Read a track history written by `save_tracks_asdf`.

    Parameters
    ----------
    path : str or path-like
        ASDF file to read.

    Returns
    -------
    times : list of float
        Timestamp for each scan.
    history : list of list of SimpleTrack
        Per-scan lists of decoded tracks, aligned with `times`.

    Raises
    ------
    DependencyError
        If asdf is not installed.

    Examples
    --------
    >>> import tempfile, os
    >>> import numpy as np
    >>> from pytcl.trackers import Track, TrackStatus
    >>> track = Track(id=1, state=np.array([1.0, 2.0]),
    ...                covariance=np.eye(2), status=TrackStatus.CONFIRMED,
    ...                hits=1, misses=0, time=0.0)
    >>> tmpdir = tempfile.TemporaryDirectory()
    >>> path = os.path.join(tmpdir.name, "tracks.asdf")
    >>> save_tracks_asdf(path, [[track]], [0.0])
    >>> times, history = load_tracks_asdf(path)
    >>> history[0][0].id
    1
    >>> tmpdir.cleanup()
    """
    asdf = _import_asdf()

    with asdf.open(path) as af:
        pytcl_tree = af.tree["pytcl"]
        times = np.asarray(pytcl_tree["times"], dtype=np.float64)
        tracks = pytcl_tree["tracks"]
        track_ids = np.asarray(tracks["track_id"])
        scan_indices = np.asarray(tracks["scan_index"])
        statuses = list(tracks["status"])
        states_arr = np.array(tracks["states"], dtype=np.float64)
        covariances_arr = np.array(tracks["covariances"], dtype=np.float64)

        history: list[list[SimpleTrack]] = [[] for _ in range(len(times))]
        for i in range(len(track_ids)):
            history[int(scan_indices[i])].append(
                SimpleTrack(
                    id=int(track_ids[i]),
                    state=states_arr[i].copy(),
                    covariance=covariances_arr[i].copy(),
                    status=statuses[i],
                )
            )

    return times.tolist(), history


def save_states_asdf(path: str | PathLike[str], x: Any, P: Any) -> None:
    """Write a single filter state estimate and covariance to an ASDF file.

    Parameters
    ----------
    path : str or path-like
        Destination ASDF file; overwritten if it exists.
    x : array_like
        State estimate vector, shape (n,).
    P : array_like
        State covariance matrix, shape (n, n).

    Raises
    ------
    DependencyError
        If asdf is not installed.

    Examples
    --------
    >>> import tempfile, os
    >>> import numpy as np
    >>> tmpdir = tempfile.TemporaryDirectory()
    >>> path = os.path.join(tmpdir.name, "state.asdf")
    >>> save_states_asdf(path, np.array([1.0, 2.0]), np.eye(2))
    >>> x2, P2 = load_states_asdf(path)
    >>> x2.tolist()
    [1.0, 2.0]
    >>> tmpdir.cleanup()
    """
    asdf = _import_asdf()

    x = np.asarray(x, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    tree = {
        "pytcl": {
            "schema_version": _SCHEMA_VERSION,
            "x": x,
            "P": P,
        }
    }
    af = asdf.AsdfFile(tree)
    af.write_to(path)


def load_states_asdf(
    path: str | PathLike[str],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Read a state estimate and covariance written by `save_states_asdf`.

    Parameters
    ----------
    path : str or path-like
        ASDF file to read.

    Returns
    -------
    x : ndarray, shape (n,)
        State estimate vector.
    P : ndarray, shape (n, n)
        State covariance matrix.

    Raises
    ------
    DependencyError
        If asdf is not installed.

    Examples
    --------
    >>> import tempfile, os
    >>> import numpy as np
    >>> tmpdir = tempfile.TemporaryDirectory()
    >>> path = os.path.join(tmpdir.name, "state.asdf")
    >>> save_states_asdf(path, np.array([1.0, 2.0, 3.0]), np.eye(3))
    >>> x2, P2 = load_states_asdf(path)
    >>> P2.shape
    (3, 3)
    >>> tmpdir.cleanup()
    """
    asdf = _import_asdf()

    with asdf.open(path) as af:
        pytcl_tree = af.tree["pytcl"]
        x = np.array(pytcl_tree["x"], dtype=np.float64)
        P = np.array(pytcl_tree["P"], dtype=np.float64)

    return x, P
