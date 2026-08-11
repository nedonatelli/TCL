"""msgspec-based serialization for filter states and track histories.

Two wire formats are supported via the ``fmt`` argument on every function:

- ``"msgpack"`` (default): compact binary, round-trips ``float64`` bit
  patterns exactly, including NaN/inf.
- ``"json"``: human-readable text. JSON has no representation for NaN/inf,
  so encoding a state or covariance containing non-finite values raises
  ``ValueError`` rather than silently producing invalid JSON.

Decoding is strict: msgspec validates the incoming bytes against the target
:class:`msgspec.Struct` and raises on missing fields, wrong types, or
malformed data rather than returning partial/garbage results.
"""

from __future__ import annotations

from typing import Any, NamedTuple, Sequence

import msgspec
import numpy as np
from numpy.typing import NDArray

__all__ = [
    "TrackRecord",
    "TrackSet",
    "StateRecord",
    "SimpleTrack",
    "encode_tracks",
    "decode_tracks",
    "encode_states",
    "decode_states",
]


class TrackRecord(msgspec.Struct):
    """One track's state at one scan, ready for msgspec encoding.

    Attributes
    ----------
    track_id : int
        Track identifier.
    t : float
        Timestamp of the scan this record belongs to.
    status : str
        Track status (``TrackStatus.value``, e.g. ``"confirmed"``).
    state : list of float
        State estimate vector.
    covariance : list of float
        Row-major flattened state covariance; ``len == len(state) ** 2``.
    """

    track_id: int
    t: float
    status: str
    state: list[float]
    covariance: list[float]


class TrackSet(msgspec.Struct):
    """A full track history: scan timestamps plus per-scan track records.

    Attributes
    ----------
    times : list of float
        Timestamp for each scan.
    scans : list of list of TrackRecord
        Per-scan lists of track records, aligned with `times`.
    """

    times: list[float]
    scans: list[list[TrackRecord]]


class StateRecord(msgspec.Struct):
    """A single filter state estimate and its flattened covariance.

    Attributes
    ----------
    x : list of float
        State estimate vector.
    p_flat : list of float
        Row-major flattened covariance; ``len == len(x) ** 2``.
    """

    x: list[float]
    p_flat: list[float]


class SimpleTrack(NamedTuple):
    """A decoded track: plain data, no tracker-class dependency.

    Attributes
    ----------
    id : int
        Track identifier.
    state : ndarray
        State estimate vector.
    covariance : ndarray, shape (n, n)
        State covariance matrix.
    status : str
        Track status value.
    """

    id: int
    state: NDArray[np.float64]
    covariance: NDArray[np.float64]
    status: str


_CODECS: dict[str, tuple[Any, Any]] = {
    "msgpack": (msgspec.msgpack.encode, msgspec.msgpack.decode),
    "json": (msgspec.json.encode, msgspec.json.decode),
}


def _codec(fmt: str) -> tuple[Any, Any]:
    """Look up the (encode, decode) callables for `fmt`, or raise."""
    try:
        return _CODECS[fmt]
    except KeyError:
        raise ValueError(
            f"unknown fmt {fmt!r}; expected one of {sorted(_CODECS)}"
        ) from None


def _check_finite(arr: NDArray[np.float64], fmt: str, name: str) -> None:
    """Raise ValueError if `arr` has non-finite values and `fmt` is JSON."""
    if fmt == "json" and not np.all(np.isfinite(arr)):
        raise ValueError(
            f"{name} contains non-finite values (NaN/inf), which JSON "
            "cannot represent; use fmt='msgpack' instead"
        )


def encode_tracks(
    history: Sequence[Sequence[Any]], times: Sequence[float], fmt: str = "msgpack"
) -> bytes:
    """Serialize a per-scan track history to bytes.

    Parameters
    ----------
    history : sequence of sequence of Track-like
        Per-scan lists of objects exposing ``id``, ``state``, ``covariance``,
        and ``status`` (a ``TrackStatus`` enum or plain ``str``).
    times : sequence of float
        Timestamp for each scan in `history`; ``len(times) == len(history)``.
    fmt : {"msgpack", "json"}, optional
        Wire format. With ``"json"``, any non-finite state or covariance
        value raises ``ValueError`` before encoding.

    Returns
    -------
    bytes
        Encoded track history, decodable with `decode_tracks`.

    Examples
    --------
    >>> import numpy as np
    >>> from pytcl.trackers import Track, TrackStatus
    >>> track = Track(id=1, state=np.array([1.0, 2.0]),
    ...                covariance=np.eye(2), status=TrackStatus.CONFIRMED,
    ...                hits=1, misses=0, time=0.0)
    >>> blob = encode_tracks([[track]], [0.0], fmt="json")
    >>> times, history = decode_tracks(blob, fmt="json")
    >>> times
    [0.0]
    >>> t2 = history[0][0]
    >>> (t2.id, t2.status)
    (1, 'confirmed')
    >>> [round(v, 3) for v in t2.state.tolist()]
    [1.0, 2.0]
    """
    encode, _ = _codec(fmt)
    scans = []
    for scan, t in zip(history, times):
        records = []
        for tr in scan:
            state = np.asarray(tr.state, dtype=np.float64)
            cov = np.asarray(tr.covariance, dtype=np.float64)
            _check_finite(state, fmt, "state")
            _check_finite(cov, fmt, "covariance")
            status = getattr(tr.status, "value", tr.status)
            records.append(
                TrackRecord(
                    track_id=int(tr.id),
                    t=float(t),
                    status=status,
                    state=state.tolist(),
                    covariance=cov.ravel().tolist(),
                )
            )
        scans.append(records)
    track_set = TrackSet(times=[float(t) for t in times], scans=scans)
    return encode(track_set)


def decode_tracks(
    data: bytes, fmt: str = "msgpack"
) -> tuple[list[float], list[list[SimpleTrack]]]:
    """Deserialize a track history produced by `encode_tracks`.

    Parameters
    ----------
    data : bytes
        Encoded track history.
    fmt : {"msgpack", "json"}, optional
        Wire format `data` was encoded with.

    Returns
    -------
    times : list of float
        Timestamp for each scan.
    history : list of list of SimpleTrack
        Per-scan lists of decoded tracks, aligned with `times`.

    Raises
    ------
    msgspec.ValidationError or msgspec.DecodeError
        If `data` does not match the expected structure.

    Examples
    --------
    >>> import numpy as np
    >>> from pytcl.trackers import Track, TrackStatus
    >>> track = Track(id=1, state=np.array([1.0, 2.0]),
    ...                covariance=np.eye(2), status=TrackStatus.CONFIRMED,
    ...                hits=1, misses=0, time=0.0)
    >>> blob = encode_tracks([[track]], [0.0], fmt="msgpack")
    >>> times, history = decode_tracks(blob, fmt="msgpack")
    >>> times
    [0.0]
    >>> history[0][0].covariance.tolist()
    [[1.0, 0.0], [0.0, 1.0]]
    """
    _, decode = _codec(fmt)
    track_set = decode(data, type=TrackSet)
    history: list[list[SimpleTrack]] = []
    for records in track_set.scans:
        scan = []
        for rec in records:
            state = np.asarray(rec.state, dtype=np.float64)
            n = state.shape[0]
            cov = np.asarray(rec.covariance, dtype=np.float64).reshape(n, n)
            scan.append(
                SimpleTrack(
                    id=rec.track_id, state=state, covariance=cov, status=rec.status
                )
            )
        history.append(scan)
    return list(track_set.times), history


def encode_states(x: Any, P: Any, fmt: str = "msgpack") -> bytes:
    """Serialize a single filter state estimate and covariance to bytes.

    Parameters
    ----------
    x : array_like
        State estimate vector, shape (n,).
    P : array_like
        State covariance matrix, shape (n, n).
    fmt : {"msgpack", "json"}, optional
        Wire format. With ``"json"``, non-finite values in `x` or `P` raise
        ``ValueError`` before encoding.

    Returns
    -------
    bytes
        Encoded state, decodable with `decode_states`.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([1.0, 2.0])
    >>> P = np.eye(2)
    >>> blob = encode_states(x, P, fmt="json")
    >>> x2, P2 = decode_states(blob, fmt="json")
    >>> [round(v, 3) for v in x2.tolist()]
    [1.0, 2.0]
    >>> P2.tolist()
    [[1.0, 0.0], [0.0, 1.0]]
    """
    encode, _ = _codec(fmt)
    x = np.asarray(x, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    _check_finite(x, fmt, "x")
    _check_finite(P, fmt, "P")
    record = StateRecord(x=x.tolist(), p_flat=P.ravel().tolist())
    return encode(record)


def decode_states(
    data: bytes, fmt: str = "msgpack"
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Deserialize a state estimate and covariance produced by `encode_states`.

    Parameters
    ----------
    data : bytes
        Encoded state.
    fmt : {"msgpack", "json"}, optional
        Wire format `data` was encoded with.

    Returns
    -------
    x : ndarray, shape (n,)
        State estimate vector.
    P : ndarray, shape (n, n)
        State covariance matrix.

    Raises
    ------
    msgspec.ValidationError or msgspec.DecodeError
        If `data` does not match the expected structure.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([1.0, 2.0, 3.0])
    >>> P = np.eye(3)
    >>> x2, P2 = decode_states(encode_states(x, P, fmt="msgpack"), fmt="msgpack")
    >>> x2.tolist()
    [1.0, 2.0, 3.0]
    >>> P2.shape
    (3, 3)
    """
    _, decode = _codec(fmt)
    record = decode(data, type=StateRecord)
    x = np.asarray(record.x, dtype=np.float64)
    n = x.shape[0]
    P = np.asarray(record.p_flat, dtype=np.float64).reshape(n, n)
    return x, P
