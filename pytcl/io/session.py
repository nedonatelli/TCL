"""Session save/restore: full tracker/filter state snapshot and resume.

A "session" is a self-describing snapshot of a stateful tracker/filter
object -- config, current estimate, and anything else needed to resume
predict/update calls exactly where they left off. Two wire formats are
supported via the ``fmt`` argument, matching :mod:`pytcl.io.serialize`:

- ``"msgpack"`` (default): round-trips ``float64`` bit patterns exactly,
  including NaN/inf.
- ``"json"``: human-readable text. JSON has no representation for NaN/inf,
  so saving a session containing non-finite values raises ``ValueError``
  rather than silently producing invalid JSON.

Decoding is strict: malformed or truncated bytes, or bytes from a newer
schema version, raise :class:`~pytcl.core.exceptions.FormatError`.

Some snapshotted objects (trackers built with callable dynamics rather
than fixed matrices) cannot be fully rebuilt from the snapshot alone --
loading such a session without the matching keyword argument raises
:class:`~pytcl.core.exceptions.ConfigurationError`.
"""

from collections import deque
from pathlib import Path
from typing import Any, Callable, Optional, Union

import msgspec
import numpy as np

import pytcl
from pytcl.core.exceptions import ConfigurationError, FormatError
from pytcl.diagnostics import NIS_WINDOW, diagnostics_enabled, logger
from pytcl.dynamic_estimation import IMMEstimator
from pytcl.dynamic_estimation.configs import IMMConfig
from pytcl.io.serialize import _check_finite, _codec
from pytcl.trackers import (
    MHTConfig,
    MHTTracker,
    MultiTargetTracker,
    SingleTargetTracker,
    TrackStatus,
)
from pytcl.trackers.configs import MultiTargetConfig, SingleTargetConfig
from pytcl.trackers.hypothesis import Hypothesis, MHTTrack, MHTTrackStatus
from pytcl.trackers.multi_target import _InternalTrack

__all__ = [
    "SESSION_SCHEMA_VERSION",
    "save_session",
    "save_session_file",
    "load_session",
    "load_session_file",
]

SESSION_SCHEMA_VERSION = 1

_log = logger.bind(site="session")


class SingleTargetSnapshot(msgspec.Struct, tag=True):
    """Snapshot of a :class:`~pytcl.trackers.SingleTargetTracker`.

    Attributes
    ----------
    config : SingleTargetConfig
        Tracker configuration. ``config.F``/``config.Q`` are ``None`` when
        the tracker was built with callable dynamics.
    initialized : bool
        Whether the tracker had been initialized.
    time : float
        Tracker's internal clock at snapshot time.
    x : list of float, optional
        Flattened state estimate; ``None`` if not initialized.
    P : list of list of float, optional
        State covariance; ``None`` if not initialized.
    """

    config: SingleTargetConfig
    initialized: bool
    time: float
    x: Optional[list[float]] = None
    P: Optional[list[list[float]]] = None


class IMMSnapshot(msgspec.Struct, tag=True):
    """Snapshot of an :class:`~pytcl.dynamic_estimation.IMMEstimator`.

    All fields are plain arrays -- an IMM estimator has no callable-dynamics
    escape hatch, so a snapshot always fully reconstructs it.

    Attributes
    ----------
    config : IMMConfig
        Mode count, state dimension, and transition matrix.
    F_list, Q_list : list of list of list of float
        Per-mode state transition matrices and process noise covariances.
    H_list, R_list : list of list of list of float
        Per-mode measurement matrices and measurement noise covariances.
    mode_states : list of list of float
        Per-mode state estimates.
    mode_covs : list of list of list of float
        Per-mode state covariances.
    mode_probs : list of float
        Mode probabilities.
    x : list of float
        Combined state estimate.
    P : list of list of float
        Combined state covariance.
    """

    config: IMMConfig
    F_list: list[list[list[float]]]
    Q_list: list[list[list[float]]]
    H_list: list[list[list[float]]]
    R_list: list[list[list[float]]]
    mode_states: list[list[float]]
    mode_covs: list[list[list[float]]]
    mode_probs: list[float]
    x: list[float]
    P: list[list[float]]


class TrackSnapshot(msgspec.Struct):
    """Snapshot of one internal track owned by a
    :class:`~pytcl.trackers.MultiTargetTracker`.

    Attributes
    ----------
    id : int
        Unique track identifier.
    state : list of float
        Flattened state estimate.
    covariance : list of list of float
        State covariance.
    status : str
        :class:`~pytcl.trackers.TrackStatus` member name (restore via
        ``TrackStatus[status]``).
    hits, misses : int
        Update/miss counters used by confirmation and deletion logic.
    time : float
        Time of the track's last update.
    nis_history : list of float, optional
        Recent normalized innovation squared values, present only when
        the track accumulated a health window under enabled diagnostics.
    """

    id: int
    state: list[float]
    covariance: list[list[float]]
    status: str
    hits: int
    misses: int
    time: float
    nis_history: Optional[list[float]] = None


class MultiTargetSnapshot(msgspec.Struct, tag=True):
    """Snapshot of a :class:`~pytcl.trackers.MultiTargetTracker`.

    Attributes
    ----------
    config : MultiTargetConfig
        Tracker configuration. ``config.F``/``config.Q`` are ``None`` when
        the tracker was built with callable dynamics.
    tracks : list of TrackSnapshot
        One entry per internal track, including deleted ones.
    next_id : int
        Next track id to be assigned; preserved so a resumed tracker never
        reissues an id already in `tracks`.
    time : float
        Tracker's internal clock at snapshot time.
    """

    config: MultiTargetConfig
    tracks: list[TrackSnapshot]
    next_id: int
    time: float


class MHTTrackSnapshot(msgspec.Struct):
    """Snapshot of one :class:`~pytcl.trackers.hypothesis.MHTTrack` branch.

    Attributes
    ----------
    id : int
        Unique track identifier.
    state : list of float
        Flattened state estimate.
    covariance : list of list of float
        State covariance.
    score : float
        Log-likelihood ratio score.
    status : str
        :class:`~pytcl.trackers.hypothesis.MHTTrackStatus` member name
        (restore via ``MHTTrackStatus[status]``).
    history : list of int
        Measurement indices associated with this track branch (-1 for a
        missed detection).
    parent_id : int
        ID of the parent track (-1 for root tracks).
    scan_created : int
        Scan number when this track branch was created.
    n_hits, n_misses : int
        Update/miss counters used by confirmation and deletion logic.
    """

    id: int
    state: list[float]
    covariance: list[list[float]]
    score: float
    status: str
    history: list[int]
    parent_id: int
    scan_created: int
    n_hits: int
    n_misses: int


class HypothesisSnapshot(msgspec.Struct):
    """Snapshot of one :class:`~pytcl.trackers.hypothesis.Hypothesis`.

    Attributes
    ----------
    id : int
        Unique hypothesis identifier.
    probability : float
        Posterior probability of this hypothesis.
    track_ids : list of int
        IDs of tracks included in this hypothesis.
    scan_created : int
        Scan number when this hypothesis was created.
    parent_id : int
        ID of the parent hypothesis (-1 for the initial hypothesis).
    """

    id: int
    probability: float
    track_ids: list[int]
    scan_created: int
    parent_id: int


class MHTSnapshot(msgspec.Struct, tag=True):
    """Snapshot of a :class:`~pytcl.trackers.MHTTracker`.

    ``MHTConfig`` carries only algorithm parameters (no F/H/Q/R), so unlike
    `SingleTargetSnapshot`/`MultiTargetSnapshot` this snapshot carries the
    construction recipe -- `state_dim`, `meas_dim`, `H`, `R`, and optional
    `F`/`Q` -- alongside the embedded config.

    Attributes
    ----------
    config : MHTConfig
        Algorithm configuration (n_scan, max_hypotheses, thresholds, ...).
    state_dim, meas_dim : int
        State and measurement vector dimensions.
    H : list of list of float
        Measurement matrix.
    R : list of list of float
        Measurement noise covariance.
    F, Q : list of list of float, optional
        State transition matrix / process noise covariance; ``None`` when
        the tracker was built with callable dynamics.
    init_covariance : list of list of float
        Initial covariance assigned to newly initiated tracks.
    time : float
        Tracker's internal clock at snapshot time.
    scan : int
        Tracker's internal scan counter at snapshot time.
    hypotheses : list of HypothesisSnapshot
        Hypothesis tree's current hypotheses.
    tracks : list of MHTTrackSnapshot
        Hypothesis tree's track branches, keyed by their own `id` on
        restore.
    current_scan : int
        Hypothesis tree's scan counter at snapshot time.
    next_hypothesis_id, next_track_id : int
        Next ids to be assigned by the hypothesis tree; preserved so a
        resumed tracker never reissues an id already in use.
    """

    config: MHTConfig
    state_dim: int
    meas_dim: int
    H: list[list[float]]
    R: list[list[float]]
    init_covariance: list[list[float]]
    time: float
    scan: int
    hypotheses: list[HypothesisSnapshot]
    tracks: list[MHTTrackSnapshot]
    current_scan: int
    next_hypothesis_id: int
    next_track_id: int
    F: Optional[list[list[float]]] = None
    Q: Optional[list[list[float]]] = None


_Snapshot = Union[SingleTargetSnapshot, IMMSnapshot, MultiTargetSnapshot, MHTSnapshot]


class SessionEnvelope(msgspec.Struct):
    """Wire envelope wrapping a tagged snapshot with schema/version info."""

    schema_version: int
    pytcl_version: str
    snapshot: _Snapshot


def _snap_single_target(t: SingleTargetTracker) -> SingleTargetSnapshot:
    cfg = SingleTargetConfig(
        state_dim=t.state_dim,
        meas_dim=t.meas_dim,
        H=t.H.tolist(),
        R=t.R.tolist(),
        F=None if t._F_matrix is None else t._F_matrix.tolist(),
        Q=None if t._Q_matrix is None else t._Q_matrix.tolist(),
        gate_threshold=t.gate_threshold,
    )
    return SingleTargetSnapshot(
        config=cfg,
        initialized=t._initialized,
        time=t._time,
        x=None if t._state is None else t._state.tolist(),
        P=None if t._covariance is None else t._covariance.tolist(),
    )


def _resolve_matrix(name: str, cfg_value: Any, kwarg_value: Any) -> Any:
    """Resolve one dynamics matrix (F or Q) between a snapshot's config and
    a `load_session` rehydration kwarg.

    The snapshot's own config wins whenever it has the matrix -- it is
    self-describing and does not need help, so a caller-supplied kwarg in
    that case is a mistake (silently overriding the saved dynamics would
    be worse) and raises rather than being applied. A kwarg is required
    (and consumed) only when the config lacks the matrix, i.e. the tracker
    had callable dynamics at save time.
    """
    if cfg_value is not None:
        if kwarg_value is not None:
            raise ConfigurationError(
                f"snapshot already carries matrix {name} dynamics; "
                f"do not pass {name}= to load_session"
            )
        return np.asarray(cfg_value, dtype=np.float64)
    if kwarg_value is None:
        raise ConfigurationError(
            f"snapshot was taken from a tracker with callable {name} "
            f"dynamics; pass {name}= to load_session to rehydrate it"
        )
    return kwarg_value


def _restore_single_target(
    s: SingleTargetSnapshot, F: Any = None, Q: Any = None
) -> SingleTargetTracker:
    cfg = s.config
    F_in = _resolve_matrix("F", cfg.F, F)
    Q_in = _resolve_matrix("Q", cfg.Q, Q)
    t = SingleTargetTracker(
        cfg.state_dim,
        cfg.meas_dim,
        F_in,
        np.asarray(cfg.H, dtype=np.float64),
        Q_in,
        np.asarray(cfg.R, dtype=np.float64),
        gate_threshold=cfg.gate_threshold,
    )
    if s.initialized:
        t._state = np.asarray(s.x, dtype=np.float64)
        t._covariance = np.asarray(s.P, dtype=np.float64)
        t._time = s.time
        t._initialized = True
    return t


def _snap_multi_target(t: MultiTargetTracker) -> MultiTargetSnapshot:
    cfg = MultiTargetConfig(
        state_dim=t.state_dim,
        meas_dim=t.meas_dim,
        H=t.H.tolist(),
        R=t.R.tolist(),
        F=None if t._F_matrix is None else t._F_matrix.tolist(),
        Q=None if t._Q_matrix is None else t._Q_matrix.tolist(),
        gate_probability=t.gate_probability,
        confirm_hits=t.confirm_hits,
        confirm_window=t.confirm_window,
        max_misses=t.max_misses,
        init_covariance=t.init_covariance.tolist(),
    )
    tracks = []
    for track in t._tracks:
        history = getattr(track, "_nis_history", None)
        tracks.append(
            TrackSnapshot(
                id=track.id,
                state=track.state.tolist(),
                covariance=track.covariance.tolist(),
                status=track.status.name,
                hits=track.hits,
                misses=track.misses,
                time=track.time,
                nis_history=None if history is None else list(history),
            )
        )
    return MultiTargetSnapshot(
        config=cfg,
        tracks=tracks,
        next_id=t._next_id,
        time=t._time,
    )


def _restore_multi_target(
    s: MultiTargetSnapshot, F: Any = None, Q: Any = None
) -> MultiTargetTracker:
    cfg = s.config
    F_in = _resolve_matrix("F", cfg.F, F)
    Q_in = _resolve_matrix("Q", cfg.Q, Q)
    t = MultiTargetTracker(
        cfg.state_dim,
        cfg.meas_dim,
        F_in,
        np.asarray(cfg.H, dtype=np.float64),
        Q_in,
        np.asarray(cfg.R, dtype=np.float64),
        gate_probability=cfg.gate_probability,
        confirm_hits=cfg.confirm_hits,
        confirm_window=cfg.confirm_window,
        max_misses=cfg.max_misses,
        init_covariance=(
            None
            if cfg.init_covariance is None
            else np.asarray(cfg.init_covariance, dtype=np.float64)
        ),
    )
    tracks = []
    for ts in s.tracks:
        track = _InternalTrack(
            id=ts.id,
            state=np.asarray(ts.state, dtype=np.float64),
            covariance=np.asarray(ts.covariance, dtype=np.float64),
            status=TrackStatus[ts.status],
            hits=ts.hits,
            misses=ts.misses,
            time=ts.time,
        )
        if ts.nis_history is not None:
            track._nis_history = deque(ts.nis_history, maxlen=NIS_WINDOW)
        tracks.append(track)
    t._tracks = tracks
    t._next_id = s.next_id
    t._time = s.time
    return t


def _snap_mht(t: MHTTracker) -> MHTSnapshot:
    tree = t.hypothesis_tree
    tracks = [
        MHTTrackSnapshot(
            id=tr.id,
            state=tr.state.tolist(),
            covariance=tr.covariance.tolist(),
            score=float(tr.score),
            status=tr.status.name,
            history=list(tr.history),
            parent_id=tr.parent_id,
            scan_created=tr.scan_created,
            n_hits=tr.n_hits,
            n_misses=tr.n_misses,
        )
        for tr in tree.tracks.values()
    ]
    hypotheses = [
        HypothesisSnapshot(
            id=h.id,
            probability=float(h.probability),
            track_ids=list(h.track_ids),
            scan_created=h.scan_created,
            parent_id=h.parent_id,
        )
        for h in tree.hypotheses
    ]
    return MHTSnapshot(
        config=t.config,
        state_dim=t.state_dim,
        meas_dim=t.meas_dim,
        H=t.H.tolist(),
        R=t.R.tolist(),
        init_covariance=t.init_covariance.tolist(),
        time=t._time,
        scan=t._scan,
        hypotheses=hypotheses,
        tracks=tracks,
        current_scan=tree.current_scan,
        next_hypothesis_id=tree._next_hypothesis_id,
        next_track_id=tree._next_track_id,
        F=None if t._F_matrix is None else t._F_matrix.tolist(),
        Q=None if t._Q_matrix is None else t._Q_matrix.tolist(),
    )


def _restore_mht(s: MHTSnapshot, F: Any = None, Q: Any = None) -> MHTTracker:
    F_in = _resolve_matrix("F", s.F, F)
    Q_in = _resolve_matrix("Q", s.Q, Q)
    t = MHTTracker(
        s.state_dim,
        s.meas_dim,
        F_in,
        np.asarray(s.H, dtype=np.float64),
        Q_in,
        np.asarray(s.R, dtype=np.float64),
        config=s.config,
        init_covariance=np.asarray(s.init_covariance, dtype=np.float64),
    )
    t._time = s.time
    t._scan = s.scan
    tree = t.hypothesis_tree
    tree.tracks = {
        ts.id: MHTTrack(
            id=ts.id,
            state=np.asarray(ts.state, dtype=np.float64),
            covariance=np.asarray(ts.covariance, dtype=np.float64),
            score=ts.score,
            status=MHTTrackStatus[ts.status],
            history=list(ts.history),
            parent_id=ts.parent_id,
            scan_created=ts.scan_created,
            n_hits=ts.n_hits,
            n_misses=ts.n_misses,
        )
        for ts in s.tracks
    }
    tree.hypotheses = [
        Hypothesis(
            id=hs.id,
            probability=hs.probability,
            track_ids=list(hs.track_ids),
            scan_created=hs.scan_created,
            parent_id=hs.parent_id,
        )
        for hs in s.hypotheses
    ]
    tree.current_scan = s.current_scan
    tree._next_hypothesis_id = s.next_hypothesis_id
    tree._next_track_id = s.next_track_id
    return t


def _snap_imm(e: IMMEstimator) -> IMMSnapshot:
    cfg = IMMConfig(
        n_modes=e.n_modes,
        state_dim=e.state_dim,
        transition_matrix=e.transition_matrix.tolist(),
    )
    return IMMSnapshot(
        config=cfg,
        F_list=[f.tolist() for f in e.F_list],
        Q_list=[q.tolist() for q in e.Q_list],
        H_list=[h.tolist() for h in e.H_list],
        R_list=[r.tolist() for r in e.R_list],
        mode_states=[x.tolist() for x in e.mode_states],
        mode_covs=[p.tolist() for p in e.mode_covs],
        mode_probs=e.mode_probs.tolist(),
        x=e.x.tolist(),
        P=e.P.tolist(),
    )


def _restore_imm(s: IMMSnapshot, **rehydration: Any) -> IMMEstimator:
    if rehydration:
        raise ConfigurationError(
            "IMM snapshots are self-contained and take no rehydration "
            f"kwargs; got: {sorted(rehydration)}"
        )
    e = IMMEstimator(config=s.config)
    e.F_list = [np.asarray(f, dtype=np.float64) for f in s.F_list]
    e.Q_list = [np.asarray(q, dtype=np.float64) for q in s.Q_list]
    e.H_list = [np.asarray(h, dtype=np.float64) for h in s.H_list]
    e.R_list = [np.asarray(r, dtype=np.float64) for r in s.R_list]
    e.mode_states = [np.asarray(x, dtype=np.float64) for x in s.mode_states]
    e.mode_covs = [np.asarray(p, dtype=np.float64) for p in s.mode_covs]
    e.mode_probs = np.asarray(s.mode_probs, dtype=np.float64)
    e.x = np.asarray(s.x, dtype=np.float64)
    e.P = np.asarray(s.P, dtype=np.float64)
    return e


_SNAPSHOTTERS: dict[type, Callable[[Any], Any]] = {
    SingleTargetTracker: _snap_single_target,
    IMMEstimator: _snap_imm,
    MultiTargetTracker: _snap_multi_target,
    MHTTracker: _snap_mht,
}
_RESTORERS: dict[str, Callable[..., Any]] = {
    "SingleTargetSnapshot": _restore_single_target,
    "IMMSnapshot": _restore_imm,
    "MultiTargetSnapshot": _restore_multi_target,
    "MHTSnapshot": _restore_mht,
}


def _check_snapshot_finite(snapshot: Any, fmt: str) -> None:
    """Walk every numeric field of `snapshot` and reject non-finite values
    under ``fmt="json"`` (mirrors ``serialize._check_finite``).

    Recurses into nested Structs (e.g. a snapshot's ``config``) and into
    lists of Structs (e.g. a snapshot's per-track entries) so new snapshot
    types registered by later tasks are covered automatically, without
    editing this function.
    """
    if fmt != "json":
        return
    for field in msgspec.structs.fields(snapshot):
        value = getattr(snapshot, field.name)
        if value is None or isinstance(value, (bool, str)):
            continue
        if isinstance(value, msgspec.Struct):
            _check_snapshot_finite(value, fmt)
            continue
        if isinstance(value, list) and value and isinstance(value[0], msgspec.Struct):
            for item in value:
                _check_snapshot_finite(item, fmt)
            continue
        if isinstance(value, (int, float, list)):
            arr = np.asarray(value, dtype=np.float64)
            if arr.size:
                _check_finite(arr, fmt, field.name)


def save_session(obj: Any, *, fmt: str = "msgpack") -> bytes:
    """Serialize a tracker/filter's full state to bytes.

    Parameters
    ----------
    obj : SingleTargetTracker, MultiTargetTracker, MHTTracker, or IMMEstimator
        The object to snapshot.
    fmt : {"msgpack", "json"}, optional
        Wire format. With ``"json"``, non-finite values anywhere in the
        snapshot raise ``ValueError`` before encoding.

    Returns
    -------
    bytes
        Encoded session, decodable with `load_session`.

    Raises
    ------
    ConfigurationError
        If `obj`'s type has no registered snapshotter.
    ValueError
        If `fmt` is ``"json"`` and `obj` contains non-finite values.
    """
    snap_fn = _SNAPSHOTTERS.get(type(obj))
    if snap_fn is None:
        raise ConfigurationError(
            f"save_session does not support {type(obj).__name__}; "
            f"supported: {sorted(c.__name__ for c in _SNAPSHOTTERS)}"
        )
    snapshot = snap_fn(obj)
    _check_snapshot_finite(snapshot, fmt)
    env = SessionEnvelope(
        schema_version=SESSION_SCHEMA_VERSION,
        pytcl_version=pytcl.__version__,
        snapshot=snapshot,
    )
    encode, _ = _codec(fmt)
    if diagnostics_enabled():
        _log.debug("saved session for {}", type(obj).__name__)
    return encode(env)


def save_session_file(obj: Any, path: Any, *, fmt: str = "msgpack") -> None:
    """Serialize a tracker/filter's full state to a file.

    Parameters
    ----------
    obj : SingleTargetTracker, MultiTargetTracker, MHTTracker, or IMMEstimator
        The object to snapshot.
    path : str or Path
        Destination file path.
    fmt : {"msgpack", "json"}, optional
        Wire format; see `save_session`.
    """
    Path(path).write_bytes(save_session(obj, fmt=fmt))


def load_session(data: bytes, *, fmt: str = "msgpack", **models: Any) -> Any:
    """Deserialize a tracker/filter's full state from bytes.

    Parameters
    ----------
    data : bytes
        Encoded session produced by `save_session`.
    fmt : {"msgpack", "json"}, optional
        Wire format `data` was encoded with.
    **models : Any
        Rehydration arguments for snapshots that could not capture
        callable dynamics (``F=``/``Q=`` for a
        :class:`~pytcl.trackers.SingleTargetTracker`,
        :class:`~pytcl.trackers.MultiTargetTracker`, or
        :class:`~pytcl.trackers.MHTTracker` built with callable
        ``F``/``Q``). Consumed only where the snapshot actually needs
        them, one matrix at a time: if the snapshot's config already has
        a matrix for ``F`` (or ``Q``), passing that kwarg raises rather
        than silently overriding the saved dynamics; if the config lacks
        it, omitting the kwarg raises rather than restoring a tracker
        that cannot predict. Snapshot types with no callable-dynamics
        escape hatch at all (:class:`~pytcl.dynamic_estimation.IMMEstimator`)
        are fully self-contained and reject every keyword argument.

    Returns
    -------
    object
        The restored tracker/filter, resumable via its normal
        predict/update API.

    Raises
    ------
    FormatError
        If `data` is malformed, or was produced by a newer schema version
        than this pytcl supports.
    ConfigurationError
        If the snapshot needs a rehydration keyword argument that was not
        supplied, or was given one it does not need (either because the
        snapshot's config already carries that matrix, or because the
        snapshot type is fully self-contained and takes none at all).
    """
    _, decode = _codec(fmt)
    try:
        env = decode(data, type=SessionEnvelope)
    except msgspec.DecodeError as exc:
        raise FormatError(f"not a pytcl session: {exc}") from exc
    if env.schema_version > SESSION_SCHEMA_VERSION:
        raise FormatError(
            f"session schema v{env.schema_version} is newer than this "
            f"pytcl supports (v{SESSION_SCHEMA_VERSION})"
        )
    restore = _RESTORERS[type(env.snapshot).__name__]
    return restore(env.snapshot, **models)


def load_session_file(path: Any, *, fmt: str = "msgpack", **models: Any) -> Any:
    """Deserialize a tracker/filter's full state from a file.

    Parameters
    ----------
    path : str or Path
        Source file path, as written by `save_session_file`.
    fmt : {"msgpack", "json"}, optional
        Wire format; see `load_session`.
    **models : Any
        Rehydration arguments; see `load_session`.

    Returns
    -------
    object
        The restored tracker/filter.
    """
    return load_session(Path(path).read_bytes(), fmt=fmt, **models)
