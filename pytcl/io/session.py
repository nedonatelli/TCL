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

from pathlib import Path
from typing import Any, Callable, Optional, Union

import msgspec
import numpy as np

import pytcl
from pytcl.core.exceptions import ConfigurationError, FormatError
from pytcl.diagnostics import diagnostics_enabled, logger
from pytcl.dynamic_estimation import IMMEstimator
from pytcl.dynamic_estimation.configs import IMMConfig
from pytcl.io.serialize import _check_finite, _codec
from pytcl.trackers import SingleTargetTracker
from pytcl.trackers.configs import SingleTargetConfig

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


_Snapshot = Union[SingleTargetSnapshot, IMMSnapshot]


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


def _restore_single_target(
    s: SingleTargetSnapshot, F: Any = None, Q: Any = None
) -> SingleTargetTracker:
    cfg = s.config
    F_in = F if F is not None else (None if cfg.F is None else np.asarray(cfg.F))
    Q_in = Q if Q is not None else (None if cfg.Q is None else np.asarray(cfg.Q))
    if F_in is None or Q_in is None:
        raise ConfigurationError(
            "snapshot was taken from a SingleTargetTracker with callable "
            "dynamics; pass F= and/or Q= to load_session to rehydrate them"
        )
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


def _restore_imm(s: IMMSnapshot) -> IMMEstimator:
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
}
_RESTORERS: dict[str, Callable[..., Any]] = {
    "SingleTargetSnapshot": _restore_single_target,
    "IMMSnapshot": _restore_imm,
}


def _check_snapshot_finite(snapshot: Any, fmt: str) -> None:
    """Walk every numeric field of `snapshot` and reject non-finite values
    under ``fmt="json"`` (mirrors ``serialize._check_finite``).

    Recurses into nested Structs (e.g. a snapshot's ``config``) so new
    snapshot types registered by later tasks are covered automatically,
    without editing this function.
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
        if isinstance(value, (int, float, list)):
            arr = np.asarray(value, dtype=np.float64)
            if arr.size:
                _check_finite(arr, fmt, field.name)


def save_session(obj: Any, *, fmt: str = "msgpack") -> bytes:
    """Serialize a tracker/filter's full state to bytes.

    Parameters
    ----------
    obj : SingleTargetTracker or IMMEstimator
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
    obj : SingleTargetTracker or IMMEstimator
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
        callable dynamics (e.g. ``F=``/``Q=`` for a
        :class:`~pytcl.trackers.SingleTargetTracker` built with callable
        ``F``/``Q``). Ignored by snapshot types that do not need them.

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
        If the snapshot needs rehydration arguments that were not
        supplied.
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
