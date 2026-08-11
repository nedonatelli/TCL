"""
Diagnostics: opt-in logging, instrumentation, and progress reporting.

pytcl is completely silent by default: importing it disables the ``pytcl``
loguru namespace and installs no handlers. Call :func:`enable_debug_logging`
to see gating rejections, association decisions, filter-health symptoms,
and data-file resolution at DEBUG level; :func:`disable_debug_logging`
returns to silence. This module is the redesigned successor to the
``pytcl.logging_config`` module removed in v2.0.0; no compatibility is
provided.

Examples
--------
>>> import pytcl
>>> pytcl.diagnostics.diagnostics_enabled()
False
"""

import sys
from typing import Any, Optional, Sequence

from loguru import logger as _logger

# Filter-health thresholds.
NIS_WINDOW = 20
NIS_OUTLIER_FACTOR = 3.0
CONDITION_WARN = 1e12

# The library never speaks unless spoken to.
_logger.disable("pytcl")

logger = _logger  # instrumentation sites import this and bind site=...

_handler_id: Optional[int] = None
_enabled: bool = False
# Id of loguru's stock stderr handler (0 at interpreter start). Tracked
# across enable/disable cycles so we only ever remove/restore the one
# handler that stands in for "the host's default logging", never our own.
_default_handler_id: Optional[int] = 0
# Whether *we* removed the handler at _default_handler_id above -- disable()
# only restores a stand-in if this is True, so a default handler the host
# application removed itself (or already had removed) is never resurrected.
_removed_default_handler: bool = False


def diagnostics_enabled() -> bool:
    """Whether diagnostic logging is currently enabled.

    Hot paths consult this before constructing log payloads, so the
    disabled path costs one boolean check.
    """
    return _enabled


def enable_debug_logging(level: str = "DEBUG") -> None:
    """
    Enable pytcl's diagnostic logging with a rich-formatted handler.

    Parameters
    ----------
    level : str, optional
        Minimum level to emit ("DEBUG", "INFO", "WARNING", ...).

    Notes
    -----
    Idempotent: calling again replaces the previous handler rather than
    stacking a second one. Output goes to stderr, ASCII-safe.

    A bad ``level`` raises before anything is touched: the new handler is
    added first, so the failure happens while state is still exactly what
    it was before the call (namespace disabled, or however a prior
    successful call last left it).

    loguru installs a default stderr handler (id 0) at import time; left
    alive, it double-prints every record next to the one pytcl's own
    handler emits. This function removes it the first time diagnostics are
    enabled and remembers that it did so; :func:`disable_debug_logging`
    then puts an equivalent handler back (``logger.add(sys.stderr)``,
    loguru's own default configuration) so the host application's logging
    is not left worse off than before pytcl touched it. That replacement
    is loguru's stock handler, not a byte-for-byte restoration of whatever
    the host had configured at id 0 -- if the host had customized it,
    this function has no way to know and does not try to reproduce it.
    """
    global _handler_id, _enabled, _default_handler_id, _removed_default_handler

    # Add the new handler FIRST: an invalid level raises here, before any
    # module state changes, so a failed call leaves everything untouched.
    new_handler_id = _logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan> - {message}"
        ),
        filter=lambda record: (
            record["name"] == "pytcl" or record["name"].startswith("pytcl.")
        ),
    )

    if _handler_id is not None:
        # Idempotent re-enable: retire our previous handler. The host may
        # have already removed it out from under us -- that is not our
        # error to raise.
        try:
            _logger.remove(_handler_id)
        except ValueError:
            pass
        _handler_id = None

    if _default_handler_id is not None:
        # First enable since interpreter start (or since the last
        # disable() restored a stand-in): take down the default handler
        # so it stops double-printing alongside ours.
        try:
            _logger.remove(_default_handler_id)
            _removed_default_handler = True
        except ValueError:
            # Already gone -- the host removed it, or a previous call did.
            _removed_default_handler = False
        _default_handler_id = None

    _handler_id = new_handler_id
    _logger.enable("pytcl")
    _enabled = True


def disable_debug_logging() -> None:
    """Return the library to complete silence. Idempotent.

    If :func:`enable_debug_logging` removed loguru's default stderr
    handler to prevent double-printing, this restores an equivalent one
    (see that function's docstring for the exact caveat).
    """
    global _handler_id, _enabled, _default_handler_id, _removed_default_handler
    if _handler_id is not None:
        try:
            _logger.remove(_handler_id)
        except ValueError:
            pass
        _handler_id = None
    _logger.disable("pytcl")
    _enabled = False
    if _removed_default_handler:
        _default_handler_id = _logger.add(sys.stderr)
        _removed_default_handler = False


def log_filter_health(
    track_id: Any,
    nis_value: float,
    nis_window: Sequence[float],
    cov_condition: float,
) -> None:
    """Log a per-track filter-health snapshot (NIS + covariance condition).

    Guarded internally by :func:`diagnostics_enabled` -- callers on a hot
    path may call this bare without checking first, since a disabled
    namespace makes the call a single boolean-check no-op. Plain
    floats/sequences only; this module takes no dependency on ``pytcl``'s
    tracker types.

    Parameters
    ----------
    track_id : Any
        Identifier of the track this health snapshot belongs to.
    nis_value : float
        Normalized innovation squared for the current update.
    nis_window : sequence of float
        Recent NIS history used as the local baseline for outlier detection.
        Includes the current sample, per call-site convention.
    cov_condition : float
        Condition number of the track's state covariance.

    Notes
    -----
    Symptomatic (logged at WARNING instead of DEBUG) when either:

    - ``nis_value`` exceeds ``NIS_OUTLIER_FACTOR`` times the mean of
      ``nis_window`` (filter diverging / mismatched noise model), or
    - ``cov_condition`` exceeds ``CONDITION_WARN`` (covariance going
      numerically singular).

    The caller owns ``nis_window``'s lifecycle; this function only reads it.
    Callers that keep a rolling window across an enable/disable toggle (as
    ``MultiTargetTracker`` does) will blend pre-disable history into the
    first post-re-enable call -- that's a call-site persistence choice, not
    something this function corrects.
    """
    if not diagnostics_enabled():
        return

    window = list(nis_window)
    # mean_nis == 0 only when every sample in the window is exactly 0 (or
    # the window is empty); guarding against it means an all-zero window
    # can never trip the NIS-outlier branch, however large nis_value gets --
    # such a track is still caught by the cov_condition branch below.
    mean_nis = sum(window) / len(window) if window else 0.0
    symptomatic = (
        nis_value > NIS_OUTLIER_FACTOR * mean_nis and mean_nis > 0
    ) or cov_condition > CONDITION_WARN

    bound = logger.bind(site="filter_health")
    message = "track {}: nis={:.4f} (window_mean={:.4f}, n={}) cov_condition={:.4e}"
    if symptomatic:
        bound.warning(
            message, track_id, nis_value, mean_nis, len(window), cov_condition
        )
    else:
        bound.debug(message, track_id, nis_value, mean_nis, len(window), cov_condition)


from pytcl.diagnostics.render import progress_bar, track_table  # noqa: E402

__all__ = [
    "logger",
    "diagnostics_enabled",
    "enable_debug_logging",
    "disable_debug_logging",
    "track_table",
    "progress_bar",
    "log_filter_health",
    "NIS_WINDOW",
    "NIS_OUTLIER_FACTOR",
    "CONDITION_WARN",
]
