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

logger = _logger  # instrumentation sites import this and pass name="pytcl"

_handler_id: Optional[int] = None
_enabled: bool = False


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
    """
    global _handler_id, _enabled
    if _handler_id is not None:
        _logger.remove(_handler_id)
    _logger.enable("pytcl")
    _handler_id = _logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan> - {message}"
        ),
        filter=lambda record: record["name"].startswith("pytcl"),
    )
    _enabled = True


def disable_debug_logging() -> None:
    """Return the library to complete silence. Idempotent."""
    global _handler_id, _enabled
    if _handler_id is not None:
        _logger.remove(_handler_id)
        _handler_id = None
    _logger.disable("pytcl")
    _enabled = False


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
