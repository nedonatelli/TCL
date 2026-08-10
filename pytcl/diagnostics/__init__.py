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
from typing import Any, Optional

from loguru import logger as _logger

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
