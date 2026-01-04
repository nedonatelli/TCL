"""
Hierarchical logging configuration for pyTCL.

Provides:
- Hierarchical loggers (pytcl.estimation, pytcl.assignment, etc.)
- Performance instrumentation decorators
- Context managers for timing critical sections
- Configurable output formats and levels

Usage
-----
>>> from pytcl.logging_config import get_logger, timed, TimingContext
>>> logger = get_logger(__name__)
>>> logger.debug("Processing measurement batch")

>>> @timed(logger, "kf_predict")
... def kf_predict(x, P, F, Q):
...     ...

>>> with TimingContext(logger, "update_loop"):
...     for _ in range(100):
...         do_update()
"""

import functools
import logging
import time
from contextlib import contextmanager
from typing import Any, Callable, Generator, Optional, TypeVar

# Type variable for decorated functions
F = TypeVar("F", bound=Callable[..., Any])


# =============================================================================
# Logger Configuration
# =============================================================================

# Root logger for pytcl namespace
PYTCL_LOGGER = "pytcl"

# Sub-loggers for major components
LOGGER_HIERARCHY = {
    "pytcl.estimation": "Dynamic estimation algorithms (Kalman, IMM, particle)",
    "pytcl.assignment": "Assignment and data association (gating, JPDA, MHT)",
    "pytcl.signal": "Signal processing functions (CFAR, matched filter)",
    "pytcl.coordinate": "Coordinate system operations (rotations, conversions)",
    "pytcl.containers": "Data containers and structures (TrackList, KDTree)",
    "pytcl.math": "Mathematical functions (special functions, transforms)",
    "pytcl.perf": "Performance instrumentation and timing",
}

# Default format strings
FORMATS = {
    "detailed": (
        "%(asctime)s - %(name)s - %(levelname)s - "
        "%(funcName)s:%(lineno)d - %(message)s"
    ),
    "simple": "%(name)s - %(levelname)s - %(message)s",
    "performance": "%(asctime)s - PERF - %(name)s - %(message)s",
    "minimal": "%(levelname)s: %(message)s",
}


def configure_logging(
    level: int = logging.WARNING,
    format_style: str = "simple",
    handler: Optional[logging.Handler] = None,
) -> logging.Logger:
    """
    Configure the pytcl logging hierarchy.

    Parameters
    ----------
    level : int
        Logging level (e.g., logging.DEBUG, logging.INFO).
    format_style : str
        One of 'detailed', 'simple', 'performance', 'minimal'.
    handler : logging.Handler, optional
        Custom handler. If None, uses StreamHandler.

    Returns
    -------
    logging.Logger
        The root pytcl logger.

    Examples
    --------
    >>> import logging
    >>> from pytcl.logging_config import configure_logging
    >>> configure_logging(level=logging.DEBUG, format_style="detailed")
    """
    root = logging.getLogger(PYTCL_LOGGER)
    root.setLevel(level)

    # Clear existing handlers
    root.handlers.clear()

    # Create handler if not provided
    if handler is None:
        handler = logging.StreamHandler()

    # Set format
    fmt = FORMATS.get(format_style, FORMATS["simple"])
    formatter = logging.Formatter(fmt)
    handler.setFormatter(formatter)
    handler.setLevel(level)

    root.addHandler(handler)

    return root


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger in the pytcl hierarchy.

    Parameters
    ----------
    name : str
        Logger name. If starts with 'pytcl.', used as-is.
        Otherwise, 'pytcl.' is prepended.

    Returns
    -------
    logging.Logger
        Logger instance.

    Examples
    --------
    >>> logger = get_logger("dynamic_estimation.kalman")
    >>> logger.name
    'pytcl.dynamic_estimation.kalman'
    """
    if not name.startswith(PYTCL_LOGGER):
        name = f"{PYTCL_LOGGER}.{name}"
    return logging.getLogger(name)


# =============================================================================
# Performance Instrumentation
# =============================================================================

# Performance logger
_perf_logger = logging.getLogger(f"{PYTCL_LOGGER}.perf")


def timed(
    logger: Optional[logging.Logger] = None,
    name: Optional[str] = None,
    level: int = logging.DEBUG,
) -> Callable[[F], F]:
    """
    Decorator to time function execution.

    Parameters
    ----------
    logger : logging.Logger, optional
        Logger to use. Defaults to pytcl.perf.
    name : str, optional
        Name to use in log message. Defaults to function name.
    level : int
        Logging level. Default is DEBUG.

    Returns
    -------
    callable
        Decorated function.

    Examples
    --------
    >>> @timed(logger, "kf_predict")
    ... def kf_predict(x, P, F, Q):
    ...     return do_prediction(x, P, F, Q)
    """

    def decorator(func: F) -> F:
        log = logger or _perf_logger
        func_name = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                elapsed = (time.perf_counter() - start) * 1000
                log.log(level, "%s completed in %.3fms", func_name, elapsed)
                return result
            except Exception as e:
                elapsed = (time.perf_counter() - start) * 1000
                log.log(level, "%s failed after %.3fms: %s", func_name, elapsed, e)
                raise

        return wrapper

    return decorator


@contextmanager
def TimingContext(
    logger: Optional[logging.Logger] = None,
    name: str = "operation",
    level: int = logging.DEBUG,
) -> Generator[None, None, None]:
    """
    Context manager for timing code blocks.

    Parameters
    ----------
    logger : logging.Logger, optional
        Logger to use. Defaults to pytcl.perf.
    name : str
        Name for the operation being timed.
    level : int
        Logging level.

    Yields
    ------
    dict
        Dictionary that will contain 'elapsed_ms' after context exits.

    Examples
    --------
    >>> with TimingContext(logger, "update_loop") as timing:
    ...     for _ in range(100):
    ...         do_update()
    >>> print(f"Elapsed: {timing['elapsed_ms']:.2f}ms")
    """
    log = logger or _perf_logger
    timing: dict[str, float] = {"elapsed_ms": 0.0}
    start = time.perf_counter()
    try:
        yield timing
    finally:
        timing["elapsed_ms"] = (time.perf_counter() - start) * 1000
        log.log(level, "%s completed in %.3fms", name, timing["elapsed_ms"])


class PerformanceTracker:
    """
    Track cumulative performance statistics.

    Useful for tracking performance across many iterations without
    logging each one individually.

    Parameters
    ----------
    name : str
        Name for the tracked operation.
    logger : logging.Logger, optional
        Logger to use. Defaults to pytcl.perf.

    Examples
    --------
    >>> tracker = PerformanceTracker("filter_cycles")
    >>> for _ in range(1000):
    ...     with tracker.track():
    ...         do_filter_step()
    >>> tracker.log_summary()
    """

    def __init__(self, name: str, logger: Optional[logging.Logger] = None):
        self.name = name
        self.logger = logger or _perf_logger
        self.count = 0
        self.total_ms = 0.0
        self.min_ms = float("inf")
        self.max_ms = 0.0

    @contextmanager
    def track(self) -> Generator[None, None, None]:
        """Track a single operation."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - start) * 1000
            self.count += 1
            self.total_ms += elapsed
            self.min_ms = min(self.min_ms, elapsed)
            self.max_ms = max(self.max_ms, elapsed)

    @property
    def mean_ms(self) -> float:
        """Get mean execution time."""
        return self.total_ms / self.count if self.count > 0 else 0.0

    def log_summary(self, level: int = logging.INFO) -> None:
        """Log performance summary."""
        if self.count == 0:
            self.logger.log(level, "%s: no data", self.name)
            return

        self.logger.log(
            level,
            "%s: count=%d, mean=%.3fms, min=%.3fms, max=%.3fms, total=%.1fms",
            self.name,
            self.count,
            self.mean_ms,
            self.min_ms,
            self.max_ms,
            self.total_ms,
        )

    def reset(self) -> None:
        """Reset statistics."""
        self.count = 0
        self.total_ms = 0.0
        self.min_ms = float("inf")
        self.max_ms = 0.0

    def __repr__(self) -> str:
        return (
            f"PerformanceTracker(name={self.name!r}, count={self.count}, "
            f"mean_ms={self.mean_ms:.3f})"
        )


__all__ = [
    "configure_logging",
    "get_logger",
    "timed",
    "TimingContext",
    "PerformanceTracker",
    "PYTCL_LOGGER",
    "LOGGER_HIERARCHY",
    "FORMATS",
]
