"""
Tracker Component Library - Python Port

A comprehensive library for target tracking algorithms, including coordinate
systems, dynamic models, estimation algorithms, and mathematical functions.

This is a Python port of the U.S. Naval Research Laboratory's Tracker Component
Library originally written in MATLAB.
**Current Version:** 2.0.0 (August 2, 2026)
**Status:** Production-ready, 4,973 tests passing, 80% line coverage
Examples
--------
>>> import pytcl as pytcl
>>> from pytcl.coordinate_systems import cart2sphere
>>> from pytcl.dynamic_estimation import kf_predict, kf_update

References
----------
- D. F. Crouse, "The Tracker Component Library: Free Routines for Rapid
  Prototyping," IEEE Aerospace and Electronic Systems Magazine, vol. 32,
  no. 5, pp. 18-27, May 2017.
"""

__version__ = "2.1.0"
__author__ = "Python Port Contributors"
__original_author__ = "David F. Crouse, Naval Research Laboratory"

# Plotting utilities
# Performance evaluation
# End-to-end trackers (Phase 7)
# Specialized domains (Phase 6)
# Assignment algorithms (Phase 5)
# I/O and storage
# Core utilities
from pytcl import (
    assignment_algorithms,
    astronomical,
    atmosphere,
    core,
    diagnostics,
    io,
    navigation,
    performance_evaluation,
    plotting,
    trackers,
)
from pytcl.diagnostics import (
    diagnostics_enabled,
    disable_debug_logging,
    enable_debug_logging,
    progress_bar,
    track_table,
)


# Version tuple for programmatic access
# Handle dev/alpha/beta/rc suffixes by extracting only numeric parts
def _parse_version(version_str: str) -> tuple[int, ...]:
    """Parse version string into tuple of integers."""
    import re

    parts = re.split(r"[.\-]", version_str)
    nums = []
    for p in parts:
        # Extract leading digits
        match = re.match(r"^(\d+)", p)
        if match:
            nums.append(int(match.group(1)))
    return tuple(nums)


VERSION = _parse_version(__version__)

__all__ = [
    "__version__",
    "__author__",
    "__original_author__",
    "VERSION",
    "core",
    "assignment_algorithms",
    "astronomical",
    "io",
    "navigation",
    "atmosphere",
    "trackers",
    "performance_evaluation",
    "plotting",
    "diagnostics",
    "diagnostics_enabled",
    "enable_debug_logging",
    "disable_debug_logging",
    "track_table",
    "progress_bar",
]
