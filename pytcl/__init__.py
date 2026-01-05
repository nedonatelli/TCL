"""
Tracker Component Library - Python Port

A comprehensive library for target tracking algorithms, including coordinate
systems, dynamic models, estimation algorithms, and mathematical functions.

This is a Python port of the U.S. Naval Research Laboratory's Tracker Component
Library originally written in MATLAB.
**Current Version:** 1.10.0 (January 4, 2026)
**Status:** Production-ready, 2,133 tests passing, 76% line coverage
Examples
--------
>>> import pytcl as pytcl
>>> from pytcl.coordinate_systems import cart2sphere
>>> from pytcl.dynamic_estimation.kalman import KalmanFilter

References
----------
.. [1] D. F. Crouse, "The Tracker Component Library: Free Routines for Rapid
       Prototyping," IEEE Aerospace and Electronic Systems Magazine, vol. 32,
       no. 5, pp. 18-27, May 2017.
"""

__version__ = "1.10.0"
__author__ = "Python Port Contributors"
__original_author__ = "David F. Crouse, Naval Research Laboratory"

# Plotting utilities
# Performance evaluation
# End-to-end trackers (Phase 7)
# Specialized domains (Phase 6)
# Assignment algorithms (Phase 5)
# Core utilities
from pytcl import (
    assignment_algorithms,
    astronomical,
    atmosphere,
    core,
    navigation,
    performance_evaluation,
    plotting,
    trackers,
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
    "navigation",
    "atmosphere",
    "trackers",
    "performance_evaluation",
    "plotting",
]
