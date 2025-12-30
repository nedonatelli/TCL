"""
Tracker Component Library - Python Port

A comprehensive library for target tracking algorithms, including coordinate
systems, dynamic models, estimation algorithms, and mathematical functions.

This is a Python port of the U.S. Naval Research Laboratory's Tracker Component
Library originally written in MATLAB.

Examples
--------
>>> import tracker_component_library as tcl
>>> from tracker_component_library.coordinate_systems import cart2sphere
>>> from tracker_component_library.dynamic_estimation.kalman import KalmanFilter

References
----------
.. [1] D. F. Crouse, "The Tracker Component Library: Free Routines for Rapid
       Prototyping," IEEE Aerospace and Electronic Systems Magazine, vol. 32,
       no. 5, pp. 18-27, May 2017.
"""

__version__ = "0.1.0"
__author__ = "Python Port Contributors"
__original_author__ = "David F. Crouse, Naval Research Laboratory"

# Core utilities
from tracker_component_library import core

# Assignment algorithms (Phase 5)
from tracker_component_library import assignment_algorithms

# Specialized domains (Phase 6)
from tracker_component_library import astronomical
from tracker_component_library import navigation
from tracker_component_library import atmosphere

# End-to-end trackers (Phase 7)
from tracker_component_library import trackers

# Version tuple for programmatic access
VERSION = tuple(int(x) for x in __version__.split("."))

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
]
