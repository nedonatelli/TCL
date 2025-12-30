"""
Tracker Component Library - Python Port

A comprehensive library for target tracking algorithms, including coordinate
systems, dynamic models, estimation algorithms, and mathematical functions.

This is a Python port of the U.S. Naval Research Laboratory's Tracker Component
Library originally written in MATLAB.

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

__version__ = "0.2.1"
__author__ = "Python Port Contributors"
__original_author__ = "David F. Crouse, Naval Research Laboratory"

# Core utilities
from pytcl import core

# Assignment algorithms (Phase 5)
from pytcl import assignment_algorithms

# Specialized domains (Phase 6)
from pytcl import astronomical
from pytcl import navigation
from pytcl import atmosphere

# End-to-end trackers (Phase 7)
from pytcl import trackers

# Performance evaluation
from pytcl import performance_evaluation

# Plotting utilities
from pytcl import plotting

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
    "performance_evaluation",
    "plotting",
]
