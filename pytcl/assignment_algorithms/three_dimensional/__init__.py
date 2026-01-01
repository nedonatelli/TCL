"""
Three-dimensional assignment algorithms.

This module provides algorithms for solving 3D assignment problems,
which arise in multi-sensor data fusion and multi-scan tracking.
"""

from pytcl.assignment_algorithms.three_dimensional.assignment import Assignment3DResult
from pytcl.assignment_algorithms.three_dimensional.assignment import assign3d
from pytcl.assignment_algorithms.three_dimensional.assignment import assign3d_auction
from pytcl.assignment_algorithms.three_dimensional.assignment import assign3d_lagrangian
from pytcl.assignment_algorithms.three_dimensional.assignment import decompose_to_2d
from pytcl.assignment_algorithms.three_dimensional.assignment import greedy_3d

__all__ = [
    "Assignment3DResult",
    "assign3d",
    "assign3d_lagrangian",
    "assign3d_auction",
    "greedy_3d",
    "decompose_to_2d",
]
