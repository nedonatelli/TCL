"""
Three-dimensional assignment algorithms.

This module provides algorithms for solving 3D assignment problems,
which arise in multi-sensor data fusion and multi-scan tracking.
"""

from pytcl.assignment_algorithms.three_dimensional.assignment import (
    Assignment3DResult,
    assign3d,
    assign3d_auction,
    assign3d_lagrangian,
    decompose_to_2d,
    greedy_3d,
)

__all__ = [
    "Assignment3DResult",
    "assign3d",
    "assign3d_lagrangian",
    "assign3d_auction",
    "greedy_3d",
    "decompose_to_2d",
]
