"""
Two-dimensional assignment algorithms.

This module provides optimal and suboptimal algorithms for solving
the 2D assignment problem (bipartite matching).
"""

from pytcl.assignment_algorithms.two_dimensional.assignment import (
    hungarian,
    auction,
    linear_sum_assignment,
    assign2d,
    AssignmentResult,
)

__all__ = [
    "hungarian",
    "auction",
    "linear_sum_assignment",
    "assign2d",
    "AssignmentResult",
]
