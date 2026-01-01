"""
Two-dimensional assignment algorithms.

This module provides optimal and suboptimal algorithms for solving
the 2D assignment problem (bipartite matching), as well as k-best
assignment algorithms for Multiple Hypothesis Tracking.
"""

from pytcl.assignment_algorithms.two_dimensional.assignment import (
    AssignmentResult,
    assign2d,
    auction,
    hungarian,
    linear_sum_assignment,
)
from pytcl.assignment_algorithms.two_dimensional.kbest import (
    KBestResult,
    kbest_assign2d,
    murty,
    ranked_assignments,
)

__all__ = [
    # 2D Assignment
    "hungarian",
    "auction",
    "linear_sum_assignment",
    "assign2d",
    "AssignmentResult",
    # K-Best Assignment
    "KBestResult",
    "murty",
    "kbest_assign2d",
    "ranked_assignments",
]
