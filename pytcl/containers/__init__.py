"""
Containers module.

This module provides spatial data structures for efficient
nearest neighbor queries and spatial indexing.
"""

from pytcl.containers.covertree import (
    CoverTree,
    CoverTreeNode,
    CoverTreeResult,
)
from pytcl.containers.kd_tree import (
    BallTree,
    KDNode,
    KDTree,
    NearestNeighborResult,
)
from pytcl.containers.rtree import (
    BoundingBox,
    RTree,
    RTreeNode,
    RTreeResult,
    box_from_point,
    box_from_points,
    merge_boxes,
)
from pytcl.containers.vptree import (
    VPNode,
    VPTree,
    VPTreeResult,
)

__all__ = [
    # K-D Tree
    "KDNode",
    "NearestNeighborResult",
    "KDTree",
    "BallTree",
    # R-Tree
    "BoundingBox",
    "merge_boxes",
    "box_from_point",
    "box_from_points",
    "RTreeNode",
    "RTreeResult",
    "RTree",
    # VP-Tree
    "VPTreeResult",
    "VPNode",
    "VPTree",
    # Cover Tree
    "CoverTreeResult",
    "CoverTreeNode",
    "CoverTree",
]
