"""
Containers module.

This module provides spatial data structures for efficient
nearest neighbor queries and spatial indexing.
"""

from pytcl.containers.kd_tree import (
    KDNode,
    NearestNeighborResult,
    KDTree,
    BallTree,
)

from pytcl.containers.rtree import (
    BoundingBox,
    merge_boxes,
    box_from_point,
    box_from_points,
    RTreeNode,
    RTreeResult,
    RTree,
)

from pytcl.containers.vptree import (
    VPTreeResult,
    VPNode,
    VPTree,
)

from pytcl.containers.covertree import (
    CoverTreeResult,
    CoverTreeNode,
    CoverTree,
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
