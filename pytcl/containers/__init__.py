"""
Containers module.

This module provides spatial data structures for efficient
nearest neighbor queries, spatial indexing, and tracking containers.

Spatial Index Hierarchy
-----------------------
All spatial index structures inherit from BaseSpatialIndex which defines
a common interface for k-nearest neighbor and radius queries:

    BaseSpatialIndex (abstract)
    ├── KDTree - K-dimensional tree (Euclidean space)
    ├── BallTree - Ball tree variant of KD-tree
    ├── RTree - Rectangle tree for bounding boxes
    └── MetricSpatialIndex (abstract)
        ├── VPTree - Vantage point tree (any metric)
        └── CoverTree - Cover tree (any metric)
"""

from pytcl.containers.base import (
    BaseSpatialIndex,
    MetricSpatialIndex,
    SpatialQueryResult,
    validate_query_input,
)
from pytcl.containers.cluster_set import (
    ClusterSet,
    ClusterStats,
    TrackCluster,
    cluster_tracks_dbscan,
    cluster_tracks_kmeans,
    compute_cluster_centroid,
)
from pytcl.containers.covertree import CoverTree, CoverTreeNode, CoverTreeResult
from pytcl.containers.kd_tree import BallTree, KDNode, KDTree, NearestNeighborResult
from pytcl.containers.measurement_set import (
    Measurement,
    MeasurementQuery,
    MeasurementSet,
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
from pytcl.containers.track_list import TrackList, TrackListStats, TrackQuery
from pytcl.containers.vptree import VPNode, VPTree, VPTreeResult

__all__ = [
    # Base classes
    "BaseSpatialIndex",
    "MetricSpatialIndex",
    "SpatialQueryResult",
    "validate_query_input",
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
    # Track List
    "TrackList",
    "TrackQuery",
    "TrackListStats",
    # Measurement Set
    "Measurement",
    "MeasurementSet",
    "MeasurementQuery",
    # Cluster Set
    "TrackCluster",
    "ClusterSet",
    "ClusterStats",
    "cluster_tracks_dbscan",
    "cluster_tracks_kmeans",
    "compute_cluster_centroid",
]
