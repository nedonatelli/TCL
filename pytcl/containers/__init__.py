"""
Containers module.

This module provides spatial data structures for efficient
nearest neighbor queries, spatial indexing, and tracking containers.
"""

from pytcl.containers.cluster_set import ClusterSet
from pytcl.containers.cluster_set import ClusterStats
from pytcl.containers.cluster_set import TrackCluster
from pytcl.containers.cluster_set import cluster_tracks_dbscan
from pytcl.containers.cluster_set import cluster_tracks_kmeans
from pytcl.containers.cluster_set import compute_cluster_centroid
from pytcl.containers.covertree import CoverTree
from pytcl.containers.covertree import CoverTreeNode
from pytcl.containers.covertree import CoverTreeResult
from pytcl.containers.kd_tree import BallTree
from pytcl.containers.kd_tree import KDNode
from pytcl.containers.kd_tree import KDTree
from pytcl.containers.kd_tree import NearestNeighborResult
from pytcl.containers.measurement_set import Measurement
from pytcl.containers.measurement_set import MeasurementQuery
from pytcl.containers.measurement_set import MeasurementSet
from pytcl.containers.rtree import BoundingBox
from pytcl.containers.rtree import RTree
from pytcl.containers.rtree import RTreeNode
from pytcl.containers.rtree import RTreeResult
from pytcl.containers.rtree import box_from_point
from pytcl.containers.rtree import box_from_points
from pytcl.containers.rtree import merge_boxes
from pytcl.containers.track_list import TrackList
from pytcl.containers.track_list import TrackListStats
from pytcl.containers.track_list import TrackQuery
from pytcl.containers.vptree import VPNode
from pytcl.containers.vptree import VPTree
from pytcl.containers.vptree import VPTreeResult

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
