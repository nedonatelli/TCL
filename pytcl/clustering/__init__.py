"""
Clustering and mixture reduction algorithms.

This module provides Gaussian mixture operations and clustering algorithms
commonly used in multi-target tracking for hypothesis management and
track clustering.
"""

from pytcl.clustering.dbscan import DBSCANResult
from pytcl.clustering.dbscan import compute_neighbors
from pytcl.clustering.dbscan import dbscan
from pytcl.clustering.dbscan import dbscan_predict
from pytcl.clustering.gaussian_mixture import GaussianComponent
from pytcl.clustering.gaussian_mixture import GaussianMixture
from pytcl.clustering.gaussian_mixture import MergeResult
from pytcl.clustering.gaussian_mixture import ReductionResult
from pytcl.clustering.gaussian_mixture import merge_gaussians
from pytcl.clustering.gaussian_mixture import moment_match
from pytcl.clustering.gaussian_mixture import prune_mixture
from pytcl.clustering.gaussian_mixture import reduce_mixture_runnalls
from pytcl.clustering.gaussian_mixture import reduce_mixture_west
from pytcl.clustering.gaussian_mixture import runnalls_merge_cost
from pytcl.clustering.gaussian_mixture import west_merge_cost
from pytcl.clustering.hierarchical import DendrogramNode
from pytcl.clustering.hierarchical import HierarchicalResult
from pytcl.clustering.hierarchical import LinkageType
from pytcl.clustering.hierarchical import agglomerative_clustering
from pytcl.clustering.hierarchical import compute_distance_matrix
from pytcl.clustering.hierarchical import cut_dendrogram
from pytcl.clustering.hierarchical import fcluster
from pytcl.clustering.kmeans import KMeansResult
from pytcl.clustering.kmeans import assign_clusters
from pytcl.clustering.kmeans import kmeans
from pytcl.clustering.kmeans import kmeans_elbow
from pytcl.clustering.kmeans import kmeans_plusplus_init
from pytcl.clustering.kmeans import update_centers

__all__ = [
    # Gaussian mixture
    "GaussianComponent",
    "MergeResult",
    "ReductionResult",
    "GaussianMixture",
    "moment_match",
    "runnalls_merge_cost",
    "merge_gaussians",
    "prune_mixture",
    "reduce_mixture_runnalls",
    "west_merge_cost",
    "reduce_mixture_west",
    # K-means
    "KMeansResult",
    "kmeans_plusplus_init",
    "assign_clusters",
    "update_centers",
    "kmeans",
    "kmeans_elbow",
    # DBSCAN
    "DBSCANResult",
    "compute_neighbors",
    "dbscan",
    "dbscan_predict",
    # Hierarchical
    "LinkageType",
    "DendrogramNode",
    "HierarchicalResult",
    "compute_distance_matrix",
    "agglomerative_clustering",
    "cut_dendrogram",
    "fcluster",
]
