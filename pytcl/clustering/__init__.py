"""
Clustering and mixture reduction algorithms.

This module provides Gaussian mixture operations and clustering algorithms
commonly used in multi-target tracking for hypothesis management and
track clustering.
"""

from pytcl.clustering.gaussian_mixture import (
    GaussianComponent,
    MergeResult,
    ReductionResult,
    GaussianMixture,
    moment_match,
    runnalls_merge_cost,
    merge_gaussians,
    prune_mixture,
    reduce_mixture_runnalls,
    west_merge_cost,
    reduce_mixture_west,
)

from pytcl.clustering.kmeans import (
    KMeansResult,
    kmeans_plusplus_init,
    assign_clusters,
    update_centers,
    kmeans,
    kmeans_elbow,
)

from pytcl.clustering.dbscan import (
    DBSCANResult,
    compute_neighbors,
    dbscan,
    dbscan_predict,
)

from pytcl.clustering.hierarchical import (
    LinkageType,
    DendrogramNode,
    HierarchicalResult,
    compute_distance_matrix,
    agglomerative_clustering,
    cut_dendrogram,
    fcluster,
)

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
