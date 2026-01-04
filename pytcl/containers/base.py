"""
Base classes for spatial data structures.

This module provides abstract base classes that define the common interface
for spatial indexing data structures like KD-trees, VP-trees, R-trees, and
Cover trees.

The unified interface ensures all spatial indices provide consistent:
- Constructor patterns (data, optional parameters)
- Query methods (query, query_radius, query_ball_point)
- Return types (NeighborResult)
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, List, NamedTuple, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

# Module logger
_logger = logging.getLogger("pytcl.containers")


class NeighborResult(NamedTuple):
    """
    Unified result type for spatial index queries.

    All spatial index implementations (KDTree, BallTree, VPTree, CoverTree,
    RTree) return this type from their query methods, ensuring a consistent
    interface across the library.

    Attributes
    ----------
    indices : ndarray of shape (n_queries, k) or (n_queries,)
        Indices of the k nearest neighbors in the original data array.
        For k=1, may be 1D. For k>1, shape is (n_queries, k).
    distances : ndarray of shape (n_queries, k) or (n_queries,)
        Distances to the k nearest neighbors.
        Same shape as indices.

    Examples
    --------
    >>> from pytcl.containers import KDTree
    >>> import numpy as np
    >>> points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    >>> tree = KDTree(points)
    >>> result = tree.query([[0.1, 0.1]], k=2)
    >>> result.indices
    array([[0, 2]])
    >>> result.distances
    array([[0.14142136, 0.9       ]])

    See Also
    --------
    BaseSpatialIndex : Abstract base class for spatial indices.
    """

    indices: NDArray[np.intp]
    distances: NDArray[np.floating]


# Backward compatibility aliases - all map to NeighborResult
SpatialQueryResult = NeighborResult
NearestNeighborResult = NeighborResult
VPTreeResult = NeighborResult
CoverTreeResult = NeighborResult
RTreeResult = NeighborResult


class BaseSpatialIndex(ABC):
    """
    Abstract base class for spatial indexing data structures.

    All spatial index implementations (KDTree, VPTree, RTree, CoverTree)
    should inherit from this class and implement the required methods.

    This provides a consistent interface for:
    - Building the index from point data
    - k-nearest neighbor queries
    - Range/radius queries
    - Dimension and size introspection

    Parameters
    ----------
    data : array_like
        Data points of shape (n_samples, n_features).

    Attributes
    ----------
    data : ndarray
        The indexed data points.
    n_samples : int
        Number of data points.
    n_features : int
        Dimensionality of data points.
    """

    def __init__(self, data: ArrayLike):
        self.data = np.asarray(data, dtype=np.float64)

        if self.data.ndim != 2:
            raise ValueError(
                f"Data must be 2-dimensional (n_samples, n_features), "
                f"got shape {self.data.shape}"
            )

        self.n_samples, self.n_features = self.data.shape
        _logger.debug(
            "%s initialized with %d points in %d dimensions",
            self.__class__.__name__,
            self.n_samples,
            self.n_features,
        )

    @abstractmethod
    def query(
        self,
        X: ArrayLike,
        k: int = 1,
    ) -> NeighborResult:
        """
        Query the index for k nearest neighbors.

        Parameters
        ----------
        X : array_like
            Query points of shape (n_queries, n_features) or (n_features,).
        k : int, optional
            Number of nearest neighbors to return. Default is 1.

        Returns
        -------
        result : NeighborResult
            Named tuple with indices and distances of k nearest neighbors
            for each query point.
        """
        pass

    @abstractmethod
    def query_radius(
        self,
        X: ArrayLike,
        r: float,
    ) -> List[List[int]]:
        """
        Query the index for all points within radius r.

        Parameters
        ----------
        X : array_like
            Query points of shape (n_queries, n_features) or (n_features,).
        r : float
            Search radius.

        Returns
        -------
        indices : list of list of int
            For each query point, a list of indices of data points
            within distance r.
        """
        pass

    def query_ball_point(
        self,
        X: ArrayLike,
        r: float,
    ) -> List[List[int]]:
        """
        Query the index for all points within radius r.

        This is an alias for :meth:`query_radius` provided for compatibility
        with scipy.spatial.KDTree.

        Parameters
        ----------
        X : array_like
            Query points of shape (n_queries, n_features) or (n_features,).
        r : float
            Search radius.

        Returns
        -------
        indices : list of list of int
            For each query point, a list of indices of data points
            within distance r.

        See Also
        --------
        query_radius : The underlying implementation.
        """
        return self.query_radius(X, r)

    def __len__(self) -> int:
        """Return number of indexed points."""
        return self.n_samples

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"n_samples={self.n_samples}, n_features={self.n_features})"
        )


class MetricSpatialIndex(BaseSpatialIndex):
    """
    Base class for metric space spatial indices.

    Extends BaseSpatialIndex with support for custom distance metrics.
    Used by VP-trees and Cover trees which can work with any metric.

    Parameters
    ----------
    data : array_like
        Data points of shape (n_samples, n_features).
    metric : callable, optional
        Distance function with signature metric(x, y) -> float.
        Default is Euclidean distance.
    """

    def __init__(
        self,
        data: ArrayLike,
        metric: Optional[Callable[[NDArray[Any], NDArray[Any]], float]] = None,
    ):
        super().__init__(data)

        if metric is None:
            self.metric = self._euclidean_distance
        else:
            self.metric = metric

    @staticmethod
    def _euclidean_distance(x: NDArray[Any], y: NDArray[Any]) -> float:
        """Default Euclidean distance metric."""
        return float(np.sqrt(np.sum((x - y) ** 2)))


def validate_query_input(
    X: ArrayLike,
    n_features: int,
) -> NDArray[np.floating]:
    """
    Validate and reshape query input.

    Parameters
    ----------
    X : array_like
        Query points.
    n_features : int
        Expected number of features.

    Returns
    -------
    X : ndarray
        Validated query array of shape (n_queries, n_features).

    Raises
    ------
    ValueError
        If query has wrong number of features.
    """
    X = np.asarray(X, dtype=np.float64)

    if X.ndim == 1:
        X = X.reshape(1, -1)

    if X.shape[1] != n_features:
        _logger.warning(
            "Query feature mismatch: got %d, expected %d", X.shape[1], n_features
        )
        raise ValueError(f"Query has {X.shape[1]} features, expected {n_features}")

    _logger.debug(
        "Validated query input: %d queries, %d features", X.shape[0], X.shape[1]
    )
    return X


__all__ = [
    # Primary types
    "NeighborResult",
    "BaseSpatialIndex",
    "MetricSpatialIndex",
    "validate_query_input",
    # Backward compatibility aliases
    "SpatialQueryResult",
    "NearestNeighborResult",
    "VPTreeResult",
    "CoverTreeResult",
    "RTreeResult",
]
