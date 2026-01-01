"""
Data association algorithms for multi-target tracking.

This module provides algorithms for associating measurements to tracks,
including Global Nearest Neighbor (GNN) and related methods.
"""

from typing import List, NamedTuple, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pytcl.assignment_algorithms.gating import mahalanobis_batch, mahalanobis_distance
from pytcl.assignment_algorithms.two_dimensional import assign2d


class AssociationResult(NamedTuple):
    """Result of data association.

    Attributes
    ----------
    track_to_measurement : ndarray
        Array of shape (n_tracks,). track_to_measurement[i] gives the
        measurement index assigned to track i, or -1 if unassigned.
    measurement_to_track : ndarray
        Array of shape (n_measurements,). measurement_to_track[j] gives the
        track index assigned to measurement j, or -1 if unassigned.
    costs : ndarray
        Array of association costs (squared Mahalanobis distances) for
        each assigned pair.
    total_cost : float
        Total assignment cost.
    """

    track_to_measurement: NDArray[np.intp]
    measurement_to_track: NDArray[np.intp]
    costs: NDArray[np.float64]
    total_cost: float


def compute_association_cost(
    track_predictions: ArrayLike,
    track_covariances: ArrayLike,
    measurements: ArrayLike,
    measurement_models: Optional[ArrayLike] = None,
) -> NDArray[np.float64]:
    """
    Compute cost matrix for track-to-measurement association.

    Parameters
    ----------
    track_predictions : array_like
        Predicted track states of shape (n_tracks, n_state).
    track_covariances : array_like
        Track covariance matrices of shape (n_tracks, n_state, n_state).
    measurements : array_like
        Measurements of shape (n_measurements, n_meas).
    measurement_models : array_like, optional
        Measurement matrices of shape (n_tracks, n_meas, n_state) or
        (n_meas, n_state) if same for all tracks. If None, assumes
        direct measurement of first n_meas states.

    Returns
    -------
    cost_matrix : ndarray
        Cost matrix of shape (n_tracks, n_measurements).
        cost_matrix[i, j] is the squared Mahalanobis distance from
        track i to measurement j.

    Examples
    --------
    >>> # 2 tracks, 3 measurements, 2D state [x, vx]
    >>> predictions = np.array([[0.0, 1.0], [5.0, -1.0]])
    >>> covariances = np.array([np.eye(2), np.eye(2)])
    >>> measurements = np.array([[0.1], [4.9], [10.0]])
    >>> H = np.array([[1.0, 0.0]])  # Measure position only
    >>> costs = compute_association_cost(predictions, covariances, measurements, H)
    """
    X = np.asarray(track_predictions, dtype=np.float64)
    P = np.asarray(track_covariances, dtype=np.float64)
    Z = np.asarray(measurements, dtype=np.float64)

    n_tracks = X.shape[0]
    n_meas = Z.shape[0]
    n_state = X.shape[1]
    meas_dim = Z.shape[1]

    # Handle measurement model
    if measurement_models is None:
        # Default: measure first meas_dim states
        H = np.zeros((meas_dim, n_state))
        H[:meas_dim, :meas_dim] = np.eye(meas_dim)
        H_per_track = np.tile(H, (n_tracks, 1, 1))
    else:
        H_arr = np.asarray(measurement_models, dtype=np.float64)
        if H_arr.ndim == 2:
            # Same H for all tracks
            H_per_track = np.tile(H_arr, (n_tracks, 1, 1))
        else:
            H_per_track = H_arr

    # Compute cost matrix using batch Mahalanobis distance for performance
    cost_matrix = np.full((n_tracks, n_meas), np.inf, dtype=np.float64)

    for i in range(n_tracks):
        H = H_per_track[i]
        z_pred = H @ X[i]
        S = H @ P[i] @ H.T  # Innovation covariance

        # Compute innovations for all measurements at once
        innovations = Z - z_pred  # (n_meas, meas_dim)

        # Use batch Mahalanobis distance (JIT-compiled)
        try:
            S_inv = np.linalg.inv(S)
            mahalanobis_batch(innovations, S_inv, cost_matrix[i])
        except np.linalg.LinAlgError:
            # Fallback if S is singular
            for j in range(n_meas):
                cost_matrix[i, j] = mahalanobis_distance(innovations[j], S)

    return cost_matrix


def nearest_neighbor(
    cost_matrix: ArrayLike,
    gate_threshold: float = np.inf,
) -> AssociationResult:
    """
    Simple nearest neighbor data association.

    Each track is assigned to its closest measurement, but a measurement
    can only be assigned to one track. Greedy assignment, not globally optimal.

    Parameters
    ----------
    cost_matrix : array_like
        Cost matrix of shape (n_tracks, n_measurements).
    gate_threshold : float, optional
        Maximum cost for valid assignment. Assignments with cost above
        this threshold are rejected.

    Returns
    -------
    result : AssociationResult
        Association result with track and measurement assignments.

    Notes
    -----
    This is a greedy algorithm that processes tracks in order of their
    minimum cost. It does not guarantee globally optimal assignment.
    Use gnn_association for globally optimal assignment.

    Examples
    --------
    >>> cost = np.array([[1.0, 5.0], [4.0, 2.0]])
    >>> result = nearest_neighbor(cost, gate_threshold=10.0)
    >>> result.track_to_measurement
    array([0, 1])
    """
    C = np.asarray(cost_matrix, dtype=np.float64)
    n_tracks, n_meas = C.shape

    track_to_meas = np.full(n_tracks, -1, dtype=np.intp)
    meas_to_track = np.full(n_meas, -1, dtype=np.intp)
    costs: List[float] = []

    # Mask for available measurements
    available = np.ones(n_meas, dtype=bool)

    # Process tracks in order of minimum cost
    min_costs = np.min(C, axis=1)
    track_order = np.argsort(min_costs)

    for i in track_order:
        # Find minimum cost to available measurements
        available_costs = np.where(available, C[i], np.inf)
        j = np.argmin(available_costs)
        cost = available_costs[j]

        if cost <= gate_threshold:
            track_to_meas[i] = j
            meas_to_track[j] = i
            available[j] = False
            costs.append(cost)

    return AssociationResult(
        track_to_measurement=track_to_meas,
        measurement_to_track=meas_to_track,
        costs=np.array(costs, dtype=np.float64),
        total_cost=float(np.sum(costs)),
    )


def gnn_association(
    cost_matrix: ArrayLike,
    gate_threshold: float = np.inf,
    cost_of_non_assignment: Optional[float] = None,
) -> AssociationResult:
    """
    Global Nearest Neighbor (GNN) data association.

    Finds the globally optimal assignment of tracks to measurements
    that minimizes the total association cost.

    Parameters
    ----------
    cost_matrix : array_like
        Cost matrix of shape (n_tracks, n_measurements).
        cost_matrix[i, j] is the cost of assigning track i to measurement j.
    gate_threshold : float, optional
        Maximum cost for valid assignment. Entries above this threshold
        are set to infinity before optimization.
    cost_of_non_assignment : float, optional
        Cost for not assigning a track or measurement. If None, all
        tracks/measurements that can be assigned will be assigned.

    Returns
    -------
    result : AssociationResult
        Association result with track and measurement assignments.

    Examples
    --------
    >>> cost = np.array([[1.0, 5.0, 2.0],
    ...                  [4.0, 2.0, 3.0]])
    >>> result = gnn_association(cost, gate_threshold=10.0)
    >>> result.track_to_measurement
    array([0, 1])
    >>> result.total_cost
    3.0

    Notes
    -----
    GNN uses the Hungarian algorithm to find the globally optimal assignment.
    It is more computationally expensive than nearest neighbor but guarantees
    optimality.

    The algorithm handles rectangular cost matrices (different numbers of
    tracks and measurements) and allows tracks/measurements to remain
    unassigned if cost_of_non_assignment is specified.

    References
    ----------
    .. [1] Blackman, S.S. and Popoli, R., "Design and Analysis of Modern
           Tracking Systems", Artech House, 1999.
    """
    C = np.asarray(cost_matrix, dtype=np.float64).copy()
    n_tracks, n_meas = C.shape

    # Apply gate threshold
    C[C > gate_threshold] = np.inf

    # Determine cost of non-assignment
    if cost_of_non_assignment is None:
        non_assign_cost = np.inf
    else:
        non_assign_cost = cost_of_non_assignment

    # Solve assignment problem
    assignment_result = assign2d(C, cost_of_non_assignment=non_assign_cost)

    # Build output
    track_to_meas = np.full(n_tracks, -1, dtype=np.intp)
    meas_to_track = np.full(n_meas, -1, dtype=np.intp)
    costs: List[float] = []

    for i, j in zip(assignment_result.row_indices, assignment_result.col_indices):
        track_to_meas[i] = j
        meas_to_track[j] = i
        costs.append(C[i, j])

    return AssociationResult(
        track_to_measurement=track_to_meas,
        measurement_to_track=meas_to_track,
        costs=np.array(costs, dtype=np.float64),
        total_cost=float(np.sum(costs)),
    )


def gated_gnn_association(
    track_predictions: ArrayLike,
    track_covariances: ArrayLike,
    measurements: ArrayLike,
    measurement_models: Optional[ArrayLike] = None,
    gate_probability: float = 0.99,
    cost_of_non_assignment: Optional[float] = None,
) -> AssociationResult:
    """
    GNN association with automatic gating.

    Combines gating and GNN association in a single function for convenience.

    Parameters
    ----------
    track_predictions : array_like
        Predicted track states of shape (n_tracks, n_state).
    track_covariances : array_like
        Track covariance matrices of shape (n_tracks, n_state, n_state).
    measurements : array_like
        Measurements of shape (n_measurements, n_meas).
    measurement_models : array_like, optional
        Measurement matrices. See compute_association_cost for details.
    gate_probability : float, optional
        Probability for chi-squared gate threshold (default: 0.99).
    cost_of_non_assignment : float, optional
        Cost for not assigning a track or measurement.

    Returns
    -------
    result : AssociationResult
        Association result.

    Examples
    --------
    >>> predictions = np.array([[0.0, 1.0], [5.0, -1.0]])
    >>> covariances = np.array([0.1 * np.eye(2), 0.1 * np.eye(2)])
    >>> measurements = np.array([[0.1], [4.9]])
    >>> H = np.array([[1.0, 0.0]])
    >>> result = gated_gnn_association(
    ...     predictions, covariances, measurements, H,
    ...     gate_probability=0.99
    ... )
    """
    from scipy.stats import chi2

    Z = np.asarray(measurements, dtype=np.float64)
    meas_dim = Z.shape[1]

    # Compute gate threshold
    gate_threshold = chi2.ppf(gate_probability, df=meas_dim)

    # Compute cost matrix
    cost_matrix = compute_association_cost(
        track_predictions,
        track_covariances,
        measurements,
        measurement_models,
    )

    # Run GNN with gating
    return gnn_association(
        cost_matrix,
        gate_threshold=gate_threshold,
        cost_of_non_assignment=cost_of_non_assignment,
    )


__all__ = [
    "AssociationResult",
    "compute_association_cost",
    "nearest_neighbor",
    "gnn_association",
    "gated_gnn_association",
]
