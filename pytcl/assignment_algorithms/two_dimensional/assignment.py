"""
Two-dimensional assignment algorithms.

This module provides algorithms for solving the 2D assignment (bipartite matching)
problem, which is fundamental to data association in target tracking.
"""

from typing import NamedTuple, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import linear_sum_assignment as scipy_lsa


class AssignmentResult(NamedTuple):
    """Result of a 2D assignment problem.

    Attributes
    ----------
    row_indices : ndarray
        Indices of assigned rows.
    col_indices : ndarray
        Indices of assigned columns (col_indices[i] is assigned to row_indices[i]).
    cost : float
        Total assignment cost.
    unassigned_rows : ndarray
        Indices of unassigned rows.
    unassigned_cols : ndarray
        Indices of unassigned columns.
    """

    row_indices: NDArray[np.intp]
    col_indices: NDArray[np.intp]
    cost: float
    unassigned_rows: NDArray[np.intp]
    unassigned_cols: NDArray[np.intp]


def linear_sum_assignment(
    cost_matrix: ArrayLike,
    maximize: bool = False,
) -> Tuple[NDArray[np.intp], NDArray[np.intp]]:
    """
    Solve the linear sum assignment problem (wrapper around scipy).

    Parameters
    ----------
    cost_matrix : array_like
        Cost matrix of shape (n, m).
    maximize : bool, optional
        If True, solve maximization problem instead of minimization.

    Returns
    -------
    row_ind : ndarray
        Row indices of optimal assignment.
    col_ind : ndarray
        Column indices of optimal assignment.

    Examples
    --------
    >>> cost = np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]])
    >>> row_ind, col_ind = linear_sum_assignment(cost)
    >>> row_ind
    array([0, 1, 2])
    >>> col_ind
    array([1, 0, 2])
    >>> cost[row_ind, col_ind].sum()
    5

    Notes
    -----
    This is a thin wrapper around scipy.optimize.linear_sum_assignment
    for API consistency.
    """
    cost = np.asarray(cost_matrix, dtype=np.float64)
    return scipy_lsa(cost, maximize=maximize)


def hungarian(
    cost_matrix: ArrayLike,
    maximize: bool = False,
) -> Tuple[NDArray[np.intp], NDArray[np.intp], float]:
    """
    Solve 2D assignment using Hungarian (Kuhn-Munkres) algorithm.

    The Hungarian algorithm finds the optimal assignment that minimizes
    (or maximizes) the total cost.

    Parameters
    ----------
    cost_matrix : array_like
        Cost matrix of shape (n, m). Can be rectangular.
    maximize : bool, optional
        If True, solve maximization problem (default: False).

    Returns
    -------
    row_ind : ndarray
        Row indices of optimal assignment.
    col_ind : ndarray
        Column indices of optimal assignment.
    total_cost : float
        Total cost of the assignment.

    Examples
    --------
    >>> cost = np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]])
    >>> row_ind, col_ind, total_cost = hungarian(cost)
    >>> total_cost
    5.0

    Notes
    -----
    Time complexity is O(n^3) for an n x n matrix.

    References
    ----------
    .. [1] Kuhn, H.W., "The Hungarian Method for the assignment problem",
           Naval Research Logistics Quarterly, 1955.
    """
    cost = np.asarray(cost_matrix, dtype=np.float64)
    row_ind, col_ind = scipy_lsa(cost, maximize=maximize)
    total_cost = cost[row_ind, col_ind].sum()
    return row_ind, col_ind, total_cost


def auction(
    cost_matrix: ArrayLike,
    epsilon: Optional[float] = None,
    max_iter: int = 1000,
    maximize: bool = False,
) -> Tuple[NDArray[np.intp], NDArray[np.intp], float]:
    """
    Solve 2D assignment using the Auction algorithm.

    The auction algorithm is an iterative method that can be faster than
    Hungarian for sparse problems and is naturally parallelizable.

    Parameters
    ----------
    cost_matrix : array_like
        Cost matrix of shape (n, m).
    epsilon : float, optional
        Price increment. If None, uses 1/(n+1) where n is matrix size.
    max_iter : int, optional
        Maximum iterations (default: 1000).
    maximize : bool, optional
        If True, solve maximization problem (default: False).

    Returns
    -------
    row_ind : ndarray
        Row indices of optimal assignment.
    col_ind : ndarray
        Column indices of optimal assignment.
    total_cost : float
        Total cost of the assignment.

    Examples
    --------
    >>> cost = np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]])
    >>> row_ind, col_ind, total_cost = auction(cost)

    Notes
    -----
    The auction algorithm treats rows as "bidders" and columns as "objects".
    Each iteration, unassigned bidders bid for their most desirable objects,
    and objects are assigned to the highest bidder.

    References
    ----------
    .. [1] Bertsekas, D.P., "The auction algorithm: A distributed relaxation
           method for the assignment problem", Annals of Operations Research, 1988.
    """
    cost = np.asarray(cost_matrix, dtype=np.float64)

    if maximize:
        cost = -cost

    n, m = cost.shape

    if n > m:
        # Transpose to ensure n <= m
        cost = cost.T
        n, m = m, n
        transposed = True
    else:
        transposed = False

    if epsilon is None:
        epsilon = 1.0 / (n + 1)

    # Prices for each column (object)
    prices = np.zeros(m, dtype=np.float64)

    # Assignment: assignment[i] = j means row i is assigned to column j
    # -1 means unassigned
    assignment = np.full(n, -1, dtype=np.intp)

    # Reverse assignment: which row is assigned to each column
    reverse_assignment = np.full(m, -1, dtype=np.intp)

    for _ in range(max_iter):
        # Find unassigned rows
        unassigned = np.where(assignment == -1)[0]
        if len(unassigned) == 0:
            break

        for i in unassigned:
            # Compute values: value[j] = -cost[i,j] - prices[j]
            values = -cost[i, :] - prices

            # Find best and second best using argpartition (O(n) vs O(n log n))
            if len(values) >= 2:
                # Get indices of top 2 values
                top2_idx = np.argpartition(values, -2)[-2:]
                # Determine which is best and second best
                if values[top2_idx[0]] > values[top2_idx[1]]:
                    best_j = top2_idx[0]
                    second_value = values[top2_idx[1]]
                else:
                    best_j = top2_idx[1]
                    second_value = values[top2_idx[0]]
                best_value = values[best_j]
            else:
                best_j = np.argmax(values)
                best_value = values[best_j]
                second_value = -np.inf

            # Bid increment
            bid = best_value - second_value + epsilon

            # Update price
            prices[best_j] += bid

            # If column was assigned, unassign it
            old_owner = reverse_assignment[best_j]
            if old_owner >= 0:
                assignment[old_owner] = -1

            # Assign row i to column best_j
            assignment[i] = best_j
            reverse_assignment[best_j] = i

    # Build result
    row_ind = np.where(assignment >= 0)[0]
    col_ind = assignment[row_ind]

    if transposed:
        row_ind, col_ind = col_ind, row_ind
        cost = cost.T

    total_cost = cost[row_ind, col_ind].sum()

    if maximize:
        total_cost = -total_cost

    return row_ind, col_ind, total_cost


def assign2d(
    cost_matrix: ArrayLike,
    cost_of_non_assignment: float = np.inf,
    maximize: bool = False,
) -> AssignmentResult:
    """
    Solve 2D assignment with cost of non-assignment.

    This extends the basic assignment problem to allow tracks and
    measurements to remain unassigned at a specified cost.

    Parameters
    ----------
    cost_matrix : array_like
        Cost matrix of shape (n_tracks, n_measurements).
    cost_of_non_assignment : float, optional
        Cost for leaving a track or measurement unassigned.
        If inf, all must be assigned (default: inf).
    maximize : bool, optional
        If True, solve maximization problem (default: False).

    Returns
    -------
    result : AssignmentResult
        Named tuple containing:
        - row_indices: Indices of assigned rows (tracks)
        - col_indices: Indices of assigned columns (measurements)
        - cost: Total assignment cost
        - unassigned_rows: Indices of unassigned rows
        - unassigned_cols: Indices of unassigned columns

    Examples
    --------
    >>> cost = np.array([[1, 10], [10, 1], [5, 5]])
    >>> result = assign2d(cost, cost_of_non_assignment=3)
    >>> result.row_indices
    array([0, 1])
    >>> result.col_indices
    array([0, 1])
    >>> result.unassigned_rows
    array([2])

    Notes
    -----
    The algorithm augments the cost matrix with dummy rows and columns
    representing non-assignment options, then solves the augmented problem.
    """
    cost = np.asarray(cost_matrix, dtype=np.float64)
    n, m = cost.shape

    if np.isinf(cost_of_non_assignment):
        # Standard assignment - must assign everything possible
        row_ind, col_ind = scipy_lsa(cost, maximize=maximize)
        total_cost = cost[row_ind, col_ind].sum()

        all_rows = set(range(n))
        all_cols = set(range(m))
        assigned_rows = set(row_ind)
        assigned_cols = set(col_ind)

        return AssignmentResult(
            row_indices=row_ind,
            col_indices=col_ind,
            cost=total_cost,
            unassigned_rows=np.array(sorted(all_rows - assigned_rows), dtype=np.intp),
            unassigned_cols=np.array(sorted(all_cols - assigned_cols), dtype=np.intp),
        )

    # Augment cost matrix with non-assignment options
    # New matrix is (n+m) x (n+m)
    # Top-left: original cost matrix
    # Top-right: diagonal with cost_of_non_assignment (row i not assigned)
    # Bottom-left: diagonal with cost_of_non_assignment (col j not assigned)
    # Bottom-right: zeros

    aug_size = n + m
    if maximize:
        # For maximization, non-assignment has negative cost
        non_assign_cost = -cost_of_non_assignment
        fill_value = -np.inf
    else:
        non_assign_cost = cost_of_non_assignment
        fill_value = np.inf

    aug_cost = np.full((aug_size, aug_size), fill_value, dtype=np.float64)

    # Original costs
    aug_cost[:n, :m] = cost

    # Row non-assignment (top-right diagonal)
    for i in range(n):
        aug_cost[i, m + i] = non_assign_cost

    # Column non-assignment (bottom-left diagonal)
    for j in range(m):
        aug_cost[n + j, j] = non_assign_cost

    # Bottom-right: zeros (dummy-to-dummy)
    aug_cost[n:, m:] = 0

    # Solve augmented problem
    row_ind, col_ind = scipy_lsa(aug_cost, maximize=maximize)

    # Extract real assignments
    real_assignments_mask = (row_ind < n) & (col_ind < m)
    assigned_row_ind = row_ind[real_assignments_mask]
    assigned_col_ind = col_ind[real_assignments_mask]

    # Compute cost
    total_cost = cost[assigned_row_ind, assigned_col_ind].sum()

    # Add non-assignment costs
    n_unassigned_rows = n - len(assigned_row_ind)
    n_unassigned_cols = m - len(assigned_col_ind)
    total_cost += (n_unassigned_rows + n_unassigned_cols) * cost_of_non_assignment

    # Find unassigned
    all_rows = set(range(n))
    all_cols = set(range(m))
    unassigned_rows = np.array(sorted(all_rows - set(assigned_row_ind)), dtype=np.intp)
    unassigned_cols = np.array(sorted(all_cols - set(assigned_col_ind)), dtype=np.intp)

    return AssignmentResult(
        row_indices=assigned_row_ind,
        col_indices=assigned_col_ind,
        cost=total_cost,
        unassigned_rows=unassigned_rows,
        unassigned_cols=unassigned_cols,
    )


__all__ = [
    "AssignmentResult",
    "linear_sum_assignment",
    "hungarian",
    "auction",
    "assign2d",
]
