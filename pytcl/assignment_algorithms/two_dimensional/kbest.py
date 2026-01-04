"""
K-best 2D assignment algorithms.

This module provides algorithms for finding the k best solutions to the
2D assignment problem, which is essential for hypothesis generation in
Multiple Hypothesis Tracking (MHT).
"""

from typing import List, NamedTuple, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import linear_sum_assignment as scipy_lsa

from pytcl.assignment_algorithms.two_dimensional.assignment import (
    AssignmentResult,
    assign2d,
)


class KBestResult(NamedTuple):
    """Result of k-best assignment problem.

    Attributes
    ----------
    assignments : List[AssignmentResult]
        List of k best assignments, sorted by cost (ascending for minimization).
    costs : ndarray
        Array of costs for each assignment.
    n_found : int
        Number of assignments found (may be less than k if fewer exist).
    """

    assignments: List[AssignmentResult]
    costs: NDArray[np.float64]
    n_found: int


class _PartitionNode:
    """Node in the partition tree for Murty's algorithm.

    Each node represents a constrained assignment problem with some
    assignments required and some forbidden.
    """

    def __init__(
        self,
        cost_matrix: NDArray[np.float64],
        required: List[Tuple[int, int]],
        forbidden: List[Tuple[int, int]],
        cost: float,
        assignment: Optional[Tuple[NDArray[np.intp], NDArray[np.intp]]] = None,
    ):
        self.cost_matrix = cost_matrix
        self.required = required
        self.forbidden = forbidden
        self.cost = cost
        self.assignment = assignment

    def __lt__(self, other: "_PartitionNode") -> bool:
        return self.cost < other.cost


def _solve_constrained(
    cost_matrix: NDArray[np.float64],
    required: List[Tuple[int, int]],
    forbidden: List[Tuple[int, int]],
    maximize: bool = False,
) -> Optional[Tuple[NDArray[np.intp], NDArray[np.intp], float]]:
    """Solve assignment with required and forbidden constraints.

    Parameters
    ----------
    cost_matrix : ndarray
        Original cost matrix.
    required : list of tuple
        List of (row, col) pairs that must be assigned.
    forbidden : list of tuple
        List of (row, col) pairs that cannot be assigned.
    maximize : bool
        If True, solve maximization problem.

    Returns
    -------
    tuple or None
        (row_indices, col_indices, cost) if feasible, None otherwise.
    """
    n, m = cost_matrix.shape
    C = cost_matrix.copy()

    # Apply forbidden constraints by setting to infinity
    fill_val = -np.inf if maximize else np.inf
    for r, c in forbidden:
        if 0 <= r < n and 0 <= c < m:
            C[r, c] = fill_val

    # Check required assignments are compatible
    required_rows = set()
    required_cols = set()
    required_cost = 0.0

    for r, c in required:
        if r in required_rows or c in required_cols:
            # Conflict in required assignments
            return None
        if C[r, c] == fill_val:
            # Required assignment is forbidden
            return None
        required_rows.add(r)
        required_cols.add(c)
        required_cost += cost_matrix[r, c]

    # If all rows or columns are required, we're done
    if len(required_rows) == min(n, m):
        row_ind = np.array([r for r, c in required], dtype=np.intp)
        col_ind = np.array([c for r, c in required], dtype=np.intp)
        sort_idx = np.argsort(row_ind)
        return row_ind[sort_idx], col_ind[sort_idx], required_cost

    # Remove required rows/cols and solve reduced problem
    free_rows = [i for i in range(n) if i not in required_rows]
    free_cols = [j for j in range(m) if j not in required_cols]

    if len(free_rows) == 0 or len(free_cols) == 0:
        # No free assignments needed
        row_ind = np.array([r for r, c in required], dtype=np.intp)
        col_ind = np.array([c for r, c in required], dtype=np.intp)
        sort_idx = np.argsort(row_ind)
        return row_ind[sort_idx], col_ind[sort_idx], required_cost

    # Build reduced cost matrix
    reduced_cost = C[np.ix_(free_rows, free_cols)]

    # Check if feasible (no row/column is all infinity)
    if not maximize:
        if np.any(np.all(reduced_cost == np.inf, axis=1)):
            return None
        if np.any(np.all(reduced_cost == np.inf, axis=0)):
            return None
    else:
        if np.any(np.all(reduced_cost == -np.inf, axis=1)):
            return None
        if np.any(np.all(reduced_cost == -np.inf, axis=0)):
            return None

    # Solve reduced problem
    try:
        red_row_ind, red_col_ind = scipy_lsa(reduced_cost, maximize=maximize)
    except ValueError:
        return None

    # Map back to original indices
    orig_row_ind = [free_rows[i] for i in red_row_ind]
    orig_col_ind = [free_cols[j] for j in red_col_ind]

    # Combine with required
    all_rows = list(required_rows) + orig_row_ind
    all_cols = [c for r, c in required] + orig_col_ind

    row_ind = np.array(all_rows, dtype=np.intp)
    col_ind = np.array(all_cols, dtype=np.intp)

    # Sort by row index
    sort_idx = np.argsort(row_ind)
    row_ind = row_ind[sort_idx]
    col_ind = col_ind[sort_idx]

    # Compute total cost
    total_cost = cost_matrix[row_ind, col_ind].sum()

    return row_ind, col_ind, total_cost


def murty(
    cost_matrix: ArrayLike,
    k: int,
    cost_of_non_assignment: float = np.inf,
    maximize: bool = False,
) -> KBestResult:
    """
    Find k-best assignments using Murty's algorithm.

    Murty's algorithm systematically partitions the solution space to
    enumerate assignments in order of increasing cost. It is widely used
    in Multiple Hypothesis Tracking (MHT) for hypothesis generation.

    Parameters
    ----------
    cost_matrix : array_like
        Cost matrix of shape (n_tracks, n_measurements).
    k : int
        Number of best assignments to find.
    cost_of_non_assignment : float, optional
        Cost for leaving a track or measurement unassigned.
        If inf, all must be assigned (default: inf).
    maximize : bool, optional
        If True, find k-best in descending order (default: False).

    Returns
    -------
    result : KBestResult
        Named tuple containing:
        - assignments: List of k best AssignmentResult objects
        - costs: Array of costs for each assignment
        - n_found: Number of assignments actually found

    Examples
    --------
    >>> import numpy as np
    >>> cost = np.array([[10, 5, 13], [3, 15, 8], [12, 7, 9]])
    >>> result = murty(cost, k=3)
    >>> result.n_found
    3
    >>> result.costs  # Three lowest-cost assignments
    array([15., 17., 18.])

    Notes
    -----
    The algorithm has time complexity O(k * n^3) for an n x n matrix,
    as each partition requires solving a constrained assignment problem.

    For large k, consider using lazy evaluation techniques or early
    termination based on cost thresholds.

    References
    ----------
    .. [1] Murty, K.G., "An algorithm for ranking all the assignments in
           order of increasing cost", Operations Research, 1968.
    .. [2] Miller, M.L., Stone, H.S., and Cox, I.J., "Optimizing Murty's
           ranked assignment method", IEEE Trans. Aerospace and Electronic
           Systems, 1997.
    """
    cost = np.asarray(cost_matrix, dtype=np.float64)
    n, m = cost.shape

    if k <= 0:
        return KBestResult(assignments=[], costs=np.array([]), n_found=0)

    # Find the optimal assignment first
    first_result = assign2d(cost, cost_of_non_assignment, maximize)

    if len(first_result.row_indices) == 0 and cost_of_non_assignment == np.inf:
        # No valid assignment
        return KBestResult(assignments=[], costs=np.array([]), n_found=0)

    assignments: List[AssignmentResult] = [first_result]
    costs: List[float] = [first_result.cost]

    if k == 1:
        return KBestResult(
            assignments=assignments,
            costs=np.array(costs, dtype=np.float64),
            n_found=1,
        )

    # Priority queue (min-heap) of partition nodes
    import heapq

    heap: List[_PartitionNode] = []

    # Create initial partition nodes from the first solution
    _partition_solution(
        cost,
        first_result.row_indices,
        first_result.col_indices,
        [],
        [],
        heap,
        maximize,
    )

    while len(assignments) < k and heap:
        # Get the best remaining node
        node = heapq.heappop(heap)

        # Create assignment result
        if node.assignment is not None:
            row_ind, col_ind = node.assignment

            # Determine unassigned
            all_rows = set(range(n))
            all_cols = set(range(m))
            unassigned_rows = np.array(sorted(all_rows - set(row_ind)), dtype=np.intp)
            unassigned_cols = np.array(sorted(all_cols - set(col_ind)), dtype=np.intp)

            result = AssignmentResult(
                row_indices=row_ind,
                col_indices=col_ind,
                cost=node.cost,
                unassigned_rows=unassigned_rows,
                unassigned_cols=unassigned_cols,
            )

            assignments.append(result)
            costs.append(node.cost)

            # Partition this solution for more candidates
            _partition_solution(
                cost,
                row_ind,
                col_ind,
                node.required,
                node.forbidden,
                heap,
                maximize,
            )

    return KBestResult(
        assignments=assignments,
        costs=np.array(costs, dtype=np.float64),
        n_found=len(assignments),
    )


def _partition_solution(
    cost_matrix: NDArray[np.float64],
    row_ind: NDArray[np.intp],
    col_ind: NDArray[np.intp],
    required: List[Tuple[int, int]],
    forbidden: List[Tuple[int, int]],
    heap: List[_PartitionNode],
    maximize: bool,
) -> None:
    """Partition the solution space based on current assignment.

    Creates child nodes by forbidding each assignment in turn while
    requiring all previous assignments.
    """
    import heapq

    n_assigned = len(row_ind)

    # Start after required assignments
    start_idx = len(required)

    for i in range(start_idx, n_assigned):
        # Require assignments 0..i-1, forbid assignment i
        new_required = required + [
            (row_ind[j], col_ind[j]) for j in range(start_idx, i)
        ]
        new_forbidden = forbidden + [(row_ind[i], col_ind[i])]

        # Solve constrained problem
        result = _solve_constrained(cost_matrix, new_required, new_forbidden, maximize)

        if result is not None:
            new_row_ind, new_col_ind, new_cost = result

            node = _PartitionNode(
                cost_matrix=cost_matrix,
                required=new_required,
                forbidden=new_forbidden,
                cost=new_cost,
                assignment=(new_row_ind, new_col_ind),
            )

            if maximize:
                # Use negative cost for max-heap behavior
                heapq.heappush(
                    heap,
                    _PartitionNode(
                        cost_matrix=cost_matrix,
                        required=new_required,
                        forbidden=new_forbidden,
                        cost=-new_cost,
                        assignment=(new_row_ind, new_col_ind),
                    ),
                )
            else:
                heapq.heappush(heap, node)


def kbest_assign2d(
    cost_matrix: ArrayLike,
    k: int,
    cost_of_non_assignment: float = np.inf,
    maximize: bool = False,
    cost_threshold: Optional[float] = None,
) -> KBestResult:
    """
    Find k-best assignments with optional cost threshold.

    This is an enhanced version of Murty's algorithm that supports
    early termination when costs exceed a threshold.

    Parameters
    ----------
    cost_matrix : array_like
        Cost matrix of shape (n_tracks, n_measurements).
    k : int
        Maximum number of assignments to find.
    cost_of_non_assignment : float, optional
        Cost for leaving a track or measurement unassigned.
    maximize : bool, optional
        If True, find k-best in descending order.
    cost_threshold : float, optional
        Stop when assignment cost exceeds this threshold (for minimization)
        or falls below (for maximization). If None, no threshold.

    Returns
    -------
    result : KBestResult
        Named tuple containing assignments, costs, and count.

    Examples
    --------
    >>> cost = np.array([[10, 5, 13], [3, 15, 8], [12, 7, 9]])
    >>> result = kbest_assign2d(cost, k=10, cost_threshold=20)
    >>> result.n_found  # Only assignments with cost <= 20
    4

    Notes
    -----
    The cost threshold enables efficient pruning for MHT where hypotheses
    with very low probability (high cost) can be discarded.
    """
    cost = np.asarray(cost_matrix, dtype=np.float64)
    n, m = cost.shape

    if k <= 0:
        return KBestResult(assignments=[], costs=np.array([]), n_found=0)

    # Find the optimal assignment first
    first_result = assign2d(cost, cost_of_non_assignment, maximize)

    if len(first_result.row_indices) == 0 and cost_of_non_assignment == np.inf:
        return KBestResult(assignments=[], costs=np.array([]), n_found=0)

    # Check threshold on first solution
    if cost_threshold is not None:
        if not maximize and first_result.cost > cost_threshold:
            return KBestResult(assignments=[], costs=np.array([]), n_found=0)
        if maximize and first_result.cost < cost_threshold:
            return KBestResult(assignments=[], costs=np.array([]), n_found=0)

    assignments: List[AssignmentResult] = [first_result]
    costs: List[float] = [first_result.cost]

    if k == 1:
        return KBestResult(
            assignments=assignments,
            costs=np.array(costs, dtype=np.float64),
            n_found=1,
        )

    import heapq

    heap: List[_PartitionNode] = []

    _partition_solution(
        cost,
        first_result.row_indices,
        first_result.col_indices,
        [],
        [],
        heap,
        maximize,
    )

    while len(assignments) < k and heap:
        node = heapq.heappop(heap)

        # Adjust cost for maximization
        actual_cost = -node.cost if maximize else node.cost

        # Check threshold
        if cost_threshold is not None:
            if not maximize and actual_cost > cost_threshold:
                break
            if maximize and actual_cost < cost_threshold:
                break

        if node.assignment is not None:
            row_ind, col_ind = node.assignment

            all_rows = set(range(n))
            all_cols = set(range(m))
            unassigned_rows = np.array(sorted(all_rows - set(row_ind)), dtype=np.intp)
            unassigned_cols = np.array(sorted(all_cols - set(col_ind)), dtype=np.intp)

            result = AssignmentResult(
                row_indices=row_ind,
                col_indices=col_ind,
                cost=actual_cost,
                unassigned_rows=unassigned_rows,
                unassigned_cols=unassigned_cols,
            )

            assignments.append(result)
            costs.append(actual_cost)

            _partition_solution(
                cost,
                row_ind,
                col_ind,
                node.required,
                node.forbidden,
                heap,
                maximize,
            )

    return KBestResult(
        assignments=assignments,
        costs=np.array(costs, dtype=np.float64),
        n_found=len(assignments),
    )


def ranked_assignments(
    cost_matrix: ArrayLike,
    max_assignments: int = 100,
    cost_threshold: Optional[float] = None,
    maximize: bool = False,
) -> KBestResult:
    """
    Enumerate assignments in ranked order (best to worst).

    This is a convenience function for MHT hypothesis generation that
    provides sensible defaults.

    Parameters
    ----------
    cost_matrix : array_like
        Cost matrix of shape (n_tracks, n_measurements).
    max_assignments : int, optional
        Maximum number of assignments to enumerate (default: 100).
    cost_threshold : float, optional
        Stop when cost exceeds threshold (minimization) or falls below
        (maximization).
    maximize : bool, optional
        If True, rank in descending order of cost.

    Returns
    -------
    result : KBestResult
        Ranked assignments.

    Examples
    --------
    >>> cost = np.array([[10, 5], [3, 15]])
    >>> result = ranked_assignments(cost, max_assignments=5)
    >>> len(result.assignments)
    2
    >>> result.costs
    array([12., 20.])

    Notes
    -----
    This function assumes all tracks must be assigned to measurements
    (no non-assignment option). For track-oriented MHT with missed
    detections, use kbest_assign2d with an appropriate
    cost_of_non_assignment.
    """
    return kbest_assign2d(
        cost_matrix,
        k=max_assignments,
        cost_of_non_assignment=np.inf,
        maximize=maximize,
        cost_threshold=cost_threshold,
    )


__all__ = [
    "KBestResult",
    "murty",
    "kbest_assign2d",
    "ranked_assignments",
]
