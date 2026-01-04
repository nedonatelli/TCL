"""
Three-dimensional assignment algorithms.

This module provides algorithms for solving the 3D assignment (axial 3-index
assignment) problem, which arises in multi-sensor data fusion and multi-scan
target tracking.

The 3D assignment problem extends the 2D problem to three dimensions:
given a cost tensor C[i,j,k], find an assignment that minimizes the total
cost subject to the constraint that each index appears in at most one
selected tuple.
"""

from typing import Any, List, NamedTuple, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import linear_sum_assignment as scipy_lsa


class Assignment3DResult(NamedTuple):
    """Result of a 3D assignment problem.

    Attributes
    ----------
    tuples : ndarray
        Array of shape (n_assignments, 3) containing assigned index tuples.
        Each row is (i, j, k) representing an assignment.
    cost : float
        Total assignment cost.
    converged : bool
        Whether the algorithm converged (for iterative methods).
    n_iterations : int
        Number of iterations used (for iterative methods).
    gap : float
        Optimality gap (upper_bound - lower_bound) for relaxation methods.
    """

    tuples: NDArray[np.intp]
    cost: float
    converged: bool
    n_iterations: int
    gap: float


def _validate_cost_tensor(
    cost_tensor: NDArray[np.float64],
) -> Tuple[int, int, int]:
    """Validate cost tensor and return dimensions."""
    if cost_tensor.ndim != 3:
        raise ValueError(f"Cost tensor must be 3-dimensional, got {cost_tensor.ndim}")
    return cost_tensor.shape


def greedy_3d(
    cost_tensor: ArrayLike,
    maximize: bool = False,
) -> Assignment3DResult:
    """
    Solve 3D assignment using a greedy algorithm.

    This algorithm iteratively selects the lowest-cost unassigned tuple
    until no more valid assignments can be made. It provides a fast but
    suboptimal solution.

    Parameters
    ----------
    cost_tensor : array_like
        Cost tensor of shape (n1, n2, n3).
    maximize : bool, optional
        If True, solve maximization problem (default: False).

    Returns
    -------
    result : Assignment3DResult
        Assignment result with tuples, cost, and metadata.

    Examples
    --------
    >>> import numpy as np
    >>> cost = np.random.rand(3, 3, 3)
    >>> result = greedy_3d(cost)
    >>> result.tuples.shape
    (3, 3)

    Notes
    -----
    Time complexity is O(n1 * n2 * n3 * min(n1, n2, n3)) in the worst case.
    The greedy solution provides a starting point for more sophisticated
    algorithms but is generally not optimal.
    """
    cost = np.asarray(cost_tensor, dtype=np.float64)
    n1, n2, n3 = _validate_cost_tensor(cost)

    if maximize:
        cost = -cost

    # Track which indices are used
    used_i = np.zeros(n1, dtype=bool)
    used_j = np.zeros(n2, dtype=bool)
    used_k = np.zeros(n3, dtype=bool)

    assignments: List[Tuple[int, int, int]] = []
    total_cost = 0.0

    # Maximum possible assignments
    max_assign = min(n1, n2, n3)

    for _ in range(max_assign):
        best_cost = np.inf
        best_tuple = None

        # Find best unassigned tuple
        for i in range(n1):
            if used_i[i]:
                continue
            for j in range(n2):
                if used_j[j]:
                    continue
                for k in range(n3):
                    if used_k[k]:
                        continue
                    if cost[i, j, k] < best_cost:
                        best_cost = cost[i, j, k]
                        best_tuple = (i, j, k)

        if best_tuple is None or np.isinf(best_cost):
            break

        i, j, k = best_tuple
        assignments.append(best_tuple)
        total_cost += best_cost
        used_i[i] = True
        used_j[j] = True
        used_k[k] = True

    if maximize:
        total_cost = -total_cost

    tuples = np.array(assignments, dtype=np.intp).reshape(-1, 3)

    return Assignment3DResult(
        tuples=tuples,
        cost=total_cost,
        converged=True,
        n_iterations=1,
        gap=np.inf,  # Unknown optimality gap
    )


def decompose_to_2d(
    cost_tensor: ArrayLike,
    fixed_dimension: int = 0,
    maximize: bool = False,
) -> Assignment3DResult:
    """
    Solve 3D assignment by decomposing into sequential 2D problems.

    This heuristic fixes one dimension and solves a sequence of 2D
    assignment problems. While not optimal, it provides a polynomial-time
    approximation.

    Parameters
    ----------
    cost_tensor : array_like
        Cost tensor of shape (n1, n2, n3).
    fixed_dimension : int, optional
        Which dimension to iterate over (0, 1, or 2). Default: 0.
    maximize : bool, optional
        If True, solve maximization problem (default: False).

    Returns
    -------
    result : Assignment3DResult
        Assignment result.

    Examples
    --------
    >>> import numpy as np
    >>> cost = np.random.rand(4, 4, 4)
    >>> result = decompose_to_2d(cost, fixed_dimension=0)
    >>> result.tuples.shape[0] <= 4
    True

    Notes
    -----
    The algorithm iterates over the fixed dimension and solves a 2D
    assignment for each slice. Indices that have been assigned in
    previous slices are excluded from subsequent problems.

    Time complexity is O(n * m^3) where n is the size of the fixed
    dimension and m is the maximum of the other two dimensions.
    """
    cost = np.asarray(cost_tensor, dtype=np.float64)
    n1, n2, n3 = _validate_cost_tensor(cost)

    if fixed_dimension not in (0, 1, 2):
        raise ValueError("fixed_dimension must be 0, 1, or 2")

    # Transpose so fixed dimension is first
    if fixed_dimension == 1:
        cost = np.transpose(cost, (1, 0, 2))
        n1, n2, n3 = n2, n1, n3
    elif fixed_dimension == 2:
        cost = np.transpose(cost, (2, 0, 1))
        n1, n2, n3 = n3, n1, n2

    # Track used indices
    used_j = np.zeros(n2, dtype=bool)
    used_k = np.zeros(n3, dtype=bool)

    assignments: List[Tuple[int, int, int]] = []
    total_cost = 0.0

    for i in range(n1):
        # Get slice
        slice_cost = cost[i].copy()

        # Mask out used indices
        if maximize:
            slice_cost[used_j, :] = -np.inf
            slice_cost[:, used_k] = -np.inf
        else:
            slice_cost[used_j, :] = np.inf
            slice_cost[:, used_k] = np.inf

        # Check if any valid assignments remain
        free_j = np.where(~used_j)[0]
        free_k = np.where(~used_k)[0]

        if len(free_j) == 0 or len(free_k) == 0:
            break

        # Extract submatrix of free indices
        sub_cost = slice_cost[np.ix_(free_j, free_k)]

        # Solve 2D assignment
        try:
            row_ind, col_ind = scipy_lsa(sub_cost, maximize=maximize)
        except ValueError:
            break

        # We only take one assignment per slice
        if len(row_ind) > 0:
            # Take the best assignment from this slice
            if maximize:
                best_idx = np.argmax(sub_cost[row_ind, col_ind])
            else:
                best_idx = np.argmin(sub_cost[row_ind, col_ind])

            j = free_j[row_ind[best_idx]]
            k = free_k[col_ind[best_idx]]
            assignment_cost = cost[i, j, k]

            if not np.isinf(assignment_cost):
                assignments.append((i, j, k))
                total_cost += assignment_cost
                used_j[j] = True
                used_k[k] = True

    # Transform back to original indexing
    if fixed_dimension == 1:
        # Was (j, i, k), transform to (i, j, k)
        assignments = [(j, i, k) for i, j, k in assignments]
    elif fixed_dimension == 2:
        # Was (k, i, j), transform to (i, j, k)
        assignments = [(j, k, i) for i, j, k in assignments]

    tuples = np.array(assignments, dtype=np.intp).reshape(-1, 3)

    return Assignment3DResult(
        tuples=tuples,
        cost=total_cost,
        converged=True,
        n_iterations=n1,
        gap=np.inf,
    )


def assign3d_lagrangian(
    cost_tensor: ArrayLike,
    max_iter: int = 100,
    tol: float = 1e-6,
    step_size: float = 1.0,
    maximize: bool = False,
) -> Assignment3DResult:
    """
    Solve 3D assignment using Lagrangian relaxation.

    This algorithm relaxes the 3D constraints and solves a sequence of
    2D assignment problems. The Lagrange multipliers are updated using
    subgradient optimization to improve the bound.

    Parameters
    ----------
    cost_tensor : array_like
        Cost tensor of shape (n1, n2, n3).
    max_iter : int, optional
        Maximum number of iterations (default: 100).
    tol : float, optional
        Convergence tolerance for gap (default: 1e-6).
    step_size : float, optional
        Initial step size for subgradient update (default: 1.0).
    maximize : bool, optional
        If True, solve maximization problem (default: False).

    Returns
    -------
    result : Assignment3DResult
        Assignment result with optimality gap information.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> cost = np.random.rand(5, 5, 5)
    >>> result = assign3d_lagrangian(cost, max_iter=50)
    >>> result.converged
    True

    Notes
    -----
    The Lagrangian relaxation dualizes the constraint that each index
    in the third dimension appears at most once. The resulting problem
    decomposes into n3 independent 2D assignment problems.

    The gap provides a measure of solution quality: gap = 0 indicates
    an optimal solution.

    References
    ----------
    .. [1] Poore, A.B., "Multidimensional assignment formulation of data
           association problems arising from multitarget and multisensor
           tracking", Computational Optimization and Applications, 1994.
    """
    cost = np.asarray(cost_tensor, dtype=np.float64)
    n1, n2, n3 = _validate_cost_tensor(cost)

    if maximize:
        cost = -cost

    # Initialize Lagrange multipliers for third dimension
    # mu[k] penalizes using index k more than once
    mu = np.zeros(n3, dtype=np.float64)

    best_cost = np.inf
    best_tuples: Optional[NDArray[np.intp]] = None
    lower_bound = -np.inf

    alpha = step_size

    for iteration in range(max_iter):
        # Solve relaxed problem: for each k, solve 2D assignment
        # with modified costs: cost[i,j,k] - mu[k]
        all_assignments: List[Tuple[int, int, int]] = []
        relaxed_cost = 0.0

        for k in range(n3):
            # Modified cost matrix for slice k
            slice_cost = cost[:, :, k] - mu[k]

            # Solve 2D assignment
            row_ind, col_ind = scipy_lsa(slice_cost, maximize=False)

            # Add assignments
            for i, j in zip(row_ind, col_ind):
                if not np.isinf(slice_cost[i, j]):
                    all_assignments.append((i, j, k))
                    relaxed_cost += slice_cost[i, j]

        # Add back the multiplier term
        relaxed_cost += np.sum(mu)

        # Update lower bound
        lower_bound = max(lower_bound, relaxed_cost)

        # Check for constraint violations (multiple uses of same i, j, or k)
        k_counts = np.zeros(n3, dtype=np.int64)
        i_counts = np.zeros(n1, dtype=np.int64)
        j_counts = np.zeros(n2, dtype=np.int64)

        for i, j, k in all_assignments:
            k_counts[k] += 1
            i_counts[i] += 1
            j_counts[j] += 1

        # Build feasible solution by resolving conflicts
        feasible_assignments: List[Tuple[int, int, int]] = []
        used_i = set()
        used_j = set()
        used_k = set()

        # Sort assignments by original cost
        sorted_assignments = sorted(
            all_assignments,
            key=lambda x: cost[x[0], x[1], x[2]],
        )

        for i, j, k in sorted_assignments:
            if i not in used_i and j not in used_j and k not in used_k:
                feasible_assignments.append((i, j, k))
                used_i.add(i)
                used_j.add(j)
                used_k.add(k)

        # Compute feasible cost
        feasible_cost = sum(cost[i, j, k] for i, j, k in feasible_assignments)

        # Update best solution
        if feasible_cost < best_cost:
            best_cost = feasible_cost
            best_tuples = np.array(feasible_assignments, dtype=np.intp).reshape(-1, 3)

        # Check convergence
        gap = best_cost - lower_bound
        if gap < tol:
            break

        # Update multipliers using subgradient
        # Subgradient: g[k] = 1 - k_counts[k] (1 if not used, 0 if used once, -1 if used twice)
        subgradient = 1 - k_counts

        # Diminishing step size
        alpha_k = alpha / (1 + iteration)

        # Update
        mu = mu + alpha_k * subgradient

    if best_tuples is None:
        best_tuples = np.array([], dtype=np.intp).reshape(0, 3)
        best_cost = 0.0

    if maximize:
        best_cost = -best_cost
        lower_bound = -lower_bound

    return Assignment3DResult(
        tuples=best_tuples,
        cost=best_cost,
        converged=(gap < tol),
        n_iterations=iteration + 1,
        gap=abs(gap),
    )


def assign3d_auction(
    cost_tensor: ArrayLike,
    epsilon: Optional[float] = None,
    max_iter: int = 1000,
    maximize: bool = False,
) -> Assignment3DResult:
    """
    Solve 3D assignment using an auction-based algorithm.

    This extends the 2D auction algorithm to 3D by treating the problem
    as a tripartite matching with iterative bidding.

    Parameters
    ----------
    cost_tensor : array_like
        Cost tensor of shape (n1, n2, n3).
    epsilon : float, optional
        Price increment. If None, uses adaptive epsilon.
    max_iter : int, optional
        Maximum iterations (default: 1000).
    maximize : bool, optional
        If True, solve maximization problem (default: False).

    Returns
    -------
    result : Assignment3DResult
        Assignment result.

    Examples
    --------
    >>> import numpy as np
    >>> cost = np.random.rand(4, 4, 4)
    >>> result = assign3d_auction(cost)
    >>> len(result.tuples) <= 4
    True

    Notes
    -----
    The auction algorithm for 3D assignment alternates between phases:
    1. Bidding phase: unassigned elements bid for their preferred matches
    2. Assignment phase: matches are updated based on bids

    This is a heuristic extension of the 2D auction algorithm and may
    not find the global optimum.

    References
    ----------
    .. [1] Bertsekas, D.P., "The auction algorithm for assignment and other
           network flow problems", Interfaces, 1990.
    """
    cost = np.asarray(cost_tensor, dtype=np.float64)
    n1, n2, n3 = _validate_cost_tensor(cost)

    if maximize:
        cost = -cost

    if epsilon is None:
        # Adaptive epsilon
        cost_range = np.max(cost[np.isfinite(cost)]) - np.min(cost[np.isfinite(cost)])
        epsilon = max(cost_range / (n1 + n2 + n3 + 1), 1e-10)

    # Prices for (j, k) pairs
    prices = np.zeros((n2, n3), dtype=np.float64)

    # Assignment: assign_i[i] = (j, k) or None
    assign_i: List[Optional[Tuple[int, int]]] = [None] * n1

    # Reverse: which i is assigned to (j, k)
    reverse: dict[tuple[int, int], int] = {}

    converged = False

    for iteration in range(max_iter):
        # Find unassigned i
        unassigned = [i for i in range(n1) if assign_i[i] is None]

        if len(unassigned) == 0:
            converged = True
            break

        for i in unassigned:
            # Find best (j, k) pair for i
            best_value = -np.inf
            best_jk = None
            second_value = -np.inf

            for j in range(n2):
                for k in range(n3):
                    value = -cost[i, j, k] - prices[j, k]
                    if value > best_value:
                        second_value = best_value
                        best_value = value
                        best_jk = (j, k)
                    elif value > second_value:
                        second_value = value

            if best_jk is None:
                continue

            j, k = best_jk

            # Bid increment
            bid = best_value - second_value + epsilon

            # Update price
            prices[j, k] += bid

            # Unassign previous owner
            if (j, k) in reverse:
                old_i = reverse[(j, k)]
                assign_i[old_i] = None

            # Assign i to (j, k)
            assign_i[i] = (j, k)
            reverse[(j, k)] = i

    # Build result
    assignments = []
    for i in range(n1):
        if assign_i[i] is not None:
            j, k = assign_i[i]
            assignments.append((i, j, k))

    tuples = np.array(assignments, dtype=np.intp).reshape(-1, 3)
    total_cost = sum(cost[i, j, k] for i, j, k in assignments) if assignments else 0.0

    if maximize:
        total_cost = -total_cost

    return Assignment3DResult(
        tuples=tuples,
        cost=total_cost,
        converged=converged,
        n_iterations=iteration + 1,
        gap=np.inf,  # Auction doesn't provide gap
    )


def assign3d(
    cost_tensor: ArrayLike,
    method: str = "lagrangian",
    maximize: bool = False,
    **kwargs: Any,
) -> Assignment3DResult:
    """
    Solve 3D assignment problem.

    This is the main entry point for 3D assignment, providing access to
    multiple algorithms through a unified interface.

    Parameters
    ----------
    cost_tensor : array_like
        Cost tensor of shape (n1, n2, n3).
    method : str, optional
        Algorithm to use:
        - "lagrangian": Lagrangian relaxation (default)
        - "auction": Auction-based algorithm
        - "greedy": Fast greedy heuristic
        - "decompose": Sequential 2D decomposition
    maximize : bool, optional
        If True, solve maximization problem (default: False).
    **kwargs
        Additional arguments passed to the specific method.

    Returns
    -------
    result : Assignment3DResult
        Assignment result.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> cost = np.random.rand(5, 5, 5)
    >>> result = assign3d(cost, method="lagrangian")
    >>> result.tuples.shape
    (5, 3)

    See Also
    --------
    assign3d_lagrangian : Lagrangian relaxation method
    assign3d_auction : Auction-based method
    greedy_3d : Greedy heuristic
    decompose_to_2d : Sequential decomposition

    Notes
    -----
    The 3D assignment problem is NP-hard, so all methods provide
    approximate solutions. The Lagrangian relaxation method generally
    provides the best quality with reasonable computation time.
    """
    methods = {
        "lagrangian": assign3d_lagrangian,
        "auction": assign3d_auction,
        "greedy": greedy_3d,
        "decompose": decompose_to_2d,
    }

    if method not in methods:
        valid = ", ".join(methods.keys())
        raise ValueError(f"Unknown method '{method}'. Valid methods: {valid}")

    return methods[method](cost_tensor, maximize=maximize, **kwargs)


__all__ = [
    "Assignment3DResult",
    "greedy_3d",
    "decompose_to_2d",
    "assign3d_lagrangian",
    "assign3d_auction",
    "assign3d",
]
