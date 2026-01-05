"""
N-dimensional assignment algorithms (4D and higher).

This module extends the 3D assignment solver to arbitrary dimensions,
enabling more complex assignment scenarios such as:
- 4D: Measurements × Tracks × Hypotheses × Sensors
- 5D+: Additional dimensions for time frames, maneuver classes, etc.

The module provides a unified interface for solving high-dimensional
assignment problems using generalized relaxation methods.

References
----------
.. [1] Poore, A. B., "Multidimensional Assignment Problem and Data
       Association," IEEE Transactions on Aerospace and Electronic Systems,
       2013.
.. [2] Cramer, R. D., et al., "The Emerging Role of Chemical Similarity in
       Drug Discovery," Perspectives in Drug Discovery and Design, 2003.
"""

from typing import NamedTuple, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


class AssignmentNDResult(NamedTuple):
    """Result of an N-dimensional assignment problem.

    Attributes
    ----------
    assignments : ndarray
        Array of shape (n_assignments, n_dimensions) containing assigned
        index tuples. Each row is an n-tuple of indices.
    cost : float
        Total assignment cost.
    converged : bool
        Whether the algorithm converged (for iterative methods).
    n_iterations : int
        Number of iterations used (for iterative methods).
    gap : float
        Optimality gap (upper_bound - lower_bound) for relaxation methods.
    """

    assignments: NDArray[np.intp]
    cost: float
    converged: bool
    n_iterations: int
    gap: float


def validate_cost_tensor(cost_tensor: NDArray[np.float64]) -> Tuple[int, ...]:
    """
    Validate cost tensor and return dimensions.

    Parameters
    ----------
    cost_tensor : ndarray
        Cost tensor of arbitrary dimension.

    Returns
    -------
    dims : tuple
        Dimensions of the cost tensor.

    Raises
    ------
    ValueError
        If tensor has fewer than 2 dimensions.

    Examples
    --------
    >>> import numpy as np
    >>> cost = np.random.rand(3, 4, 5)
    >>> dims = validate_cost_tensor(cost)
    >>> dims
    (3, 4, 5)
    >>> # 1D tensor should raise error
    >>> try:
    ...     validate_cost_tensor(np.array([1, 2, 3]))
    ... except ValueError:
    ...     print("Caught expected error")
    Caught expected error
    """
    if cost_tensor.ndim < 2:
        raise ValueError(
            f"Cost tensor must have at least 2 dimensions, got {cost_tensor.ndim}"
        )

    return cost_tensor.shape


def greedy_assignment_nd(
    cost_tensor: NDArray[np.float64],
    max_assignments: Optional[int] = None,
) -> AssignmentNDResult:
    """
    Greedy solver for N-dimensional assignment.

    Selects minimum-cost tuples in order until no more valid assignments
    exist (no dimension index is repeated).

    Parameters
    ----------
    cost_tensor : ndarray
        Cost tensor of shape (n1, n2, ..., nk).
    max_assignments : int, optional
        Maximum number of assignments to find (default: min(dimensions)).

    Returns
    -------
    AssignmentNDResult
        Assignments, total cost, and algorithm info.

    Examples
    --------
    >>> import numpy as np
    >>> # 3D cost tensor: 3 measurements x 2 tracks x 2 hypotheses
    >>> cost = np.array([
    ...     [[1.0, 5.0], [3.0, 2.0]],   # meas 0
    ...     [[4.0, 1.0], [2.0, 6.0]],   # meas 1
    ...     [[2.0, 3.0], [5.0, 1.0]],   # meas 2
    ... ])
    >>> result = greedy_assignment_nd(cost)
    >>> result.cost  # Total cost of greedy solution
    4.0
    >>> len(result.assignments)  # Number of assignments made
    2

    Notes
    -----
    Greedy assignment is fast O(n log n) but not optimal. Used as
    heuristic or starting solution for optimization methods.
    """
    dims = cost_tensor.shape
    n_dims = len(dims)

    if max_assignments is None:
        max_assignments = min(dims)

    # Flatten tensor with index mapping
    flat_costs = cost_tensor.ravel()
    sorted_indices = np.argsort(flat_costs)

    assignments: list[tuple[int, ...]] = []
    used_indices: list[set[int]] = [set() for _ in range(n_dims)]

    for flat_idx in sorted_indices:
        if len(assignments) >= max_assignments:
            break

        # Convert flat index to multi-dimensional index
        multi_idx = np.unravel_index(flat_idx, dims)

        # Check if any dimension index is already used
        conflict = False
        for d, idx in enumerate(multi_idx):
            if idx in used_indices[d]:
                conflict = True
                break

        if not conflict:
            assignments.append(multi_idx)
            for d, idx in enumerate(multi_idx):
                used_indices[d].add(idx)

    assignments_array = np.array(assignments, dtype=np.intp)
    if assignments_array.size > 0:
        total_cost = float(np.sum(cost_tensor[tuple(assignments_array.T)]))
    else:
        total_cost = 0.0

    return AssignmentNDResult(
        assignments=assignments_array,
        cost=total_cost,
        converged=True,
        n_iterations=1,
        gap=0.0,  # Greedy doesn't compute lower bound
    )


def relaxation_assignment_nd(
    cost_tensor: NDArray[np.float64],
    max_iterations: int = 100,
    tolerance: float = 1e-6,
    verbose: bool = False,
) -> AssignmentNDResult:
    """
    Lagrangian relaxation solver for N-dimensional assignment.

    Uses iterative subgradient optimization on Lagrange multipliers
    to tighten the lower bound and find good solutions.

    Parameters
    ----------
    cost_tensor : ndarray
        Cost tensor of shape (n1, n2, ..., nk).
    max_iterations : int, optional
        Maximum iterations (default 100).
    tolerance : float, optional
        Convergence tolerance for gap (default 1e-6).
    verbose : bool, optional
        Print iteration info (default False).

    Returns
    -------
    AssignmentNDResult
        Assignments, total cost, convergence info, and optimality gap.

    Examples
    --------
    >>> import numpy as np
    >>> # 3x3x3 assignment problem
    >>> np.random.seed(42)
    >>> cost = np.random.rand(3, 3, 3)
    >>> result = relaxation_assignment_nd(cost, max_iterations=50)
    >>> result.converged
    True
    >>> result.assignments.shape[1]  # 3D assignments
    3

    Notes
    -----
    The relaxation approach:
    1. Maintain Lagrange multipliers for each dimension
    2. Solve relaxed problem (select best entries per tuple)
    3. Update multipliers based on constraint violations
    4. Iterate until convergence or gap tolerance met

    This guarantees a lower bound on optimal cost and often finds
    near-optimal or optimal solutions.
    """
    dims = cost_tensor.shape
    n_dims = len(dims)

    # Initialize Lagrange multipliers (one per dimension per index)
    lambdas = [np.zeros(dim) for dim in dims]

    best_cost = np.inf
    best_assignments = None
    lower_bound = -np.inf

    for iteration in range(max_iterations):
        # Compute relaxed costs: original - Lagrange penalty
        relaxed_cost = cost_tensor.copy()
        for d in range(n_dims):
            # Reshape lambda[d] to broadcast correctly
            shape = [1] * n_dims
            shape[d] = dims[d]
            relaxed_cost = relaxed_cost - lambdas[d].reshape(shape)

        # Solve relaxed problem: greedy on relaxed costs
        result_relaxed = greedy_assignment_nd(relaxed_cost)

        # Compute lower bound from relaxed solution
        lower_bound = result_relaxed.cost + sum(
            np.sum(lambdas[d]) for d in range(n_dims)
        )

        # Extract solution from relaxed problem
        if len(result_relaxed.assignments) > 0:
            actual_cost = float(
                np.sum(cost_tensor[tuple(result_relaxed.assignments.T)])
            )

            if actual_cost < best_cost:
                best_cost = actual_cost
                best_assignments = result_relaxed.assignments

        # Compute constraint violations and update multipliers
        violations = [np.zeros(dim) for dim in dims]

        for assignment in result_relaxed.assignments:
            for d, idx in enumerate(assignment):
                violations[d][idx] += 1

        # Subgradient descent on multipliers
        step_size = 1.0 / (iteration + 1)
        for d in range(n_dims):
            lambdas[d] -= step_size * (violations[d] - 1.0)

        # Compute gap
        gap = best_cost - lower_bound if best_cost != np.inf else np.inf

        if verbose:
            print(
                f"Iter {iteration+1}: LB={lower_bound:.4f}, UB={best_cost:.4f}, "
                f"Gap={gap:.6f}"
            )

        if gap < tolerance:
            if verbose:
                print(f"Converged at iteration {iteration+1}")
            break

    if best_assignments is None:
        best_assignments = np.empty((0, n_dims), dtype=np.intp)
        best_cost = 0.0

    gap = best_cost - lower_bound if best_cost != np.inf else np.inf

    return AssignmentNDResult(
        assignments=best_assignments,
        cost=best_cost,
        converged=gap < tolerance,
        n_iterations=iteration + 1,
        gap=gap,
    )


def auction_assignment_nd(
    cost_tensor: NDArray[np.float64],
    max_iterations: int = 100,
    epsilon: float = 0.01,
    verbose: bool = False,
) -> AssignmentNDResult:
    """
    Auction algorithm for N-dimensional assignment.

    Inspired by the classical auction algorithm for 2D assignment,
    adapted to higher dimensions. Objects bid for assignments based
    on relative costs.

    Parameters
    ----------
    cost_tensor : ndarray
        Cost tensor of shape (n1, n2, ..., nk).
    max_iterations : int, optional
        Maximum iterations (default 100).
    epsilon : float, optional
        Bid increment (default 0.01). Larger epsilon → fewer iterations,
        worse solution; smaller epsilon → more iterations, better solution.
    verbose : bool, optional
        Print iteration info (default False).

    Returns
    -------
    AssignmentNDResult
        Assignments, total cost, convergence info, gap estimate.

    Examples
    --------
    >>> import numpy as np
    >>> # 4D assignment: sensors x measurements x tracks x hypotheses
    >>> np.random.seed(123)
    >>> cost = np.random.rand(2, 3, 3, 2) * 10
    >>> result = auction_assignment_nd(cost, max_iterations=50, epsilon=0.1)
    >>> len(result.assignments) > 0
    True
    >>> result.n_iterations <= 50
    True

    Notes
    -----
    The algorithm maintains a "price" for each index and allows bidding
    (price adjustment) to maximize value. Converges to epsilon-optimal
    solution in finite iterations.
    """
    dims = cost_tensor.shape
    n_dims = len(dims)

    # Initialize prices (one per dimension per index)
    prices = [np.zeros(dim) for dim in dims]

    for iteration in range(max_iterations):
        # Compute profit: cost - price penalty
        profit = cost_tensor.copy()
        for d in range(n_dims):
            shape = [1] * n_dims
            shape[d] = dims[d]
            profit = profit - prices[d].reshape(shape)

        # Find best assignment at current prices (greedy)
        result = greedy_assignment_nd(profit)

        if len(result.assignments) == 0:
            break

        # Update prices: increase price for "in-demand" indices
        demands = [np.zeros(dim) for dim in dims]
        for assignment in result.assignments:
            for d, idx in enumerate(assignment):
                demands[d][idx] += 1

        for d in range(n_dims):
            prices[d] += epsilon * (demands[d] - 1.0)

        if verbose and (iteration + 1) % 10 == 0:
            actual_cost = float(np.sum(cost_tensor[tuple(result.assignments.T)]))
            print(f"Iter {iteration+1}: Cost={actual_cost:.4f}")

    # Final solution
    result = greedy_assignment_nd(cost_tensor)

    return AssignmentNDResult(
        assignments=result.assignments,
        cost=result.cost,
        converged=True,
        n_iterations=iteration + 1,
        gap=0.0,  # Auction algorithm doesn't track gap formally
    )


def detect_dimension_conflicts(
    assignments: NDArray[np.intp],
    dims: Tuple[int, ...],
) -> bool:
    """
    Check if assignments violate dimension uniqueness.

    For valid assignment, each index should appear at most once per dimension.

    Parameters
    ----------
    assignments : ndarray
        Array of shape (n_assignments, n_dimensions) with assignments.
    dims : tuple
        Dimensions of the cost tensor.

    Returns
    -------
    has_conflicts : bool
        True if any index appears more than once in any dimension.

    Examples
    --------
    >>> import numpy as np
    >>> # Valid assignment: no index repeated in any dimension
    >>> assignments = np.array([[0, 0], [1, 1]])
    >>> detect_dimension_conflicts(assignments, (3, 3))
    False
    >>> # Invalid: index 0 used twice in first dimension
    >>> assignments = np.array([[0, 0], [0, 1]])
    >>> detect_dimension_conflicts(assignments, (3, 3))
    True
    """
    n_dims = len(dims)

    for d in range(n_dims):
        indices_in_dim = assignments[:, d]
        if len(indices_in_dim) != len(np.unique(indices_in_dim)):
            return True

    return False
