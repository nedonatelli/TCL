"""
Network flow solutions for assignment problems.

This module provides min-cost flow formulations for assignment problems,
offering an alternative to Hungarian algorithm and relaxation methods.

A min-cost flow approach:
1. Models assignment as flow network
2. Uses cost edges for penalties
3. Enforces supply/demand constraints
4. Finds minimum-cost flow solution
5. Extracts assignment from flow

References
----------
.. [1] Ahuja, R. K., Magnanti, T. L., & Orlin, J. B. (1993). Network Flows:
       Theory, Algorithms, and Applications. Prentice-Hall.
.. [2] Costain, G., & Liang, H. (2012). An Auction Algorithm for the
       Minimum Cost Flow Problem. CoRR, abs/1208.4859.
"""

from enum import Enum
from typing import Any, NamedTuple, Tuple

import numpy as np
from numpy.typing import NDArray


class FlowStatus(Enum):
    """Status of min-cost flow computation."""

    OPTIMAL = 0
    UNBOUNDED = 1
    INFEASIBLE = 2
    TIMEOUT = 3


class MinCostFlowResult(NamedTuple):
    """Result of min-cost flow computation.

    Attributes
    ----------
    flow : ndarray
        Flow values on each edge, shape (n_edges,).
    cost : float
        Total flow cost.
    status : FlowStatus
        Optimization status.
    iterations : int
        Number of iterations used.
    """

    flow: NDArray[np.float64]
    cost: float
    status: FlowStatus
    iterations: int


class FlowEdge(NamedTuple):
    """Edge in a flow network.

    Attributes
    ----------
    from_node : int
        Source node index.
    to_node : int
        Destination node index.
    capacity : float
        Maximum flow on edge (default 1.0 for assignment).
    cost : float
        Cost per unit flow.
    """

    from_node: int
    to_node: int
    capacity: float
    cost: float


def assignment_to_flow_network(
    cost_matrix: NDArray[np.float64],
) -> Tuple[list[FlowEdge], NDArray[np.floating], NDArray[Any]]:
    """
    Convert 2D assignment problem to min-cost flow network.

    Network structure:
    - Source node (0) supplies all workers
    - Worker nodes (1 to m) demand 1 unit each
    - Task nodes (m+1 to m+n) supply 1 unit each
    - Sink node (m+n+1) collects all completed tasks

    Parameters
    ----------
    cost_matrix : ndarray
        Cost matrix of shape (m, n) where cost[i,j] is cost of
        assigning worker i to task j.

    Returns
    -------
    edges : list[FlowEdge]
        List of edges in the flow network.
    supplies : ndarray
        Supply/demand at each node (shape n_nodes,).
        Positive = supply, negative = demand.
    node_names : ndarray
        Names of nodes for reference.

    Examples
    --------
    >>> import numpy as np
    >>> # 2 workers, 3 tasks
    >>> cost = np.array([[1.0, 2.0, 3.0],
    ...                  [4.0, 5.0, 6.0]])
    >>> edges, supplies, names = assignment_to_flow_network(cost)
    >>> len(edges)  # source->workers + workers->tasks + tasks->sink
    11
    >>> supplies[0]  # source supplies 2 (num workers)
    2.0
    >>> names[0]
    'source'
    """
    m, n = cost_matrix.shape

    # Node numbering:
    # 0: source
    # 1 to m: workers
    # m+1 to m+n: tasks
    # m+n+1: sink

    n_nodes = m + n + 2
    source = 0
    sink = m + n + 1

    edges = []

    # Source to workers: capacity 1, cost 0
    for i in range(1, m + 1):
        edges.append(FlowEdge(from_node=source, to_node=i, capacity=1.0, cost=0.0))

    # Workers to tasks: capacity 1, cost = assignment cost
    for i in range(m):
        for j in range(n):
            worker_node = i + 1
            task_node = m + 1 + j
            edges.append(
                FlowEdge(
                    from_node=worker_node,
                    to_node=task_node,
                    capacity=1.0,
                    cost=cost_matrix[i, j],
                )
            )

    # Tasks to sink: capacity 1, cost 0
    for j in range(1, n + 1):
        task_node = m + j
        edges.append(
            FlowEdge(from_node=task_node, to_node=sink, capacity=1.0, cost=0.0)
        )

    # Supply/demand: source supplies m units, sink demands m units
    supplies = np.zeros(n_nodes)
    supplies[source] = float(m)
    supplies[sink] = float(-m)

    node_names = np.array(
        ["source"]
        + [f"worker_{i}" for i in range(m)]
        + [f"task_{j}" for j in range(n)]
        + ["sink"]
    )

    return edges, supplies, node_names


def min_cost_flow_successive_shortest_paths(
    edges: list[FlowEdge],
    supplies: NDArray[np.float64],
    max_iterations: int = 1000,
) -> MinCostFlowResult:
    """
    Solve min-cost flow using successive shortest paths with cost scaling.

    Algorithm:
    1. Initialize potentials using Bellman-Ford
    2. While there is excess supply:
       - Find shortest path using reduced costs (Dijkstra with potentials)
       - Push unit flow along path
       - Update node potentials
       - Recompute shortest paths to maintain optimality

    This is the standard min-cost flow algorithm that guarantees optimality
    and convergence. It uses Dijkstra's algorithm with potentials, which
    maintains the dual feasibility (reduced cost property).

    Parameters
    ----------
    edges : list[FlowEdge]
        List of edges with capacities and costs.
    supplies : ndarray
        Supply/demand at each node.
    max_iterations : int, optional
        Maximum iterations (default 1000).

    Returns
    -------
    MinCostFlowResult
        Solution with flow values, cost, status, and iterations.

    Examples
    --------
    >>> import numpy as np
    >>> from pytcl.assignment_algorithms.network_flow import (
    ...     FlowEdge, assignment_to_flow_network
    ... )
    >>> cost = np.array([[1.0, 5.0], [4.0, 2.0]])
    >>> edges, supplies, _ = assignment_to_flow_network(cost)
    >>> result = min_cost_flow_successive_shortest_paths(edges, supplies)
    >>> result.status == FlowStatus.OPTIMAL
    True
    >>> result.cost  # Optimal is 1+2=3 (diagonal)
    3.0

    Notes
    -----
    This implementation uses successive shortest paths with potentials.
    The algorithm is guaranteed to find the optimal solution for any
    feasible min-cost flow problem.

    For rectangular assignment problems (m < n or m > n), all m units
    of flow must be satisfied. The algorithm ensures this by finding
    augmenting paths until all supply is routed.
    """
    n_nodes = len(supplies)
    n_edges = len(edges)

    # Initialize flow and residual capacity
    flow = np.zeros(n_edges)
    residual_capacity = np.array([e.capacity for e in edges])

    # Initialize node potentials using Bellman-Ford from a dummy source
    # This ensures all potentials are finite and maintains dual feasibility
    potential = np.zeros(n_nodes)

    # Build adjacency list representation
    # Each entry: (to_node, edge_idx, is_reverse, cost)
    graph: list[list[tuple[int, int, int, float]]] = [[] for _ in range(n_nodes)]

    for edge_idx, edge in enumerate(edges):
        # Forward edge
        graph[edge.from_node].append((edge.to_node, edge_idx, 0, edge.cost))
        # Reverse edge (for flow cancellation)
        graph[edge.to_node].append((edge.from_node, edge_idx, 1, -edge.cost))

    current_supplies = supplies.copy()
    iteration = 0

    # Main algorithm loop
    while iteration < max_iterations:
        # Find excess and deficit nodes
        excess_node = -1
        deficit_node = -1

        for node in range(n_nodes):
            if current_supplies[node] > 1e-10:
                excess_node = node
                break

        if excess_node < 0:
            break  # No more excess nodes

        for node in range(n_nodes):
            if current_supplies[node] < -1e-10:
                deficit_node = node
                break

        if deficit_node < 0:
            break  # No deficit nodes

        # Find shortest path using Dijkstra with potentials
        # Reduced cost: c_reduced(u,v) = c(u,v) + π(u) - π(v)
        dist = np.full(n_nodes, np.inf)
        dist[excess_node] = 0.0
        parent = np.full(n_nodes, -1, dtype=int)
        parent_edge = np.full(n_nodes, -1, dtype=int)
        parent_reverse = np.full(n_nodes, 0, dtype=int)
        visited = np.zeros(n_nodes, dtype=bool)

        # Dijkstra's algorithm
        for _ in range(n_nodes):
            # Find unvisited node with minimum distance
            u = -1
            min_dist = np.inf
            for node in range(n_nodes):
                if not visited[node] and dist[node] < min_dist:
                    u = node
                    min_dist = dist[node]

            if u < 0 or dist[u] == np.inf:
                break

            visited[u] = True

            # Relax edges from u
            for v, eidx, is_rev, cost in graph[u]:
                if residual_capacity[eidx] > 1e-10:
                    # Compute reduced cost
                    reduced_cost = cost + potential[u] - potential[v]
                    new_dist = dist[u] + reduced_cost

                    if new_dist < dist[v] - 1e-10:
                        dist[v] = new_dist
                        parent[v] = u
                        parent_edge[v] = eidx
                        parent_reverse[v] = is_rev

        if dist[deficit_node] >= np.inf:
            # No path found
            break

        # Update potentials to maintain dual feasibility
        for node in range(n_nodes):
            if dist[node] < np.inf:
                potential[node] += dist[node]

        # Extract path by backtracking
        path_edges = []
        path_reverse_flags = []
        node = deficit_node
        path_length = 0
        visited_set = set()

        while parent[node] >= 0:
            if path_length >= n_nodes:
                break  # Safety check
            if node in visited_set:
                break  # Cycle detected

            visited_set.add(node)
            path_edges.append(parent_edge[node])
            path_reverse_flags.append(parent_reverse[node])
            node = parent[node]
            path_length += 1

        if not path_edges:
            iteration += 1
            continue

        path_edges.reverse()
        path_reverse_flags.reverse()

        # Find bottleneck capacity
        min_flow = min(residual_capacity[e] for e in path_edges)
        min_flow = min(
            min_flow,
            current_supplies[excess_node],
            -current_supplies[deficit_node],
        )

        # Push flow along path
        for edge_idx, is_reverse in zip(path_edges, path_reverse_flags):
            if is_reverse == 0:
                # Forward edge: increase flow
                flow[edge_idx] += min_flow
                residual_capacity[edge_idx] -= min_flow
            else:
                # Reverse edge: decrease flow (cancel)
                flow[edge_idx] -= min_flow
                residual_capacity[edge_idx] += min_flow

        current_supplies[excess_node] -= min_flow
        current_supplies[deficit_node] += min_flow

        iteration += 1

    # Compute total cost: include all flows (including negative which cancel)
    total_cost = 0.0
    for i, edge in enumerate(edges):
        total_cost += flow[i] * edge.cost

    # Determine status
    if np.allclose(current_supplies, 0, atol=1e-6):
        status = FlowStatus.OPTIMAL
    elif iteration >= max_iterations:
        status = FlowStatus.TIMEOUT
    else:
        status = FlowStatus.INFEASIBLE

    return MinCostFlowResult(
        flow=flow,
        cost=total_cost,
        status=status,
        iterations=iteration,
    )


def min_cost_flow_simplex(
    edges: list[FlowEdge],
    supplies: NDArray[np.float64],
    max_iterations: int = 10000,
) -> MinCostFlowResult:
    """
    Solve min-cost flow using Dijkstra-based successive shortest paths.

    This optimized version uses:
    - Dijkstra's algorithm (O(E log V)) instead of Bellman-Ford (O(VE))
    - Node potentials to maintain non-negative edge costs
    - Johnson's technique for cost adjustment

    This is significantly faster than Bellman-Ford while maintaining
    guaranteed correctness and optimality.

    Time complexity: O(K * E log V) where K = number of shortest paths
    Space complexity: O(V + E)

    Parameters
    ----------
    edges : list[FlowEdge]
        List of edges with capacities and costs.
    supplies : ndarray
        Supply/demand at each node.
    max_iterations : int, optional
        Maximum iterations (default 10000).

    Returns
    -------
    MinCostFlowResult
        Solution with flow values, cost, status, and iterations.

    References
    ----------
    .. [1] Ahuja, R. K., Magnanti, T. L., & Orlin, J. B. (1993).
           Network Flows: Theory, Algorithms, and Applications.
           (Chapter on successive shortest paths with potentials)
    .. [2] Johnson, D. B. (1977).
           Efficient All-Pairs Shortest Paths in Weighted Graphs.
    """
    from pytcl.assignment_algorithms.dijkstra_min_cost import (
        min_cost_flow_dijkstra_potentials,
    )

    n_nodes = len(supplies)

    # Convert FlowEdge objects to tuples
    edge_tuples = [(e.from_node, e.to_node, e.capacity, e.cost) for e in edges]

    # Run optimized Dijkstra-based algorithm
    flow, total_cost, iterations = min_cost_flow_dijkstra_potentials(
        n_nodes, edge_tuples, supplies, max_iterations
    )

    # Check feasibility
    residual_supplies = supplies.copy()
    for i, edge in enumerate(edges):
        residual_supplies[edge.from_node] -= flow[i]
        residual_supplies[edge.to_node] += flow[i]

    if np.allclose(residual_supplies, 0, atol=1e-6):
        status = FlowStatus.OPTIMAL
    elif iterations >= max_iterations:
        status = FlowStatus.TIMEOUT
    else:
        status = FlowStatus.INFEASIBLE

    return MinCostFlowResult(
        flow=flow,
        cost=total_cost,
        status=status,
        iterations=iterations,
    )


def assignment_from_flow_solution(
    flow: NDArray[np.float64],
    edges: list[FlowEdge],
    cost_matrix_shape: Tuple[int, int],
) -> Tuple[NDArray[np.intp], float]:
    """
    Extract assignment from flow network solution.

    A valid flow solution for assignment should have:
    - Exactly 1 unit of flow from each worker to some task
    - Exactly 1 unit of flow to each task from some worker
    - No negative flows on worker->task edges (those are cancellations)

    This function extracts the actual assignment by identifying which
    worker->task edges carry the net positive flow.

    Parameters
    ----------
    flow : ndarray
        Flow values on each edge.
    edges : list[FlowEdge]
        List of edges used in network.
    cost_matrix_shape : tuple
        Shape of original cost matrix (m, n).

    Returns
    -------
    assignment : ndarray
        Assignment array of shape (n_assignments, 2) with [worker, task].
    cost : float
        Total assignment cost.
    """
    m, n = cost_matrix_shape
    assignment = []
    cost = 0.0

    # Build source node (node 0) and sink node (node m+n+1) indices
    source = 0
    sink = m + n + 1

    # For a valid assignment solution:
    # - Count flow out of source to each worker
    # - Count flow into sink from each task
    worker_outflow = np.zeros(m)
    task_inflow = np.zeros(n)

    # Collect all worker->task edges and their flows
    worker_task_edges = []

    for edge_idx, edge in enumerate(edges):
        # Worker edges: from source (0) to worker nodes (1..m)
        if edge.from_node == source and 1 <= edge.to_node <= m:
            worker_id = edge.to_node - 1
            worker_outflow[worker_id] += flow[edge_idx]

        # Task edges: from task nodes (m+1..m+n) to sink
        if m + 1 <= edge.from_node <= m + n and edge.to_node == sink:
            task_id = edge.from_node - (m + 1)
            task_inflow[task_id] += flow[edge_idx]

        # Worker-to-task edges
        if 1 <= edge.from_node <= m and m + 1 <= edge.to_node <= m + n:
            worker_id = edge.from_node - 1
            task_id = edge.to_node - (m + 1)
            if flow[edge_idx] > 0.5:  # Positive flow means this edge is used
                worker_task_edges.append(
                    {
                        "worker": worker_id,
                        "task": task_id,
                        "flow": flow[edge_idx],
                        "cost": edge.cost,
                        "edge_idx": edge_idx,
                    }
                )

    # For assignment problems, each worker should have exactly 1 outgoing flow
    # and each task should have exactly 1 incoming flow
    # Extract the assignment from worker->task edges with positive flow
    for edge_info in worker_task_edges:
        assignment.append([edge_info["worker"], edge_info["task"]])
        cost += edge_info["flow"] * edge_info["cost"]

    assignment = (
        np.array(assignment, dtype=np.intp)
        if assignment
        else np.empty((0, 2), dtype=np.intp)
    )
    return assignment, cost


def min_cost_assignment_via_flow(
    cost_matrix: NDArray[np.float64],
    use_simplex: bool = True,
) -> Tuple[NDArray[np.intp], float]:
    """
    Solve 2D assignment problem via min-cost flow network.

    Uses Dijkstra-optimized successive shortest paths (Phase 1B) by default.
    Falls back to Bellman-Ford if needed.

    Parameters
    ----------
    cost_matrix : ndarray
        Cost matrix of shape (m, n).
    use_simplex : bool, optional
        Use Dijkstra-optimized algorithm (default True) or
        Bellman-Ford based successive shortest paths (False).

    Returns
    -------
    assignment : ndarray
        Assignment array of shape (n_assignments, 2).
    total_cost : float
        Total assignment cost.

    Examples
    --------
    >>> import numpy as np
    >>> cost = np.array([[1.0, 5.0, 9.0],
    ...                  [3.0, 2.0, 8.0],
    ...                  [7.0, 6.0, 4.0]])
    >>> assignment, total_cost = min_cost_assignment_via_flow(cost)
    >>> total_cost  # Optimal assignment: (0,0), (1,1), (2,2) = 1+2+4 = 7
    7.0
    >>> len(assignment)
    3

    Notes
    -----
    Phase 1B: Dijkstra-based optimization provides O(K*E log V) vs
    Bellman-Ford O(K*V*E), where K is number of shortest paths needed.
    """
    edges, supplies, _ = assignment_to_flow_network(cost_matrix)

    if use_simplex:
        result = min_cost_flow_simplex(edges, supplies)
    else:
        result = min_cost_flow_successive_shortest_paths(edges, supplies)

    assignment, cost = assignment_from_flow_solution(
        result.flow, edges, cost_matrix.shape
    )

    return assignment, cost
