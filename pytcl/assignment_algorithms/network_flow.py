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
    Solve min-cost flow using successive shortest paths.

    Algorithm:
    1. While there is excess supply:
       - Find shortest path from a supply node to a demand node
       - Push maximum feasible flow along path
       - Update supplies and residual capacities

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

    Notes
    -----
    This is a simplified implementation using Bellman-Ford for shortest
    paths. Production code would use more efficient implementations.
    """
    n_nodes = len(supplies)
    n_edges = len(edges)

    # Build adjacency lists for residual graph
    graph: list[list[tuple[int, int, float]]] = [[] for _ in range(n_nodes)]
    flow = np.zeros(n_edges)
    residual_capacity = np.array([e.capacity for e in edges])

    for edge_idx, edge in enumerate(edges):
        graph[edge.from_node].append((edge.to_node, edge_idx, edge.cost))
        # Add reverse edge with negative cost
        graph[edge.to_node].append((edge.from_node, edge_idx, -edge.cost))

    current_supplies = supplies.copy()
    iteration = 0

    while iteration < max_iterations:
        # Find a node with excess supply
        excess_node = None
        for node in range(n_nodes):
            if current_supplies[node] > 1e-10:
                excess_node = node
                break

        if excess_node is None:
            break

        # Find a node with deficit
        deficit_node = None
        for node in range(n_nodes):
            if current_supplies[node] < -1e-10:
                deficit_node = node
                break

        if deficit_node is None:
            break

        # Find shortest path using Bellman-Ford relaxation
        dist = np.full(n_nodes, np.inf)
        dist[excess_node] = 0.0
        parent = np.full(n_nodes, -1, dtype=int)
        parent_edge = np.full(n_nodes, -1, dtype=int)

        for _ in range(n_nodes - 1):
            for u in range(n_nodes):
                if dist[u] == np.inf:
                    continue
                for v, edge_idx, cost in graph[u]:
                    if residual_capacity[edge_idx] > 1e-10:
                        new_dist = dist[u] + cost
                        if new_dist < dist[v]:
                            dist[v] = new_dist
                            parent[v] = u
                            parent_edge[v] = edge_idx

        if dist[deficit_node] == np.inf:
            # No path found
            break

        # Extract path and find bottleneck capacity
        path_edges = []
        node = deficit_node
        while parent[node] != -1:
            path_edges.append(parent_edge[node])
            node = parent[node]

        path_edges.reverse()

        # Find minimum capacity along path
        min_flow = min(residual_capacity[e] for e in path_edges)
        min_flow = min(
            min_flow, current_supplies[excess_node], -current_supplies[deficit_node]
        )

        # Push flow along path
        total_cost = 0.0
        for edge_idx in path_edges:
            flow[edge_idx] += min_flow
            residual_capacity[edge_idx] -= min_flow
            total_cost += min_flow * edges[edge_idx].cost

        current_supplies[excess_node] -= min_flow
        current_supplies[deficit_node] += min_flow

        iteration += 1

    # Compute total cost
    total_cost = float(np.sum(flow[i] * edges[i].cost for i in range(n_edges)))

    # Determine status
    if np.allclose(current_supplies, 0):
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


def assignment_from_flow_solution(
    flow: NDArray[np.float64],
    edges: list[FlowEdge],
    cost_matrix_shape: Tuple[int, int],
) -> Tuple[NDArray[np.intp], float]:
    """
    Extract assignment from flow network solution.

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

    for edge_idx, edge in enumerate(edges):
        # Worker-to-task edges: from_node in [1, m], to_node in [m+1, m+n]
        if 1 <= edge.from_node <= m and m + 1 <= edge.to_node <= m + n:
            if flow[edge_idx] > 0.5:  # Flow > 0 (allowing for numerical tolerance)
                worker_idx = edge.from_node - 1
                task_idx = edge.to_node - m - 1
                assignment.append([worker_idx, task_idx])

    assignment = np.array(assignment, dtype=np.intp)
    cost = 0.0
    if len(assignment) > 0:
        cost = float(
            np.sum(
                flow[edge_idx] * edges[edge_idx].cost for edge_idx in range(len(edges))
            )
        )

    return assignment, cost


def min_cost_assignment_via_flow(
    cost_matrix: NDArray[np.float64],
) -> Tuple[NDArray[np.intp], float]:
    """
    Solve 2D assignment problem via min-cost flow network.

    Parameters
    ----------
    cost_matrix : ndarray
        Cost matrix of shape (m, n).

    Returns
    -------
    assignment : ndarray
        Assignment array of shape (n_assignments, 2).
    total_cost : float
        Total assignment cost.
    """
    edges, supplies, _ = assignment_to_flow_network(cost_matrix)
    result = min_cost_flow_successive_shortest_paths(edges, supplies)
    assignment, cost = assignment_from_flow_solution(
        result.flow, edges, cost_matrix.shape
    )

    return assignment, cost
