"""
Dijkstra-based minimum cost flow using potentials (Johnson's algorithm).

This implements the successive shortest paths algorithm using Dijkstra's algorithm
instead of Bellman-Ford, which is much faster when costs can be non-negative
after potential adjustments.

Algorithm:
1. Maintain node potentials that preserve optimality
2. Use potentials to ensure all edge costs are non-negative
3. Run Dijkstra (O(E log V)) instead of Bellman-Ford (O(VE))
4. Update potentials after each shortest path

Time complexity: O(K * E log V) where K is number of shortest paths needed
Space complexity: O(V + E)

This is based on:
- Johnson's algorithm for all-pairs shortest paths
- Successive shortest paths with potentials
- Published in: "Efficient Implementation of the Bellman-Ford Algorithm"
"""

import heapq
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray


def min_cost_flow_dijkstra_potentials(
    n_nodes: int,
    edges: list[tuple[int, int, float, float]],
    supplies: NDArray[np.float64],
    max_iterations: int = 1000,
) -> tuple[NDArray[np.float64], float, int]:
    """
    Solve min-cost flow using Dijkstra with potentials.

    Uses Johnson's method to maintain non-negative reduced costs,
    allowing efficient Dijkstra instead of Bellman-Ford.

    Parameters
    ----------
    n_nodes : int
        Number of nodes
    edges : list of tuple
        Each tuple is (from_node, to_node, capacity, cost)
    supplies : ndarray
        Supply/demand for each node
    max_iterations : int
        Maximum iterations

    Returns
    -------
    flow : ndarray
        Flow on each edge
    total_cost : float
        Total cost
    iterations : int
        Iterations used
    """
    # Build edge structures
    graph: list[list[int]] = [[] for _ in range(n_nodes)]
    edge_data: list[dict[str, Any]] = []

    for idx, (u, v, cap, cost) in enumerate(edges):
        edge_data.append(
            {
                "from": u,
                "to": v,
                "capacity": cap,
                "cost": float(cost),
                "flow": 0.0,
            }
        )
        graph[u].append(idx)

    # Initialize potentials to zero
    potential = np.zeros(n_nodes)

    # Single Bellman-Ford pass to initialize potentials
    # This ensures all reduced costs are non-negative at start
    for _ in range(n_nodes - 1):
        for u in range(n_nodes):
            for edge_idx in graph[u]:
                e = edge_data[edge_idx]
                v = e["to"]
                if e["flow"] < e["capacity"] - 1e-10:
                    reduced = e["cost"] + potential[u] - potential[v]
                    if reduced < -1e-10:
                        potential[v] = potential[u] + e["cost"]

    # Main loop
    current_supplies = supplies.copy()
    iteration = 0

    for iteration in range(max_iterations):
        # Find source (excess) and sink (deficit) nodes
        source = -1
        sink = -1

        for node in range(n_nodes):
            if current_supplies[node] > 1e-10 and source == -1:
                source = node
            if current_supplies[node] < -1e-10 and sink == -1:
                sink = node

        if source == -1 or sink == -1:
            break

        # Dijkstra with potentials
        dist = np.full(n_nodes, np.inf)
        dist[source] = 0.0
        parent = np.full(n_nodes, -1, dtype=int)
        parent_edge = np.full(n_nodes, -1, dtype=int)

        pq = [(0.0, source)]
        visited = set()

        while pq:
            d, u = heapq.heappop(pq)

            if u in visited:
                continue
            visited.add(u)

            if d > dist[u] + 1e-10:
                continue

            for edge_idx in graph[u]:
                e = edge_data[edge_idx]
                v = e["to"]

                if e["flow"] < e["capacity"] - 1e-10:
                    # Reduced cost using potentials
                    reduced = e["cost"] + potential[u] - potential[v]
                    new_dist = dist[u] + reduced

                    if new_dist < dist[v] - 1e-10:
                        dist[v] = new_dist
                        parent[v] = u
                        parent_edge[v] = edge_idx
                        heapq.heappush(pq, (new_dist, v))

        if dist[sink] == np.inf:
            break

        # Extract path
        path_edges = []
        node = sink
        while parent[node] != -1:
            path_edges.append(parent_edge[node])
            node = parent[node]
        path_edges.reverse()

        # Find bottleneck
        min_flow = min(
            edge_data[e]["capacity"] - edge_data[e]["flow"] for e in path_edges
        )
        min_flow = min(
            min_flow,
            current_supplies[source],
            -current_supplies[sink],
        )

        # Push flow
        for edge_idx in path_edges:
            edge_data[edge_idx]["flow"] += min_flow

        current_supplies[source] -= min_flow
        current_supplies[sink] += min_flow

        # Update potentials for next iteration
        # New potential = old potential + distance from Dijkstra
        for node in range(n_nodes):
            if dist[node] < np.inf:
                potential[node] += dist[node]

    # Extract solution
    result_flow = np.array([e["flow"] for e in edge_data])
    total_cost = sum(e["flow"] * e["cost"] for e in edge_data)

    return result_flow, total_cost, iteration + 1
