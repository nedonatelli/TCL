"""``min_cost_flow_dijkstra_potentials`` against a linear-programming oracle.

Surfaced by gh-53. This solver lives in a module with no ``__all__``, so the
public-API coverage gate could not see it, and no test named it -- yet it is
live code: ``min_cost_flow_simplex`` in ``network_flow`` delegates to it for
every call. The tests that exist go through that wrapper and assert on the
result object, so the solver was executed but never checked against anything
that knows the right answer.

That matters more than usual here. Its sibling in the same package,
``min_cost_flow_cost_scaling``, is a second min-cost-flow implementation and
gh-18 records it as incorrect. Two solvers, one known wrong, neither compared
against an independent optimum.

The oracle is `scipy.optimize.linprog`. Min-cost flow *is* a linear program --
minimize ``c.x`` subject to flow conservation ``A x = supplies`` and capacity
bounds ``0 <= x <= cap`` -- so the LP optimum is the exact answer, not an
approximation to it. Comparing total cost rather than the flow vector is
deliberate: a network can have several distinct optimal flows, and the cost is
the part that is uniquely determined.

Note the name: ``min_cost_flow_simplex`` does not use the simplex method. It is
successive shortest paths with Johnson potentials, which is what this module
implements. The name is misleading but out of scope here.
"""

from typing import NamedTuple

import numpy as np
import pytest

from pytcl.assignment_algorithms.dijkstra_min_cost import (
    min_cost_flow_dijkstra_potentials,
)

pytest.importorskip("scipy.optimize", reason="scipy provides the LP oracle")


def _linear_program_optimum(
    n_nodes: int,
    edges: list[tuple[int, int, float, float]],
    supplies: np.ndarray,
) -> float:
    """The exact minimum cost, from a general-purpose LP solver.

    Flow conservation is written as ``A x = supplies`` with ``A[i, j] = +1``
    when edge ``j`` leaves node ``i`` and ``-1`` when it enters, matching the
    sign convention the solver under test uses: a positive supply is a source.
    """
    from scipy.optimize import linprog

    n_edges = len(edges)
    conservation = np.zeros((n_nodes, n_edges))
    costs = np.zeros(n_edges)
    capacities = np.zeros(n_edges)

    for index, (source, target, capacity, cost) in enumerate(edges):
        conservation[source, index] += 1.0
        conservation[target, index] -= 1.0
        costs[index] = cost
        capacities[index] = capacity

    solution = linprog(
        costs,
        A_eq=conservation,
        b_eq=supplies,
        bounds=[(0.0, capacity) for capacity in capacities],
        method="highs",
    )
    assert solution.success, f"the LP oracle itself failed: {solution.message}"
    return float(solution.fun)


class Network(NamedTuple):
    """A named min-cost-flow instance, for readable parametrization."""

    label: str
    n_nodes: int
    edges: list[tuple[int, int, float, float]]
    supplies: np.ndarray


NETWORKS = [
    Network(
        "single-path",
        3,
        [(0, 1, 10.0, 2.0), (1, 2, 10.0, 3.0), (0, 2, 5.0, 10.0)],
        np.array([8.0, 0.0, -8.0]),
    ),
    Network(
        # Two routes of equal cost: the optimum is degenerate, so only the
        # total cost is well defined. A test asserting a particular flow vector
        # would be asserting an implementation detail.
        "equal-cost-parallel-routes",
        4,
        [(0, 1, 5.0, 1.0), (0, 2, 5.0, 2.0), (1, 3, 5.0, 1.0), (2, 3, 5.0, 1.0)],
        np.array([6.0, 0.0, 0.0, -6.0]),
    ),
    Network(
        # The cheap route cannot carry the whole supply, so the solver has to
        # split across a more expensive one. A greedy shortest-path solver that
        # never revisits its choice gets this wrong.
        "capacity-forces-a-split",
        4,
        [(0, 1, 3.0, 1.0), (0, 2, 10.0, 5.0), (1, 3, 10.0, 1.0), (2, 3, 10.0, 1.0)],
        np.array([7.0, 0.0, 0.0, -7.0]),
    ),
    Network(
        # An assignment problem expressed as a flow, which is what the
        # assignment package uses this solver for.
        "bipartite-assignment",
        6,
        [
            (0, 1, 1.0, 0.0),
            (0, 2, 1.0, 0.0),
            (1, 3, 1.0, 4.0),
            (1, 4, 1.0, 1.0),
            (2, 3, 1.0, 2.0),
            (2, 4, 1.0, 3.0),
            (3, 5, 1.0, 0.0),
            (4, 5, 1.0, 0.0),
        ],
        np.array([2.0, 0.0, 0.0, 0.0, 0.0, -2.0]),
    ),
    Network(
        # Zero supply: the optimum is to ship nothing, at zero cost. A solver
        # that always pushes at least one unit fails here.
        "no-supply",
        3,
        [(0, 1, 5.0, 1.0), (1, 2, 5.0, 1.0)],
        np.array([0.0, 0.0, 0.0]),
    ),
    Network(
        "multiple-sources-and-sinks",
        6,
        [
            (0, 2, 4.0, 1.0),
            (1, 2, 4.0, 2.0),
            (2, 3, 8.0, 1.0),
            (3, 4, 4.0, 1.0),
            (3, 5, 4.0, 3.0),
        ],
        np.array([3.0, 3.0, 0.0, 0.0, -4.0, -2.0]),
    ),
]


class TestAgainstTheLinearProgram:
    """The optimum is a defined quantity; check the solver finds it."""

    @pytest.mark.parametrize("network", NETWORKS, ids=[n.label for n in NETWORKS])
    def test_total_cost_is_the_linear_program_optimum(self, network):
        _, cost, _ = min_cost_flow_dijkstra_potentials(
            network.n_nodes, network.edges, network.supplies
        )
        expected = _linear_program_optimum(
            network.n_nodes, network.edges, network.supplies
        )
        assert cost == pytest.approx(expected, abs=1e-6), (
            f"{network.label}: solver returned cost {cost}, the true optimum is "
            f"{expected}"
        )

    @pytest.mark.parametrize("network", NETWORKS, ids=[n.label for n in NETWORKS])
    def test_the_returned_flow_conserves_supply_at_every_node(self, network):
        """A cheaper-than-optimal cost usually means the flow is infeasible.

        Checking cost alone would accept a solver that quietly failed to route
        all the supply, which is the cheapest possible answer.
        """
        flow, _, _ = min_cost_flow_dijkstra_potentials(
            network.n_nodes, network.edges, network.supplies
        )
        residual = network.supplies.copy()
        for index, (source, target, _, _) in enumerate(network.edges):
            residual[source] -= flow[index]
            residual[target] += flow[index]

        np.testing.assert_allclose(
            residual,
            0.0,
            atol=1e-6,
            err_msg=f"{network.label}: supply is not conserved, so the flow is "
            f"not a valid solution regardless of its cost",
        )

    @pytest.mark.parametrize("network", NETWORKS, ids=[n.label for n in NETWORKS])
    def test_no_edge_exceeds_its_capacity_or_runs_backwards(self, network):
        flow, _, _ = min_cost_flow_dijkstra_potentials(
            network.n_nodes, network.edges, network.supplies
        )
        for index, (source, target, capacity, _) in enumerate(network.edges):
            assert -1e-9 <= flow[index] <= capacity + 1e-9, (
                f"{network.label}: edge {index} ({source}->{target}) carries "
                f"{flow[index]}, outside [0, {capacity}]"
            )

    @pytest.mark.parametrize("network", NETWORKS, ids=[n.label for n in NETWORKS])
    def test_the_reported_cost_matches_the_returned_flow(self, network):
        """The scalar and the vector must describe the same solution.

        They are computed separately, so a solver could return an optimal cost
        alongside a flow that does not produce it.
        """
        flow, cost, _ = min_cost_flow_dijkstra_potentials(
            network.n_nodes, network.edges, network.supplies
        )
        recomputed = sum(
            flow[index] * edge_cost
            for index, (_, _, _, edge_cost) in enumerate(network.edges)
        )
        assert cost == pytest.approx(recomputed, abs=1e-6)


class TestRandomizedAgainstTheLinearProgram:
    """Hand-built cases test what their author thought of. These do not."""

    @pytest.mark.parametrize("seed", range(12))
    def test_a_random_feasible_network_reaches_the_optimum(self, seed):
        """Random layered networks, always feasible by construction.

        The layered shape guarantees a path from every source to the sink, so a
        failure here is a wrong answer rather than an infeasible instance.
        """
        rng = np.random.default_rng(seed)
        width = int(rng.integers(2, 5))
        n_nodes = 2 + 2 * width

        edges = []
        for node in range(1, width + 1):  # source -> first layer
            edges.append(
                (0, node, float(rng.integers(2, 6)), float(rng.integers(1, 9)))
            )
        for left in range(1, width + 1):  # first layer -> second layer
            for right in range(width + 1, 2 * width + 1):
                edges.append(
                    (left, right, float(rng.integers(2, 6)), float(rng.integers(1, 9)))
                )
        for node in range(width + 1, 2 * width + 1):  # second layer -> sink
            edges.append(
                (
                    node,
                    n_nodes - 1,
                    float(rng.integers(2, 6)),
                    float(rng.integers(1, 9)),
                )
            )

        supply = float(rng.integers(1, width * 2 + 1))
        supplies = np.zeros(n_nodes)
        supplies[0], supplies[-1] = supply, -supply

        _, cost, _ = min_cost_flow_dijkstra_potentials(n_nodes, edges, supplies)
        expected = _linear_program_optimum(n_nodes, edges, supplies)
        assert cost == pytest.approx(expected, abs=1e-6), (
            f"seed {seed}: solver returned {cost}, LP optimum is {expected}"
        )


def test_the_public_wrapper_agrees_with_the_solver_it_delegates_to():
    """``min_cost_flow_simplex`` is the exported entry point.

    It converts edge objects to tuples and hands off to this solver, so the two
    must report the same cost. A conversion that lost an edge or transposed
    capacity and cost would show up only here.
    """
    from pytcl.assignment_algorithms.network_flow import (
        FlowEdge,
        min_cost_flow_simplex,
    )

    network = NETWORKS[2]  # capacity-forces-a-split
    flow_edges = [
        FlowEdge(from_node=source, to_node=target, capacity=capacity, cost=cost)
        for source, target, capacity, cost in network.edges
    ]

    wrapped = min_cost_flow_simplex(flow_edges, network.supplies)
    _, direct_cost, _ = min_cost_flow_dijkstra_potentials(
        network.n_nodes, network.edges, network.supplies
    )

    assert wrapped.cost == pytest.approx(direct_cost, abs=1e-9)
    assert wrapped.cost == pytest.approx(
        _linear_program_optimum(network.n_nodes, network.edges, network.supplies),
        abs=1e-6,
    )
