"""Correctness audit tests for assignment_algorithms and static_estimation.

Every test validates a public function against an independent reference:
scipy, brute-force enumeration, closed-form algebra, or hand-computed
posteriors. Written as part of the v2 correctness audit.
"""

import itertools

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.optimize import linear_sum_assignment as scipy_lsa
from scipy.special import gamma as gamma_func
from scipy.stats import chi2, norm

from pytcl.assignment_algorithms.data_association import (
    compute_association_cost,
    gated_gnn_association,
    gnn_association,
    nearest_neighbor,
)
from pytcl.assignment_algorithms.gating import (
    chi2_gate_threshold,
    compute_gate_volume,
    ellipsoidal_gate,
    gate_measurements,
    mahalanobis_batch,
    mahalanobis_distance,
    rectangular_gate,
)
from pytcl.assignment_algorithms.jpda import (
    compute_likelihood_matrix,
    compute_measurement_likelihood,
    jpda,
    jpda_probabilities,
    jpda_update,
)
from pytcl.assignment_algorithms.nd_assignment import (
    SparseCostTensor,
    assignment_nd,
    auction_assignment_nd,
    detect_dimension_conflicts,
    greedy_assignment_nd,
    greedy_assignment_nd_sparse,
    relaxation_assignment_nd,
    validate_cost_tensor,
)
from pytcl.assignment_algorithms.network_flow import (
    FlowStatus,
    assignment_to_flow_network,
    min_cost_assignment_via_flow,
    min_cost_flow_successive_shortest_paths,
)
from pytcl.assignment_algorithms.three_dimensional.assignment import (
    assign3d,
    assign3d_auction,
    assign3d_lagrangian,
    decompose_to_2d,
    greedy_3d,
)
from pytcl.assignment_algorithms.two_dimensional.assignment import (
    assign2d,
    auction,
    hungarian,
    linear_sum_assignment,
)
from pytcl.assignment_algorithms.two_dimensional.kbest import (
    kbest_assign2d,
    murty,
    ranked_assignments,
)
from pytcl.static_estimation.least_squares import (
    generalized_least_squares,
    ordinary_least_squares,
    recursive_least_squares,
    ridge_regression,
    total_least_squares,
    weighted_least_squares,
)
from pytcl.static_estimation.maximum_likelihood import (
    aic,
    aicc,
    bic,
    cramer_rao_bound,
    cramer_rao_bound_biased,
    efficiency,
    fisher_information_exponential_family,
    fisher_information_gaussian,
    fisher_information_numerical,
    mle_gaussian,
    mle_newton_raphson,
    mle_scoring,
    observed_fisher_information,
)
from pytcl.static_estimation.robust import (
    huber_regression,
    huber_rho,
    huber_weight,
    irls,
    mad,
    ransac,
    ransac_n_trials,
    tau_scale,
    tukey_regression,
    tukey_rho,
    tukey_weight,
)

# =============================================================================
# Brute-force references
# =============================================================================


def _brute_kbest_costs(C, k):
    """All full-assignment costs of a rectangular matrix, sorted ascending."""
    n, m = C.shape
    costs = []
    if n <= m:
        for perm in itertools.permutations(range(m), n):
            costs.append(C[np.arange(n), list(perm)].sum())
    else:
        for perm in itertools.permutations(range(n), m):
            costs.append(C[list(perm), np.arange(m)].sum())
    return np.sort(costs)[:k]


def _brute_assign2d_cost(C, cna):
    """Optimal cost with finite cost of non-assignment, by subset enumeration."""
    n, m = C.shape
    best = np.inf
    for k in range(0, min(n, m) + 1):
        for rows in itertools.combinations(range(n), k):
            for cols in itertools.permutations(range(m), k):
                c = sum(C[r, cc] for r, cc in zip(rows, cols))
                c += (n - k + m - k) * cna
                best = min(best, c)
    return best


def _brute_3d_optimal(C):
    """Optimal full axial 3D assignment cost for a cubic tensor."""
    n = C.shape[0]
    best = np.inf
    for pj in itertools.permutations(range(n)):
        for pk in itertools.permutations(range(n)):
            best = min(best, sum(C[i, pj[i], pk[i]] for i in range(n)))
    return best


def _assert_3d_feasible(result, C):
    """Assert a 3D result satisfies per-dimension uniqueness and its cost."""
    t = result.tuples
    for d in range(3):
        assert len(np.unique(t[:, d])) == len(t), f"dimension {d} reused"
    recomputed = sum(C[i, j, k] for i, j, k in t)
    assert_allclose(result.cost, recomputed, atol=1e-9)


# =============================================================================
# 2D assignment
# =============================================================================


class TestHungarian2D:
    def test_vs_scipy_200_random(self):
        """hungarian/linear_sum_assignment/assign2d(inf) match scipy exactly
        on 200 seeded matrices: square, rectangular, ties, negative costs."""
        rng = np.random.default_rng(42)
        for trial in range(200):
            n = int(rng.integers(2, 31))
            m = int(rng.integers(2, 31))
            if trial % 3 == 0:
                C = rng.integers(-10, 10, size=(n, m)).astype(float)  # ties
            else:
                C = rng.uniform(-50, 50, size=(n, m))
            ri, ci = scipy_lsa(C)
            opt = C[ri, ci].sum()

            r, c, cost = hungarian(C)
            assert_allclose(cost, opt, atol=1e-9)
            assert len(r) == min(n, m)
            assert len(np.unique(r)) == len(r) and len(np.unique(c)) == len(c)

            r2, c2 = linear_sum_assignment(C)
            assert_allclose(C[r2, c2].sum(), opt, atol=1e-9)

            res = assign2d(C)
            assert_allclose(res.cost, opt, atol=1e-9)

    def test_hungarian_maximize(self):
        rng = np.random.default_rng(1)
        for _ in range(20):
            C = rng.uniform(-5, 5, size=(4, 6))
            ri, ci = scipy_lsa(C, maximize=True)
            r, c, cost = hungarian(C, maximize=True)
            assert_allclose(cost, C[ri, ci].sum(), atol=1e-9)


class TestAuction2D:
    def test_epsilon_optimality_bound(self):
        """Auction is epsilon-optimal: cost <= optimal + n*epsilon."""
        rng = np.random.default_rng(7)
        for _ in range(100):
            n = int(rng.integers(2, 12))
            m = int(rng.integers(n, 13))
            C = rng.uniform(-5, 10, size=(n, m))
            eps = 1.0 / (n + 1)
            r, c, cost = auction(C, epsilon=eps, max_iter=100000)
            ri, ci = scipy_lsa(C)
            assert len(r) == n, "auction left rows unassigned"
            assert len(np.unique(c)) == len(c)
            assert cost >= C[ri, ci].sum() - 1e-9
            assert cost <= C[ri, ci].sum() + n * eps + 1e-9

    def test_exact_for_integer_costs(self):
        """With integer costs and epsilon < 1/n the auction is exact."""
        rng = np.random.default_rng(8)
        for _ in range(50):
            n = int(rng.integers(2, 10))
            C = rng.integers(0, 50, size=(n, n)).astype(float)
            r, c, cost = auction(C, epsilon=0.9 / (n + 1), max_iter=1000000)
            ri, ci = scipy_lsa(C)
            assert_allclose(cost, C[ri, ci].sum(), atol=1e-9)

    def test_maximize_matches_negated_min(self):
        rng = np.random.default_rng(9)
        C = rng.integers(0, 20, size=(5, 5)).astype(float)
        _, _, cost = auction(C, epsilon=0.9 / 6, maximize=True, max_iter=100000)
        ri, ci = scipy_lsa(C, maximize=True)
        assert_allclose(cost, C[ri, ci].sum(), atol=1e-9)


class TestAssign2DNonAssignment:
    def test_finite_cost_vs_brute_force(self):
        """assign2d with finite cost_of_non_assignment matches subset
        enumeration."""
        rng = np.random.default_rng(3)
        for _ in range(60):
            n = int(rng.integers(1, 4))
            m = int(rng.integers(1, 4))
            C = np.round(rng.uniform(0, 10, size=(n, m)), 2)
            cna = float(np.round(rng.uniform(0.5, 6), 2))
            res = assign2d(C, cost_of_non_assignment=cna)
            assert_allclose(res.cost, _brute_assign2d_cost(C, cna), atol=1e-9)
            # bookkeeping consistency
            assert len(res.row_indices) + len(res.unassigned_rows) == n
            assert len(res.col_indices) + len(res.unassigned_cols) == m


# =============================================================================
# Murty k-best
# =============================================================================


class TestMurtyKBest:
    def test_vs_brute_force_square_and_rectangular(self):
        """k best costs must match the sorted enumeration exactly."""
        rng = np.random.default_rng(0)
        for trial in range(120):
            n = int(rng.integers(2, 5))
            m = int(rng.integers(2, 6))
            C = np.round(rng.uniform(-5, 10, size=(n, m)), 3)
            k = 6
            ref = _brute_kbest_costs(C, k)
            res = murty(C, k=k)
            assert res.n_found == min(k, len(ref))
            assert np.all(np.diff(res.costs) >= -1e-12), "costs not sorted"
            assert_allclose(res.costs, ref[: res.n_found], atol=1e-9)

    def test_assignments_are_valid_and_distinct(self):
        rng = np.random.default_rng(5)
        C = rng.uniform(0, 10, size=(4, 4))
        res = murty(C, k=10)
        seen = set()
        for a in res.assignments:
            key = tuple(zip(a.row_indices.tolist(), a.col_indices.tolist()))
            assert key not in seen, "duplicate assignment returned"
            seen.add(key)
            assert_allclose(C[a.row_indices, a.col_indices].sum(), a.cost)

    def test_maximize_returns_positive_descending_costs(self):
        C = np.array([[10.0, 5, 13], [3, 15, 8], [12, 7, 9]])
        ref = sorted(
            (C[np.arange(3), list(p)].sum() for p in itertools.permutations(range(3))),
            reverse=True,
        )[:3]
        res = murty(C, k=3, maximize=True)
        assert_allclose(res.costs, ref, atol=1e-9)

    def test_kbest_assign2d_vs_brute(self):
        rng = np.random.default_rng(2)
        for _ in range(40):
            n = int(rng.integers(2, 5))
            m = int(rng.integers(n, 6))
            C = np.round(rng.uniform(0, 10, size=(n, m)), 3)
            ref = _brute_kbest_costs(C, 4)
            res = kbest_assign2d(C, k=4)
            assert_allclose(res.costs, ref[: res.n_found], atol=1e-9)

    def test_kbest_threshold(self):
        C = np.array([[10.0, 5, 13], [3, 15, 8], [12, 7, 9]])
        ref = _brute_kbest_costs(C, 6)
        thresh = float(ref[2]) + 0.5
        res = kbest_assign2d(C, k=10, cost_threshold=thresh)
        assert res.n_found == int(np.sum(ref <= thresh))
        assert np.all(res.costs <= thresh)

    def test_ranked_assignments_enumerates_all(self):
        C = np.array([[10.0, 5], [3, 15]])
        res = ranked_assignments(C, max_assignments=5)
        assert res.n_found == 2
        assert_allclose(res.costs, _brute_kbest_costs(C, 2), atol=1e-9)


# =============================================================================
# Network-flow assignment wrappers
# =============================================================================


class TestFlowAssignment:
    def test_via_flow_vs_scipy_both_algorithms(self):
        rng = np.random.default_rng(12)
        for use_simplex in (True, False):
            for _ in range(30):
                n = int(rng.integers(2, 8))
                C = np.round(rng.uniform(0, 10, size=(n, n)), 3)
                ri, ci = scipy_lsa(C)
                assignment, cost = min_cost_assignment_via_flow(
                    C, use_simplex=use_simplex
                )
                assert len(assignment) == n
                assert_allclose(cost, C[ri, ci].sum(), atol=1e-6)

    def test_via_flow_adversarial_requires_flow_cancellation(self):
        """Greedy shortest paths without residual arcs fail on this matrix."""
        C = np.array([[1.0, 2.0], [1.0, 100.0]])
        for use_simplex in (True, False):
            _, cost = min_cost_assignment_via_flow(C, use_simplex=use_simplex)
            assert_allclose(cost, 3.0, atol=1e-9)

    def test_assignment_to_flow_network_structure(self):
        C = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        edges, supplies, names = assignment_to_flow_network(C)
        m, n = C.shape
        assert len(edges) == m + m * n + n
        assert supplies[0] == m and supplies[-1] == -m
        assert np.sum(supplies) == 0
        assert names[0] == "source" and names[-1] == "sink"

    def test_ssp_wrapper_status_and_cost(self):
        rng = np.random.default_rng(13)
        C = np.round(rng.uniform(0, 9, size=(4, 4)), 2)
        edges, supplies, _ = assignment_to_flow_network(C)
        result = min_cost_flow_successive_shortest_paths(edges, supplies)
        ri, ci = scipy_lsa(C)
        assert result.status == FlowStatus.OPTIMAL
        assert_allclose(result.cost, C[ri, ci].sum(), atol=1e-6)


# =============================================================================
# 3D assignment
# =============================================================================


class Test3DAssignment:
    def test_heuristics_feasible_and_bounded_below_by_optimum(self):
        rng = np.random.default_rng(21)
        for _ in range(15):
            C = rng.uniform(0, 10, size=(3, 3, 3))
            opt = _brute_3d_optimal(C)
            for fn in (greedy_3d, decompose_to_2d, assign3d_auction):
                res = fn(C)
                _assert_3d_feasible(res, C)
                if len(res.tuples) == 3:
                    assert res.cost >= opt - 1e-9

    def test_lagrangian_feasible_and_bounded(self):
        rng = np.random.default_rng(22)
        for _ in range(15):
            C = rng.uniform(0, 10, size=(3, 3, 3))
            opt = _brute_3d_optimal(C)
            res = assign3d_lagrangian(C, max_iter=100)
            _assert_3d_feasible(res, C)
            if len(res.tuples) == 3:
                assert res.cost >= opt - 1e-9

    def test_lagrangian_finds_obvious_diagonal(self):
        C = np.full((3, 3, 3), 100.0)
        for i in range(3):
            C[i, i, i] = 1.0
        res = assign3d_lagrangian(C, max_iter=100)
        assert_allclose(res.cost, 3.0, atol=1e-9)

    def test_auction_feasible_on_structured_conflict_case(self):
        """Cost tensor that used to trigger j/k reuse in assign3d_auction."""
        rng = np.random.default_rng(23)
        for _ in range(20):
            C = rng.uniform(0, 10, size=(4, 4, 4))
            res = assign3d_auction(C, max_iter=500)
            _assert_3d_feasible(res, C)

    def test_decompose_all_fixed_dimensions(self):
        rng = np.random.default_rng(24)
        C = rng.uniform(0, 10, size=(3, 4, 5))
        for d in (0, 1, 2):
            res = decompose_to_2d(C, fixed_dimension=d)
            _assert_3d_feasible(res, C)
            assert np.all(res.tuples[:, 0] < 3)
            assert np.all(res.tuples[:, 1] < 4)
            assert np.all(res.tuples[:, 2] < 5)

    def test_assign3d_dispatch(self):
        rng = np.random.default_rng(25)
        C = rng.uniform(0, 10, size=(3, 3, 3))
        for method in ("lagrangian", "auction", "greedy", "decompose"):
            res = assign3d(C, method=method)
            _assert_3d_feasible(res, C)
        with pytest.raises(ValueError, match="Unknown method"):
            assign3d(C, method="nope")

    def test_greedy_3d_maximize(self):
        rng = np.random.default_rng(26)
        C = rng.uniform(0, 10, size=(3, 3, 3))
        res_max = greedy_3d(C, maximize=True)
        _assert_3d_feasible(res_max, C)
        assert res_max.cost >= greedy_3d(C).cost


# =============================================================================
# N-D assignment
# =============================================================================


class TestNDAssignment:
    def test_validate_cost_tensor(self):
        assert validate_cost_tensor(np.zeros((3, 4, 5))) == (3, 4, 5)
        with pytest.raises(ValueError):
            validate_cost_tensor(np.zeros(3))

    def test_greedy_nd_2d_is_feasible_upper_bound(self):
        """Greedy on a matrix is feasible and >= the scipy optimum."""
        rng = np.random.default_rng(31)
        for _ in range(30):
            n = int(rng.integers(2, 7))
            C = rng.uniform(0, 10, size=(n, n))
            res = greedy_assignment_nd(C)
            assert not detect_dimension_conflicts(res.assignments, C.shape)
            recomputed = C[tuple(res.assignments.T)].sum()
            assert_allclose(res.cost, recomputed, atol=1e-9)
            ri, ci = scipy_lsa(C)
            assert res.cost >= C[ri, ci].sum() - 1e-9

    def test_greedy_nd_known_tensor(self):
        cost = np.array(
            [
                [[1.0, 5.0], [3.0, 2.0]],
                [[4.0, 1.0], [2.0, 6.0]],
                [[2.0, 3.0], [5.0, 1.0]],
            ]
        )
        res = greedy_assignment_nd(cost)
        # Greedy picks (0,0,0)=1 first, then cheapest disjoint (2,1,1)=1
        assert_allclose(res.cost, 2.0)
        assert len(res.assignments) == 2

    def test_relaxation_nd_feasible_upper_bound(self):
        # NOTE: relaxation_assignment_nd's "lower bound" is not a valid
        # Lagrangian bound (greedy is not the relaxed minimizer), so its
        # converged/gap certificate can be wrong; here we validate only
        # feasibility and that its cost is a true upper bound.
        rng = np.random.default_rng(32)
        for _ in range(10):
            C = rng.uniform(0, 10, size=(3, 3, 3))
            opt = _brute_3d_optimal(C)
            res = relaxation_assignment_nd(C, max_iterations=50)
            assert not detect_dimension_conflicts(res.assignments, C.shape)
            recomputed = C[tuple(res.assignments.T)].sum()
            assert_allclose(res.cost, recomputed, atol=1e-9)
            if len(res.assignments) == 3:
                assert res.cost >= opt - 1e-9

    def test_auction_nd_feasible_and_no_worse_than_greedy(self):
        rng = np.random.default_rng(33)
        for _ in range(10):
            C = rng.uniform(0, 10, size=(4, 4, 4))
            res = auction_assignment_nd(C, max_iterations=50, epsilon=0.1)
            assert not detect_dimension_conflicts(res.assignments, C.shape)
            recomputed = C[tuple(res.assignments.T)].sum()
            assert_allclose(res.cost, recomputed, atol=1e-9)
            greedy = greedy_assignment_nd(C)
            assert res.cost <= greedy.cost + 1e-9

    def test_detect_dimension_conflicts_truth_table(self):
        assert not detect_dimension_conflicts(np.array([[0, 0], [1, 1]]), (3, 3))
        assert detect_dimension_conflicts(np.array([[0, 0], [0, 1]]), (3, 3))
        assert detect_dimension_conflicts(np.array([[0, 1], [1, 1]]), (3, 3))

    def test_sparse_matches_dense_greedy(self):
        rng = np.random.default_rng(34)
        dense = np.full((6, 6, 6), np.inf)
        idx = rng.integers(0, 6, size=(40, 3))
        for row in idx:
            dense[tuple(row)] = rng.uniform(0, 10)
        sparse = SparseCostTensor.from_dense(dense)
        res_sparse = greedy_assignment_nd_sparse(sparse)
        res_dense = greedy_assignment_nd(dense)
        # dense greedy may pick inf entries only after all finite exhausted;
        # compare finite-cost solutions
        assert_allclose(res_sparse.cost, res_dense.cost, atol=1e-9)
        assert not detect_dimension_conflicts(res_sparse.assignments, dense.shape)

    def test_sparse_round_trip(self):
        dense = np.full((3, 3), np.inf)
        dense[0, 1] = 2.0
        dense[2, 0] = 5.0
        sparse = SparseCostTensor.from_dense(dense)
        assert sparse.n_valid == 2
        assert sparse.get_cost((0, 1)) == 2.0
        assert sparse.get_cost((1, 1)) == np.inf
        assert_allclose(sparse.to_dense(), dense)

    def test_assignment_nd_dispatch(self):
        rng = np.random.default_rng(35)
        C = rng.uniform(0, 10, size=(3, 3, 3))
        for method in ("auto", "greedy", "relaxation", "auction"):
            res = assignment_nd(C, method=method)
            assert not detect_dimension_conflicts(res.assignments, C.shape)
        with pytest.raises(ValueError):
            assignment_nd(C, method="bogus")


# =============================================================================
# Gating
# =============================================================================


class TestGating:
    def test_mahalanobis_vs_manual(self):
        rng = np.random.default_rng(41)
        for dim in (1, 2, 3, 5):
            M = rng.normal(size=(dim, dim))
            S = M @ M.T + dim * np.eye(dim)
            nu = rng.normal(size=dim)
            expected = float(nu @ np.linalg.inv(S) @ nu)
            assert_allclose(mahalanobis_distance(nu, S), expected, rtol=1e-10)

    def test_mahalanobis_batch_vs_loop(self):
        rng = np.random.default_rng(42)
        S = np.array([[2.0, 0.3], [0.3, 1.0]])
        innovations = rng.normal(size=(10, 2))
        out = np.zeros(10)
        mahalanobis_batch(innovations, np.linalg.inv(S), out)
        expected = [mahalanobis_distance(v, S) for v in innovations]
        assert_allclose(out, expected, rtol=1e-10)

    def test_chi2_threshold_matches_scipy(self):
        for p, df in [(0.95, 1), (0.99, 2), (0.99, 3), (0.999, 4)]:
            assert_allclose(chi2_gate_threshold(p, df), chi2.ppf(p, df))

    def test_gate_probability_monte_carlo(self):
        """Fraction of true measurements inside the ellipsoidal gate matches
        the chi-squared gate probability."""
        rng = np.random.default_rng(43)
        S = np.array([[2.0, 0.3, 0.0], [0.3, 1.0, 0.2], [0.0, 0.2, 1.5]])
        L = np.linalg.cholesky(S)
        for p in (0.9, 0.99):
            thr = chi2_gate_threshold(p, 3)
            samples = (L @ rng.normal(size=(3, 20000))).T
            inside = np.mean(
                [ellipsoidal_gate(s, S, gate_threshold=thr) for s in samples]
            )
            assert abs(inside - p) < 0.01

    def test_rectangular_gate_manual(self):
        S = np.array([[4.0, 0.0], [0.0, 1.0]])  # sigmas: 2, 1
        assert rectangular_gate(np.array([5.9, 2.9]), S, num_sigmas=3.0)
        assert not rectangular_gate(np.array([6.1, 0.0]), S, num_sigmas=3.0)
        assert not rectangular_gate(np.array([0.0, 3.1]), S, num_sigmas=3.0)

    def test_gate_measurements_indices_and_distances(self):
        z_pred = np.zeros(2)
        S = np.eye(2)
        Z = np.array([[0.5, 0.5], [5.0, 5.0], [1.0, -1.0]])
        idx, d2 = gate_measurements(z_pred, S, Z, gate_threshold=9.21)
        assert list(idx) == [0, 2]
        assert_allclose(d2, [0.5, 2.0])
        with pytest.raises(ValueError):
            gate_measurements(z_pred, S, Z, 9.21, gate_type="octagon")

    def test_gate_volume_analytic(self):
        """V = c_m * gamma^{m/2} * sqrt(det S); checked against the closed
        forms in 1D (2*sqrt(g*S)) and 2D (pi*g*sqrt(det S)) plus Monte Carlo."""
        g = 9.21
        S1 = np.array([[4.0]])
        assert_allclose(compute_gate_volume(S1, g), 2 * np.sqrt(g * 4.0), rtol=1e-12)

        S2 = np.array([[4.0, 1.0], [1.0, 2.0]])
        expected2 = np.pi * g * np.sqrt(np.linalg.det(S2))
        assert_allclose(compute_gate_volume(S2, g), expected2, rtol=1e-12)

        # 3D against the unit-sphere-volume formula
        S3 = np.diag([2.0, 1.0, 1.5])
        c3 = np.pi**1.5 / gamma_func(2.5)
        expected3 = c3 * g**1.5 * np.sqrt(np.linalg.det(S3))
        assert_allclose(compute_gate_volume(S3, g), expected3, rtol=1e-12)

        # Monte Carlo cross-check in 2D
        rng = np.random.default_rng(44)
        lo, hi = -12.0, 12.0
        pts = rng.uniform(lo, hi, size=(100000, 2))
        d2 = np.einsum("ij,ij->i", pts @ np.linalg.inv(S2), pts)
        mc_vol = np.mean(d2 <= g) * (hi - lo) ** 2
        assert abs(mc_vol - expected2) / expected2 < 0.03


# =============================================================================
# JPDA
# =============================================================================


class TestJPDA:
    def test_exact_probabilities_hand_enumerated_2x2(self):
        """2 tracks / 2 measurements against explicit hypothesis enumeration.

        Hypotheses (assignment of measurements to origin, tracks distinct):
          (C,C):    lam^2 (1-Pd)^2
          (T0,C):   Pd*g00 * lam * (1-Pd)
          (T1,C):   Pd*g10 * lam * (1-Pd)
          (C,T0):   lam * Pd*g01 * (1-Pd)
          (C,T1):   lam * Pd*g11 * (1-Pd)
          (T0,T1):  Pd*g00 * Pd*g11
          (T1,T0):  Pd*g10 * Pd*g01
        beta[i,j] = sum of normalized probs of hypotheses with meas j from
        track i; beta[i,2] = sum of hypotheses where track i is undetected.
        """
        g = np.array([[0.5, 0.1], [0.2, 0.6]])
        Pd = 0.8
        lam = 0.1
        gated = np.ones((2, 2), dtype=bool)

        q = 1 - Pd
        h = {
            "cc": lam * lam * q * q,
            "t0c": Pd * g[0, 0] * lam * q,
            "t1c": Pd * g[1, 0] * lam * q,
            "ct0": lam * Pd * g[0, 1] * q,
            "ct1": lam * Pd * g[1, 1] * q,
            "t0t1": Pd * g[0, 0] * Pd * g[1, 1],
            "t1t0": Pd * g[1, 0] * Pd * g[0, 1],
        }
        Z = sum(h.values())
        expected = np.array(
            [
                [
                    (h["t0c"] + h["t0t1"]) / Z,
                    (h["ct0"] + h["t1t0"]) / Z,
                    (h["cc"] + h["t1c"] + h["ct1"]) / Z,
                ],
                [
                    (h["t1c"] + h["t1t0"]) / Z,
                    (h["ct1"] + h["t0t1"]) / Z,
                    (h["cc"] + h["t0c"] + h["ct0"]) / Z,
                ],
            ]
        )

        beta = jpda_probabilities(g, gated, detection_prob=Pd, clutter_density=lam)
        assert_allclose(beta, expected, rtol=1e-12)
        assert_allclose(beta.sum(axis=1), [1.0, 1.0], rtol=1e-12)

    def test_full_jpda_single_track_closed_form(self):
        """jpda() end-to-end: 1 track / 1 measurement posterior must equal
        Pd*g / (Pd*g + lam*(1-Pd)) with g the Gaussian innovation density.
        (Guards against double-counting Pd in the likelihood matrix.)"""
        Pd = 0.7
        lam = 0.1
        P = np.eye(2) * 0.5
        H = np.array([[1.0, 0.0]])
        R = np.array([[0.1]])
        res = jpda(
            [np.array([0.0, 0.0])],
            [P],
            np.array([[0.1]]),
            H,
            R,
            detection_prob=Pd,
            clutter_density=lam,
            gate_probability=0.9999,
        )
        g = norm.pdf(0.1, 0.0, np.sqrt(0.6))  # S = HPH' + R = 0.6
        expected = Pd * g / (Pd * g + lam * (1 - Pd))
        assert_allclose(res.association_probs[0, 0], expected, rtol=1e-10)

    def test_probability_rows_sum_to_one(self):
        rng = np.random.default_rng(51)
        # exact path (small) and approximate path (large)
        for n_tracks, n_meas in [(3, 4), (7, 8)]:
            L = rng.uniform(0, 1, size=(n_tracks, n_meas))
            gated = rng.uniform(size=(n_tracks, n_meas)) < 0.8
            beta = jpda_probabilities(
                L * gated, gated, detection_prob=0.9, clutter_density=0.01
            )
            assert beta.shape == (n_tracks, n_meas + 1)
            assert np.all(beta >= -1e-12)
            assert_allclose(beta.sum(axis=1), np.ones(n_tracks), rtol=1e-6)

    def test_no_measurements_all_miss(self):
        beta = jpda_probabilities(np.zeros((2, 0)), np.zeros((2, 0), dtype=bool))
        assert_allclose(beta[:, 0], [1.0, 1.0])

    def test_measurement_likelihood_gaussian_density(self):
        S = np.array([[2.0, 0.3], [0.3, 1.0]])
        nu = np.array([0.4, -0.2])
        expected = np.exp(-0.5 * nu @ np.linalg.inv(S) @ nu) / np.sqrt(
            (2 * np.pi) ** 2 * np.linalg.det(S)
        )
        got = compute_measurement_likelihood(nu, S, detection_prob=1.0)
        assert_allclose(got, expected, rtol=1e-12)

    def test_likelihood_matrix_vs_manual(self):
        states = [np.array([0.0, 1.0]), np.array([5.0, 0.0])]
        covs = [np.eye(2) * 0.5, np.eye(2) * 0.5]
        Z = np.array([[0.1], [5.2]])
        H = np.array([[1.0, 0.0]])
        R = np.array([[0.1]])
        L, gated = compute_likelihood_matrix(states, covs, Z, H, R, 1.0)
        for i in range(2):
            S = H @ covs[i] @ H.T + R
            for j in range(2):
                nu = Z[j] - H @ states[i]
                expected = norm.pdf(nu[0], 0.0, np.sqrt(S[0, 0]))
                assert_allclose(L[i, j], expected, rtol=1e-10)
        assert gated.all()

    def test_jpda_update_covariance_standard_pda_formula(self):
        """P_upd must equal beta0*P + (1-beta0)*Pc + K(sum_j b_j y_j y_j'
        - y y')K' (Bar-Shalom spread of innovations)."""
        x1 = np.array([0.0, 1.0])
        P1 = np.eye(2) * 0.5
        Z = np.array([[0.3], [1.5]])
        H = np.array([[1.0, 0.0]])
        R = np.array([[0.2]])
        res = jpda_update(
            [x1],
            [P1],
            Z,
            H,
            R,
            detection_prob=0.8,
            clutter_density=0.05,
            gate_probability=0.9999,
        )
        beta = res.association_probs[0]
        S = H @ P1 @ H.T + R
        K = P1 @ H.T @ np.linalg.inv(S)
        ys = [Z[j] - H @ x1 for j in range(2)]
        y_comb = beta[0] * ys[0] + beta[1] * ys[1]
        assert_allclose(res.innovations[0], y_comb, rtol=1e-12)
        assert_allclose(res.states[0], x1 + K @ y_comb, rtol=1e-12)
        Pc = P1 - K @ S @ K.T
        spread = (
            K
            @ (
                beta[0] * np.outer(ys[0], ys[0])
                + beta[1] * np.outer(ys[1], ys[1])
                - np.outer(y_comb, y_comb)
            )
            @ K.T
        )
        P_expected = beta[2] * P1 + (1 - beta[2]) * Pc + spread
        assert_allclose(res.covariances[0], P_expected, rtol=1e-10)
        # PSD sanity
        assert np.all(np.linalg.eigvalsh(res.covariances[0]) > 0)


# =============================================================================
# Data association
# =============================================================================


class TestDataAssociation:
    def test_gnn_matches_scipy_optimum(self):
        rng = np.random.default_rng(61)
        for _ in range(30):
            n = int(rng.integers(2, 8))
            m = int(rng.integers(2, 8))
            C = rng.uniform(0, 10, size=(n, m))
            res = gnn_association(C)
            ri, ci = scipy_lsa(C)
            assert_allclose(res.total_cost, C[ri, ci].sum(), atol=1e-9)
            # cross-consistency of the two index maps
            for i, j in enumerate(res.track_to_measurement):
                if j >= 0:
                    assert res.measurement_to_track[j] == i

    def test_nearest_neighbor_feasible_and_dominated_by_gnn(self):
        rng = np.random.default_rng(62)
        for _ in range(30):
            C = rng.uniform(0, 10, size=(5, 5))
            nn = nearest_neighbor(C)
            gnn = gnn_association(C)
            assigned = nn.track_to_measurement[nn.track_to_measurement >= 0]
            assert len(np.unique(assigned)) == len(assigned)
            assert nn.total_cost >= gnn.total_cost - 1e-9

    def test_compute_association_cost_vs_manual(self):
        preds = np.array([[0.0, 1.0], [5.0, -1.0]])
        covs = np.array([np.eye(2), 2 * np.eye(2)])
        Z = np.array([[0.1], [4.9]])
        H = np.array([[1.0, 0.0]])
        C = compute_association_cost(preds, covs, Z, H)
        for i in range(2):
            S = H @ covs[i] @ H.T  # NOTE: no measurement noise R in this API
            for j in range(2):
                nu = Z[j] - H @ preds[i]
                assert_allclose(C[i, j], float(nu @ np.linalg.inv(S) @ nu), rtol=1e-9)

    def test_gated_gnn_associates_obvious_pairs(self):
        preds = np.array([[0.0, 1.0], [5.0, -1.0]])
        covs = np.array([0.1 * np.eye(2), 0.1 * np.eye(2)])
        Z = np.array([[0.1], [4.9]])
        H = np.array([[1.0, 0.0]])
        res = gated_gnn_association(preds, covs, Z, H, gate_probability=0.99)
        assert list(res.track_to_measurement) == [0, 1]


# =============================================================================
# Least squares
# =============================================================================


class TestLeastSquares:
    def test_ols_vs_lstsq(self):
        rng = np.random.default_rng(71)
        for _ in range(20):
            A = rng.normal(size=(12, 4))
            b = rng.normal(size=12)
            res = ordinary_least_squares(A, b)
            x_ref = np.linalg.lstsq(A, b, rcond=None)[0]
            assert_allclose(res.x, x_ref, atol=1e-10)
            assert res.rank == 4

    def test_wls_vs_normal_equations(self):
        rng = np.random.default_rng(72)
        for _ in range(20):
            A = rng.normal(size=(12, 4))
            b = rng.normal(size=12)
            w = rng.uniform(0.1, 3.0, size=12)
            res = weighted_least_squares(A, b, weights=w)
            W = np.diag(w)
            x_ref = np.linalg.solve(A.T @ W @ A, A.T @ W @ b)
            assert_allclose(res.x, x_ref, atol=1e-9)
            assert_allclose(res.covariance, np.linalg.inv(A.T @ W @ A), atol=1e-9)
            r = b - A @ x_ref
            assert_allclose(res.weighted_residual_sum, r @ W @ r, rtol=1e-9)

    def test_wls_full_matrix_and_default_identity(self):
        rng = np.random.default_rng(73)
        A = rng.normal(size=(10, 3))
        b = rng.normal(size=10)
        M = rng.normal(size=(10, 10))
        W = M @ M.T + 10 * np.eye(10)
        res = weighted_least_squares(A, b, W=W)
        x_ref = np.linalg.solve(A.T @ W @ A, A.T @ W @ b)
        assert_allclose(res.x, x_ref, atol=1e-8)
        res_id = weighted_least_squares(A, b)
        assert_allclose(res_id.x, np.linalg.lstsq(A, b, rcond=None)[0], atol=1e-9)

    def test_gls_equals_wls_with_inverse_sigma(self):
        rng = np.random.default_rng(74)
        A = rng.normal(size=(10, 3))
        b = rng.normal(size=10)
        M = rng.normal(size=(10, 10))
        Sigma = M @ M.T + 10 * np.eye(10)
        res = generalized_least_squares(A, b, Sigma)
        Si = np.linalg.inv(Sigma)
        x_ref = np.linalg.solve(A.T @ Si @ A, A.T @ Si @ b)
        assert_allclose(res.x, x_ref, atol=1e-8)

    def test_tls_vs_svd_construction(self):
        rng = np.random.default_rng(75)
        for _ in range(20):
            A = rng.normal(size=(15, 3))
            b = rng.normal(size=15)
            res = total_least_squares(A, b)
            C = np.column_stack([A, b])
            _, _, Vt = np.linalg.svd(C)
            v = Vt[-1]
            assert_allclose(res.x, -v[:3] / v[3], atol=1e-9)
            # corrections satisfy the documented constraint (A+E)x = b+r
            assert_allclose(
                (A + res.residuals_A) @ res.x, b + res.residuals_b, atol=1e-8
            )

    def test_rls_converges_to_batch_ols(self):
        rng = np.random.default_rng(76)
        A = rng.normal(size=(30, 3))
        b = rng.normal(size=30)
        x = np.zeros(3)
        P = np.eye(3) * 1e8
        for i in range(30):
            x, P = recursive_least_squares(x, P, A[i], b[i])
        x_ref = np.linalg.lstsq(A, b, rcond=None)[0]
        assert_allclose(x, x_ref, atol=1e-5)
        assert_allclose(P, np.linalg.inv(A.T @ A), atol=1e-5)

    def test_ridge_closed_form(self):
        rng = np.random.default_rng(77)
        A = rng.normal(size=(10, 4))
        b = rng.normal(size=10)
        alpha = 0.7
        x = ridge_regression(A, b, alpha=alpha)
        x_ref = np.linalg.solve(A.T @ A + alpha * np.eye(4), A.T @ b)
        assert_allclose(x, x_ref, atol=1e-10)
        # alpha -> 0 recovers OLS
        assert_allclose(
            ridge_regression(A, b, alpha=1e-12),
            np.linalg.lstsq(A, b, rcond=None)[0],
            atol=1e-6,
        )


# =============================================================================
# Robust estimation
# =============================================================================


class TestRobust:
    def test_irls_unit_weights_reduces_to_ols(self):
        """IRLS with quadratic rho (constant weights) is exactly OLS."""
        rng = np.random.default_rng(81)
        A = rng.normal(size=(20, 3))
        b = rng.normal(size=20)
        res = irls(A, b, weight_func=lambda r: np.ones_like(r))
        x_ref = np.linalg.lstsq(A, b, rcond=None)[0]
        assert_allclose(res.x, x_ref, atol=1e-8)

    def test_rho_functions_analytic_values(self):
        # Huber: r^2/2 inside, c|r| - c^2/2 outside
        c = 1.345
        assert_allclose(huber_rho(np.array([0.5]), c), [0.125])
        assert_allclose(huber_rho(np.array([3.0]), c), [c * 3 - c**2 / 2])
        # Tukey: saturates at c^2/6
        c = 4.685
        assert_allclose(tukey_rho(np.array([0.0]), c), [0.0])
        assert_allclose(tukey_rho(np.array([10.0]), c), [c**2 / 6])

    def test_rho_gradient_is_weight_times_r(self):
        """psi(r) = rho'(r) = w(r)*r for both Huber and Tukey."""
        r = np.linspace(-6, 6, 4001)
        for rho_f, w_f, c in [
            (huber_rho, huber_weight, 1.345),
            (tukey_rho, tukey_weight, 4.685),
        ]:
            num_grad = np.gradient(rho_f(r, c), r)
            psi = w_f(r, c) * r
            # exclude points adjacent to the kinks at |r| = c
            mask = np.abs(np.abs(r) - c) > 0.01
            assert np.max(np.abs(num_grad[mask] - psi[mask])) < 1e-3

    def test_mad_known_value(self):
        r = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
        # median 3, |r - 3| = [2,1,0,1,97], median 1
        assert_allclose(mad(r, c=1.4826), 1.4826)
        assert_allclose(mad(r, c=1.0), 1.0)

    def test_tau_scale_robust_to_outlier(self):
        r = np.array([1.0, 1.1, 0.9, 1.0, 1.2, 100.0])
        assert 0 < tau_scale(r) < 10.0

    def test_huber_tukey_regression_reject_outlier(self):
        A = np.column_stack([np.ones(8), np.arange(8.0)])
        b = 1.0 + 2.0 * np.arange(8.0)
        b[4] += 50.0
        for fn in (huber_regression, tukey_regression):
            res = fn(A, b)
            assert_allclose(res.x, [1.0, 2.0], atol=0.1)
            assert res.weights[4] < 0.2

    def test_ransac_recovers_planted_model(self):
        rng = np.random.default_rng(82)
        for seed in range(5):
            m = 60
            A = np.column_stack([np.ones(m), rng.uniform(-5, 5, m)])
            x_true = np.array([1.5, -2.0])
            b = A @ x_true + rng.normal(0, 0.05, m)
            out_idx = rng.choice(m, size=18, replace=False)
            b[out_idx] += rng.uniform(20, 50, 18) * rng.choice([-1, 1], 18)
            res = ransac(
                A, b, residual_threshold=0.5, max_trials=200, random_state=seed
            )
            assert_allclose(res.x, x_true, atol=0.05)
            assert res.n_inliers >= m - 18
            # planted outliers must be excluded
            assert not res.inliers[out_idx].any()

    def test_ransac_n_trials_formula(self):
        p, e, s = 0.99, 0.3, 2
        expected = int(np.ceil(np.log(1 - p) / np.log(1 - (1 - e) ** s)))
        assert ransac_n_trials(100, 30, 2, probability=p) == expected
        assert ransac_n_trials(10, 10, 2) == 1  # degenerate: all outliers


# =============================================================================
# Maximum likelihood / information
# =============================================================================


class TestMaximumLikelihood:
    @staticmethod
    def _gaussian_setup(seed=90, n=50):
        rng = np.random.default_rng(seed)
        data = rng.normal(2.0, 1.5, n)

        def loglik(th):
            mu, s2 = th
            return -0.5 * n * np.log(2 * np.pi * s2) - np.sum((data - mu) ** 2) / (
                2 * s2
            )

        def score(th):
            mu, s2 = th
            return np.array(
                [
                    np.sum(data - mu) / s2,
                    -n / (2 * s2) + np.sum((data - mu) ** 2) / (2 * s2**2),
                ]
            )

        return data, loglik, score

    def test_fisher_numerical_gaussian_analytic(self):
        """I(mu, s2) = diag(n/s2, n/(2 s2^2)) at the MLE."""
        data, loglik, _ = self._gaussian_setup()
        n = len(data)
        mu, s2 = np.mean(data), np.var(data)
        F = fisher_information_numerical(loglik, np.array([mu, s2]), h=1e-4)
        assert_allclose(F, np.diag([n / s2, n / (2 * s2**2)]), rtol=1e-2)

    def test_observed_fisher_equals_numerical(self):
        data, loglik, _ = self._gaussian_setup()
        theta = np.array([np.mean(data), np.var(data)])
        assert_allclose(
            observed_fisher_information(loglik, theta),
            fisher_information_numerical(loglik, theta),
        )

    def test_fisher_gaussian_linear_model(self):
        rng = np.random.default_rng(91)
        J = rng.normal(size=(6, 3))
        R = np.diag(rng.uniform(0.5, 2.0, 6))
        assert_allclose(
            fisher_information_gaussian(J, R),
            J.T @ np.linalg.inv(R) @ J,
            rtol=1e-12,
        )

    def test_fisher_exponential_family_is_stat_covariance(self):
        rng = np.random.default_rng(92)
        data = rng.normal(0, 1, 500)
        F = fisher_information_exponential_family(
            lambda x, th: np.array([x, x**2]), np.array([0.0, 1.0]), data
        )
        T = np.array([[x, x**2] for x in data])
        assert_allclose(F, np.cov(T.T), rtol=1e-10)

    def test_crb_and_efficiency(self):
        F = np.array([[10.0, 0.0], [0.0, 5.0]])
        res = cramer_rao_bound(F)
        assert_allclose(res.crb_matrix, np.diag([0.1, 0.2]))
        assert_allclose(res.variances, [0.1, 0.2])
        assert_allclose(res.std_bounds, np.sqrt([0.1, 0.2]))
        assert_allclose(efficiency([0.12, 0.25], [0.1, 0.2]), [0.1 / 0.12, 0.2 / 0.25])

    def test_crb_inequality_monte_carlo_sample_mean(self):
        """Sample-mean variance over Monte Carlo runs respects the CRB
        sigma^2/n (the sample mean is efficient, so it is also close)."""
        rng = np.random.default_rng(93)
        sigma, n = 1.5, 30
        means = [np.mean(rng.normal(0, sigma, n)) for _ in range(2000)]
        mc_var = np.var(means)
        crb = sigma**2 / n
        assert mc_var >= crb * 0.9  # bound holds (allowing MC noise)
        assert mc_var <= crb * 1.15  # and is attained (efficient estimator)

    def test_crb_biased_formula(self):
        F = np.array([[10.0, 0.0], [0.0, 5.0]])
        db = np.array([[0.1, 0.0], [0.0, 0.2]])
        expected = (np.eye(2) + db) @ np.linalg.inv(F) @ (np.eye(2) + db).T
        assert_allclose(cramer_rao_bound_biased(F, db), expected, rtol=1e-12)

    def test_mle_newton_raphson_gaussian(self):
        data, loglik, score = self._gaussian_setup()
        res = mle_newton_raphson(loglik, score, np.array([1.0, 1.0]))
        assert res.converged
        assert_allclose(res.theta, [np.mean(data), np.var(data)], atol=1e-6)

    def test_mle_scoring_gaussian(self):
        data, loglik, score = self._gaussian_setup()
        n = len(data)

        def fisher(th):
            return np.diag([n / th[1], n / (2 * th[1] ** 2)])

        res = mle_scoring(loglik, score, fisher, np.array([1.0, 1.0]))
        assert res.converged
        assert_allclose(res.theta, [np.mean(data), np.var(data)], atol=1e-6)
        assert_allclose(res.covariance, np.linalg.inv(fisher(res.theta)), rtol=1e-8)

    def test_mle_gaussian_closed_form(self):
        data, loglik, _ = self._gaussian_setup()
        n = len(data)
        res = mle_gaussian(data)
        mu, s2 = np.mean(data), np.var(data)
        assert_allclose(res.theta, [mu, s2], rtol=1e-12)
        assert_allclose(res.log_likelihood, loglik(np.array([mu, s2])), rtol=1e-12)
        assert_allclose(res.fisher_info, np.diag([n / s2, n / (2 * s2**2)]))

    def test_information_criteria_analytic(self):
        assert aic(-100.0, 3) == 206.0
        assert_allclose(bic(-100.0, 3, 100), 200.0 + 3 * np.log(100))
        assert_allclose(aicc(-50.0, 3, 20), aic(-50.0, 3) + 2 * 3 * 4 / 16)
        assert aicc(-50.0, 5, 6) == np.inf  # n - k - 1 <= 0
