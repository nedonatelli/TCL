"""Property-based tests for 2D assignment optimality against brute force.

Target: ``hungarian`` in
``pytcl/assignment_algorithms/two_dimensional/assignment.py``. Its docstring
gives the signature (``cost_matrix: array_like`` of shape ``(n, m)``, "Can be
rectangular") but says nothing further about rectangular semantics -- which
dimension gets fully matched, what happens to the leftover indices on the
other side. ``hungarian`` is a thin wrapper: it calls
``scipy.optimize.linear_sum_assignment`` directly and sums the selected
entries, so its rectangular behavior *is* scipy's. Confirmed by direct
probing (not assumed from scipy's own docs):

>>> lsa(np.array([[1., 2., 3.], [4., 5., 6.]]))          # 2x3 (n < m)
(array([0, 1]), array([0, 1]))          # both rows used, 2 of 3 cols used
>>> lsa(np.array([[1., 2.], [3., 4.], [5., 6.]]))        # 3x2 (n > m)
(array([0, 1]), array([0, 1]))          # 2 of 3 rows used, both cols used
>>> lsa(np.zeros((0, 3)))
(array([], dtype=int64), array([], dtype=int64))         # empty -> empty

So for an (n, m) matrix, exactly ``k = min(n, m)`` pairs are returned, and
*every* index along the smaller dimension is matched -- the returned
``row_ind``/``col_ind`` are not independently truncated, they encode an
injective map from the smaller dimension into the larger one. The brute-force
oracle below mirrors this exactly: it enumerates all injective maps from the
smaller dimension to the larger one (``itertools.permutations(range(larger),
smaller)``), not all-pairs-of-both, so it never gives hungarian credit or
blame for a semantics it doesn't implement.

Ties (deliberately generated -- see ``_cost_element``) make the arg-optimal
*pairing* non-unique in general, so, per Task 2's precedent for this
situation, only the achieved *cost* is asserted against the oracle -- never a
specific set of pairing indices.
"""

from __future__ import annotations

import itertools

import numpy as np
from hypothesis import given, note
from hypothesis import strategies as st

from pytcl.assignment_algorithms.two_dimensional.assignment import hungarian

MAX_DIM = 6
MAX_ABS_COST = 1e6
_EPS = np.finfo(np.float64).eps

# hungarian's total_cost and the brute-force oracle below both sum up to
# MAX_DIM float64 terms bounded in magnitude by MAX_ABS_COST, but not
# necessarily in the same order: hungarian sums via numpy fancy indexing
# (``cost[row_ind, col_ind].sum()``), the oracle sums the same multiset of
# entries via a Python generator expression. Float64 addition isn't
# associative, so two sums over the *same* values can differ by a few ULP
# depending on accumulation order. Standard bound for summing k terms each
# bounded by M: at most (k - 1) rounding steps, each introducing error
# <= eps/2 * |partial sum|, and every partial sum is itself bounded by k*M.
# That gives a worst-case absolute error of (k - 1) * k * M * eps.
_SUM_ATOL = (MAX_DIM - 1) * MAX_DIM * MAX_ABS_COST * _EPS


def _cost_element() -> st.SearchStrategy:
    """Bounded float64 cost entries, biased toward exact ties.

    Two branches:
    - A small integer-valued pool (-5..5, as floats) so exact duplicate
      costs -- the case where the optimal *pairing* is not unique -- show up
      routinely, rather than relying on the vanishing probability of two
      independent continuous draws colliding exactly.
    - Bounded finite floats over [-MAX_ABS_COST, MAX_ABS_COST] (including
      negatives and zero) for general coverage. Magnitude is capped here
      rather than reusing Task 1/2's unbounded ``finite_floats()`` so that
      summing up to MAX_DIM of them can never overflow to +/-inf -- an
      overflow "counterexample" would be a generator artifact having nothing
      to do with hungarian's correctness, and would make the _SUM_ATOL
      derivation above (which assumes no overflow) meaningless.
    """
    return st.one_of(
        st.integers(min_value=-5, max_value=5).map(float),
        st.floats(
            min_value=-MAX_ABS_COST,
            max_value=MAX_ABS_COST,
            allow_nan=False,
            allow_infinity=False,
            width=64,
        ),
    )


@st.composite
def cost_matrices(draw, *, min_dim: int = 1, max_dim: int = MAX_DIM) -> np.ndarray:
    """An (n, m) float64 cost matrix, n and m independently in [min_dim, max_dim]."""
    n = draw(st.integers(min_value=min_dim, max_value=max_dim))
    m = draw(st.integers(min_value=min_dim, max_value=max_dim))
    values = draw(st.lists(_cost_element(), min_size=n * m, max_size=n * m))
    return np.array(values, dtype=np.float64).reshape(n, m)


def _brute_force_optimum(cost: np.ndarray, *, maximize: bool) -> float:
    """The true optimal total cost via exhaustive permutation search.

    Mirrors hungarian's (== scipy's) rectangular semantics documented in the
    module docstring above: exactly ``k = min(n, m)`` pairs are made, and
    every index of the smaller dimension participates. Enumerating
    ``itertools.permutations(range(larger), smaller)`` covers exactly the
    injective maps from the smaller dimension into the larger one -- both
    which subset of the larger dimension is used and in what order -- so at
    n, m <= 6 this is a genuine, exhaustive oracle, not an approximation.
    """
    n, m = cost.shape
    k = min(n, m)
    if k == 0:
        return 0.0
    reduce_fn = max if maximize else min
    if n <= m:
        totals = (
            cost[np.arange(n), cols].sum()
            for cols in itertools.permutations(range(m), n)
        )
    else:
        totals = (
            cost[rows, np.arange(m)].sum()
            for rows in itertools.permutations(range(n), m)
        )
    return reduce_fn(totals)


class TestHungarianOptimality:
    """hungarian's total cost equals the brute-force optimum, min and max."""

    @given(cost_matrices())
    def test_minimize_matches_brute_force(self, cost):
        _row_ind, _col_ind, total_cost = hungarian(cost, maximize=False)
        optimum = _brute_force_optimum(cost, maximize=False)
        note(f"shape={cost.shape} total_cost={total_cost} optimum={optimum}")
        assert abs(total_cost - optimum) <= _SUM_ATOL

    @given(cost_matrices())
    def test_maximize_matches_brute_force(self, cost):
        _row_ind, _col_ind, total_cost = hungarian(cost, maximize=True)
        optimum = _brute_force_optimum(cost, maximize=True)
        note(f"shape={cost.shape} total_cost={total_cost} optimum={optimum}")
        assert abs(total_cost - optimum) <= _SUM_ATOL


class TestHungarianValidity:
    """Each row/col index appears at most once; exactly min(n, m) pairs are
    returned, per the rectangular semantics established in the module
    docstring."""

    @given(cost_matrices(), st.booleans())
    def test_indices_unique_and_sized(self, cost, maximize):
        row_ind, col_ind, _total_cost = hungarian(cost, maximize=maximize)
        note(f"shape={cost.shape} row_ind={row_ind} col_ind={col_ind}")
        assert len(row_ind) == len(col_ind) == min(cost.shape)
        assert len(set(row_ind.tolist())) == len(row_ind)
        assert len(set(col_ind.tolist())) == len(col_ind)
        assert np.all((row_ind >= 0) & (row_ind < cost.shape[0]))
        assert np.all((col_ind >= 0) & (col_ind < cost.shape[1]))


class TestHungarianCostConsistency:
    """total_cost matches an independently-summed cost.sum() over the
    returned (row_ind, col_ind) pairing, within the same accumulation-order
    tolerance used above.

    ``hungarian`` computes ``total_cost`` internally as
    ``cost[row_ind, col_ind].sum()`` -- asserting that exact expression
    against itself here would compare an operation against itself and could
    never fail for any input. Instead this recomputes the sum independently
    in Python (a generator expression over ``zip(row_ind, col_ind)``, not
    numpy fancy indexing), so a future refactor that permutes or transposes
    the returned indices relative to the reported cost -- e.g. swapping
    ``row_ind``/``col_ind``, or returning a cost computed against a
    different pairing -- shows up as a mismatch here even though it would
    not change ``hungarian``'s own internal expression.
    """

    @given(cost_matrices(), st.booleans())
    def test_total_cost_matches_selected_entries(self, cost, maximize):
        row_ind, col_ind, total_cost = hungarian(cost, maximize=maximize)
        independent_sum = sum(
            cost[i, j] for i, j in zip(row_ind.tolist(), col_ind.tolist())
        )
        note(
            f"shape={cost.shape} total_cost={total_cost} "
            f"independent_sum={independent_sum}"
        )
        assert abs(total_cost - independent_sum) <= _SUM_ATOL
