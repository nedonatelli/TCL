"""Tests for region-cubature point generators (Cube_Space subset).

PROPERTY class: a degree-d cube rule must integrate every monomial of total
degree <= d over [-1, 1]^n exactly, and weights must sum to the cube's
measure 2**n (NOT 1 -- see region_cubature.py's module docstring for the
pinned region-measure convention). Sharpness (some degree-(d+1) monomial
must fail) guards every exactness class against a vacuous loop.
"""

import itertools
import math
import os
from math import comb, gamma
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pytcl.mathematical_functions.numerical_integration.region_cubature import (
    cube_cubature_points,
    simplex_cubature_points,
)


def ninth_order_true_point_count(n):
    """True (non-redundant) degree-9 cube point count, from the actual
    fullSymPerms block sizes -- see region_cubature.py's module docstring
    ("A related, non-crashing point-count discrepancy"). MATLAB's own
    preallocation formula (Stroud's published point count) overcounts by
    padding with dead zero-weight origin duplicates; this is the true
    count both the port and (after stripping padding) the MATLAB fixture
    comparison must match.
    """
    return 1 + 4 * n + 16 * comb(n, 2) + 8 * comb(n, 3) + 32 * comb(n, 4)


def cube_monomial_integral(alpha, half_width=1.0):
    """integral_{[-h,h]^n} prod_i x_i^{a_i} dx, closed form.

    prod_i (h^(a_i+1) - (-h)^(a_i+1)) / (a_i + 1) -- degenerates to
    0 if a_i is odd, 2*h^(a_i+1)/(a_i+1) if a_i is even.
    """
    h = half_width
    result = 1.0
    for a in alpha:
        result *= (h ** (a + 1) - (-h) ** (a + 1)) / (a + 1)
    return result


def simplex_monomial_integral(alpha):
    """integral_{simplex} prod_i x_i^{a_i} dx over the standard n-simplex
    {x>=0, sum(x)<=1}, closed Dirichlet form (design spec Section 5.1):

    prod_i gamma(a_i+1) / gamma(n + 1 + sum(a_i))
    = prod_i (a_i!) / (n + sum(a_i))!  for integer a_i.

    Degenerates to the simplex volume 1/n! at alpha=(0,...,0) (n zeros),
    the cheapest regression case before the harder monomial sweep -- same
    role cube_monomial_integral's degree-0 check plays for the cube.
    """
    n = len(alpha)
    numerator = 1.0
    for a in alpha:
        numerator *= gamma(a + 1)
    return numerator / gamma(n + 1 + sum(alpha))


def monomials_up_to(n, degree):
    """All exponent tuples of length n with total degree <= degree."""
    for total in range(degree + 1):
        for alpha in itertools.product(range(total + 1), repeat=n):
            if sum(alpha) == total:
                yield alpha


def rule_moment(points, weights, alpha):
    """Sum_i w_i * prod_j points[i, j]**alpha_j."""
    return float(np.sum(weights * np.prod(points ** np.asarray(alpha), axis=1)))


def assert_region_rule_exact(pts, w, integral_fn, n, degree, atol=1e-9):
    for alpha in monomials_up_to(n, degree):
        assert_allclose(
            rule_moment(pts, w, alpha),
            integral_fn(alpha),
            atol=atol,
            err_msg=f"monomial {alpha} not integrated exactly",
        )


def assert_some_monomial_of_degree_fails(
    pts, w, integral_fn, n, degree, threshold=1e-4
):
    """At least one monomial of EXACTLY `degree` fails.

    A single-axis probe like (degree, 0, ..., 0) can coincidentally be
    zeroed by a rule's symmetry even past its true exactness limit (seen
    here for degree-2's n=3 case); an exhaustive per-degree scan is robust
    to that.
    """
    worst = 0.0
    for alpha in itertools.product(range(degree + 1), repeat=n):
        if sum(alpha) != degree:
            continue
        worst = max(worst, abs(rule_moment(pts, w, alpha) - integral_fn(alpha)))
    assert worst > threshold, (
        f"expected some degree-{degree} monomial to fail, none did"
    )


class TestCubeMonomialIntegralOracle:
    def test_known_values_unit_half_width(self):
        # integral_{-1}^{1} x^2 dx = 2/3; x^4 -> 2/5; odd -> 0.
        assert_allclose(cube_monomial_integral((2,)), 2.0 / 3.0)
        assert_allclose(cube_monomial_integral((4,)), 2.0 / 5.0)
        assert_allclose(cube_monomial_integral((1,)), 0.0)
        assert_allclose(cube_monomial_integral((0,)), 2.0)
        # separable: integral over [-1,1]^2 of x^2*y^2 = (2/3)*(2/3)
        assert_allclose(cube_monomial_integral((2, 2)), (2.0 / 3.0) ** 2)
        # degree-0 monomial (constant 1) over [-1,1]^n is the volume 2^n
        assert_allclose(cube_monomial_integral((0, 0, 0)), 8.0)

    def test_half_width_scaling(self):
        # integral_{-h}^{h} x^2 dx = 2h^3/3
        assert_allclose(cube_monomial_integral((2,), half_width=2.0), 2.0 * 8.0 / 3.0)


class TestCubeDegree1:
    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
    def test_algorithm0_shape_and_weight_sum(self, n):
        pts, w = cube_cubature_points(n, 1, algorithm=0)
        assert pts.shape == (1, n)
        assert_allclose(w.sum(), 2.0**n, atol=1e-12)

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
    def test_algorithm0_exact_through_degree_1(self, n):
        pts, w = cube_cubature_points(n, 1, algorithm=0)
        assert_region_rule_exact(pts, w, cube_monomial_integral, n, degree=1)

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
    def test_algorithm1_shape_and_weight_sum(self, n):
        # Default algorithm; corrected from MATLAB's shape-mismatched
        # weight vector -- see module docstring defect 1.
        pts, w = cube_cubature_points(n, 1)
        assert pts.shape == (2**n, n)
        assert_allclose(w.sum(), 2.0**n, atol=1e-12)
        assert_allclose(w, np.full(2**n, 2.0**n / 2.0**n), atol=1e-12)

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
    def test_algorithm1_exact_through_degree_1(self, n):
        pts, w = cube_cubature_points(n, 1, algorithm=1)
        assert_region_rule_exact(pts, w, cube_monomial_integral, n, degree=1)

    def test_algorithm1_matches_default(self):
        pts1, w1 = cube_cubature_points(3, 1)
        pts2, w2 = cube_cubature_points(3, 1, algorithm=1)
        assert np.array_equal(pts1, pts2)
        assert np.array_equal(w1, w2)

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_sharpness_degree_2_fails(self, n):
        # A single-point rule (algorithm 0) cannot be degree-2 exact:
        # integral of x^2 over the cube is 2/3 * 2^(n-1) != 0 (the rule's
        # value at the origin).
        pts, w = cube_cubature_points(n, 1, algorithm=0)
        alpha = (2,) + (0,) * (n - 1)
        assert abs(rule_moment(pts, w, alpha) - cube_monomial_integral(alpha)) > 1e-6

    def test_antipodal_symmetry_algorithm1(self):
        pts, _ = cube_cubature_points(4, 1, algorithm=1)
        for p in pts:
            assert np.any(np.all(np.isclose(pts, -p), axis=1))

    def test_unknown_algorithm_raises(self):
        with pytest.raises(ValueError):
            cube_cubature_points(2, 1, algorithm=2)


class TestCubeDegree2:
    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    def test_algorithm0_shape_and_weight_sum(self, n):
        pts, w = cube_cubature_points(n, 2)
        assert pts.shape == (n + 1, n)
        assert_allclose(w.sum(), 2.0**n, atol=1e-10)

    @pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
    def test_algorithm0_exact_through_degree_2(self, n):
        pts, w = cube_cubature_points(n, 2)
        assert_region_rule_exact(pts, w, cube_monomial_integral, n, degree=2)

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_algorithm0_sharpness_degree_3_fails(self, n):
        # Single-axis probe: NOT usable here -- (3,0,...,0) is coincidentally
        # zeroed by this rule's rotational symmetry at n=3 (measured 6.7e-48,
        # not roundoff noise around a nonzero true value); the exhaustive
        # scan below is the discriminating check.
        pts, w = cube_cubature_points(n, 2)
        assert_some_monomial_of_degree_fails(
            pts, w, cube_monomial_integral, n, degree=3
        )

    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    def test_algorithm1_shape_and_weight_sum(self, n):
        pts, w = cube_cubature_points(n, 2, algorithm=1)
        assert pts.shape == (2 * n + 1, n)
        assert_allclose(w.sum(), 2.0**n, atol=1e-10)

    @pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
    def test_algorithm1_exact_through_degree_2(self, n):
        pts, w = cube_cubature_points(n, 2, algorithm=1)
        assert_region_rule_exact(pts, w, cube_monomial_integral, n, degree=2)

    def test_invalid_dimension_raises(self):
        with pytest.raises(ValueError):
            cube_cubature_points(1, 2)

    def test_unknown_algorithm_raises(self):
        with pytest.raises(ValueError):
            cube_cubature_points(2, 2, algorithm=5)


class TestCubeDegree3:
    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    def test_algorithm0_shape_and_weight_sum(self, n):
        pts, w = cube_cubature_points(n, 3)
        assert pts.shape == (2 * n, n)
        assert_allclose(w.sum(), 2.0**n, atol=1e-10)

    @pytest.mark.parametrize("n", [2, 3, 4, 5, 6, 7])
    def test_algorithm0_exact_through_degree_3(self, n):
        # n = 3, 5, 7 (odd) exercise the corrected odd-dimension row --
        # see module docstring defect 2.
        pts, w = cube_cubature_points(n, 3)
        assert_region_rule_exact(pts, w, cube_monomial_integral, n, degree=3)

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_algorithm0_sharpness_degree_4_fails(self, n):
        pts, w = cube_cubature_points(n, 3)
        alpha = (4,) + (0,) * (n - 1)
        assert abs(rule_moment(pts, w, alpha) - cube_monomial_integral(alpha)) > 1e-6

    @pytest.mark.parametrize(
        "algorithm,expected_points", [(1, "2n+1"), (2, "2^n"), (3, "2^n+1")]
    )
    def test_general_n_algorithms_exact_through_degree_3(
        self, algorithm, expected_points
    ):
        for n in (2, 3, 4, 5):
            pts, w = cube_cubature_points(n, 3, algorithm=algorithm)
            assert_allclose(w.sum(), 2.0**n, atol=1e-9)
            assert_region_rule_exact(pts, w, cube_monomial_integral, n, degree=3)

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_algorithm4_tensor_grid_exact_through_degree_3(self, n):
        pts, w = cube_cubature_points(n, 3, algorithm=4)
        assert pts.shape == (3**n, n)
        assert_allclose(w.sum(), 2.0**n, atol=1e-10)
        assert_region_rule_exact(pts, w, cube_monomial_integral, n, degree=3)

    def test_algorithm1_point_count(self):
        n = 4
        pts, _ = cube_cubature_points(n, 3, algorithm=1)
        assert pts.shape == (2 * n + 1, n)

    def test_algorithm2_point_count(self):
        n = 4
        pts, _ = cube_cubature_points(n, 3, algorithm=2)
        assert pts.shape == (2**n, n)

    def test_algorithm3_point_count(self):
        n = 4
        pts, _ = cube_cubature_points(n, 3, algorithm=3)
        assert pts.shape == (2**n + 1, n)

    def test_odd_dimension_row_matches_even_row_convention(self):
        # Regression pin for the corrected defect 2: at n=3 the last
        # coordinate must take values +-1/sqrt(3), not 0 (which is what
        # MATLAB's uncorrected `i+1` offset would silently leave column 1
        # holding).
        pts, _ = cube_cubature_points(3, 3, algorithm=0)
        last_coord = np.sort(pts[:, 2])
        assert_allclose(np.abs(last_coord), np.full(6, 1.0 / np.sqrt(3.0)), atol=1e-12)

    def test_invalid_dimension_raises(self):
        with pytest.raises(ValueError):
            cube_cubature_points(1, 3)

    def test_unknown_algorithm_raises(self):
        with pytest.raises(ValueError):
            cube_cubature_points(2, 3, algorithm=5)


class TestCubeDegree5:
    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    def test_algorithm0_shape_and_weight_sum(self, n):
        pts, w = cube_cubature_points(n, 5)
        assert pts.shape == (2 * n * n + 1, n)
        assert_allclose(w.sum(), 2.0**n, atol=1e-9)

    @pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
    def test_algorithm0_exact_through_degree_5(self, n):
        pts, w = cube_cubature_points(n, 5)
        assert_region_rule_exact(pts, w, cube_monomial_integral, n, degree=5)

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_algorithm0_sharpness_degree_6_fails(self, n):
        pts, w = cube_cubature_points(n, 5)
        alpha = (6,) + (0,) * (n - 1)
        assert abs(rule_moment(pts, w, alpha) - cube_monomial_integral(alpha)) > 1e-6

    @pytest.mark.parametrize("algorithm", [1, 2, 3, 4, 5, 7])
    def test_remaining_algorithms_exact_through_degree_5(self, algorithm):
        for n in (2, 3, 4):
            pts, w = cube_cubature_points(n, 5, algorithm=algorithm)
            assert_allclose(w.sum(), 2.0**n, atol=1e-8)
            assert_region_rule_exact(
                pts, w, cube_monomial_integral, n, degree=5, atol=1e-8
            )

    def test_algorithm6_exact_through_degree_5(self):
        # Cn 5-8 requires n >= 3 (see module docstring); n=2 is a domain
        # violation, not an exactness case, so it is excluded from the
        # sweep above and tested separately here.
        for n in (3, 4, 5):
            pts, w = cube_cubature_points(n, 5, algorithm=6)
            assert_allclose(w.sum(), 2.0**n, atol=1e-8)
            assert_region_rule_exact(
                pts, w, cube_monomial_integral, n, degree=5, atol=1e-8
            )

    def test_algorithm6_requires_n3(self):
        with pytest.raises(ValueError):
            cube_cubature_points(2, 5, algorithm=6)

    def test_algorithm1_point_count(self):
        n = 4
        pts, _ = cube_cubature_points(n, 5, algorithm=1)
        assert pts.shape == (3 * n * n + 3 * n + 1, n)

    def test_algorithm2_point_count(self):
        n = 4
        pts, _ = cube_cubature_points(n, 5, algorithm=2)
        assert pts.shape == (2**n + 2 * n, n)

    def test_algorithm3_point_count(self):
        n = 4
        pts, _ = cube_cubature_points(n, 5, algorithm=3)
        assert pts.shape == (2**n + 2 * n + 1, n)

    def test_algorithm4_point_count(self):
        n = 4
        pts, _ = cube_cubature_points(n, 5, algorithm=4)
        assert pts.shape == (2 ** (n + 1) - 1, n)

    def test_algorithm5_point_count(self):
        n = 4
        pts, _ = cube_cubature_points(n, 5, algorithm=5)
        assert pts.shape == (n * 2**n + 1, n)

    def test_algorithm6_point_count(self):
        n = 4
        pts, _ = cube_cubature_points(n, 5, algorithm=6)
        assert pts.shape == (2**n * (n + 1), n)

    def test_algorithm7_tensor_grid_point_count(self):
        n = 3
        pts, w = cube_cubature_points(n, 5, algorithm=7)
        assert pts.shape == (3**n, n)
        assert_allclose(w.sum(), 2.0**n, atol=1e-10)

    def test_antipodal_symmetry_algorithm0(self):
        pts, _ = cube_cubature_points(4, 5)
        nonzero = pts[np.any(pts != 0.0, axis=1)]
        for p in nonzero:
            assert np.any(np.all(np.isclose(nonzero, -p), axis=1))

    def test_invalid_dimension_raises(self):
        with pytest.raises(ValueError):
            cube_cubature_points(1, 5)

    def test_unknown_algorithm_raises(self):
        with pytest.raises(ValueError):
            cube_cubature_points(2, 5, algorithm=8)


class TestCubeDegree7:
    """No general-n formula exists in MATLAB at degree 7 for this region
    (spec Section 2.1) -- n=2 (algorithm 0) and n=3 (algorithm 5) are the
    only ported cases, matching capture_region_rules.m's case list."""

    def test_algorithm0_n2_shape_and_weight_sum(self):
        pts, w = cube_cubature_points(2, 7)
        assert pts.shape == (12, 2)
        assert_allclose(w.sum(), 4.0, atol=1e-9)

    def test_algorithm0_n2_exact_through_degree_7(self):
        pts, w = cube_cubature_points(2, 7)
        assert_region_rule_exact(pts, w, cube_monomial_integral, 2, degree=7)

    def test_algorithm0_n2_sharpness_degree_8_fails(self):
        pts, w = cube_cubature_points(2, 7)
        alpha = (8, 0)
        assert abs(rule_moment(pts, w, alpha) - cube_monomial_integral(alpha)) > 1e-6

    def test_default_dispatch_n2(self):
        pts1, w1 = cube_cubature_points(2, 7)
        pts2, w2 = cube_cubature_points(2, 7, algorithm=0)
        assert np.array_equal(pts1, pts2)
        assert np.array_equal(w1, w2)

    def test_algorithm5_n3_shape_and_weight_sum(self):
        pts, w = cube_cubature_points(3, 7, algorithm=5)
        assert pts.shape == (34, 3)
        assert_allclose(w.sum(), 8.0, atol=1e-8)

    def test_algorithm5_n3_exact_through_degree_7(self):
        pts, w = cube_cubature_points(3, 7, algorithm=5)
        assert_region_rule_exact(pts, w, cube_monomial_integral, 3, degree=7, atol=1e-8)

    def test_algorithm5_n3_sharpness_degree_8_fails(self):
        pts, w = cube_cubature_points(3, 7, algorithm=5)
        alpha = (8, 0, 0)
        assert abs(rule_moment(pts, w, alpha) - cube_monomial_integral(alpha)) > 1e-6

    def test_default_dispatch_n3(self):
        pts1, w1 = cube_cubature_points(3, 7)
        pts2, w2 = cube_cubature_points(3, 7, algorithm=5)
        assert np.array_equal(pts1, pts2)
        assert np.array_equal(w1, w2)

    def test_algorithm0_requires_n2(self):
        with pytest.raises(ValueError):
            cube_cubature_points(3, 7, algorithm=0)

    def test_algorithm5_requires_n3(self):
        with pytest.raises(ValueError):
            cube_cubature_points(2, 7, algorithm=5)

    def test_unported_algorithm_raises(self):
        with pytest.raises(ValueError):
            cube_cubature_points(2, 7, algorithm=1)

    def test_unsupported_n_raises(self):
        with pytest.raises(ValueError):
            cube_cubature_points(4, 7)


class TestCubeDegree9:
    @pytest.mark.parametrize("n", [4, 5, 6])
    def test_shape_and_weight_sum(self, n):
        pts, w = cube_cubature_points(n, 9)
        assert pts.shape == (ninth_order_true_point_count(n), n)
        assert_allclose(w.sum(), 2.0**n, atol=1e-6)

    @pytest.mark.parametrize("n", [4, 5])
    def test_exact_through_degree_9(self, n):
        pts, w = cube_cubature_points(n, 9)
        assert_region_rule_exact(pts, w, cube_monomial_integral, n, degree=9, atol=1e-6)

    def test_sharpness_degree_10_fails(self):
        pts, w = cube_cubature_points(4, 9)
        alpha = (10, 0, 0, 0)
        assert abs(rule_moment(pts, w, alpha) - cube_monomial_integral(alpha)) > 1e-3

    @pytest.mark.parametrize("n", [4, 5, 6])
    def test_algorithm1_shape_and_weight_sum(self, n):
        pts, w = cube_cubature_points(n, 9, algorithm=1)
        assert pts.shape == (ninth_order_true_point_count(n), n)
        assert_allclose(w.sum(), 2.0**n, atol=1e-6)

    @pytest.mark.parametrize("n", [4, 5, 6])
    def test_algorithm1_exact_through_degree_9(self, n):
        # Algorithm 1 ("variant 2" per MATLAB's own comment) is a
        # genuinely different rule from algorithm 0, not a relabeling --
        # see module docstring -- and must independently satisfy the
        # degree-9 oracle.
        pts, w = cube_cubature_points(n, 9, algorithm=1)
        assert_region_rule_exact(pts, w, cube_monomial_integral, n, degree=9, atol=1e-6)

    def test_algorithm1_sharpness_degree_10_fails(self):
        pts, w = cube_cubature_points(4, 9, algorithm=1)
        alpha = (10, 0, 0, 0)
        assert abs(rule_moment(pts, w, alpha) - cube_monomial_integral(alpha)) > 1e-3

    @pytest.mark.parametrize("n", [4, 5, 6])
    def test_algorithm0_and_algorithm1_differ(self, n):
        # NOT a relabeling: algorithm 0 places the triple-repeated
        # (v, v, v, 0, ...) block at the larger root of the shared
        # quartic, algorithm 1 at the smaller -- the two point sets
        # genuinely differ (measured, not merely asserted). A prior
        # version of this test asserted the opposite (pts0 == pts1) by
        # comparing the port to itself, which was vacuous; see the
        # module docstring's "genuinely different rules" note.
        pts0, w0 = cube_cubature_points(n, 9, algorithm=0)
        pts1, w1 = cube_cubature_points(n, 9, algorithm=1)
        o0 = np.lexsort(pts0.T)
        o1 = np.lexsort(pts1.T)
        assert not np.allclose(pts0[o0], pts1[o1], atol=1e-6)
        assert not np.allclose(w0[o0], w1[o1], atol=1e-6)

    def test_antipodal_symmetry(self):
        pts, _ = cube_cubature_points(4, 9)
        nonzero = pts[np.any(pts != 0.0, axis=1)]
        for p in nonzero:
            assert np.any(np.all(np.isclose(nonzero, -p, atol=1e-8), axis=1))

    def test_invalid_dimension_raises(self):
        with pytest.raises(ValueError):
            cube_cubature_points(3, 9)

    def test_unknown_algorithm_raises(self):
        with pytest.raises(ValueError):
            cube_cubature_points(4, 9, algorithm=2)


class TestCubeCubaturePointsGeneral:
    def test_unsupported_degree_raises(self):
        with pytest.raises(ValueError):
            cube_cubature_points(3, 4)
        with pytest.raises(ValueError):
            cube_cubature_points(3, 6)
        with pytest.raises(ValueError):
            cube_cubature_points(3, 8)

    def test_invalid_n_raises(self):
        with pytest.raises(ValueError):
            cube_cubature_points(0, 1)


class TestSimplexMonomialIntegralOracle:
    def test_hand_case_integral_of_x_over_2_simplex(self):
        # The brief's own hand case: integral of x over the standard
        # 2-simplex {x>=0,y>=0,x+y<=1} is 1/6.
        assert_allclose(simplex_monomial_integral((1, 0)), 1.0 / 6.0)
        assert_allclose(simplex_monomial_integral((0, 1)), 1.0 / 6.0)

    def test_degree_0_monomial_is_simplex_volume(self):
        # alpha=(0,...,0): integral of the constant 1 over the simplex is
        # its volume 1/n! -- the cheapest regression case, per the design
        # spec (Section 5.1) and mirroring cube_monomial_integral's
        # equivalent degree-0 check.
        for n in range(1, 6):
            assert_allclose(
                simplex_monomial_integral((0,) * n), 1.0 / math.factorial(n)
            )

    def test_known_values(self):
        # integral_0^1 integral_0^{1-x} x^2 dy dx = 1/12.
        assert_allclose(simplex_monomial_integral((2, 0)), 1.0 / 12.0)
        # integral over the 2-simplex of x*y = 1/24.
        assert_allclose(simplex_monomial_integral((1, 1)), 1.0 / 24.0)
        # 3-simplex, integral of x = 1/24.
        assert_allclose(simplex_monomial_integral((1, 0, 0)), 1.0 / 24.0)


class TestSimplexDegree2:
    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    def test_shape_and_weight_sum(self, n):
        pts, w = simplex_cubature_points(n, 2)
        assert pts.shape == ((n + 1) * (n + 2) // 2, n)
        assert_allclose(w.sum(), 1.0 / math.factorial(n), atol=1e-12)

    @pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
    def test_exact_through_degree_2(self, n):
        pts, w = simplex_cubature_points(n, 2)
        assert_region_rule_exact(pts, w, simplex_monomial_integral, n, degree=2)

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_sharpness_degree_3_fails(self, n):
        pts, w = simplex_cubature_points(n, 2)
        assert_some_monomial_of_degree_fails(
            pts, w, simplex_monomial_integral, n, degree=3
        )

    def test_default_algorithm_is_only_algorithm(self):
        pts1, w1 = simplex_cubature_points(3, 2)
        pts2, w2 = simplex_cubature_points(3, 2, algorithm=0)
        assert np.array_equal(pts1, pts2)
        assert np.array_equal(w1, w2)

    def test_unknown_algorithm_raises(self):
        with pytest.raises(ValueError):
            simplex_cubature_points(3, 2, algorithm=1)

    def test_invalid_dimension_raises(self):
        with pytest.raises(ValueError):
            simplex_cubature_points(1, 2)


class TestSimplexDegree3:
    """thirdOrderSimplexCubPoints has 12 MATLAB algorithms; 0-9 are
    general-n (each with its own dimension restriction, some hardened
    beyond MATLAB's own checks -- see region_cubature.py's module
    docstring); 10 (T_2 3-1, n=2 only) and 11 (T_5 3-1, n=5 only) are
    fixed-dimension literature variants, not ported."""

    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    def test_algorithm0_shape_and_weight_sum(self, n):
        pts, w = simplex_cubature_points(n, 3)
        assert pts.shape == (n + 2, n)
        assert_allclose(w.sum(), 1.0 / math.factorial(n), atol=1e-10)

    @pytest.mark.parametrize("n", [2, 3, 4, 5, 6, 7])
    def test_algorithm0_exact_through_degree_3(self, n):
        pts, w = simplex_cubature_points(n, 3)
        assert_region_rule_exact(pts, w, simplex_monomial_integral, n, degree=3)

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_algorithm0_sharpness_degree_4_fails(self, n):
        # n=4's worst degree-4 residual is real (measured ~6.2e-5) but
        # smaller than the default threshold=1e-4 -- still 11 orders of
        # magnitude above roundoff (~1e-16), so a lower threshold still
        # discriminates a genuine failure from noise, it just needs to be
        # smaller here than the default catches.
        pts, w = simplex_cubature_points(n, 3)
        assert_some_monomial_of_degree_fails(
            pts, w, simplex_monomial_integral, n, degree=4, threshold=1e-6
        )

    def test_default_dispatch_is_algorithm0(self):
        pts1, w1 = simplex_cubature_points(4, 3)
        pts2, w2 = simplex_cubature_points(4, 3, algorithm=0)
        assert np.array_equal(pts1, pts2)
        assert np.array_equal(w1, w2)

    @pytest.mark.parametrize(
        "algorithm,expected_points",
        [
            (1, lambda n: 2 * n + 2),
            (2, lambda n: 2 * n + 3),
            (6, lambda n: (n**3 + 11 * n + 12) // 6),
            (7, lambda n: (n + 1) * (n + 2) * (n + 3) // 6),
        ],
    )
    def test_unrestricted_algorithms_exact_through_degree_3(
        self, algorithm, expected_points
    ):
        for n in (2, 3, 4, 5, 6):
            pts, w = simplex_cubature_points(n, 3, algorithm=algorithm)
            assert pts.shape == (expected_points(n), n)
            assert_allclose(w.sum(), 1.0 / math.factorial(n), atol=1e-9)
            assert_region_rule_exact(pts, w, simplex_monomial_integral, n, degree=3)

    @pytest.mark.parametrize(
        "algorithm,expected_points",
        [
            (4, lambda n: (n + 1) * (n + 4) // 2),
            (5, lambda n: (n**3 + 5 * n + 12) // 6),
        ],
    )
    def test_n_ge_3_algorithms_exact_through_degree_3(self, algorithm, expected_points):
        for n in (3, 4, 5, 6):
            pts, w = simplex_cubature_points(n, 3, algorithm=algorithm)
            assert pts.shape == (expected_points(n), n)
            assert_allclose(w.sum(), 1.0 / math.factorial(n), atol=1e-9)
            assert_region_rule_exact(pts, w, simplex_monomial_integral, n, degree=3)

    @pytest.mark.parametrize(
        "algorithm,expected_points",
        [
            (8, lambda n: (n**3 - n + 3) // 3),
            (9, lambda n: (n**3 + 2 * n + 3) // 3),
        ],
    )
    def test_n_ge_3_n_ne_5_algorithms_exact_through_degree_3(
        self, algorithm, expected_points
    ):
        for n in (3, 4, 6, 7):
            pts, w = simplex_cubature_points(n, 3, algorithm=algorithm)
            assert pts.shape == (expected_points(n), n)
            assert_allclose(w.sum(), 1.0 / math.factorial(n), atol=1e-8)
            assert_region_rule_exact(
                pts, w, simplex_monomial_integral, n, degree=3, atol=1e-8
            )

    def test_algorithm4_requires_n3(self):
        with pytest.raises(ValueError):
            simplex_cubature_points(2, 3, algorithm=4)

    def test_algorithm5_requires_n3(self):
        with pytest.raises(ValueError):
            simplex_cubature_points(2, 3, algorithm=5)

    def test_algorithm8_and_9_require_n3_and_reject_n5(self):
        with pytest.raises(ValueError):
            simplex_cubature_points(2, 3, algorithm=8)
        with pytest.raises(ValueError):
            simplex_cubature_points(5, 3, algorithm=8)
        with pytest.raises(ValueError):
            simplex_cubature_points(2, 3, algorithm=9)
        with pytest.raises(ValueError):
            simplex_cubature_points(5, 3, algorithm=9)

    def test_algorithm10_and_11_not_ported(self):
        # Fixed-dimension literature variants (T_2 3-1, T_5 3-1) -- see
        # region_cubature.py's module docstring for why these specifically
        # are deferred even though the rest of this file is general-n.
        with pytest.raises(ValueError):
            simplex_cubature_points(2, 3, algorithm=10)
        with pytest.raises(ValueError):
            simplex_cubature_points(5, 3, algorithm=11)

    def test_unknown_algorithm_raises(self):
        with pytest.raises(ValueError):
            simplex_cubature_points(3, 3, algorithm=12)


class TestSimplexDegree3Algorithm3Defect:
    """Algorithm 3 (T_n 3-4) hits a MATLAB defect at n=2: both real roots
    of the file's own parameter cubic make the weight formula an exact
    0/0, verified symbolically (region_cubature.py module docstring,
    "third corrected defect"). This class pins that hardened guard and
    independently verifies the corrected rule's exactness and the n=5
    double-root tie-break at every n this port supports (3<=n<7)."""

    @pytest.mark.parametrize("n", [3, 4, 5, 6])
    def test_exact_through_degree_3(self, n):
        pts, w = simplex_cubature_points(n, 3, algorithm=3)
        assert pts.shape == ((n + 1) * (n + 2) // 2, n)
        assert_allclose(w.sum(), 1.0 / math.factorial(n), atol=1e-9)
        assert_region_rule_exact(
            pts, w, simplex_monomial_integral, n, degree=3, atol=1e-8
        )

    def test_n2_raises_instead_of_returning_nan(self):
        # The confirmed 0/0 defect: MATLAB's own domain guard (numDim<7)
        # does not exclude n=2, but running it there would poison every
        # weight with NaN (0.0/0.0 in IEEE 754). The port hardens the
        # guard rather than propagate NaN.
        with pytest.raises(ValueError, match="3 <= n < 7"):
            simplex_cubature_points(2, 3, algorithm=3)

    def test_n_ge_7_raises(self):
        with pytest.raises(ValueError):
            simplex_cubature_points(7, 3, algorithm=3)


class TestSimplexDegree4:
    @pytest.mark.parametrize("n", [3, 5, 6, 7])
    def test_shape_and_weight_sum(self, n):
        pts, w = simplex_cubature_points(n, 4)
        assert pts.shape == (comb(n + 4, 4), n)
        assert_allclose(w.sum(), 1.0 / math.factorial(n), atol=1e-9)

    def test_n4_zero_weight_points_stripped(self):
        # B4's exact (4-n) factor vanishes only at n=4: MATLAB itself
        # strips these (module docstring: "not a defect"), reducing the
        # point count below the general C(n+4,4) formula.
        pts, w = simplex_cubature_points(4, 4)
        assert pts.shape == (40, 4)
        assert pts.shape[0] < comb(4 + 4, 4)
        assert np.all(w != 0.0)
        assert_allclose(w.sum(), 1.0 / math.factorial(4), atol=1e-9)

    @pytest.mark.parametrize("n", [3, 4, 5, 6, 7])
    def test_exact_through_degree_4(self, n):
        pts, w = simplex_cubature_points(n, 4)
        assert_region_rule_exact(pts, w, simplex_monomial_integral, n, degree=4)

    @pytest.mark.parametrize("n", [3, 4, 5])
    def test_sharpness_degree_5_fails(self, n):
        # n=4,5's worst degree-5 residuals are real (measured ~5.8e-5 and
        # ~1.3e-5) but smaller than the default threshold=1e-4 -- see the
        # analogous note on TestSimplexDegree3's degree-4 sharpness test.
        pts, w = simplex_cubature_points(n, 4)
        assert_some_monomial_of_degree_fails(
            pts, w, simplex_monomial_integral, n, degree=5, threshold=1e-6
        )

    def test_default_algorithm_is_only_algorithm(self):
        pts1, w1 = simplex_cubature_points(5, 4)
        pts2, w2 = simplex_cubature_points(5, 4, algorithm=0)
        assert np.array_equal(pts1, pts2)
        assert np.array_equal(w1, w2)

    def test_unknown_algorithm_raises(self):
        with pytest.raises(ValueError):
            simplex_cubature_points(5, 4, algorithm=1)

    def test_requires_n3(self):
        with pytest.raises(ValueError):
            simplex_cubature_points(2, 4)


class TestSimplexDegree5:
    @pytest.mark.parametrize("n", [4, 5, 6, 7])
    def test_algorithm0_shape_and_weight_sum(self, n):
        pts, w = simplex_cubature_points(n, 5, algorithm=0)
        assert pts.shape == (comb(n + 5, 5), n)
        assert_allclose(w.sum(), 1.0 / math.factorial(n), atol=1e-8)

    @pytest.mark.parametrize("n", [4, 5, 6, 7])
    def test_algorithm0_exact_through_degree_5(self, n):
        pts, w = simplex_cubature_points(n, 5, algorithm=0)
        assert_region_rule_exact(
            pts, w, simplex_monomial_integral, n, degree=5, atol=1e-8
        )

    def test_algorithm0_sharpness_degree_6_fails(self):
        # Worst degree-6 residual at n=4 is real (measured ~1.5e-5) but
        # smaller than the default threshold=1e-4 -- see the analogous
        # note on TestSimplexDegree3's degree-4 sharpness test.
        pts, w = simplex_cubature_points(4, 5, algorithm=0)
        assert_some_monomial_of_degree_fails(
            pts, w, simplex_monomial_integral, 4, degree=6, threshold=1e-6
        )

    def test_algorithm0_requires_n4(self):
        with pytest.raises(ValueError):
            simplex_cubature_points(3, 5, algorithm=0)

    def test_algorithm1_n2_exact_through_degree_5(self):
        pts, w = simplex_cubature_points(2, 5, algorithm=1)
        assert pts.shape == (7, 2)
        assert_allclose(w.sum(), 0.5, atol=1e-10)
        assert_region_rule_exact(pts, w, simplex_monomial_integral, 2, degree=5)

    def test_algorithm1_requires_n2(self):
        with pytest.raises(ValueError):
            simplex_cubature_points(3, 5, algorithm=1)

    def test_algorithm2_n3_exact_through_degree_5(self):
        pts, w = simplex_cubature_points(3, 5, algorithm=2)
        assert pts.shape == (15, 3)
        assert_allclose(w.sum(), 1.0 / 6.0, atol=1e-10)
        assert_region_rule_exact(pts, w, simplex_monomial_integral, 3, degree=5)

    def test_algorithm2_requires_n3(self):
        with pytest.raises(ValueError):
            simplex_cubature_points(4, 5, algorithm=2)

    def test_default_dispatch_matches_matlabs_own_per_n_selection(self):
        # n=2 -> algorithm 1, n=3 -> algorithm 2, n>=4 -> algorithm 0,
        # matching fifthOrderSimplexCubPoints.m's own nargin<2 branch.
        pts_a, w_a = simplex_cubature_points(2, 5)
        pts_b, w_b = simplex_cubature_points(2, 5, algorithm=1)
        assert np.array_equal(pts_a, pts_b)
        assert np.array_equal(w_a, w_b)

        pts_a, w_a = simplex_cubature_points(3, 5)
        pts_b, w_b = simplex_cubature_points(3, 5, algorithm=2)
        assert np.array_equal(pts_a, pts_b)
        assert np.array_equal(w_a, w_b)

        pts_a, w_a = simplex_cubature_points(4, 5)
        pts_b, w_b = simplex_cubature_points(4, 5, algorithm=0)
        assert np.array_equal(pts_a, pts_b)
        assert np.array_equal(w_a, w_b)

    def test_unknown_algorithm_raises(self):
        with pytest.raises(ValueError):
            simplex_cubature_points(4, 5, algorithm=3)


class TestSimplexCubaturePointsGeneral:
    def test_unsupported_degree_raises(self):
        with pytest.raises(ValueError):
            simplex_cubature_points(3, 1)
        with pytest.raises(ValueError):
            simplex_cubature_points(3, 6)

    def test_invalid_n_raises(self):
        with pytest.raises(ValueError):
            simplex_cubature_points(1, 2)
        with pytest.raises(ValueError):
            simplex_cubature_points(0, 2)


_MATLAB_FIXTURES = Path(__file__).parents[1] / "fixtures" / "matlab"


def _load_matlab_fixture(name):
    path = _MATLAB_FIXTURES / name
    if not path.exists():
        if os.environ.get("PYTCL_REQUIRE_MATLAB_FIXTURES") == "1":
            pytest.fail(
                f"fixture {name} required but absent -- run "
                "scripts/matlab_capture/capture_region_rules.m in MATLAB "
                "with the TCL library on path to generate it"
            )
        pytest.skip(
            f"MATLAB fixture {name} not captured yet -- run "
            "scripts/matlab_capture/capture_region_rules.m in MATLAB "
            "with the TCL library on path to generate it"
        )
    data = np.loadtxt(path, delimiter=",")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data[:, :-1], data[:, -1]


class TestCubeMatlabFixtures:
    """Cross-check against real MATLAB output (capture_region_rules.m).

    firstOrderNDimCubPoints is captured at algorithm 0 (1 point), not
    MATLAB's default algorithm 1: MATLAB's algorithm 1, as pinned at
    commit 593ce51, cannot produce a valid fixture at all (a genuine
    shape-mismatch crash, not merely a wrong answer) -- see this module's
    defect-1 note and capture_region_rules.m's comment at that case.
    thirdOrderNDimCubPoints's default algorithm is captured only at even n
    (n=2, 4) for the same reason (defect 2) -- see capture_region_rules.m.
    """

    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    def test_first_order_algorithm0_matches_matlab(self, n):
        ref_pts, ref_w = _load_matlab_fixture(
            f"region_cube_firstOrderNDimCubPoints_n{n}_alg0.csv"
        )
        pts, w = cube_cubature_points(n, 1, algorithm=0)
        assert_allclose(pts, ref_pts, atol=1e-10)
        assert_allclose(w, ref_w, atol=1e-8)

    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    def test_second_order_matches_matlab(self, n):
        ref_pts, ref_w = _load_matlab_fixture(
            f"region_cube_secondOrderNDimCubPoints_n{n}_alg0.csv"
        )
        pts, w = cube_cubature_points(n, 2)
        oa = np.lexsort(pts.T)
        ob = np.lexsort(ref_pts.T)
        assert_allclose(pts[oa], ref_pts[ob], atol=1e-10)
        assert_allclose(w[oa], ref_w[ob], atol=1e-8)

    @pytest.mark.parametrize("n", [2, 4])
    def test_third_order_matches_matlab(self, n):
        ref_pts, ref_w = _load_matlab_fixture(
            f"region_cube_thirdOrderNDimCubPoints_n{n}_alg0.csv"
        )
        pts, w = cube_cubature_points(n, 3)
        oa = np.lexsort(pts.T)
        ob = np.lexsort(ref_pts.T)
        assert_allclose(pts[oa], ref_pts[ob], atol=1e-10)
        assert_allclose(w[oa], ref_w[ob], atol=1e-8)

    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    def test_fifth_order_matches_matlab(self, n):
        ref_pts, ref_w = _load_matlab_fixture(
            f"region_cube_fifthOrderNDimCubPoints_n{n}_alg0.csv"
        )
        pts, w = cube_cubature_points(n, 5)
        oa = np.lexsort(pts.T)
        ob = np.lexsort(ref_pts.T)
        assert_allclose(pts[oa], ref_pts[ob], atol=1e-8)
        assert_allclose(w[oa], ref_w[ob], atol=1e-7)

    def test_seventh_order_n2_matches_matlab(self):
        ref_pts, ref_w = _load_matlab_fixture(
            "region_cube_seventhOrderNDimCubPoints_n2_alg0.csv"
        )
        pts, w = cube_cubature_points(2, 7, algorithm=0)
        oa = np.lexsort(pts.T)
        ob = np.lexsort(ref_pts.T)
        assert_allclose(pts[oa], ref_pts[ob], atol=1e-8)
        assert_allclose(w[oa], ref_w[ob], atol=1e-7)

    def test_seventh_order_n3_matches_matlab(self):
        ref_pts, ref_w = _load_matlab_fixture(
            "region_cube_seventhOrderNDimCubPoints_n3_alg5.csv"
        )
        pts, w = cube_cubature_points(3, 7, algorithm=5)
        oa = np.lexsort(pts.T)
        ob = np.lexsort(ref_pts.T)
        assert_allclose(pts[oa], ref_pts[ob], atol=1e-8)
        assert_allclose(w[oa], ref_w[ob], atol=1e-7)

    @pytest.mark.parametrize("n", [4, 5])
    def test_ninth_order_matches_matlab(self, n):
        # MATLAB preallocates xi/w with Stroud's own admittedly-incorrect
        # published point-count formula and never overwrites the trailing
        # slots beyond the true count -- they're captured verbatim as
        # dead zero-weight, zero-coordinate rows (see region_cubature.py's
        # module docstring). Strip them before comparing so this test
        # checks the real rule, not MATLAB's padding; the stripped count
        # must equal the same true-count formula the port itself uses.
        ref_pts, ref_w = _load_matlab_fixture(
            f"region_cube_ninthOrderNDimCubPoints_n{n}_alg0.csv"
        )
        true_count = ninth_order_true_point_count(n)
        nonzero = ~np.isclose(ref_w, 0.0, atol=1e-12)
        assert nonzero.sum() == true_count, (
            f"expected {true_count} true (nonzero-weight) MATLAB points at "
            f"n={n}, found {nonzero.sum()} of {len(ref_w)}"
        )
        ref_pts = ref_pts[nonzero]
        ref_w = ref_w[nonzero]

        pts, w = cube_cubature_points(n, 9)
        oa = np.lexsort(pts.T)
        ob = np.lexsort(ref_pts.T)
        assert_allclose(pts[oa], ref_pts[ob], atol=1e-6)
        assert_allclose(w[oa], ref_w[ob], atol=1e-5)


class TestSimplexMatlabFixtures:
    """Cross-check against real MATLAB output (capture_region_rules.m).

    Every case captured for Simplex is a direct port (no "functionally
    covered, different algorithm" cases exist for this region, unlike
    Spherical_Surface), so every comparison here is value-for-value at
    float64 tolerance after order-independent lexsort, matching
    TestCubeMatlabFixtures's pattern exactly.
    """

    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    def test_second_order_matches_matlab(self, n):
        ref_pts, ref_w = _load_matlab_fixture(
            f"region_simplex_secondOrderSimplexCubPoints_n{n}.csv"
        )
        pts, w = simplex_cubature_points(n, 2)
        oa = np.lexsort(pts.T)
        ob = np.lexsort(ref_pts.T)
        assert_allclose(pts[oa], ref_pts[ob], atol=1e-10)
        assert_allclose(w[oa], ref_w[ob], atol=1e-9)

    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    def test_third_order_algorithm0_matches_matlab(self, n):
        ref_pts, ref_w = _load_matlab_fixture(
            f"region_simplex_thirdOrderSimplexCubPoints_n{n}_alg0.csv"
        )
        pts, w = simplex_cubature_points(n, 3, algorithm=0)
        oa = np.lexsort(pts.T)
        ob = np.lexsort(ref_pts.T)
        assert_allclose(pts[oa], ref_pts[ob], atol=1e-10)
        assert_allclose(w[oa], ref_w[ob], atol=1e-9)

    @pytest.mark.parametrize("n", [3, 4, 5])
    def test_fourth_order_matches_matlab(self, n):
        ref_pts, ref_w = _load_matlab_fixture(
            f"region_simplex_fourthOrderSimplexCubPoints_n{n}.csv"
        )
        pts, w = simplex_cubature_points(n, 4)
        if n == 4:
            # MATLAB's own zero-weight stripping (module docstring: not a
            # defect) already removes the dead block before it's written,
            # so no extra filtering is needed here, unlike the ninth-order
            # cube fixture -- but the true (post-strip) count is pinned
            # explicitly regardless, since this is the one n where the
            # general C(n+4,4) formula would be wrong.
            assert ref_pts.shape[0] == 40
        oa = np.lexsort(pts.T)
        ob = np.lexsort(ref_pts.T)
        assert_allclose(pts[oa], ref_pts[ob], atol=1e-9)
        assert_allclose(w[oa], ref_w[ob], atol=1e-8)

    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    def test_fifth_order_default_algorithm_matches_matlab(self, n):
        # MATLAB's fifthOrderSimplexCubPoints auto-selects its algorithm
        # from n when omitted (algdefault in the fixture name) -- the same
        # default dispatch simplex_cubature_points(n, 5) reproduces.
        ref_pts, ref_w = _load_matlab_fixture(
            f"region_simplex_fifthOrderSimplexCubPoints_n{n}_algdefault.csv"
        )
        pts, w = simplex_cubature_points(n, 5)
        oa = np.lexsort(pts.T)
        ob = np.lexsort(ref_pts.T)
        assert_allclose(pts[oa], ref_pts[ob], atol=1e-8)
        assert_allclose(w[oa], ref_w[ob], atol=1e-7)
