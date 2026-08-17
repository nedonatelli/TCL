"""Tests for Gaussian-weight cubature point generators.

PROPERTY class: a degree-d rule must integrate every monomial of total
degree <= d against N(0, I) exactly, and must FAIL on some degree d+1
monomial (sharpness -- guards against a vacuous exactness loop).
"""

import itertools
import os
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.special import gamma, gammaln

from pytcl.mathematical_functions.numerical_integration.cubature_points import (
    _gk_1d,
    _sphere_surface_points,
    cubature_point_moments,
    fifth_order_cubature_points,
    fourteenth_order_cubature_points,
    genz_keister_points,
    second_order_cubature_points,
    seventh_order_cubature_points,
    smolyak_points,
    sphere_surface_to_gauss_points,
    spherical_radial_points,
    student_t_cubature_points,
    transform_cubature_points,
)


def gaussian_moment(alpha):
    """E[prod x_i^alpha_i] for x ~ N(0, I): prod (a-1)!! over even a, else 0."""
    m = 1.0
    for a in alpha:
        if a % 2 == 1:
            return 0.0
        for k in range(a - 1, 0, -2):  # (a-1)!! = (a-1)(a-3)...1
            m *= k
    return m


def weighted_gaussian_moment(alpha, beta):
    """E[prod x_i^alpha_i * |x|^beta] for x ~ N(0, I_n), n = len(alpha).

    Closed form via polar coordinates: the surface integral of u^alpha over
    S^(n-1) is zero unless every alpha_i is even, and otherwise separates
    from the radial integral of the chi_n density times |x|^beta. beta=0
    reduces to gaussian_moment(alpha).
    """
    n = len(alpha)
    d = sum(alpha)
    if any(a % 2 == 1 for a in alpha):
        return 0.0
    surface_part = 1.0
    for a in alpha:
        surface_part *= gamma((a + 1) / 2.0)
    return (
        np.pi ** (-n / 2.0)
        * 2.0 ** ((d + beta) / 2.0)
        * gamma((d + beta + n) / 2.0)
        / gamma((n + d) / 2.0)
        * surface_part
    )


def student_t_moment(alpha, dof):
    """E[prod x_i^alpha_i] for x ~ multivariate Student-t(0, I, dof).

    Derivation: represent X = Z * sqrt(dof / W) with Z ~ N(0, I) and
    W ~ chi2(dof) independent (the standard normal-variance-mixture
    construction of the multivariate t). The scalar factor sqrt(dof / W)
    multiplies every coordinate of Z, so for alpha with total degree
    d = sum(alpha):

        prod x_i^alpha_i = (dof / W)^(d/2) * prod z_i^alpha_i

    and by independence of Z and W:

        E[prod x^alpha] = gaussian_moment(alpha) * dof^(d/2) * E[W^(-d/2)]

    A chi2(dof) variable has E[W^-s] = 2^-s * Gamma(dof/2 - s) / Gamma(dof/2)
    for dof > 2s. Substituting s = d/2:

        E[prod x^alpha] = gaussian_moment(alpha)
                           * (dof/2)^(d/2) * Gamma((dof-d)/2) / Gamma(dof/2)

    which requires dof > d. Sanity checks: d=2 reduces to
    dof/(dof-2) = E[x_i^2]; d=4 reduces to
    3 * dof^2 / ((dof-2)(dof-4)) = E[x_i^4], the standard univariate
    Student-t fourth moment.
    """
    d = sum(alpha)
    if d == 0:
        return 1.0
    g = gaussian_moment(alpha)
    if g == 0.0:
        # Any alpha with an odd exponent is 0 by the distribution's
        # reflection symmetry regardless of the dof existence threshold
        # below (which only bites the nonzero, all-even-exponent case).
        # Note this is a symmetry CONVENTION, not always a literal fact:
        # for 2 < dof <= d with odd d, E[|x|^d] itself is infinite (not
        # absolutely convergent), so the signed integral isn't a Lebesgue
        # expectation at all -- "0" here is its principal value, the same
        # value the antipodally-symmetric cubature rule produces, which is
        # what this oracle exists to validate against.
        return 0.0
    if not dof > d:
        raise ValueError(f"moment of degree {d} does not exist for dof={dof}")
    # log-space gamma ratio: gamma(dof/2) overflows float64 for dof > ~342,
    # which the dof -> inf convergence check below needs to stay finite for.
    log_ratio = gammaln((dof - d) / 2.0) - gammaln(dof / 2.0)
    return g * (dof / 2.0) ** (d / 2.0) * np.exp(log_ratio)


def rule_moment(points, weights, alpha):
    """Sum_i w_i * prod_j points[i, j]**alpha_j."""
    return float(np.sum(weights * np.prod(points ** np.asarray(alpha), axis=1)))


def monomials_up_to(n, degree):
    """All exponent tuples of length n with total degree <= degree."""
    for total in range(degree + 1):
        for alpha in itertools.product(range(total + 1), repeat=n):
            if sum(alpha) == total:
                yield alpha


def assert_rule_exact(points, weights, n, degree):
    for alpha in monomials_up_to(n, degree):
        assert_allclose(
            rule_moment(points, weights, alpha),
            gaussian_moment(alpha),
            atol=1e-9,
            err_msg=f"monomial {alpha} not integrated exactly",
        )


def assert_rule_exact_weighted(points, weights, n, degree, beta):
    for alpha in monomials_up_to(n, degree):
        assert_allclose(
            rule_moment(points, weights, alpha),
            weighted_gaussian_moment(alpha, beta),
            atol=1e-8,
            err_msg=f"monomial {alpha} (beta={beta}) not integrated exactly",
        )


class TestHelpers:
    def test_gaussian_moment_known_values(self):
        # E[x^2]=1, E[x^4]=3, E[x^6]=15, E[x^2 y^2]=1, odd -> 0
        assert gaussian_moment((2,)) == 1.0
        assert gaussian_moment((4,)) == 3.0
        assert gaussian_moment((6,)) == 15.0
        assert gaussian_moment((8,)) == 105.0
        assert gaussian_moment((2, 2)) == 1.0
        assert gaussian_moment((1, 2)) == 0.0

    def test_weighted_gaussian_moment_beta_0_matches_gaussian_moment(self):
        for alpha in monomials_up_to(3, 6):
            assert_allclose(
                weighted_gaussian_moment(alpha, 0.0), gaussian_moment(alpha)
            )

    def test_weighted_gaussian_moment_known_chi_moments(self):
        # E[|x|^beta] for x ~ N(0, I_n) is the beta-th absolute moment of
        # the chi_n distribution: 2^(beta/2) Gamma((n+beta)/2)/Gamma(n/2).
        # n=1: E[|x|^2] = Var(x) = 1.
        assert_allclose(weighted_gaussian_moment((0,), 2.0), 1.0)
        # n=3: E[|x|^2] = trace(I_3) = 3.
        assert_allclose(weighted_gaussian_moment((0, 0, 0), 2.0), 3.0)

    def test_student_t_moment_known_second_moments(self):
        # E[x_i^2] = dof/(dof-2); cross terms and odd moments vanish.
        assert_allclose(student_t_moment((2,), 6.0), 6.0 / 4.0)
        assert_allclose(student_t_moment((2, 0), 10.0), 10.0 / 8.0)
        assert student_t_moment((1, 1), 10.0) == 0.0
        assert student_t_moment((3, 0), 10.0) == 0.0
        assert student_t_moment((0, 0), 10.0) == 1.0

    def test_student_t_moment_known_fourth_moment(self):
        # Standard univariate Student-t fourth moment: 3 dof^2 / ((dof-2)(dof-4)).
        for dof in (6.0, 10.0, 20.0):
            expected = 3.0 * dof**2 / ((dof - 2.0) * (dof - 4.0))
            assert_allclose(student_t_moment((4,), dof), expected)

    def test_student_t_moment_known_mixed_second_moment(self):
        # E[x_i^2 * x_j^2] for i != j: x_i and x_j are uncorrelated but NOT
        # independent (they share the mixing variable W), so the correct
        # value is NOT the naive product of the two marginal E[x_i^2]
        # terms. Closed form via the same chi2-mixture derivation as the
        # docstring above: E[x_i^2 x_j^2] = gaussian_moment((2,2)) * dof^2
        # * E[W^-2] = dof^2 / ((dof-2)*(dof-4)) (gaussian_moment((2,2))=1).
        for dof in (10.0, 20.0):
            expected = dof**2 / ((dof - 2.0) * (dof - 4.0))
            assert_allclose(student_t_moment((2, 2), dof), expected)
            # Discriminating power: a regression that reintroduced
            # independence-factoring for mixed even-degree moments would
            # give this naive product instead -- confirm it does NOT
            # satisfy the assertion above (i.e. this test isn't vacuous).
            naive_independence_factored = (dof / (dof - 2.0)) ** 2
            assert abs(naive_independence_factored - expected) > 0.1

    def test_student_t_moment_converges_to_gaussian_as_dof_grows(self):
        # dof -> inf recovers the N(0, I) moments.
        for alpha in [(2,), (4,), (2, 2), (6,)]:
            assert_allclose(
                student_t_moment(alpha, 1e8), gaussian_moment(alpha), rtol=1e-5
            )

    def test_student_t_moment_nonexistent_degree_raises(self):
        # The 4th moment does not exist for dof <= 4.
        with pytest.raises(ValueError):
            student_t_moment((4,), 4.0)
        with pytest.raises(ValueError):
            student_t_moment((4,), 3.0)


class TestTransformCubaturePoints:
    def test_affine_map_moments(self):
        # Any exact-through-degree-2 rule pushed through the transform must
        # reproduce the target mean and covariance.
        n = 3
        rng = np.random.default_rng(7)
        A = rng.normal(size=(n, n))
        cov = A @ A.T
        mean = rng.normal(size=n)
        sqrt_cov = np.linalg.cholesky(cov)

        # 3rd-degree unit rule: points +-sqrt(n) e_i, weights 1/(2n)
        unit = np.sqrt(n) * np.vstack([np.eye(n), -np.eye(n)])
        w = np.full(2 * n, 1.0 / (2 * n))

        pts, wts = transform_cubature_points(unit, w, mean, sqrt_cov)
        assert pts.shape == unit.shape
        assert_allclose(wts, w)  # weights unchanged
        assert_allclose(np.sum(wts[:, None] * pts, axis=0), mean, atol=1e-12)
        resid = pts - mean
        assert_allclose(resid.T @ (wts[:, None] * resid), cov, atol=1e-10)

    def test_shape_mismatch_raises(self):
        unit = np.zeros((4, 2))
        w = np.full(3, 1 / 3)  # wrong length
        with pytest.raises(ValueError):
            transform_cubature_points(unit, w, np.zeros(2), np.eye(2))
        with pytest.raises(ValueError):
            transform_cubature_points(
                np.zeros((4, 2)), np.full(4, 0.25), np.zeros(3), np.eye(3)
            )


class TestSecondOrder:
    """MATLAB's ``secondOrderCubPoints`` -- the scaled unscented
    transformation's minimal (n+2)-point spherical simplex rule. See the
    reconciliation note in ``second_order_cubature_points``'s docstring:
    this is a genuinely different construction from both
    ``unscented_transform_points`` (2n+1 points) and
    ``ckf_spherical_cubature_points`` (2n points), not an alternate
    parameterization of either -- exact through degree 2 only, not 3.
    """

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
    def test_shape_and_weight_sum(self, n):
        pts, w = second_order_cubature_points(n)
        assert pts.shape == (n + 2, n)
        assert_allclose(w.sum(), 1.0, atol=1e-12)

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
    def test_exact_through_degree_2(self, n):
        pts, w = second_order_cubature_points(n)
        assert_rule_exact(pts, w, n, degree=2)

    @pytest.mark.parametrize("w0,alpha", [(1 / 3, 1.0), (0.7, 1.0), (0.2, 1.3)])
    def test_exact_through_degree_2_nondefault_params(self, w0, alpha):
        # Mean/covariance exactness must hold regardless of w0/alpha.
        # All three (w0, alpha) pairs here give a positive center weight
        # (+0.333, +0.700, +0.527); the negative-center-weight case is
        # covered separately by
        # test_negative_center_weight_is_not_suppressed, which also checks
        # degree-2 exactness there.
        n = 4
        pts, w = second_order_cubature_points(n, w0=w0, alpha=alpha)
        assert_allclose(w.sum(), 1.0, atol=1e-12)
        assert_rule_exact(pts, w, n, degree=2)

    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    def test_sharpness_degree_3_fails(self, n):
        # Unlike the symmetric 2n+1- and 2n-point rules, this (n+2)-point
        # simplex set is NOT antipodally symmetric in every dimension, so
        # third moments are not zeroed out in general. n=3 exhibits it on
        # x2: E[x2^3] = sqrt(2) against a true value of 0 -- verified
        # directly here rather than merely asserted. n=1 is excluded: its
        # single dimension IS the one axis with a mirrored pair, so
        # E[x^3] = 0 there even though the rule is still only degree-2
        # exact in general (n >= 2).
        pts, w = second_order_cubature_points(n)
        alpha = (0,) * (n - 1) + (3,)
        rule_val = rule_moment(pts, w, alpha)
        assert abs(rule_val - gaussian_moment(alpha)) > 1e-6

    def test_sharpness_degree_3_known_value_n3(self):
        pts, w = second_order_cubature_points(3)
        assert_allclose(rule_moment(pts, w, (0, 0, 3)), np.sqrt(2.0), atol=1e-12)

    def test_negative_center_weight_is_not_suppressed(self):
        # alpha**2 < 1 - w0 drives the center weight negative; this is an
        # inherent property of the scaled construction, documented in the
        # docstring, not clamped or suppressed here.
        pts, w = second_order_cubature_points(3, w0=1 / 3, alpha=0.5)
        assert w[0] < 0.0
        assert_allclose(w[0], -5.0 / 3.0, atol=1e-12)
        assert_allclose(w.sum(), 1.0, atol=1e-12)
        # Degree-2 exactness must still hold with the negative weight.
        assert_rule_exact(pts, w, 3, degree=2)

    def test_default_alpha_positive_weights(self):
        # alpha=1 (unscaled spherical simplex points) keeps every weight
        # positive for any valid w0, unlike the scaled case above.
        for w0 in (0.1, 1 / 3, 0.5, 0.9):
            _, w = second_order_cubature_points(5, w0=w0, alpha=1.0)
            assert np.all(w > 0.0)

    def test_published_values_n1(self):
        # Hand-derivable at n=1: points at 0, -1/sqrt(2 w0), 1/sqrt(2 w0),
        # rescaled by 1/sqrt((1/w0 - 1)/2); default w0=1/3, alpha=1 gives
        # points at 0, -sqrt(3/2), sqrt(3/2) and weights [1/3, 1/3, 1/3].
        pts, w = second_order_cubature_points(1)
        assert_allclose(
            sorted(pts.ravel()), sorted([0.0, -np.sqrt(1.5), np.sqrt(1.5)]), atol=1e-12
        )
        assert_allclose(w, np.full(3, 1.0 / 3.0), atol=1e-12)

    def test_invalid_n_raises(self):
        with pytest.raises(ValueError):
            second_order_cubature_points(0)

    @pytest.mark.parametrize("w0", [0.0, 1.0, -0.1, 1.1])
    def test_invalid_w0_raises(self, w0):
        with pytest.raises(ValueError):
            second_order_cubature_points(3, w0=w0)

    @pytest.mark.parametrize("alpha", [0.0, -1.0])
    def test_invalid_alpha_raises(self, alpha):
        with pytest.raises(ValueError):
            second_order_cubature_points(3, alpha=alpha)

    def test_point_count_differs_from_symmetric_ut(self):
        # Documents the reconciliation finding: this is not the same
        # construction as unscented_transform_points under a different
        # parameterization -- the point counts themselves differ (n+2 vs
        # 2n+1), so no choice of w0/alpha can make them coincide.
        from pytcl.mathematical_functions.numerical_integration.quadrature import (
            unscented_transform_points,
        )

        n = 4
        pts, _ = second_order_cubature_points(n)
        ut_pts, _, _ = unscented_transform_points(n)
        assert pts.shape[0] == n + 2
        assert ut_pts.shape[0] == 2 * n + 1
        assert pts.shape[0] != ut_pts.shape[0]


class TestFifthOrder:
    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6])
    def test_exact_through_degree_5(self, n):
        pts, w = fifth_order_cubature_points(n)
        assert pts.shape == (2 * n * n + 1, n)
        assert_allclose(w.sum(), 1.0, atol=1e-12)
        assert_rule_exact(pts, w, n, degree=5)

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6])
    def test_sharpness_degree_6_fails(self, n):
        # E[x1^6] = 15 must NOT be matched -- otherwise the exactness test
        # above could be vacuous.
        pts, w = fifth_order_cubature_points(n)
        alpha = (6,) + (0,) * (n - 1)
        assert abs(rule_moment(pts, w, alpha) - 15.0) > 1e-3

    def test_antipodal_symmetry(self):
        pts, _ = fifth_order_cubature_points(4)
        nonzero = pts[np.any(pts != 0.0, axis=1)]
        # every non-origin point's negation is also a point
        for p in nonzero:
            assert np.any(np.all(np.isclose(nonzero, -p), axis=1))

    def test_negative_axis_weights_above_n4_documented(self):
        # (4 - n)/(2 (n+2)^2) < 0 for n > 4: present, not suppressed.
        _, w = fifth_order_cubature_points(5)
        assert w.min() < 0.0

    def test_invalid_n_raises(self):
        with pytest.raises(ValueError):
            fifth_order_cubature_points(0)

    def test_published_values_n2(self):
        # REFERENCE: Stroud (1971), rule E_n^{r^2} 5-3 at n = 2:
        # lambda = sqrt(n+2) = 2, mu = sqrt((n+2)/2) = sqrt(2),
        # w_center = 2/(n+2) = 1/2, w_axis = (4-n)/(2(n+2)^2) = 1/16,
        # w_pair = 1/(n+2)^2 = 1/16.
        pts, w = fifth_order_cubature_points(2)
        assert_allclose(
            sorted(np.abs(pts).max(axis=1)),
            sorted([0.0] + [2.0] * 4 + [np.sqrt(2.0)] * 4),
            atol=1e-12,
        )
        assert_allclose(w[0], 0.5, atol=1e-15)  # center
        assert_allclose(w[1:5], 1.0 / 16.0, atol=1e-15)  # axis
        assert_allclose(w[5:], 1.0 / 16.0, atol=1e-15)  # pairs

    def test_reference_expectation_vs_tensor_gauss_hermite(self):
        # REFERENCE: E[cos(a.T x)] = exp(-|a|^2 / 2) has a closed form and is
        # not polynomial; compare rule vs dense tensor GH vs closed form.
        # cubature_gauss_hermite is RAW physicists' GH: scale by sqrt(2),
        # normalize by pi^(n/2).
        from pytcl.mathematical_functions.numerical_integration import (
            cubature_gauss_hermite,
        )

        n = 3
        a = np.array([0.3, -0.5, 0.2])
        exact = np.exp(-0.5 * float(a @ a))

        pts, w = fifth_order_cubature_points(n)
        rule_val = float(np.sum(w * np.cos(pts @ a)))

        gh_pts, gh_w = cubature_gauss_hermite(n, 10)
        gh_val = float(
            np.sum(gh_w * np.cos((np.sqrt(2.0) * gh_pts) @ a)) / np.pi ** (n / 2)
        )

        assert_allclose(gh_val, exact, atol=1e-9)  # oracle sanity
        assert_allclose(rule_val, exact, atol=5e-3)  # degree-5 approximation


class TestSeventhOrder:
    @pytest.mark.parametrize("n", [3, 4, 5, 6])
    def test_exact_through_degree_7(self, n):
        pts, w = seventh_order_cubature_points(n)
        assert_allclose(w.sum(), 1.0, atol=1e-12)
        assert_rule_exact(pts, w, n, degree=7)

    @pytest.mark.parametrize("n", [3, 4, 5, 6])
    def test_sharpness_degree_8_fails(self, n):
        pts, w = seventh_order_cubature_points(n)
        alpha = (8,) + (0,) * (n - 1)
        assert abs(rule_moment(pts, w, alpha) - 105.0) > 1e-3

    @pytest.mark.parametrize("n", [3, 4, 5, 6])
    def test_point_count_polynomial(self, n):
        # The O(n^3) fully-symmetric family, not an exponential product rule.
        pts, _ = seventh_order_cubature_points(n)
        assert len(pts) < 8 * n**3 + 1

    def test_antipodal_symmetry(self):
        pts, _ = seventh_order_cubature_points(4)
        nonzero = pts[np.any(pts != 0.0, axis=1)]
        # every non-origin point's negation is also a point
        for p in nonzero:
            assert np.any(np.all(np.isclose(nonzero, -p), axis=1))

    def test_invalid_n_raises(self):
        with pytest.raises(ValueError):
            seventh_order_cubature_points(0)
        # n=2 no longer raises under the default dispatch (it now selects
        # algorithm 2, MATLAB parity -- see TestSeventhOrderAlgorithms);
        # algorithm 0 explicitly still requires n>=3.
        with pytest.raises(ValueError):
            seventh_order_cubature_points(2, algorithm=0)


class TestSphericalRadial:
    @pytest.mark.parametrize("n", [1, 2, 3, 4])
    @pytest.mark.parametrize("degree", [3, 5, 7, 9])
    def test_exact_through_degree(self, n, degree):
        pts, w = spherical_radial_points(n, degree)
        assert_allclose(w.sum(), 1.0, atol=1e-12)
        assert_rule_exact(pts, w, n, degree)

    @pytest.mark.parametrize("n,degree", [(2, 3), (3, 5), (2, 7)])
    def test_sharpness(self, n, degree):
        pts, w = spherical_radial_points(n, degree)
        alpha = (degree + 1,) + (0,) * (n - 1)
        assert abs(rule_moment(pts, w, alpha) - gaussian_moment(alpha)) > 1e-4

    def test_degree_3_matches_known_radius(self):
        # Single radial node at r = sqrt(n) -- same radius the CKF uses.
        pts, _ = spherical_radial_points(3, 3)
        radii = np.linalg.norm(pts, axis=1)
        assert_allclose(radii, np.sqrt(3.0), atol=1e-12)

    def test_antipodal_symmetry(self):
        pts, _ = spherical_radial_points(3, 5)
        nonzero = pts[np.any(pts != 0.0, axis=1)]
        # every non-origin point's negation is also a point
        for p in nonzero:
            assert np.any(np.all(np.isclose(nonzero, -p, atol=1e-10), axis=1))

    def test_invalid_degree_raises(self):
        with pytest.raises(ValueError):
            spherical_radial_points(2, 4)  # even
        with pytest.raises(ValueError):
            spherical_radial_points(2, 1)  # < 3
        with pytest.raises(ValueError):
            spherical_radial_points(0, 3)  # bad dim
        with pytest.raises(ValueError):
            spherical_radial_points(2, 3.7)  # non-integer


class TestFourteenthOrder:
    """MATLAB's ``fourteenthOrderCubPoints`` hardcodes
    ``if(numDim~=3) error('Only 3D points are supported'); end`` -- there is no
    n-dimensional generalization in the source; the 72-point spherical-surface
    rule it builds on (Stroud's U3 14-1) is specific to S^2. So n=3 is not a
    lower bound here, it is the only supported dimension.
    """

    def test_shape_and_weight_sum(self):
        pts, w = fourteenth_order_cubature_points(3)
        assert pts.shape == (288, 3)  # 72 surface points * 4 radial nodes
        assert_allclose(w.sum(), 1.0, atol=1e-10)

    def test_exact_through_degree_14(self):
        # Exhaustive: C(14+3,3) = 680 monomials against 288 points -- cheap
        # (well under a second), so the plan's random-sampling fallback for
        # large-n degree-14 grids is not needed: MATLAB only supports n=3.
        # Uses the shared assert_rule_exact (atol=1e-9); measured worst
        # err/tol ratio here is 5.3e-2, comfortably inside it.
        pts, w = fourteenth_order_cubature_points(3)
        assert_rule_exact(pts, w, 3, degree=14)

    def test_sharpness_degree_15_fails(self):
        # Every degree-15 monomial has at least one odd exponent (a tuple of
        # all-even exponents can't sum to the odd number 15), so its true
        # Gaussian moment is 0 for ANY rule order -- a single-axis probe like
        # (15,0,0) is NOT a useful sharpness test here, because this rule's
        # 60-point icosahedral-symmetry block (see
        # _fourteenth_order_unit_sphere_points_3d) is built from a *chiral*
        # rotation orbit, not individually antipodal pairs (confirmed: 240 of
        # the 288 points have no exact negation among the others), and that
        # weaker symmetry still exactly zeroes single-axis odd moments by
        # coincidence. A mixed-exponent monomial is not so lucky and exposes
        # the rule's actual degree-14 limit: (1, 1, 13) rule-integrates to
        # ~-127.3 against a true value of 0.
        pts, w = fourteenth_order_cubature_points(3)
        alpha = (1, 1, 13)
        assert abs(rule_moment(pts, w, alpha) - gaussian_moment(alpha)) > 1e-3

    def test_sharpness_degree_16_fails(self):
        pts, w = fourteenth_order_cubature_points(3)
        alpha = (16, 0, 0)
        assert abs(rule_moment(pts, w, alpha) - gaussian_moment(alpha)) > 1e-3

    @pytest.mark.parametrize("n", [1, 2, 4, 5])
    def test_invalid_dimension_raises(self, n):
        with pytest.raises(ValueError):
            fourteenth_order_cubature_points(n)

    def test_invalid_beta_raises(self):
        with pytest.raises(ValueError):
            fourteenth_order_cubature_points(3, beta=-3.0)  # <= -n

    def test_axis_block_antipodal_symmetry(self):
        # sphere_surface_to_gauss_points tiles the 72 surface points (12
        # axis-plane points followed by 60 icosahedral-symmetry points, see
        # _fourteenth_order_unit_sphere_points_3d) once per radial node, so
        # the axis points occupy the first 12 slots of each 72-point block.
        # Only those are individually antipodal-paired -- see
        # test_sharpness_degree_15_fails for why the other 60 per block are
        # not.
        pts, _ = fourteenth_order_cubature_points(3)
        axis_pts = pts.reshape(4, 72, 3)[:, :12, :].reshape(-1, 3)
        for p in axis_pts:
            assert np.any(np.all(np.isclose(pts, -p, atol=1e-10), axis=1))

    @pytest.mark.parametrize("beta", [1.0, 2.0, -0.5])
    def test_nonzero_beta_exact_through_degree_14(self, beta):
        pts, w = fourteenth_order_cubature_points(3, beta=beta)
        assert_allclose(
            w.sum(),
            2.0 ** (beta / 2.0) * gamma((3 + beta) / 2.0) / gamma(3 / 2.0),
            atol=1e-8,
        )
        for alpha in monomials_up_to(3, 14):
            assert_allclose(
                rule_moment(pts, w, alpha),
                weighted_gaussian_moment(alpha, beta),
                rtol=1e-6,
                atol=1e-7,
                err_msg=f"monomial {alpha} (beta={beta}) not integrated exactly",
            )

    @pytest.mark.parametrize("beta", [1.0, 2.0, -0.5])
    def test_nonzero_beta_sharpness(self, beta):
        # Mirrors test_sharpness_degree_15_fails's beta=0 mixed-exponent
        # probe: (1, 1, 13) is degree 14, still within bounds, but exposes
        # the rule's true degree-14 limit even under the beta weighting
        # (measured |rule - true| is ~5.3e2/2.3e3/6.3e1 for beta=1/2/-0.5).
        pts, w = fourteenth_order_cubature_points(3, beta=beta)
        alpha = (1, 1, 13)
        assert (
            abs(rule_moment(pts, w, alpha) - weighted_gaussian_moment(alpha, beta))
            > 1.0
        )


class TestSphereSurfaceToGaussPoints:
    """Adapter: spherical-surface rule -> Gaussian(-times-|x|^beta) rule."""

    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    @pytest.mark.parametrize("degree", [3, 5, 7])
    def test_beta_0_exact_through_degree(self, n, degree):
        surf_pts, surf_w = _sphere_surface_points(n, degree)
        pts, w = sphere_surface_to_gauss_points(surf_pts, surf_w, degree)
        assert_allclose(w.sum(), 1.0, atol=1e-10)
        assert_rule_exact(pts, w, n, degree)

    @pytest.mark.parametrize("n,degree", [(2, 3), (3, 5), (5, 7)])
    def test_beta_0_sharpness(self, n, degree):
        surf_pts, surf_w = _sphere_surface_points(n, degree)
        pts, w = sphere_surface_to_gauss_points(surf_pts, surf_w, degree)
        alpha = (degree + 1,) + (0,) * (n - 1)
        assert abs(rule_moment(pts, w, alpha) - gaussian_moment(alpha)) > 1e-4

    def test_beta_0_default_matches_explicit(self):
        surf_pts, surf_w = _sphere_surface_points(3, 5)
        pts1, w1 = sphere_surface_to_gauss_points(surf_pts, surf_w, 5)
        pts2, w2 = sphere_surface_to_gauss_points(surf_pts, surf_w, 5, beta=0.0)
        assert np.array_equal(pts1, pts2)
        assert np.array_equal(w1, w2)

    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    @pytest.mark.parametrize("degree", [3, 5, 7])
    @pytest.mark.parametrize("beta", [1.0, 2.0, -0.5, 3.0])
    def test_nonzero_beta_exact_through_degree(self, n, degree, beta):
        surf_pts, surf_w = _sphere_surface_points(n, degree)
        pts, w = sphere_surface_to_gauss_points(surf_pts, surf_w, degree, beta=beta)
        # Total weight is E[|x|^beta] under N(0, I), not 1 in general.
        assert_allclose(
            w.sum(),
            2.0 ** (beta / 2.0) * gamma((n + beta) / 2.0) / gamma(n / 2.0),
            atol=1e-9,
        )
        assert_rule_exact_weighted(pts, w, n, degree, beta)

    @pytest.mark.parametrize("n,degree,beta", [(2, 3, 1.0), (3, 5, 2.0), (4, 7, -0.5)])
    def test_nonzero_beta_sharpness(self, n, degree, beta):
        surf_pts, surf_w = _sphere_surface_points(n, degree)
        pts, w = sphere_surface_to_gauss_points(surf_pts, surf_w, degree, beta=beta)
        alpha = (degree + 1,) + (0,) * (n - 1)
        assert (
            abs(rule_moment(pts, w, alpha) - weighted_gaussian_moment(alpha, beta))
            > 1e-4
        )

    def test_invalid_beta_raises(self):
        surf_pts, surf_w = _sphere_surface_points(3, 3)
        with pytest.raises(ValueError):
            sphere_surface_to_gauss_points(surf_pts, surf_w, 3, beta=-3.0)  # <= -n
        with pytest.raises(ValueError):
            sphere_surface_to_gauss_points(surf_pts, surf_w, 3, beta=-5.0)

    def test_mismatched_weights_shape_raises(self):
        # This is the only public function in the module that used to
        # accept a mismatched points/weights pair silently -- e.g. a
        # (2, 12)-shaped weights array against 12 surface points used to
        # return (12, 3) points paired with (2, 12) weights and no
        # exception. Now validated the same way transform_cubature_points
        # and cubature_point_moments already are.
        surf_pts, surf_w = _sphere_surface_points(3, 3)
        with pytest.raises(ValueError):
            sphere_surface_to_gauss_points(surf_pts, np.tile(surf_w, (2, 1)), 3)

    def test_wrong_length_weights_raises(self):
        surf_pts, surf_w = _sphere_surface_points(3, 3)
        with pytest.raises(ValueError):
            sphere_surface_to_gauss_points(surf_pts, surf_w[:-1], 3)

    def test_noninteger_degree_raises(self):
        # Unlike spherical_radial_points, EVEN degree is legitimately
        # accepted here (fourteenth_order_cubature_points calls with
        # degree=14) -- see the docstring's `degree` parameter -- but a
        # non-integer degree is not a valid "polynomial degree" under
        # either function and must be rejected the same way.
        surf_pts, surf_w = _sphere_surface_points(3, 3)
        with pytest.raises(ValueError):
            sphere_surface_to_gauss_points(surf_pts, surf_w, 3.7)

    def test_even_degree_accepted(self):
        # The construction fourteenth_order_cubature_points relies on:
        # degree need not be odd here.
        surf_pts, surf_w = _sphere_surface_points(3, 4)
        pts, w = sphere_surface_to_gauss_points(surf_pts, surf_w, 4)
        assert_allclose(w.sum(), 1.0, atol=1e-8)


class TestSphericalRadialBetaGeneralization:
    """spherical_radial_points(n, degree, beta=0.0) -- pre-existing callers
    (beta omitted or 0.0) must be unaffected by the beta generalization.

    The durable, platform-independent invariant is the runtime one below:
    ``beta=0.0`` is bit-identical to omitting ``beta``, computed fresh on
    whatever platform runs the test. Bit-identity against the
    PRE-generalization code was verified once, at migration time, on the
    machine that performed the port (arm64, scipy 1.17.1) via stored
    SHA-256 digests; those digests were then removed because they encode
    that one environment's floating-point bits -- scipy version bumps
    alone (1.17.1 -> 1.18.0 changes roots_genlaguerre/roots_jacobi
    output) and architecture/libm differences flip them, so as a CI
    assertion they claimed far more than was ever measured.
    """

    _DEFAULT_CASES = [(n, degree) for n in (1, 2, 3, 4) for degree in (3, 5, 7, 9)]

    @pytest.mark.parametrize("n,degree", _DEFAULT_CASES)
    def test_explicit_beta_0_bit_identical_to_omitted(self, n, degree):
        pts1, w1 = spherical_radial_points(n, degree)
        pts2, w2 = spherical_radial_points(n, degree, beta=0.0)
        assert np.array_equal(pts1, pts2)
        assert np.array_equal(w1, w2)

    @pytest.mark.parametrize("n", [2, 3, 4])
    @pytest.mark.parametrize("degree", [3, 5, 7])
    @pytest.mark.parametrize("beta", [1.0, 2.0, -0.5])
    def test_nonzero_beta_exact_through_degree(self, n, degree, beta):
        pts, w = spherical_radial_points(n, degree, beta=beta)
        assert_allclose(
            w.sum(),
            2.0 ** (beta / 2.0) * gamma((n + beta) / 2.0) / gamma(n / 2.0),
            atol=1e-9,
        )
        assert_rule_exact_weighted(pts, w, n, degree, beta)

    @pytest.mark.parametrize("n,degree,beta", [(2, 3, 1.0), (3, 5, 2.0), (4, 7, -0.5)])
    def test_nonzero_beta_sharpness(self, n, degree, beta):
        # TestSphericalRadialBetaGeneralization above only had exactness
        # coverage for beta != 0; mirrors
        # TestSphereSurfaceToGaussPoints.test_nonzero_beta_sharpness's
        # single-axis degree+1 probe (measured |rule - true| ~2.8/21.6/10.2
        # for these three (n, degree, beta) combinations).
        pts, w = spherical_radial_points(n, degree, beta=beta)
        alpha = (degree + 1,) + (0,) * (n - 1)
        assert (
            abs(rule_moment(pts, w, alpha) - weighted_gaussian_moment(alpha, beta))
            > 1.0
        )

    def test_invalid_beta_raises(self):
        with pytest.raises(ValueError):
            spherical_radial_points(3, 3, beta=-5.0)


class TestCKFWithCustomPoints:
    def _setup(self):
        x = np.array([1.0, -0.5, 0.2])
        P = np.diag([0.20, 0.30, 0.15])
        Q = np.eye(3) * 0.01

        def f(v):  # mildly nonlinear dynamics
            return np.array([v[0] + 0.1 * v[1], v[1] + 0.05 * v[0] ** 2, v[2]])

        return x, P, Q, f

    def test_explicit_none_matches_default(self):
        # Confirms explicitly passing points=None, weights=None is
        # equivalent to omitting them entirely -- both hit the same
        # default-rule branch in ckf_predict. This does NOT guard against
        # drift from the pre-branch CKF behavior (both calls run on the
        # same new code); that regression guard is the pre-existing CKF
        # unit/validation tests, which are unchanged by this branch.
        from pytcl.dynamic_estimation.kalman.unscented import ckf_predict

        x, P, Q, f = self._setup()
        base = ckf_predict(x, P, f, Q)
        again = ckf_predict(x, P, f, Q, points=None, weights=None)
        assert_allclose(again.x, base.x, atol=0)
        assert_allclose(again.P, base.P, atol=0)

    def test_fifth_order_matches_tensor_gauss_hermite_moments(self):
        # REFERENCE: dense tensor GH (scaled to N(0,I) convention) is the
        # oracle for the propagated mean/covariance.
        from pytcl.dynamic_estimation.kalman.unscented import ckf_predict
        from pytcl.mathematical_functions.numerical_integration import (
            cubature_gauss_hermite,
        )
        from pytcl.mathematical_functions.numerical_integration.cubature_points import (
            fifth_order_cubature_points,
        )

        x, P, Q, f = self._setup()
        pts5, w5 = fifth_order_cubature_points(3)
        pred5 = ckf_predict(x, P, f, Q, points=pts5, weights=w5)

        gh_pts, gh_w = cubature_gauss_hermite(3, 8)
        unit = np.sqrt(2.0) * gh_pts
        wts = gh_w / np.pi**1.5
        S = np.linalg.cholesky(P)
        prop = np.array([f(x + S @ p) for p in unit])
        mean_ref = np.sum(wts[:, None] * prop, axis=0)
        resid = prop - mean_ref
        cov_ref = resid.T @ (wts[:, None] * resid) + Q

        # f is quadratic, so a degree-5 rule propagates mean and covariance
        # of a quadratic map exactly up to the rule's degree; tight tol.
        assert_allclose(pred5.x, mean_ref, atol=1e-10)
        assert_allclose(pred5.P, cov_ref, atol=1e-8)

    def test_negative_weights_covariance_is_sign_safe(self):
        # n = 5 makes the fifth-order axis weight negative; the covariance
        # assembly must not sqrt() weights.
        from pytcl.dynamic_estimation.kalman.unscented import ckf_predict
        from pytcl.mathematical_functions.numerical_integration.cubature_points import (
            fifth_order_cubature_points,
        )

        n = 5
        x = np.zeros(n)
        P = np.eye(n)
        Q = np.zeros((n, n))
        F = np.diag([1.0, 2.0, 0.5, 1.5, 1.0])
        pts, w = fifth_order_cubature_points(n)
        assert w.min() < 0
        pred = ckf_predict(x, P, lambda v: F @ v, Q, points=pts, weights=w)
        # Linear map: exact answers are F x and F P F.T
        assert_allclose(pred.x, F @ x, atol=1e-10)
        assert_allclose(pred.P, F @ P @ F.T, atol=1e-9)
        assert not np.any(np.isnan(pred.P))

    def test_update_with_custom_points(self):
        from pytcl.dynamic_estimation.kalman.unscented import ckf_update
        from pytcl.mathematical_functions.numerical_integration.cubature_points import (
            fifth_order_cubature_points,
        )

        x = np.array([3.0, 1.0])
        P = np.eye(2) * 0.5
        z = np.array([3.1])
        R = np.array([[0.1]])
        pts, w = fifth_order_cubature_points(2)
        upd = ckf_update(x, P, z, lambda v: np.array([v[0]]), R, points=pts, weights=w)
        assert upd.x.shape == (2,)
        # Linear measurement: must agree with the default CKF closely.
        from pytcl.dynamic_estimation.kalman.unscented import (
            ckf_update as default_update,
        )

        base = default_update(x, P, z, lambda v: np.array([v[0]]), R)
        assert_allclose(upd.x, base.x, atol=1e-9)
        assert_allclose(upd.P, base.P, atol=1e-9)

    def test_shape_mismatch_raises(self):
        from pytcl.dynamic_estimation.kalman.unscented import ckf_predict

        x, P, Q, f = self._setup()
        pts, w = fifth_order_cubature_points(2)  # wrong dimension (2 != 3)
        with pytest.raises(ValueError):
            ckf_predict(x, P, f, Q, points=pts, weights=w)
        pts3, w3 = fifth_order_cubature_points(3)
        with pytest.raises(ValueError):
            ckf_predict(x, P, f, Q, points=pts3, weights=w3[:-1])
        with pytest.raises(ValueError):
            ckf_predict(x, P, f, Q, points=pts3)  # points without weights


class TestGenzKeister:
    """MATLAB's ``GenzKeisterPoints`` -- a fully-symmetric interpolatory
    rule (Genz [2]_ extended by Genz and Keister [1]_) built from a single
    NESTED 1-D generator sequence, so the n-D point sets nest across
    increasing ``m`` too. This nesting is the entire reason this rule is
    worth having (it is the prerequisite for Smolyak sparse grids, deferred
    to a follow-on effort) -- see
    `test_nesting_holds_for_every_consecutive_pair_except_the_last` below
    and the "Notes" section of ``genz_keister_points``'s docstring, which
    the (flat, not class-mirrored) methods below cover point for point:

    - the *generic* exactness bound, degree ``2m + 1``, holding for any n
      and any m EXCEPT the maximum m of each algorithm
      (`test_exact_through_generic_degree`,
      `test_sharp_at_generic_degree_plus_one`);
    - the n=1-only *bonus* at specific "milestone" m values, where
      single-axis moments (equivalently, the n=1 rule) are exact well
      beyond ``2m + 1`` (`test_milestone_exact_through_bonus_degree`,
      `test_milestone_sharp_at_bonus_degree_plus_one`);
    - the documented precision/nesting breakdown at the very top of each
      algorithm's valid ``m`` range
      (`test_nesting_breaks_at_the_documented_top_of_range`,
      `test_top_of_range_precision_documented`).
    """

    # ---- input validation ----

    def test_invalid_n_raises(self):
        with pytest.raises(ValueError):
            genz_keister_points(0, 1)

    @pytest.mark.parametrize(
        "algorithm,bad_m", [(0, 0), (0, 18), (0, -1), (1, 0), (1, 16), (1, -3)]
    )
    def test_invalid_m_raises(self, algorithm, bad_m):
        with pytest.raises(ValueError):
            genz_keister_points(1, bad_m, algorithm=algorithm)

    def test_noninteger_m_raises(self):
        with pytest.raises(ValueError):
            genz_keister_points(1, 1.5)

    @pytest.mark.parametrize("algorithm", [2, -1, 3, "nu"])
    def test_invalid_algorithm_raises(self, algorithm):
        # Covers MATLAB's third path (a custom nu-increment vector), which
        # this port deliberately does not implement -- see the docstring's
        # `algorithm` parameter description for why.
        with pytest.raises(ValueError):
            genz_keister_points(1, 1, algorithm=algorithm)

    # ---- structural invariants across the full valid m range ----

    @pytest.mark.parametrize("algorithm,max_m", [(0, 17), (1, 15)])
    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_weight_sum_full_range(self, algorithm, max_m, n):
        for m in range(1, max_m + 1):
            _, w = genz_keister_points(n, m, algorithm=algorithm)
            assert_allclose(
                w.sum(), 1.0, atol=1e-8, err_msg=f"n={n} m={m} algorithm={algorithm}"
            )

    @pytest.mark.parametrize("algorithm,max_m", [(0, 17), (1, 15)])
    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_antipodal_symmetry_full_range(self, algorithm, max_m, n):
        # Every point's full negation (all coordinates flipped at once) is
        # also a point, with the SAME weight -- guaranteed by construction
        # (PMCombos enumerates every sign combination of a partition's
        # nonzero magnitudes, which includes the all-negative one), not
        # merely a numerical coincidence.
        for m in range(1, max_m + 1):
            pts, w = genz_keister_points(n, m, algorithm=algorithm)
            for p, wt in zip(pts, w):
                match = np.all(np.isclose(pts, -p, atol=1e-8), axis=1)
                assert np.any(match), f"n={n} m={m} point {p} has no antipodal partner"
                assert_allclose(w[match], wt, atol=1e-8)

    def test_negative_weights_present_and_not_suppressed(self):
        # Genz-Keister rules commonly have negative weights; documented in
        # the docstring's Returns section, never clamped or dropped (only
        # numerically-zero weights are dropped, via eps_val).
        _, w = genz_keister_points(1, 9)
        assert w.min() < 0.0

    def test_points_are_rows_default_algorithm(self):
        pts, w = genz_keister_points(3, 2)
        assert pts.ndim == 2 and pts.shape[1] == 3
        assert w.shape == (pts.shape[0],)

    def test_eps_val_controls_pruning(self):
        # A very large eps_val prunes aggressively (fewer points survive);
        # eps_val=0 prunes only exact zeros (MATLAB's a=0 lambda=0 term at
        # the origin's own partition never even gets a chance to appear
        # negative/positive-near-zero the way extremal points do, but every
        # OTHER near-zero weight should survive when the threshold is 0).
        pts_default, w_default = genz_keister_points(1, 9)
        pts_loose, w_loose = genz_keister_points(1, 9, eps_val=1e-3)
        assert len(w_loose) < len(w_default)
        # The real claim eps_val is supposed to guarantee: every point the
        # tighter threshold prunes away must actually have had |weight| at
        # or below it in the unpruned rule (checking only the survivors, as
        # the old assertion here did, restates the implementation's own
        # filter and cannot fail even if pruning were broken).
        kept = np.array(
            [
                np.any(np.all(np.isclose(pts_loose, p, atol=1e-12), axis=1))
                for p in pts_default
            ]
        )
        dropped_weights = w_default[~kept]
        assert len(dropped_weights) > 0
        assert np.all(np.abs(dropped_weights) <= 1e-3)

    # ---- NESTING: the property this port exists for ----

    @pytest.mark.parametrize("algorithm,max_m", [(0, 17), (1, 15)])
    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_nesting_holds_for_every_consecutive_pair_except_the_last(
        self, algorithm, max_m, n
    ):
        # Explicit, consecutive-level assertion: the point set at level m
        # contains EVERY point of level m-1, to floating tolerance -- for
        # every m from 2 up to (but not including) the documented top-of-
        # range exception (see test_nesting_breaks_at_the_documented_top).
        prev_pts, _ = genz_keister_points(n, 1, algorithm=algorithm)
        for m in range(2, max_m):
            pts, _ = genz_keister_points(n, m, algorithm=algorithm)
            for p in prev_pts:
                assert np.any(np.all(np.isclose(pts, p, atol=1e-9), axis=1)), (
                    f"n={n} algorithm={algorithm}: point {p} from level "
                    f"m={m - 1} is missing from level m={m}"
                )
            prev_pts = pts

    @pytest.mark.parametrize("algorithm,max_m", [(0, 17), (1, 15)])
    def test_nesting_breaks_at_the_documented_top_of_range(self, algorithm, max_m):
        # A genuine, verified property of the published double-precision
        # generator tables (see the docstring's "Precision at the top of
        # the range" note), not a porting bug: at the very last m for each
        # algorithm, recomputing the shared weight table with the larger m
        # shifts an existing extremal point's weight across the eps_val
        # pruning threshold, so it does NOT simply extend the previous
        # point set. Asserted explicitly (rather than merely omitted from
        # the loop above) so the limitation stays visible and tested.
        pts_prev, _ = genz_keister_points(1, max_m - 1, algorithm=algorithm)
        pts_top, _ = genz_keister_points(1, max_m, algorithm=algorithm)
        missing = [
            p
            for p in pts_prev
            if not np.any(np.all(np.isclose(pts_top, p, atol=1e-9), axis=1))
        ]
        assert missing, (
            "expected the documented top-of-range nesting break to "
            "reproduce; if this now passes, the docstring's Notes section "
            "should be revisited"
        )

    def test_top_of_range_precision_documented(self):
        # Pins the OTHER documented consequence of the top-of-range
        # precision breakdown (see the docstring's Notes item 1, now
        # qualified to exclude the maximum m of each algorithm, and its
        # "Precision at the top of the range" note): the generic
        # exact-through-2m+1 guarantee itself is FALSE here, not merely
        # unimproved -- both monomials probed sit strictly within their
        # rule's generic bound and still fail by orders of magnitude more
        # than roundoff. If this now passes, the docstring's qualified
        # guarantee should be revisited.
        #
        # algorithm 0, n=2, m=17: E[x1^34], degree 34 <= 2*17+1 = 35.
        pts0, w0 = genz_keister_points(2, 17, algorithm=0)
        alpha0 = (34, 0)
        relerr0 = abs(rule_moment(pts0, w0, alpha0) - gaussian_moment(alpha0)) / abs(
            gaussian_moment(alpha0)
        )
        assert relerr0 > 1e-3, (
            f"expected the documented generic-bound failure to reproduce "
            f"at algorithm=0, n=2, m=17 (measured relerr {relerr0:.3e})"
        )

        # algorithm 1, n=1, m=15: E[x^30], degree 30 <= 2*15+1 = 31.
        pts1, w1 = genz_keister_points(1, 15, algorithm=1)
        alpha1 = (30,)
        relerr1 = abs(rule_moment(pts1, w1, alpha1) - gaussian_moment(alpha1)) / abs(
            gaussian_moment(alpha1)
        )
        assert relerr1 > 1e-6, (
            f"expected the documented generic-bound failure to reproduce "
            f"at algorithm=1, n=1, m=15 (measured relerr {relerr1:.3e})"
        )

    # ---- generic exactness bound: degree 2m+1, any n, any m ----

    @staticmethod
    def _assert_some_monomial_of_degree_fails(
        points, weights, n, degree, threshold=1e-4
    ):
        """At least one monomial of EXACTLY `degree` fails.

        Deliberately does not hand-pick a single probe monomial: WHICH
        monomial fails first varies by (n, m) -- single-axis moments can be
        elevated well beyond the generic bound at "milestone" m values
        (test_milestone_exact_through_bonus_degree below), while
        mixed-exponent ones never are
        (test_milestone_bonus_does_not_extend_to_mixed_exponents).
        An exhaustive per-degree scan is robust to that without needing a
        separate exclusion list per (n, m).
        """
        worst = 0.0
        for alpha in monomials_up_to(n, degree):
            if sum(alpha) != degree:
                continue
            worst = max(
                worst, abs(rule_moment(points, weights, alpha) - gaussian_moment(alpha))
            )
        assert worst > threshold, (
            f"expected some degree-{degree} monomial (n={n}) to fail by "
            f"more than {threshold}, but the worst was {worst:.3e}"
        )

    # (n, m) pairs where the generic 2m+1 bound is exactly the TRUE degree,
    # not merely a safe lower bound -- so sharpness at 2m+2 is a genuine,
    # non-vacuous claim. For n=1, m in {1, 4, 5, 6, 9, 10, ..., 13} sit
    # strictly BELOW their milestone's actual (elevated) degree -- see
    # `TestGenzKeister` docstring / the module docstring's Notes -- so
    # sharpness at their generic 2m+2 would be false (that degree is still
    # exact there); m=2 lands on the same elevated 3-point rule as m=1 but
    # 2*2+1 happens to equal that rule's actual degree (5) by arithmetic
    # coincidence, so it IS safe to include. n=2 and n=3 have no such
    # elevation at any m (test_milestone_bonus_does_not_extend_to_mixed_exponents's
    # finding: the bonus is single-axis-only), so every m is safe there.
    _GENERIC_CASES = (
        [(1, m) for m in (2, 3, 7, 8)]
        + [(2, m) for m in (1, 2, 3, 4, 5, 6)]
        + [(3, m) for m in (1, 2, 3, 4, 5, 6)]
    )

    @pytest.mark.parametrize("n,m", _GENERIC_CASES)
    def test_exact_through_generic_degree(self, n, m):
        pts, w = genz_keister_points(n, m)
        assert_rule_exact(pts, w, n, degree=2 * m + 1)

    @pytest.mark.parametrize("n,m", _GENERIC_CASES)
    def test_sharp_at_generic_degree_plus_one(self, n, m):
        pts, w = genz_keister_points(n, m)
        self._assert_some_monomial_of_degree_fails(pts, w, n, degree=2 * m + 2)

    # ---- n-dependent floor of the generic bound (round-2/round-3 finding) ----
    #
    # A first attempt at documenting the generic bound's top-of-range
    # breakdown (round 1) claimed it held for every m except each
    # algorithm's true maximum (17 / 15). That was independently found
    # false: the breakdown starts BEFORE the true maximum once n >= 2, and
    # `_GENERIC_CASES` above never exercised that region (it stops at m=8
    # for n=1, m=6 for n>=2) -- exactly why the false claim survived
    # review once already. A round-2 fix narrowed this to a floor verified
    # for n<=4 -- ALSO false, one dimension past where it was checked: at
    # n=5, algorithm=0, m=15 breaks on (6,6,6,6,6), a monomial no n<=4 case
    # can express. The cases and helper below cover the region the
    # docstring's Notes item 1 now actually makes a claim about: the
    # largest m, per (algorithm, n) with n=1..8, at which every
    # degree-2m monomial (mixed-exponent included) is still exact.

    @staticmethod
    def _assert_exact_at_degree_relative(points, weights, n, degree, rtol):
        """Every monomial of EXACTLY `degree` is exact to a RELATIVE
        tolerance. `assert_rule_exact`'s fixed atol=1e-9 cannot be reused
        here (task constraint: keep it byte-identical) and would be far
        too tight regardless once the raw N(0,1) moments reach into the
        1e10+ range at these degrees.

        Divides by each monomial's OWN true value, not a single per-degree
        scale: a per-degree scale (e.g. the pure single-axis moment, as
        `_moment_scale` returns) can be orders of magnitude larger than a
        mixed-exponent monomial's own true value at the same total degree
        -- e.g. at degree 28, gaussian_moment((28,)) is ~2.13e14 but
        gaussian_moment((0,6,22)) is ~2.06e11, three orders smaller -- so
        dividing every monomial by the single-axis scale would silently
        mask exactly the mixed-exponent failures this test exists to
        catch (caught by hand while writing this test; see
        test_generic_bound_fails_one_m_past_floor's alpha=(0,6,22), 4.5e-2
        relative to its own true value but only 4.3e-5 relative to
        gaussian_moment((28,))). `_moment_scale` is used only as a
        fallback for monomials whose true value is exactly 0 (some
        individual exponent odd), where a relative comparison is
        undefined.

        Precomputes each axis's power column (``points[:, j] ** k`` for
        every ``k`` up to `degree`) once and looks those up per monomial,
        instead of recomputing ``points ** alpha`` from scratch for each
        of the (up to tens of thousands of) monomials `monomials_up_to`
        enumerates -- at n=5, degree=28 (round 3's `(0, 5, 14)` floor
        case) this is the difference between ~104s and ~4s; needed once
        `n` climbed past 4, not merely a nicety.
        """
        scale = TestGenzKeister._moment_scale(degree)
        num_points = points.shape[0]
        pow_table = np.empty((n, degree + 1, num_points))
        for j in range(n):
            col = points[:, j]
            pow_table[j, 0] = 1.0
            for k in range(1, degree + 1):
                pow_table[j, k] = pow_table[j, k - 1] * col

        worst = 0.0
        worst_alpha = None
        for alpha in monomials_up_to(n, degree):
            if sum(alpha) != degree:
                continue
            true_val = gaussian_moment(alpha)
            prod = pow_table[0, alpha[0]]
            for j in range(1, n):
                prod = prod * pow_table[j, alpha[j]]
            rule_val = float(np.dot(weights, prod))
            denom = abs(true_val) if true_val != 0.0 else scale
            err = abs(rule_val - true_val) / denom
            if err > worst:
                worst, worst_alpha = err, alpha
        assert worst < rtol, (
            f"n={n} degree={degree}: worst relative error {worst:.3e} at "
            f"{worst_alpha} >= {rtol:.1e}"
        )

    # (algorithm, n, floor m) -- the largest m, measured for n=1..8, at
    # which the degree-2m generic bound above still holds (see the
    # docstring's Notes item 1 table this reproduces). Below each
    # algorithm's true maximum (17 / 15) once n >= 2, contrary to the
    # round-1 docstring text -- and, for algorithm 0, below the n<=4-only
    # floor of 15 a round-2 fix claimed once n >= 5, contrary to THAT
    # docstring text: (0, 5, 14) below is what catches it, because the
    # monomial that breaks m=15 there, (6, 6, 6, 6, 6), needs five
    # simultaneously nonzero exponents that no n<=4 case here could ever
    # construct -- the exact blind spot that let the n<=4-only floor ship.
    _FLOOR_CASES = [
        (0, 1, 16),
        (0, 2, 15),
        (0, 3, 15),
        (0, 4, 15),
        (0, 5, 14),
        (1, 1, 14),
        (1, 2, 15),  # algorithm 1's true maximum -- no violation found at n=2
        (1, 3, 13),
        (1, 4, 13),
    ]

    @pytest.mark.parametrize("algorithm,n,floor_m", _FLOOR_CASES)
    def test_exact_at_generic_bound_floor(self, algorithm, n, floor_m):
        # Measured worst-case relative error at these (algorithm, n,
        # floor_m) points is ~1e-12 or tighter (see the fix report); 1e-8
        # leaves several orders of margin above that noise floor while
        # sitting far below the ~1e-2+ degradation one m past the floor
        # (test_generic_bound_fails_one_m_past_floor below).
        pts, w = genz_keister_points(n, floor_m, algorithm=algorithm)
        self._assert_exact_at_degree_relative(pts, w, n, degree=2 * floor_m, rtol=1e-8)

    def test_generic_bound_fails_one_m_past_floor(self):
        # Pins the degradation one m step past n=3's floor for BOTH
        # algorithms -- the worked examples the docstring's Notes item 1
        # and "Precision at the top of the range" now cite directly. Uses
        # the WORST monomial at that degree (not a single-axis probe): for
        # algorithm 1 here, the true worst case is the mixed-exponent
        # (0, 6, 22), which fails far more severely (4.5e-2) than the
        # single-axis (28, 0, 0) probe alone would suggest (3.1e-3) --
        # exactly the mixed-vs-single-axis gap the docstring calls out.
        pts0, w0 = genz_keister_points(3, 16, algorithm=0)  # floor is 15
        relerr0 = abs(
            rule_moment(pts0, w0, (0, 32, 0)) - gaussian_moment((0, 32, 0))
        ) / gaussian_moment((0, 32, 0))
        assert relerr0 > 1e-3, (
            f"expected the documented n=3 algorithm=0 m=16 breakdown to "
            f"reproduce (measured relerr {relerr0:.3e})"
        )

        pts1, w1 = genz_keister_points(3, 14, algorithm=1)  # floor is 13
        alpha1 = (0, 6, 22)  # all-even exponents: true value is nonzero
        relerr1 = abs(
            rule_moment(pts1, w1, alpha1) - gaussian_moment(alpha1)
        ) / gaussian_moment(alpha1)
        assert relerr1 > 1e-3, (
            f"expected the documented n=3 algorithm=1 m=14 breakdown to "
            f"reproduce (measured relerr {relerr1:.3e})"
        )

    def test_generic_bound_fails_at_n5_needs_five_nonzero_exponents(self):
        # Round-3 finding: a round-2 fix claimed the generic bound's floor
        # was m=15 (algorithm 0) verified for n<=4 -- false one dimension
        # past where it was checked. At n=5, algorithm=0, m=15 (the OLD
        # claimed floor; the corrected floor is 14, see _FLOOR_CASES), the
        # worst monomial is (6, 6, 6, 6, 6) -- it needs FIVE simultaneously
        # nonzero exponents, which no n<=4 monomial can ever have, which is
        # exactly why `_GENERIC_CASES`/`_FLOOR_CASES`-style sweeps stopping
        # at n=4 could not catch this. The single-axis probe at the
        # identical rule is clean (~3.8e-12), demonstrating the failure is
        # invisible to any single-axis-only or n<=4 check.
        pts, w = genz_keister_points(5, 15, algorithm=0)  # corrected floor is 14
        alpha = (6, 6, 6, 6, 6)
        relerr = abs(
            rule_moment(pts, w, alpha) - gaussian_moment(alpha)
        ) / gaussian_moment(alpha)
        assert relerr > 1e-3, (
            f"expected the documented n=5 algorithm=0 m=15 breakdown to "
            f"reproduce (measured relerr {relerr:.3e})"
        )

        axis_alpha = (30, 0, 0, 0, 0)
        axis_relerr = abs(
            rule_moment(pts, w, axis_alpha) - gaussian_moment(axis_alpha)
        ) / gaussian_moment(axis_alpha)
        assert axis_relerr < 1e-8, (
            f"expected the single-axis probe to stay clean at this same "
            f"rule (measured relerr {axis_relerr:.3e}) -- this is the "
            f"point: the failure is invisible to a single-axis-only check"
        )

    def test_milestone_bonus_does_not_extend_to_mixed_exponents(self):
        # REFERENCE (hand-verified value, see also
        # test_milestone_exact_through_bonus_degree below):
        # n=1's marginal at m=4 (algorithm 0) is exact through degree 15.
        # In 2D, the axis-only moment at that degree is STILL exact (the
        # bonus is a property of the shared 1-D generator, inherited by
        # single-axis moments regardless of n) -- but a MIXED-exponent
        # monomial of the SAME total degree is not, which is what actually
        # caps the multi-dimensional rule's degree at the generic 2m+1=9.
        pts, w = genz_keister_points(2, 4)
        axis_alpha = (14, 0)  # degree 14 <= 15: still exact (axis-only)
        assert_allclose(
            rule_moment(pts, w, axis_alpha), gaussian_moment(axis_alpha), atol=1e-6
        )
        mixed_alpha = (8, 2)  # degree 10 > 9 (generic bound): must fail
        rule_val = rule_moment(pts, w, mixed_alpha)
        true_val = gaussian_moment(mixed_alpha)
        assert_allclose(rule_val, 153.37847512583247, atol=1e-6)
        assert abs(rule_val - true_val) > 1.0

    # ---- milestone bonus: n=1 marginal exceeds the generic bound ----

    _MILESTONE_CASES = [
        (0, 1, 5),  # base rule: 0, +-sqrt(3) IS the 3-point Gauss-Hermite rule
        (0, 4, 15),  # nu=[3,...] stage complete
        (0, 9, 29),  # nu=[3,5,...] stage complete
        (1, 1, 5),  # base rule, shared with algorithm 0
        (1, 5, 19),  # nu=[4,...] stage complete
    ]

    @staticmethod
    def _moment_scale(degree):
        """A magnitude reference for degree `degree`, even or odd.

        The raw N(0,1) moments here reach ~1e15 by degree 30 (E[x^30] =
        29!! = 6.19e15); `assert_rule_exact`'s fixed ``atol=1e-9`` is far
        too tight for genuine float64 summation noise at that scale (this
        module's docstring's "Precision at the top of the range" note
        documents the same effect at the extreme top of the m range --
        this is the same phenomenon at more moderate degrees, verified
        directly: the *relative* error stays ~1e-15 across every degree up
        to each milestone's claimed bound, but the raw values it's relative
        TO grow by 15 orders of magnitude, so a fixed absolute tolerance
        cannot track it). Odd degrees have a true value of 0, so relative
        error is undefined there too; this returns the true (even) moment
        at `degree` or `degree - 1`, whichever is even, as the scale to
        measure noise against instead.
        """
        even_degree = degree - (degree % 2)
        return max(1.0, gaussian_moment((even_degree,)))

    @pytest.mark.parametrize("algorithm,m,degree", _MILESTONE_CASES)
    def test_milestone_exact_through_bonus_degree(self, algorithm, m, degree):
        pts, w = genz_keister_points(1, m, algorithm=algorithm)
        for d in range(degree + 1):
            alpha = (d,)
            rule_val = rule_moment(pts, w, alpha)
            true_val = gaussian_moment(alpha)
            scale = self._moment_scale(d)
            assert abs(rule_val - true_val) < 1e-6 * scale, (
                f"algorithm={algorithm} m={m} degree={d}: rule={rule_val} "
                f"true={true_val}"
            )

    @pytest.mark.parametrize("algorithm,m,degree", _MILESTONE_CASES)
    def test_milestone_sharp_at_bonus_degree_plus_one(self, algorithm, m, degree):
        pts, w = genz_keister_points(1, m, algorithm=algorithm)
        alpha = (degree + 1,)
        rule_val = rule_moment(pts, w, alpha)
        true_val = gaussian_moment(alpha)
        scale = self._moment_scale(degree + 1)
        assert abs(rule_val - true_val) > 1e-5 * scale, (
            f"algorithm={algorithm} m={m} degree={degree + 1}: rule={rule_val} "
            f"true={true_val} (expected a real failure, not noise)"
        )

    def test_milestone_point_counts(self):
        # REFERENCE point counts (independently re-derived, not copied from
        # Table 3.4 -- see the docstring's Notes): the classic nested
        # Genz-Keister growth 1, 3, 9, 19, ... for algorithm 0's milestones.
        for m, expected_npts in [(1, 3), (4, 9), (9, 19)]:
            pts, _ = genz_keister_points(1, m, algorithm=0)
            assert len(pts) == expected_npts
        for m, expected_npts in [(1, 3), (5, 11)]:
            pts, _ = genz_keister_points(1, m, algorithm=1)
            assert len(pts) == expected_npts

    # ---- cross-check against an unoptimized brute-force reconstruction ----

    @staticmethod
    def _matlab_inner_terms_4_weights(a, lam2, m):
        """Literal, line-for-line translation of MATLAB's
        ``innerTerms4Weights`` loop (``GenzKeisterPoints.m`` lines 223-239),
        re-indexed from MATLAB's 1-based ``pSubi``/``pkSum`` to 0-based
        ``p``/``k`` (``p = pSubi - 1``, ``k = pkSum - 1``, so MATLAB's
        ``pkSum > pSubi`` becomes ``k > p`` and its ``pkSum - 1`` index
        becomes ``k - 1``). Deliberately NOT the closed-form/optimized
        ``_gk_inner_terms`` the shipped implementation uses -- see
        `test_matches_matlab_reconstruction` below for why this
        independence matters.
        """
        b = np.zeros((m + 1, m + 1))
        b[0, 0] = a[0]
        for p in range(m + 1):
            prod_val = 1.0
            for k in range(1, m + 1):
                if k > p:
                    prod_val *= lam2[p] - lam2[k]
                else:
                    prod_val *= lam2[p] - lam2[k - 1]
                if k >= p:
                    b[p, k] = a[k] / prod_val
        return b

    @staticmethod
    def _matlab_compute_w(m, p, inner_terms):
        """Literal translation of MATLAB's ``computeW`` odometer
        (``GenzKeisterPoints.m`` lines 298-346), preserving its mutable
        ``prodSums``/``kpSum``/``cardinalityLeft`` state machine and its
        "for loop with a break, then inspect the loop variable" control
        flow (``i`` is 0-based here vs. MATLAB's 1-based, so ``i == n``
        becomes ``i == n - 1`` and ``p(n)`` becomes ``p[-1]``) -- NOT the
        polynomial-convolution closed form ``_gk_partition_weight`` uses.
        """
        n = len(p)
        active = sum(1 for pi in p if pi != 0)
        cardinality_left = m - sum(p)
        kp_sum = list(p)
        prod_sums = [0.0] * (n + 1)
        while True:
            broke = False
            for i in range(n):
                prod_sums[0] = 1.0
                if cardinality_left >= 0:
                    prod_sums[i + 1] += inner_terms[p[i], kp_sum[i]] * prod_sums[i]
                    prod_sums[i] = 0.0
                    kp_sum[i] += 1
                    cardinality_left -= 1
                    broke = True
                    break
                cardinality_left += kp_sum[i] - p[i]
                kp_sum[i] = p[i]
            if not broke:
                i = n - 1
            if i == n - 1 and kp_sum[n - 1] == p[n - 1]:
                break
        return 2.0 ** (-active) * prod_sums[n]

    def test_matches_matlab_reconstruction(self):
        # GENUINE independent oracle (unlike a since-fixed earlier version
        # of this test, which imported `_gk_inner_terms`/`_gk_partition_weight`
        # from the module under test and so re-derived only the point
        # enumeration, taking every weight value from the code it was
        # meant to validate): `_matlab_inner_terms_4_weights` and
        # `_matlab_compute_w` above are literal translations of MATLAB's
        # own routines, transcribed directly from
        # GenzKeisterPoints.m, not derived from or cross-checked against
        # this module's implementation. This also evaluates every raw
        # n-tuple in {0,...,m}^n directly, rather than computing one weight
        # per partition *class* and reusing it for every point in that
        # partition's symmetry orbit (what the shipped implementation does,
        # mirroring MATLAB's own optimization) -- so both the enumeration
        # AND the weight formula are independently re-derived here. Covers
        # every valid m for both algorithms at n=1..3 (agreement to
        # roundoff throughout, ~1 second total).
        def brute_force(n, m, algorithm=0):
            from pytcl.mathematical_functions.numerical_integration.cubature_points import (
                _GK_TABLES,
                _pm_combos,
            )

            lam, a = _GK_TABLES[algorithm]
            lam2 = lam[: m + 1] ** 2
            inner = self._matlab_inner_terms_4_weights(a, lam2, m)
            pts, ws = [], []
            for p in itertools.product(range(m + 1), repeat=n):
                if sum(p) > m:
                    continue
                w_p = self._matlab_compute_w(m, p, inner)
                combos = _pm_combos(lam[list(p)])
                pts.append(combos)
                ws.append(np.full(len(combos), w_p))
            pts = np.vstack(pts)
            ws = np.concatenate(ws)
            eps_val = np.finfo(np.float64).eps
            sel = np.abs(ws) > eps_val
            return pts[sel], ws[sel]

        def canonicalize(pts, w):
            idx = np.lexsort(pts.T[::-1])
            return pts[idx], w[idx]

        for algorithm, max_m in [(0, 17), (1, 15)]:
            for n in (1, 2, 3):
                for m in range(1, max_m + 1):
                    pts_opt, w_opt = genz_keister_points(n, m, algorithm=algorithm)
                    pts_brute, w_brute = brute_force(n, m, algorithm=algorithm)
                    c_opt = canonicalize(pts_opt, w_opt)
                    c_brute = canonicalize(pts_brute, w_brute)
                    assert c_opt[0].shape == c_brute[0].shape, (
                        f"algorithm={algorithm} n={n} m={m}: point count mismatch"
                    )
                    assert_allclose(
                        c_opt[0],
                        c_brute[0],
                        atol=1e-8,
                        err_msg=f"algorithm={algorithm} n={n} m={m}",
                    )
                    assert_allclose(
                        c_opt[1],
                        c_brute[1],
                        atol=1e-8,
                        err_msg=f"algorithm={algorithm} n={n} m={m}",
                    )


def assert_rule_exact_student_t(points, weights, n, degree, dof):
    for alpha in monomials_up_to(n, degree):
        assert_allclose(
            rule_moment(points, weights, alpha),
            student_t_moment(alpha, dof),
            atol=1e-9,
            err_msg=f"monomial {alpha} not integrated exactly (dof={dof})",
        )


class TestStudentT:
    """MATLAB's ``thirdOrderStudentTCubPoints`` -- the Student-t analogue of
    ``ckf_spherical_cubature_points``. The Gaussian moment harness does NOT
    apply here; ``student_t_moment`` (module-level, above) is the oracle.
    """

    @pytest.mark.parametrize("n", [1, 2, 3, 4])
    def test_shape_and_weight_sum(self, n):
        pts, w = student_t_cubature_points(n, 6.0)
        assert pts.shape == (2 * n, n)
        assert_allclose(w.sum(), 1.0, atol=1e-12)
        assert_allclose(w, np.full(2 * n, 1.0 / (2 * n)), atol=1e-12)

    @pytest.mark.parametrize("n", [1, 2, 3, 4])
    @pytest.mark.parametrize("dof", [2.5, 6.0, 30.0])
    def test_exact_through_degree_3(self, n, dof):
        pts, w = student_t_cubature_points(n, dof)
        assert_rule_exact_student_t(pts, w, n, degree=3, dof=dof)

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_sharpness_degree_4_fails(self, n):
        # dof=10 keeps the true 4th moment defined (needs dof>4) so the gap
        # is a genuine rule limitation, not a missing oracle value.
        dof = 10.0
        pts, w = student_t_cubature_points(n, dof)
        alpha = (4,) + (0,) * (n - 1)
        rule_val = rule_moment(pts, w, alpha)
        true_val = student_t_moment(alpha, dof)
        assert abs(rule_val - true_val) > 1e-3

    def test_sharpness_degree_4_mixed_exponent_fails(self):
        # Beyond the pure (4,0,...) case above: every point has only one
        # nonzero coordinate, so ANY monomial touching >=2 distinct axes
        # (e.g. (2,2,0)) integrates to exactly 0 under the rule, while the
        # true mixed moment is nonzero (see
        # TestHelpers.test_student_t_moment_known_mixed_second_moment) --
        # a stronger, free sharpness check beyond the single-axis probe.
        dof = 10.0
        pts, w = student_t_cubature_points(3, dof)
        alpha = (2, 2, 0)
        rule_val = rule_moment(pts, w, alpha)
        true_val = student_t_moment(alpha, dof)
        assert rule_val == 0.0
        assert abs(rule_val - true_val) > 1e-3

    def test_antipodal_symmetry(self):
        pts, w = student_t_cubature_points(4, 6.0)
        for p, wt in zip(pts, w):
            match = np.all(np.isclose(pts, -p), axis=1)
            assert np.any(match)
            assert_allclose(w[match], wt)

    def test_nu_const_matches_known_value(self):
        # nuConst = sqrt(dof * n / (dof - 2)); rule points sit at +-nuConst
        # along each axis (MATLAB's SR=I reduction).
        n, dof = 3, 6.0
        nu_const = np.sqrt(dof * n / (dof - 2.0))
        pts, _ = student_t_cubature_points(n, dof)
        assert_allclose(np.sort(np.abs(pts[pts != 0])), np.full(2 * n, nu_const))

    def test_converges_to_ckf_points_as_dof_grows(self):
        # As dof -> inf the Student-t distribution -> N(0, I), so the rule
        # should converge to ckf_spherical_cubature_points's +-sqrt(n) e_i.
        from pytcl.dynamic_estimation.kalman.unscented import (
            ckf_spherical_cubature_points,
        )

        n = 3
        pts_t, w_t = student_t_cubature_points(n, 1e8)
        pts_g, w_g = ckf_spherical_cubature_points(n)
        assert_allclose(w_t, w_g, atol=1e-12)
        assert_allclose(np.sort(np.abs(pts_t).ravel()), np.sort(np.abs(pts_g).ravel()))

    def test_invalid_n_raises(self):
        with pytest.raises(ValueError):
            student_t_cubature_points(0, 6.0)

    @pytest.mark.parametrize("dof", [2.0, 1.0, -1.0, 0.0])
    def test_dof_at_or_below_boundary_raises(self, dof):
        with pytest.raises(ValueError):
            student_t_cubature_points(3, dof)

    def test_dof_just_above_boundary_is_valid(self):
        pts, w = student_t_cubature_points(2, 2.0001)
        assert_allclose(w.sum(), 1.0, atol=1e-12)
        assert np.all(np.isfinite(pts))


class TestCubaturePointMoments:
    """MATLAB's ``calcCubPointMoments`` -- the public, filter-independent
    form of the transform-then-propagate pattern ``ckf_predict``/
    ``ckf_update`` apply internally.
    """

    def test_linear_transform_matches_closed_form(self):
        # E[Ax+b] = A mu + b, Cov = A P A^T -- exact for any rule that
        # integrates degree <= 2 exactly (fifth_order_cubature_points
        # comfortably does).
        n = 3
        rng = np.random.default_rng(11)
        mean = rng.normal(size=n)
        Araw = rng.normal(size=(n, n))
        cov = Araw @ Araw.T + np.eye(n)
        A = rng.normal(size=(n, n))
        b = rng.normal(size=n)

        pts, w = fifth_order_cubature_points(n)
        mu, P = cubature_point_moments(pts, w, lambda x: A @ x + b, mean, cov)

        assert_allclose(mu, A @ mean + b, atol=1e-10)
        assert_allclose(P, A @ cov @ A.T, atol=1e-9)

    def test_no_mean_cov_uses_points_directly(self):
        # mean/cov omitted: points are passed to func unchanged, no affine
        # map applied first.
        pts = np.array([[0.0], [1.0], [-1.0]])
        w = np.full(3, 1.0 / 3.0)
        mu, P = cubature_point_moments(pts, w, lambda x: x)
        assert_allclose(mu, np.array([0.0]), atol=1e-12)
        assert_allclose(P, np.array([[2.0 / 3.0]]), atol=1e-12)

    def test_agrees_with_ckf_internal_propagation(self):
        # Shared point set, shared nonlinear function: the utility and the
        # filter's own internal moment propagation must not drift apart.
        from pytcl.dynamic_estimation.kalman.unscented import (
            ckf_predict,
            ckf_spherical_cubature_points,
        )

        x = np.array([1.0, -0.5, 0.2])
        P = np.diag([0.2, 0.3, 0.15])
        Q = np.zeros((3, 3))  # no added process noise -> pure moment propagation

        def f(v):
            return np.array([v[0] + 0.1 * v[1], v[1] + 0.05 * v[0] ** 2, v[2]])

        pts, w = ckf_spherical_cubature_points(3)
        mu, cov = cubature_point_moments(pts, w, f, x, P)
        pred = ckf_predict(x, P, f, Q, points=pts, weights=w)

        assert_allclose(mu, pred.x, atol=1e-12)
        assert_allclose(cov, pred.P, atol=1e-12)

    def test_agrees_with_ckf_using_fifth_order_points(self):
        from pytcl.dynamic_estimation.kalman.unscented import ckf_predict

        x = np.array([3.0, 1.0])
        P = np.eye(2) * 0.5
        Q = np.zeros((2, 2))

        def f(v):
            return np.array([v[0] ** 2, v[0] + v[1]])

        pts, w = fifth_order_cubature_points(2)
        mu, cov = cubature_point_moments(pts, w, f, x, P)
        pred = ckf_predict(x, P, f, Q, points=pts, weights=w)

        assert_allclose(mu, pred.x, atol=1e-12)
        assert_allclose(cov, pred.P, atol=1e-12)

    def test_negative_weights_not_suppressed(self):
        # n=5 makes fifth_order_cubature_points' axis weight negative; the
        # covariance assembly must not sqrt() weights (mirrors ckf_predict's
        # own sign-safe assembly).
        n = 5
        mean = np.zeros(n)
        cov = np.eye(n)
        pts, w = fifth_order_cubature_points(n)
        assert w.min() < 0
        F = np.diag([1.0, 2.0, 0.5, 1.5, 1.0])
        mu, P = cubature_point_moments(pts, w, lambda x: F @ x, mean, cov)
        assert_allclose(mu, F @ mean, atol=1e-10)
        assert_allclose(P, F @ cov @ F.T, atol=1e-9)
        assert not np.any(np.isnan(P))

    def test_rank_deficient_covariance_does_not_raise(self):
        # A raw np.linalg.cholesky would raise LinAlgError on a singular
        # (positive semi-definite but not definite) covariance -- exactly
        # the situation MATLAB's own docstring anticipates by recommending
        # cholSemiDef(R,'lower'), and the same situation ckf_predict
        # already falls back to eigh for with a filter's own (possibly
        # rank-deficient) state covariance. Build a genuinely singular 3x3
        # covariance (rank 2: outer product of two vectors) and confirm
        # the linear closed form still holds exactly.
        n = 3
        rng = np.random.default_rng(23)
        B = rng.normal(size=(n, 2))  # rank <= 2 in 3 dimensions
        cov = B @ B.T
        assert np.linalg.matrix_rank(cov) < n
        mean = rng.normal(size=n)
        A = rng.normal(size=(n, n))
        b = rng.normal(size=n)

        pts, w = fifth_order_cubature_points(n)
        mu, P = cubature_point_moments(pts, w, lambda x: A @ x + b, mean, cov)

        assert_allclose(mu, A @ mean + b, atol=1e-9)
        assert_allclose(P, A @ cov @ A.T, atol=1e-8)

    def test_shape_mismatch_raises(self):
        pts, w = fifth_order_cubature_points(3)
        with pytest.raises(ValueError):
            cubature_point_moments(pts, w[:-1], lambda x: x)
        with pytest.raises(ValueError):
            cubature_point_moments(pts.ravel(), w, lambda x: x)

    def test_mean_without_cov_raises(self):
        pts, w = fifth_order_cubature_points(3)
        with pytest.raises(ValueError):
            cubature_point_moments(pts, w, lambda x: x, mean=np.zeros(3))
        with pytest.raises(ValueError):
            cubature_point_moments(pts, w, lambda x: x, cov=np.eye(3))

    def test_mean_cov_dimension_mismatch_raises(self):
        pts, w = fifth_order_cubature_points(3)
        with pytest.raises(ValueError):
            cubature_point_moments(pts, w, lambda x: x, mean=np.zeros(2), cov=np.eye(2))


class TestSeventhOrderAlgorithms:
    """seventh_order_cubature_points algorithm surface (MATLAB parity).

    Exactness verified per (algorithm, n) cell listed here; the docstring
    must claim no more than this table.
    """

    CASES = [  # (algorithm, n, expected_num_points)
        (1, 3, 2**3 + 2 * 9 + 1),
        (1, 4, 2**4 + 2 * 16 + 1),
        (1, 6, 2**6 + 2 * 36 + 1),
        (1, 7, 2**7 + 2 * 49 + 1),
        (2, 2, 12),
        # 17, not MATLAB's 16: MATLAB's 16-point, no-origin layout for
        # algorithm 3 is provably incapable of degree-7 exactness for any
        # choice of its parameters (see _e2_7_2's docstring); a 17th point
        # at the origin supplies the missing degree of freedom.
        (3, 2, 17),
        (4, 3, 27),
        (5, 3, 27),
        (6, 3, 33),
        (7, 3, 33),
        (8, 4, 49),
        (9, 1, 4),
    ]

    @pytest.mark.parametrize("algorithm,n,num_points", CASES)
    def test_exact_through_degree_7(self, algorithm, n, num_points):
        pts, w = seventh_order_cubature_points(n, algorithm=algorithm)
        assert pts.shape == (num_points, n)
        assert abs(w.sum() - 1.0) < 1e-12
        assert_rule_exact(pts, w, n, 7)

    @pytest.mark.parametrize("algorithm,n,_", CASES)
    def test_sharpness_degree_8_fails(self, algorithm, n, _):
        # Mirrors TestSeventhOrder's degree-8 sharpness check for algorithm
        # 0: every algorithm here is a degree-7 rule, not degree-8, so a
        # single-axis degree-8 monomial must miss E[x1^8] = 105 by more
        # than roundoff. Measured misses range from ~4.0 (algorithm 1,
        # n=6) to ~96 (algorithm 6); all comfortably clear 1.0.
        pts, w = seventh_order_cubature_points(n, algorithm=algorithm)
        alpha = (8,) + (0,) * (n - 1)
        assert abs(rule_moment(pts, w, alpha) - 105.0) > 1.0

    def test_default_algorithm_dispatch(self):
        # MATLAB default logic: n==1 -> 9, n==2 -> 2, else 0
        p1, _ = seventh_order_cubature_points(1)
        assert p1.shape[0] == 4
        p2, _ = seventh_order_cubature_points(2)
        assert p2.shape[0] == 12
        p3, w3 = seventh_order_cubature_points(3)
        e3, we3 = seventh_order_cubature_points(3, algorithm=0)
        np.testing.assert_array_equal(p3, e3)
        np.testing.assert_array_equal(w3, we3)

    @pytest.mark.parametrize(
        "algorithm,bad_n",
        [(1, 2), (1, 5), (1, 8), (2, 3), (3, 1), (4, 2), (8, 3), (9, 2)],
    )
    def test_invalid_dimension_raises(self, algorithm, bad_n):
        with pytest.raises(ValueError):
            seventh_order_cubature_points(bad_n, algorithm=algorithm)

    def test_unknown_algorithm_raises(self):
        with pytest.raises(ValueError):
            seventh_order_cubature_points(3, algorithm=10)


_MATLAB_FIXTURES = Path(__file__).parents[1] / "fixtures" / "matlab"


def _load_matlab_fixture(name):
    path = _MATLAB_FIXTURES / name
    if not path.exists():
        if os.environ.get("PYTCL_REQUIRE_MATLAB_FIXTURES") == "1":
            pytest.fail(
                f"fixture {name} required but absent -- run "
                "scripts/matlab_capture/capture_seventh_order.m in MATLAB "
                "with the TCL library on path to generate it"
            )
        pytest.skip(
            f"MATLAB fixture {name} not captured yet -- run "
            "scripts/matlab_capture/capture_seventh_order.m in MATLAB "
            "with the TCL library on path to generate it"
        )
    data = np.loadtxt(path, delimiter=",")
    return data[:, :-1], data[:, -1]


class TestSeventhOrderMatlabFixtures:
    # Algorithms 3 and 8 are excluded: this port's _e2_7_2 and _e4_7_1
    # deliberately do NOT reproduce MATLAB's seventhOrderCubPoints output
    # for those two algorithms, because that output does not satisfy the
    # function's own degree-7-exactness contract (see _e2_7_2/_e4_7_1
    # docstrings and TestSeventhOrderAlgorithms.CASES's inline note) --
    # a "matches MATLAB" comparison for them would necessarily fail by
    # design and would not indicate a defect in this port.
    _PARITY_CASES = [c for c in TestSeventhOrderAlgorithms.CASES if c[0] not in (3, 8)]

    @pytest.mark.parametrize("algorithm,n,_", _PARITY_CASES)
    def test_matches_matlab(self, algorithm, n, _):
        ref_pts, ref_w = _load_matlab_fixture(f"seventh_order_alg{algorithm}_n{n}.csv")
        pts, w = seventh_order_cubature_points(n, algorithm=algorithm)
        # order-independent comparison: sort both point sets lexicographically
        oa = np.lexsort(pts.T)
        ob = np.lexsort(ref_pts.T)
        np.testing.assert_allclose(pts[oa], ref_pts[ob], atol=1e-12)
        np.testing.assert_allclose(w[oa], ref_w[ob], atol=1e-12)


class TestSmolyakPoints:
    """Smolyak combination over the nested Genz-Keister milestone sequences.

    Original design (no MATLAB counterpart to transcribe); every exactness
    claim below is bounded to the measured (algorithm, n, level) grid in
    MEASURED -- see the claims-inherit-measurement-range note in
    `smolyak_points`' docstring.
    """

    # The brief's verified grid: every (n, level) here (algorithm 0) is
    # what the docstring may claim without consulting the MEASURED table.
    SWEEP = [(n, lv) for n in (1, 2, 3, 4, 5) for lv in (0, 1, 2, 3)]

    # Measured total-degree exactness per (algorithm, n, level), with a
    # measured-FAILING monomial of degree+1 pinning sharpness (guards
    # against a vacuous exactness loop). Produced by scanning every
    # all-even-exponent monomial per total degree at relative tolerance
    # 1e-12 (odd exponents are exact by the grid's antipodal symmetry;
    # the through-degree test below re-checks them anyway); see
    # `smolyak_points`' docstring for the same table and the measurement
    # description. Note the standard degree-(2*level+1) mapping HOLDS in
    # every measured cell; cells with n <= level additionally exceed it
    # (the GK milestone bonus degrees surfacing through the combination),
    # which is why this table, not 2*level+1, is the claim.
    MEASURED = [
        # algorithm 0: levels 0..3, n = 1..8
        (0, 1, 0, 1, (2,)),
        (0, 1, 1, 5, (6,)),
        (0, 1, 2, 15, (16,)),
        (0, 1, 3, 29, (30,)),
        (0, 2, 0, 1, (0, 2)),
        (0, 2, 1, 3, (2, 2)),
        (0, 2, 2, 7, (2, 6)),
        (0, 2, 3, 11, (6, 6)),
        (0, 3, 0, 1, (0, 0, 2)),
        (0, 3, 1, 3, (0, 2, 2)),
        (0, 3, 2, 5, (2, 2, 2)),
        (0, 3, 3, 9, (2, 6, 2)),
        (0, 4, 0, 1, (0, 0, 0, 2)),
        (0, 4, 1, 3, (0, 0, 2, 2)),
        (0, 4, 2, 5, (0, 2, 2, 2)),
        (0, 4, 3, 7, (2, 2, 2, 2)),
        (0, 5, 0, 1, (0, 0, 0, 0, 2)),
        (0, 5, 1, 3, (0, 0, 0, 2, 2)),
        (0, 5, 2, 5, (0, 0, 2, 2, 2)),
        (0, 5, 3, 7, (0, 2, 2, 2, 2)),
        (0, 6, 2, 5, (0, 0, 0, 2, 2, 2)),
        (0, 6, 3, 7, (0, 0, 2, 2, 2, 2)),
        (0, 7, 2, 5, (0, 0, 0, 0, 2, 2, 2)),
        (0, 7, 3, 7, (0, 0, 0, 2, 2, 2, 2)),
        (0, 8, 2, 5, (0, 0, 0, 0, 0, 2, 2, 2)),
        (0, 8, 3, 7, (0, 0, 0, 0, 2, 2, 2, 2)),
        # algorithm 1: levels 0..2 (its level cap -- see _SMOLYAK_GK_M),
        # n = 1..6
        (1, 1, 0, 1, (2,)),
        (1, 1, 1, 5, (6,)),
        (1, 1, 2, 19, (20,)),
        (1, 2, 0, 1, (0, 2)),
        (1, 2, 1, 3, (2, 2)),
        (1, 2, 2, 7, (2, 6)),
        (1, 3, 2, 5, (2, 2, 2)),
        (1, 4, 2, 5, (0, 2, 2, 2)),
        (1, 5, 2, 5, (0, 0, 2, 2, 2)),
        (1, 6, 2, 5, (0, 0, 0, 2, 2, 2)),
    ]

    @staticmethod
    def _assert_exact_through_degree_relative(points, weights, n, degree, rtol):
        """Every monomial of total degree <= `degree` is exact to a
        RELATIVE tolerance (each monomial measured against its own true
        value; zero-true monomials against the same-degree single-axis
        moment scale). `assert_rule_exact`'s fixed atol=1e-9 cannot serve
        here: the n=1 cells reach degree 29, where E[x^28] = 27!! ~ 2e14
        makes a fixed absolute tolerance nonsensical -- the same reasoning
        as TestGenzKeister._assert_exact_at_degree_relative, which this
        mirrors (through-degree instead of at-degree)."""
        num_points = points.shape[0]
        pow_table = np.empty((n, degree + 1, num_points))
        for j in range(n):
            col = points[:, j]
            pow_table[j, 0] = 1.0
            for k in range(1, degree + 1):
                pow_table[j, k] = pow_table[j, k - 1] * col
        worst = 0.0
        worst_alpha = None
        for alpha in monomials_up_to(n, degree):
            true_val = gaussian_moment(alpha)
            prod = pow_table[0, alpha[0]]
            for j in range(1, n):
                prod = prod * pow_table[j, alpha[j]]
            rule_val = float(np.dot(weights, prod))
            denom = (
                abs(true_val)
                if true_val != 0.0
                else TestGenzKeister._moment_scale(sum(alpha))
            )
            err = abs(rule_val - true_val) / denom
            if err > worst:
                worst, worst_alpha = err, alpha
        assert worst < rtol, (
            f"n={n} degree={degree}: worst relative error {worst:.3e} at "
            f"{worst_alpha} >= {rtol:.1e}"
        )

    @pytest.mark.parametrize("n,level", SWEEP)
    def test_weights_sum_to_one(self, n, level):
        pts, w = smolyak_points(n, level)
        assert pts.shape[1] == n
        assert abs(w.sum() - 1.0) < 1e-10

    @pytest.mark.parametrize("n,level", SWEEP)
    def test_exactness_degree_2level_plus_1(self, n, level):
        # standard Smolyak result for nested 1-D rules whose level-q rule
        # is exact to degree >= 2q+1: total-degree exactness 2*level+1.
        # Verified to actually hold in every MEASURED cell (the GK
        # milestone degrees 1, 5, 15, 29 dominate 2q+1 at every q), so
        # the brief's standard-mapping test survives unweakened; MEASURED
        # additionally claims the per-cell surplus.
        pts, w = smolyak_points(n, level)
        assert_rule_exact(pts, w, n, 2 * level + 1)

    @pytest.mark.parametrize("algorithm,n,level,degree,fail_alpha", MEASURED)
    def test_measured_exactness(self, algorithm, n, level, degree, fail_alpha):
        pts, w = smolyak_points(n, level, algorithm=algorithm)
        assert abs(w.sum() - 1.0) < 1e-10
        # Measured worst relative error inside the claimed range is
        # 2.7e-14 (algorithm 0, n=8, level 3); 1e-8 leaves margin above
        # that noise floor while sitting far below the smallest measured
        # past-degree failure (2.1e-4, next test).
        self._assert_exact_through_degree_relative(pts, w, n, degree, rtol=1e-8)

    @pytest.mark.parametrize("algorithm,n,level,degree,fail_alpha", MEASURED)
    def test_sharp_past_measured_degree(self, algorithm, n, level, degree, fail_alpha):
        # fail_alpha is the measured first-failing monomial (degree+1);
        # smallest measured failure is 2.1e-4 relative (algorithm 0, n=1,
        # level 3, (30,)), so 1e-5 separates cleanly from the 1e-8 pass
        # tolerance above.
        assert sum(fail_alpha) == degree + 1
        pts, w = smolyak_points(n, level, algorithm=algorithm)
        true_val = gaussian_moment(fail_alpha)
        rule_val = rule_moment(pts, w, fail_alpha)
        assert abs(rule_val - true_val) > 1e-5 * abs(true_val), (
            f"algorithm={algorithm} n={n} level={level}: expected "
            f"{fail_alpha} to fail past the measured degree {degree} "
            f"(rule={rule_val}, true={true_val})"
        )

    def test_level_0_is_single_point(self):
        pts, w = smolyak_points(4, 0)
        np.testing.assert_array_equal(pts, np.zeros((1, 4)))
        np.testing.assert_allclose(w, [1.0])

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_duplicates_merged(self, n):
        pts, _ = smolyak_points(n, 3)
        assert len(np.unique(pts.round(12), axis=0)) == len(pts)

    @pytest.mark.parametrize("n", [4, 6, 8])
    def test_fewer_points_than_tensor(self, n):
        # the value proposition: sparse vs tensor Gauss-Hermite at matched
        # degree (level 2 -> degree 5 -> 3-point 1-D rule tensorized)
        pts, _ = smolyak_points(n, 2)
        assert len(pts) < 3**n

    def test_point_count_n8_level2(self):
        # pins the docstring's worked point-count example (structural
        # property of the construction, not an external golden value)
        pts, _ = smolyak_points(8, 2)
        assert len(pts) == 177

    @pytest.mark.parametrize("algorithm,max_level", [(0, 3), (1, 2)])
    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_grids_nest_across_levels(self, algorithm, max_level, n):
        # the point of building on GK: each level's grid contains the
        # previous level's grid EXACTLY (same table constants, so exact
        # float equality, no tolerance needed)
        for level in range(max_level):
            lo, _ = smolyak_points(n, level, algorithm=algorithm)
            hi, _ = smolyak_points(n, level + 1, algorithm=algorithm)
            hi_keys = {tuple(p) for p in hi}
            assert all(tuple(p) in hi_keys for p in lo)
            assert len(hi) > len(lo)

    def test_negative_weights_present_and_disclosed(self):
        # already at n=3, level=1 the combination coefficients drive one
        # weight negative -- the docstring's negative-weights caveat is
        # about real behavior, not a hypothetical
        _, w = smolyak_points(3, 1)
        assert bool((w < 0).any())

    def test_gk1d_matches_nd_generator(self):
        # _gk_1d must agree with genz_keister_points collapsed to 1-D
        for m in (1, 2, 4, 8):
            x1, w1 = _gk_1d(0, m)
            xg, wg = genz_keister_points(1, m, algorithm=0)
            order = np.argsort(xg[:, 0])
            np.testing.assert_allclose(x1, xg[order, 0], atol=1e-14)
            np.testing.assert_allclose(w1, wg[order], atol=1e-14)

    def test_gk1d_m0_is_single_point(self):
        for algorithm in (0, 1):
            x, w = _gk_1d(algorithm, 0)
            np.testing.assert_array_equal(x, [0.0])
            np.testing.assert_allclose(w, [1.0])

    def test_invalid_args_raise(self):
        with pytest.raises(ValueError):
            smolyak_points(0, 1)
        with pytest.raises(ValueError):
            smolyak_points(3, -1)
        with pytest.raises(ValueError):
            smolyak_points(3, 99)  # past algorithm 0's level cap (3)
        with pytest.raises(ValueError):
            smolyak_points(3, 3, algorithm=1)  # past algorithm 1's cap (2)
        with pytest.raises(ValueError):
            smolyak_points(3, 1.5)  # non-integer level
        with pytest.raises(ValueError):
            smolyak_points(3, 1, algorithm=2)  # no such GK table
