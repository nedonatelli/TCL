"""Tests for Gaussian-weight cubature point generators.

PROPERTY class: a degree-d rule must integrate every monomial of total
degree <= d against N(0, I) exactly, and must FAIL on some degree d+1
monomial (sharpness -- guards against a vacuous exactness loop).
"""

import hashlib
import itertools

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.special import gamma, gammaln

from pytcl.mathematical_functions.numerical_integration.cubature_points import (
    _sphere_surface_points,
    cubature_point_moments,
    fifth_order_cubature_points,
    fourteenth_order_cubature_points,
    genz_keister_points,
    second_order_cubature_points,
    seventh_order_cubature_points,
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
        # Mean/covariance exactness must hold regardless of w0/alpha,
        # including when alpha drives the center weight negative (see
        # test_negative_center_weight_is_not_suppressed).
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
        with pytest.raises(ValueError):
            seventh_order_cubature_points(2)


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
        pts, w = fourteenth_order_cubature_points(3)
        for alpha in monomials_up_to(3, 14):
            assert_allclose(
                rule_moment(pts, w, alpha),
                gaussian_moment(alpha),
                rtol=1e-6,
                atol=1e-8,
                err_msg=f"monomial {alpha} not integrated exactly",
            )

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


class TestSphericalRadialBetaGeneralization:
    """spherical_radial_points(n, degree, beta=0.0) -- pre-existing callers
    (beta omitted or 0.0) must reproduce bit-for-bit the pre-generalization
    output. These SHA-256 digests were captured from spherical_radial_points
    BEFORE the beta parameter was added.
    """

    _BASELINE_SHA256 = {
        (1, 3): "800d279e3e66a5a9cc5ebba21408f63ac39151ff609d827c475b775832a8891c",
        (1, 5): "756565d3af37c593033924a35309ef0105a122e11a6e1be616972548c9c8b5c9",
        (1, 7): "756565d3af37c593033924a35309ef0105a122e11a6e1be616972548c9c8b5c9",
        (1, 9): "db212c0d7f896eca7a2ea54ec854ef7af183ec2ff99cda003c56759557b9b2b3",
        (2, 3): "7c144cc11bc583514574968f5ff0cdc92cd0d5be3d25bb457862b3fc70a0af4e",
        (2, 5): "0e59d08cca88aa4565a0533eb4ee71a0efcd48635f6d24fe95d017dae58911eb",
        (2, 7): "12d8037c982cc68c937ca0a59ddf553de0c4409fedc6c5a9f58c88bf9172aa57",
        (2, 9): "131b674dfc120b6160ac5e37746fb8472c7c8ddd460c2dda2b65b26075697e73",
        (3, 3): "e83e8b72c9c59da14d90f324722ea8dcbe7bd574f7636df0c84b82729ebd1486",
        (3, 5): "6c2d703888317a15eb1dff2c23306d4687e315b109d520c79ddf9e1154a71e59",
        (3, 7): "64f632263e1a7521370f3edc44fe5ad9dd39d9371d6a0f0f85a59a8ab941f7b4",
        (3, 9): "9b074a352e8dab99ad9afb54e598648071ba2a67687ddc8c3746ab8538702e1a",
        (4, 3): "b5b98b69c1fe51bf7d84c0fa167f8b3905ca1c87f446af96e68556002ad8b00f",
        (4, 5): "013b9269ee841980113bf7509c487bb1f9ea2e472b7f215c1e2b51050e1d6ac1",
        (4, 7): "b162270d7810b4fa74012029621a9ddda0394307b02ac5c455a28c703bf3ff8d",
        (4, 9): "816bd69e26060699e996052a64954f0a97b24c7a8e7001bc6cde119f19e7068e",
    }

    @pytest.mark.parametrize("n,degree", list(_BASELINE_SHA256.keys()))
    def test_default_beta_bit_identical_to_pre_generalization(self, n, degree):
        pts, w = spherical_radial_points(n, degree)
        digest = hashlib.sha256(pts.tobytes() + w.tobytes()).hexdigest()
        assert digest == self._BASELINE_SHA256[(n, degree)]

    @pytest.mark.parametrize("n,degree", list(_BASELINE_SHA256.keys())[:4])
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
    to a follow-on effort) -- see `TestNesting` below and the "Notes"
    section of ``genz_keister_points``'s docstring, which this test class
    mirrors section for section:

    - the *generic* exactness bound, degree ``2m + 1``, holding for any
      n and any m (`TestGenericDegreeBound`);
    - the n=1-only *bonus* at specific "milestone" m values, where
      single-axis moments (equivalently, the n=1 rule) are exact well
      beyond ``2m + 1`` (`TestMilestoneBonus`);
    - the documented precision/nesting breakdown at the very top of each
      algorithm's valid ``m`` range (`TestNesting`,
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
        _, w_default = genz_keister_points(1, 9)
        pts_loose, w_loose = genz_keister_points(1, 9, eps_val=1e-3)
        assert len(w_loose) < len(w_default)
        assert np.all(np.abs(w_loose) > 1e-3)

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

    # ---- generic exactness bound: degree 2m+1, any n, any m ----

    @staticmethod
    def _assert_some_monomial_of_degree_fails(
        points, weights, n, degree, threshold=1e-4
    ):
        """At least one monomial of EXACTLY `degree` fails.

        Deliberately does not hand-pick a single probe monomial: WHICH
        monomial fails first varies by (n, m) -- single-axis moments can be
        elevated well beyond the generic bound at "milestone" m values
        (TestMilestoneBonus below), while mixed-exponent ones never are
        (TestGenericDegreeBound.test_milestone_bonus_does_not_extend_to_mixed_exponents).
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
    # elevation at any m (TestMilestoneBonus's mixed-exponent finding: the
    # bonus is single-axis-only), so every m is safe there.
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

    def test_milestone_bonus_does_not_extend_to_mixed_exponents(self):
        # REFERENCE (hand-verified value, see also TestMilestoneBonus):
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

    def test_matches_brute_force_reconstruction(self):
        # Independent re-derivation used to gain confidence in the port
        # (see the docstring's Notes): instead of computing one weight per
        # partition *class* and reusing it for every point in that
        # partition's symmetry orbit (what the shipped implementation
        # does, mirroring MATLAB's own optimization), evaluate every raw
        # n-tuple in {0,...,m}^n directly and independently. Both must
        # produce the identical point/weight set.
        def brute_force(n, m, algorithm=0):
            from pytcl.mathematical_functions.numerical_integration.cubature_points import (
                _GK_TABLES,
                _gk_inner_terms,
                _gk_partition_weight,
                _pm_combos,
            )

            lam, a = _GK_TABLES[algorithm]
            lam2 = lam[: m + 1] ** 2
            inner = _gk_inner_terms(a, lam2, m)
            pts, ws = [], []
            for p in itertools.product(range(m + 1), repeat=n):
                if sum(p) > m:
                    continue
                w_p = _gk_partition_weight(m, p, inner)
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

        for n, m in [(1, 6), (2, 5), (3, 4)]:
            pts_opt, w_opt = genz_keister_points(n, m)
            pts_brute, w_brute = brute_force(n, m)
            c_opt = canonicalize(pts_opt, w_opt)
            c_brute = canonicalize(pts_brute, w_brute)
            assert c_opt[0].shape == c_brute[0].shape
            assert_allclose(c_opt[0], c_brute[0], atol=1e-10)
            assert_allclose(c_opt[1], c_brute[1], atol=1e-10)


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
