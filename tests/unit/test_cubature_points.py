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
from scipy.special import gamma

from pytcl.mathematical_functions.numerical_integration.cubature_points import (
    _sphere_surface_points,
    fifth_order_cubature_points,
    seventh_order_cubature_points,
    sphere_surface_to_gauss_points,
    spherical_radial_points,
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
