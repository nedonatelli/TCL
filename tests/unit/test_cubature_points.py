"""Tests for Gaussian-weight cubature point generators.

PROPERTY class: a degree-d rule must integrate every monomial of total
degree <= d against N(0, I) exactly, and must FAIL on some degree d+1
monomial (sharpness -- guards against a vacuous exactness loop).
"""

import itertools

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pytcl.mathematical_functions.numerical_integration.cubature_points import (
    fifth_order_cubature_points,
    seventh_order_cubature_points,
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


class TestHelpers:
    def test_gaussian_moment_known_values(self):
        # E[x^2]=1, E[x^4]=3, E[x^6]=15, E[x^2 y^2]=1, odd -> 0
        assert gaussian_moment((2,)) == 1.0
        assert gaussian_moment((4,)) == 3.0
        assert gaussian_moment((6,)) == 15.0
        assert gaussian_moment((8,)) == 105.0
        assert gaussian_moment((2, 2)) == 1.0
        assert gaussian_moment((1, 2)) == 0.0


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
