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
