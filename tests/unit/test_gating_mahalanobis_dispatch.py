"""Behavior-equality tests for the Mahalanobis dispatch in gating.py.

Context (v2.5.0 campaign, task C2): `mahalanobis_distance()` always took the
generic `np.linalg.solve(S, nu)` LAPACK path regardless of dimension, even
though `gating.py` already defined `@njit` fast-path kernels for 2D/3D/
general that nothing called. These tests pin the *old* generic path (computed
inline here, not via the dispatch) as the reference and assert the dispatched
implementation reproduces it to a measured tolerance, for both well-
conditioned and near-singular covariances across dims {1, 2, 3, 6, 9}. They
were written and passing (trivially, pre-refactor) before the dispatch in
gating.py was touched.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pytcl.assignment_algorithms.gating import mahalanobis_distance


def _old_generic_path(nu: np.ndarray, S: np.ndarray) -> float:
    """The pre-dispatch `mahalanobis_distance` body, verbatim: generic solve."""
    S_inv_nu = np.linalg.solve(S, nu)
    return float(nu @ S_inv_nu)


DIMS = [1, 2, 3, 6, 9]


class TestMahalanobisDispatchMatchesGenericPath:
    """Well-conditioned SPD covariances: dispatch must match old path tightly.

    Bound: measured empirically over dims {1,2,3,6,9} x 20 seeds -- max
    relative error 6.12e-16 at dim=6 (dims 2/3 use closed-form Cramer's-rule/
    adjugate solves at ~3e-16; dims 6/9 use `np.linalg.inv` + a njit
    quadratic-form kernel at ~4-6e-16; dim=1 is exact at 0.0). All of these
    are float64-ULP-level differences against LAPACK's LU-based `solve`.
    rtol=1e-12 keeps ~1e6x margin over the measured worst case.
    """

    @pytest.mark.parametrize("dim", DIMS)
    @pytest.mark.parametrize("seed", list(range(20)))
    def test_well_conditioned_spd(self, dim, seed):
        rng = np.random.default_rng(seed * 1_000 + dim)
        A = rng.normal(size=(dim, dim))
        S = A @ A.T + dim * np.eye(dim)  # SPD, condition number bounded
        nu = rng.normal(size=dim)

        expected = _old_generic_path(nu, S)
        actual = mahalanobis_distance(nu, S)

        assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


class TestMahalanobisDispatchNearSingular:
    """Near-singular covariances (condition number ~1e10).

    Closed-form Cramer's-rule/adjugate inversion and LAPACK's LU-based solve
    diverge more as the matrix approaches singularity -- both are "correct"
    in the sense of solving the same ill-conditioned system, but they are not
    bitwise identical. Bound: measured empirically over dims {2,3} x 10
    seeds with condition numbers up to ~1.17e10 -- max relative error
    2.79e-7. rtol=1e-5 keeps ~35x margin over the measured worst case.
    """

    @pytest.mark.parametrize("dim", [2, 3])
    @pytest.mark.parametrize("seed", list(range(10)))
    def test_near_singular_spd(self, dim, seed):
        rng = np.random.default_rng(seed * 7 + dim * 100)
        A = rng.normal(size=(dim, dim))
        S = A @ A.T
        eigvals, eigvecs = np.linalg.eigh(S)
        eigvals = np.clip(eigvals, 1e-3, None)
        eigvals[0] = 1e-9  # push condition number to ~1e9-1e10
        S = (eigvecs * eigvals) @ eigvecs.T
        nu = rng.normal(size=dim)

        expected = _old_generic_path(nu, S)
        actual = mahalanobis_distance(nu, S)

        assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


class TestMahalanobisDispatchSingular:
    """Exactly-singular covariances must fail exactly as before: LinAlgError."""

    @pytest.mark.parametrize("dim", [2, 3])
    def test_exactly_singular_raises_linalgerror(self, dim):
        # Rank-1 PSD covariance (outer product of a single vector) -- exactly
        # singular for any dim > 1.
        v = np.arange(1, dim + 1, dtype=np.float64)
        S = np.outer(v, v)
        nu = np.ones(dim)

        with pytest.raises(np.linalg.LinAlgError):
            _old_generic_path(nu, S)
        with pytest.raises(np.linalg.LinAlgError):
            mahalanobis_distance(nu, S)
