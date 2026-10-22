"""Guards for silent-wrong-answer defects in decompositions, Romberg, the MLE
ridge and zero-iteration budgets (v2.11.1, task 5.7)."""

import warnings

import numpy as np
import pytest

from pytcl.mathematical_functions.basic_matrix.decompositions import (
    chol_semi_def,
    matrix_sqrt,
)
from pytcl.mathematical_functions.numerical_integration.quadrature import romberg
from pytcl.static_estimation.maximum_likelihood import mle_newton_raphson
from pytcl.static_estimation.robust import irls, ransac

_rng = np.random.default_rng(0)
_A = np.column_stack([np.ones(20), np.arange(20.0)])
_b = 1.0 + 2.0 * np.arange(20.0) + _rng.normal(0.0, 0.1, 20)


def test_chol_semi_def_rejects_indefinite_input():
    with pytest.raises(np.linalg.LinAlgError, match="not positive semi-definite"):
        chol_semi_def(np.diag([1.0, -1.0]))


def test_chol_semi_def_factors_rank_deficient_psd_exactly():
    A = np.diag([1.0, 0.0])
    L = chol_semi_def(A)
    assert np.allclose(L @ L.T, A, atol=1e-15)
    assert np.all(np.diag(L) >= 0.0)


def test_chol_semi_def_rank_deficient_dense_psd_reconstructs():
    v = np.array([[1.0], [2.0], [3.0]])
    A = v @ v.T
    L = chol_semi_def(A)
    assert np.allclose(L @ L.T, A, atol=1e-12)
    assert np.all(np.diag(L) >= 0.0)


def _with_eigenvalues(eigs, seed=7):
    q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((3, 3)))
    a = q @ np.diag(eigs) @ q.T
    return 0.5 * (a + a.T), q


@pytest.mark.parametrize("neg", [-1e-14, -5e-11])
def test_chol_semi_def_accepts_rounding_level_negative_eigenvalue(neg):
    # cholSemiDef.m is meant for filter covariances that are "very slightly
    # non-positive semi-definite due to finite precision errors"; those must
    # still factor, to the matrix with the negative eigenvalue floored at 0.
    A, q = _with_eigenvalues([1.0, 0.5, neg])
    floored = q @ np.diag([1.0, 0.5, 0.0]) @ q.T
    L = chol_semi_def(A)
    assert np.max(np.abs(L @ L.T - floored)) <= 1e-15 + abs(neg)
    assert np.all(np.diag(L) >= 0.0)


def test_chol_semi_def_rejects_eigenvalue_just_past_the_band():
    A, _ = _with_eigenvalues([1.0, 0.5, -2e-10])
    with pytest.raises(np.linalg.LinAlgError, match="not positive semi-definite"):
        chol_semi_def(A)


@pytest.mark.parametrize("seed", range(5))
def test_chol_semi_def_accepts_rank_deficient_outer_products(seed):
    # B @ B.T of rank 2 in 6x6 has computed eigenvalues down to -1.6e-16 of
    # the largest, so the plain Cholesky fails and the guard must pass it.
    B = np.random.default_rng(seed).standard_normal((6, 2)) * [[1e3, 1e-2]]
    P = B @ B.T
    L = chol_semi_def(P)
    assert np.max(np.abs(L @ L.T - P)) <= 1e-14 * np.max(np.abs(P))
    assert np.all(np.diag(L) >= 0.0)


@pytest.mark.parametrize("method", ["schur", "eigenvalue", "denman_beavers"])
def test_matrix_sqrt_rejects_negative_eigenvalues(method):
    with pytest.raises(
        np.linalg.LinAlgError,
        match=r"negative eigenvalue \(-4\); no real square root exists",
    ):
        matrix_sqrt(np.diag([-4.0, 9.0]), method=method)


@pytest.mark.parametrize("method", ["schur", "eigenvalue", "denman_beavers"])
def test_matrix_sqrt_still_squares_back_for_spd(method):
    A = np.array([[4.0, 1.0], [1.0, 3.0]])
    S = matrix_sqrt(A, method=method)
    assert np.allclose(S @ S, A, atol=1e-10)


def test_matrix_sqrt_denman_beavers_warns_on_nonconvergence():
    # Eigenvalue ratio 1e80 needs far more than 50 Denman-Beavers iterations.
    with pytest.warns(RuntimeWarning, match="converge"):
        matrix_sqrt(np.diag([1e-40, 1e40]), method="denman_beavers")


def test_romberg_warns_when_it_cannot_reach_the_tolerance():
    # The true integral is 0.49998586; six steps stop at 0.50098244.
    with pytest.warns(RuntimeWarning, match=r"tolerance 1e-15.*estimate 0\.50098243"):
        got = romberg(lambda t: abs(t - 0.3) ** 0.5, 0.0, 1.0, tol=1e-15, max_steps=6)
    assert got == pytest.approx(0.50098244, rel=1e-8, abs=0)


def test_romberg_converges_silently_on_smooth_integrand():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = romberg(lambda t: t**2, 0.0, 1.0)
    assert result == pytest.approx(1.0 / 3.0, abs=1e-8)


def _flat_problem():
    def ll(t):
        return -0.5 * t[0] ** 2

    def score(t):
        return np.array([-t[0], 0.0])

    def hess(t):
        return np.diag([-1.0, 0.0])

    return ll, score, hess


def test_mle_reports_unridged_fisher_and_honest_covariance_on_flat_direction():
    ll, score, hess = _flat_problem()
    with pytest.warns(RuntimeWarning, match="not positive definite"):
        res = mle_newton_raphson(ll, score, np.array([1.0, 0.0]), hessian=hess)
    assert res.fisher_info == pytest.approx(np.diag([1.0, 0.0]), abs=0)
    assert not res.converged
    assert res.covariance[0, 0] == pytest.approx(1.0, abs=1e-12)
    assert np.isinf(res.covariance[1, 1])
    assert np.isnan(res.covariance[0, 1]) and np.isnan(res.covariance[1, 0])


def test_mle_well_conditioned_is_unchanged_and_converged():
    data = np.array([1.0, 2.0, 3.0, 4.0])

    def ll(t):
        return -0.5 * np.sum((data - t[0]) ** 2)

    def score(t):
        return np.array([np.sum(data - t[0])])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = mle_newton_raphson(
            ll, score, np.array([0.0]), hessian=lambda t: np.array([[-4.0]])
        )
    assert res.converged
    assert res.theta[0] == pytest.approx(2.5, abs=1e-12)
    assert res.fisher_info == pytest.approx(np.array([[4.0]]), abs=0)
    assert res.covariance == pytest.approx(np.array([[0.25]]), abs=1e-15)


@pytest.mark.parametrize(
    "fn,kwargs", [(irls, {"max_iter": 0}), (ransac, {"max_trials": 0})]
)
def test_zero_iteration_budget_raises_a_real_error(fn, kwargs):
    with pytest.raises(ValueError, match="at least one"):
        fn(_A, _b, **kwargs)
