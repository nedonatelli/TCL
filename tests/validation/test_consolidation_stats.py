"""Statistics helpers consolidated to one body each (v2.12.0 Task 6.2).

NEES/NIS, MAD and the Mahalanobis family are pinned against the code shipped
in v2.11.1 at rtol=1e-14 on seeded inputs.
"""

import math
import warnings

import numpy as np
import pytest

from pytcl.assignment_algorithms import gating
from pytcl.dynamic_estimation.kalman import matrix_utils
from pytcl.mathematical_functions.combinatorics import combinatorics
from pytcl.mathematical_functions.statistics import distributions, estimators
from pytcl.static_estimation import robust
from tests.validation._v2111 import shipped_module

N_CASES = 200
RTOL = 1e-14


@pytest.fixture(scope="module")
def old_estimators():
    return shipped_module("pytcl/mathematical_functions/statistics/estimators.py")


@pytest.fixture(scope="module")
def old_robust():
    return shipped_module("pytcl/static_estimation/robust.py")


@pytest.fixture(scope="module")
def old_matrix_utils():
    return shipped_module("pytcl/dynamic_estimation/kalman/matrix_utils.py")


@pytest.fixture(scope="module")
def old_distributions():
    return shipped_module("pytcl/mathematical_functions/statistics/distributions.py")


@pytest.fixture(scope="module")
def old_combinatorics():
    return shipped_module("pytcl/mathematical_functions/combinatorics/combinatorics.py")


def _spd(rng, n):
    A = rng.standard_normal((n, n))
    return A @ A.T + n * np.eye(n)


def _nees_cases():
    rng = np.random.default_rng(2130)
    for i in range(N_CASES):
        n = int(rng.integers(1, 7))
        P = _spd(rng, n)
        kind = i % 4
        if kind == 0:
            yield rng.standard_normal(n), P
        elif kind == 1:
            yield rng.standard_normal((int(rng.integers(1, 9)), n)), P
        elif kind == 2:
            yield rng.integers(-5, 6, size=n), np.round(P)
        else:
            yield rng.integers(-5, 6, size=(3, n)), np.round(P)


@pytest.mark.parametrize("name", ["nees", "nis"])
def test_batch_nees_nis_match_v2111(old_estimators, name):
    old, new = getattr(old_estimators, name), getattr(estimators, name)
    for e, P in _nees_cases():
        if np.linalg.cond(P.astype(float)) > 1e6:
            continue
        a, b = old(e, P), new(e, P)
        assert np.shape(a) == np.shape(b)
        assert type(a) is type(b)
        assert np.asarray(b).dtype == np.float64
        np.testing.assert_allclose(b, a, rtol=RTOL, atol=0)


def test_batch_nees_empty_batch_keeps_shape(old_estimators):
    P = np.eye(3)
    a = old_estimators.nees(np.zeros((0, 3)), P)
    b = estimators.nees(np.zeros((0, 3)), P)
    assert a.shape == b.shape == (0,)


@pytest.mark.parametrize("name", ["nees", "nis"])
def test_singular_covariance_raises_linalg_error(old_estimators, name):
    P = np.array([[1.0, 1.0], [1.0, 1.0]])
    for e in (np.array([1.0, 2.0]), np.ones((3, 2))):
        with pytest.raises(np.linalg.LinAlgError):
            getattr(old_estimators, name)(e, P)
        with pytest.raises(np.linalg.LinAlgError):
            getattr(estimators, name)(e, P)


def test_nees_is_unnormalized():
    P = 0.5 * np.eye(4)
    assert estimators.nees(np.ones(4), P) == pytest.approx(8.0)


def test_mad_matches_v2111(old_robust):
    rng = np.random.default_rng(2131)
    for i in range(N_CASES):
        r = rng.standard_normal(int(rng.integers(1, 40))) * 10 ** rng.uniform(-3, 3)
        if i % 3 == 0:
            r = np.round(r).astype(np.int64)
        if i % 5 == 0:
            r = r.tolist()
        c = 1.4826 if i % 2 else float(rng.uniform(0.5, 3.0))
        a, b = old_robust.mad(r, c), robust.mad(r, c)
        assert type(b) is float and type(a) is float
        np.testing.assert_allclose(b, a, rtol=RTOL, atol=0)
    np.testing.assert_allclose(robust.mad(np.arange(5.0)), 1.4826, rtol=RTOL)


def test_mad_matches_v2111_on_a_column_vector(old_robust):
    r = np.array([[1.0], [2.0], [4.0], [8.0], [16.0]])
    assert robust.mad(r) == old_robust.mad(r)


def test_mad_keyword_c_still_accepted():
    assert robust.mad([1.0, 2.0, 3.0], c=2.0) == 2.0


def _mahal_cases():
    rng = np.random.default_rng(2132)
    for i in range(N_CASES):
        n = int(rng.integers(1, 8))
        S = _spd(rng, n)
        y = rng.standard_normal(n)
        if i % 4 == 1:
            y = y.reshape(-1, 1)
        elif i % 4 == 2:
            y = np.round(y * 3).astype(np.int64)
        elif i % 4 == 3:
            y = y.tolist()
        yield y, S


def test_compute_mahalanobis_distance_matches_v2111(old_matrix_utils):
    for y, S in _mahal_cases():
        a = old_matrix_utils.compute_mahalanobis_distance(y, S)
        b = matrix_utils.compute_mahalanobis_distance(y, S)
        assert type(a) is type(b) is float
        np.testing.assert_allclose(b, a, rtol=RTOL, atol=1e-300)


def test_compute_mahalanobis_distance_cholesky_path_matches_v2111(old_matrix_utils):
    for y, S in _mahal_cases():
        L = np.linalg.cholesky(S)
        a = old_matrix_utils.compute_mahalanobis_distance(y, L, S_is_cholesky=True)
        b = matrix_utils.compute_mahalanobis_distance(y, L, S_is_cholesky=True)
        np.testing.assert_allclose(b, a, rtol=RTOL, atol=1e-300)


def test_compute_mahalanobis_distance_non_pd_warns_and_returns_nan(old_matrix_utils):
    S = np.diag([1.0, -1.0])
    y = np.array([1.0, 1.0])
    for mod in (old_matrix_utils, matrix_utils):
        with pytest.warns(RuntimeWarning, match="not positive definite"):
            assert math.isnan(mod.compute_mahalanobis_distance(y, S))


def test_gating_is_squared_and_matrix_utils_is_not():
    y = np.array([1.0, 0.5, -0.25])
    S = np.diag([2.0, 1.0, 0.5])
    squared = gating.mahalanobis_distance(y, S)
    assert matrix_utils.compute_mahalanobis_distance(y, S) == pytest.approx(
        math.sqrt(squared), rel=RTOL
    )
    assert squared == pytest.approx(0.5 + 0.25 + 0.125)


def test_gating_value_is_unchanged_from_v2111():
    old_gating = shipped_module("pytcl/assignment_algorithms/gating.py")
    rng = np.random.default_rng(2133)
    for _ in range(N_CASES):
        n = int(rng.integers(1, 12))
        y, S = rng.standard_normal(n), _spd(rng, n)
        assert gating.mahalanobis_distance(y, S) == old_gating.mahalanobis_distance(
            y, S
        )


def test_multivariate_gaussian_mahalanobis_matches_v2111(old_distributions):
    rng = np.random.default_rng(2134)
    for _ in range(50):
        n = int(rng.integers(1, 6))
        mean, S = rng.standard_normal(n), _spd(rng, n)
        old = old_distributions.MultivariateGaussian(mean, S)
        new = distributions.MultivariateGaussian(mean, S)
        for x in (
            rng.standard_normal(n),
            rng.standard_normal((4, n)),
            rng.standard_normal((2, 3, n)),
            rng.standard_normal((0, n)),
            np.round(rng.standard_normal(n) * 3).astype(np.int64),
        ):
            a, b = old.mahalanobis(x), new.mahalanobis(x)
            assert np.shape(a) == np.shape(b)
            assert type(a) is type(b)
            np.testing.assert_allclose(b, a, rtol=RTOL, atol=1e-300)


def test_docstrings_say_squared_or_not():
    first = lambda f: f.__doc__.strip().splitlines()[0]  # noqa: E731
    assert "the squared Mahalanobis distance" in first(gating.mahalanobis_distance)
    assert "not squared" in first(matrix_utils.compute_mahalanobis_distance)
    assert "not squared" in first(distributions.MultivariateGaussian.mahalanobis)


def test_factorial_matches_v2111(old_combinatorics):
    for n in list(range(0, 30)) + [np.int64(7), True, 0.0, 1.0]:
        assert combinatorics.factorial(n) == old_combinatorics.factorial(n)
        assert type(combinatorics.factorial(n)) is type(old_combinatorics.factorial(n))
    for bad in (-1, -7, np.int64(-2)):
        with pytest.raises(ValueError, match="non-negative"):
            combinatorics.factorial(bad)
    for bad in (5.0, 2.5):
        with pytest.raises(TypeError):
            old_combinatorics.factorial(bad)
        with pytest.raises(TypeError):
            combinatorics.factorial(bad)
    assert combinatorics.factorial(25) == math.factorial(25)


def test_factorial_contracts_differ_between_modules():
    from pytcl.mathematical_functions.special_functions import factorial

    assert factorial(-3) == 0
    with pytest.raises(ValueError):
        combinatorics.factorial(-3)
    assert "special_functions" in combinatorics.factorial.__doc__
    assert "combinatorics" in factorial.__doc__


def test_no_deprecation_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        estimators.nees(np.ones(2), np.eye(2))
        robust.mad([1.0, 2.0, 3.0])


def _outcome(fn, *args):
    try:
        return fn(*args)
    except Exception as exc:  # noqa: BLE001 - the class is what is compared
        return exc


@pytest.mark.parametrize("name", ["nees", "nis"])
def test_singular_and_near_singular_covariances_raise_as_v2111(old_estimators, name):
    rng = np.random.default_rng(2135)
    raised = 0
    for i in range(N_CASES):
        n = int(rng.integers(2, 6))
        B = rng.standard_normal((n, n - 1))
        P = B @ B.T
        if i % 2:
            P = P + 10.0 ** rng.uniform(-18, -10) * np.eye(n)
        e = rng.standard_normal((3, n)) if i % 3 else rng.standard_normal(n)
        a = _outcome(getattr(old_estimators, name), e, P)
        b = _outcome(getattr(estimators, name), e, P)
        assert type(a) is type(b)
        if isinstance(a, Exception):
            raised += 1
        else:
            np.testing.assert_allclose(b, a, rtol=RTOL, atol=0)
    assert raised > 5


@pytest.mark.parametrize("name", ["nees", "nis"])
def test_undocumented_nees_inputs_match_v2111(old_estimators, name):
    P3 = np.stack([np.eye(2), 2 * np.eye(2)])
    cases = [
        (2.0, [[2.0]]),
        (np.ones(2), P3),
        (np.ones((3, 2)), P3),
        (np.ones((2, 3, 2)), np.eye(2)),
        (np.ones(3), np.eye(2)),
        (np.ones((4, 3)), np.eye(2)),
    ]
    for e, P in cases:
        a = _outcome(getattr(old_estimators, name), e, P)
        b = _outcome(getattr(estimators, name), e, P)
        assert type(a) is type(b), (e, P)
        if not isinstance(a, Exception):
            np.testing.assert_allclose(b, a, rtol=RTOL, atol=0)


def test_compute_mahalanobis_distance_shape_mismatch_raises_as_v2111(
    old_matrix_utils,
):
    rng = np.random.default_rng(2136)
    S = _spd(rng, 6)
    for y in (rng.standard_normal(3), rng.standard_normal(7)):
        for chol in (False, True):
            M = np.linalg.cholesky(S) if chol else S
            a = _outcome(old_matrix_utils.compute_mahalanobis_distance, y, M, chol)
            b = _outcome(matrix_utils.compute_mahalanobis_distance, y, M, chol)
            assert isinstance(a, Exception) and type(a) is type(b)


def test_multivariate_gaussian_mahalanobis_edge_inputs_match_v2111(old_distributions):
    rng = np.random.default_rng(2137)
    mean, S = rng.standard_normal(3), _spd(rng, 3)
    old = old_distributions.MultivariateGaussian(mean, S)
    new = distributions.MultivariateGaussian(mean, S)
    for x in (
        rng.standard_normal((3, 1)),
        rng.standard_normal((1, 3)),
        rng.standard_normal(5),
        rng.standard_normal((4, 5)),
        rng.standard_normal(3).tolist(),
    ):
        a, b = _outcome(old.mahalanobis, x), _outcome(new.mahalanobis, x)
        assert type(a) is type(b)
        if isinstance(a, Exception):
            continue
        assert np.shape(a) == np.shape(b)
        np.testing.assert_allclose(b, a, rtol=RTOL, atol=1e-300)


def test_multivariate_gaussian_non_pd_covariance_matches_v2111(old_distributions):
    S = np.diag([1.0, -1.0])
    a = _outcome(old_distributions.MultivariateGaussian, [0.0, 0.0], S)
    b = _outcome(distributions.MultivariateGaussian, [0.0, 0.0], S)
    assert isinstance(a, Exception) and type(a) is type(b)


def test_batch_paths_are_not_slower_than_v2111(old_estimators, old_distributions):
    import time

    rng = np.random.default_rng(2138)
    m, n = 10000, 6
    P, e = _spd(rng, n), rng.standard_normal((m, n))
    mean = rng.standard_normal(n)
    old_g = old_distributions.MultivariateGaussian(mean, P)
    new_g = distributions.MultivariateGaussian(mean, P)

    def best(fn):
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t0)
        return min(times)

    assert best(lambda: estimators.nees(e, P)) < 20 * best(
        lambda: old_estimators.nees(e, P)
    )
    assert best(lambda: new_g.mahalanobis(e)) < 20 * best(lambda: old_g.mahalanobis(e))
