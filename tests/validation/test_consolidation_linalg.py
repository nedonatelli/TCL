"""Linear-algebra helpers consolidated to one body each (v2.12.0 Task 6.1).

Each delegation is pinned against the code shipped in v2.11.1: the old and
new paths must agree bit for bit on 200 seeded inputs.
"""

import warnings

import numpy as np
import pytest

from pytcl.coordinate_systems.jacobians.jacobians import (
    numerical_jacobian as jac_numerical_jacobian,
)
from pytcl.core import array_utils
from pytcl.core.exceptions import PytclDeprecationWarning, ValidationError
from pytcl.core.validation import check_symmetric, ensure_symmetric
from pytcl.dynamic_estimation import batch_estimation
from pytcl.dynamic_estimation.kalman import extended, unscented
from pytcl.mathematical_functions.basic_matrix import special_matrices
from tests.validation._v2111 import shipped_module

N_CASES = 200


@pytest.fixture(scope="module")
def old_unscented():
    return shipped_module("pytcl/dynamic_estimation/kalman/unscented.py")


@pytest.fixture(scope="module")
def old_extended():
    return shipped_module("pytcl/dynamic_estimation/kalman/extended.py")


@pytest.fixture(scope="module")
def old_special():
    return shipped_module(
        "pytcl/mathematical_functions/basic_matrix/special_matrices.py"
    )


@pytest.fixture(scope="module")
def old_batch():
    return shipped_module("pytcl/dynamic_estimation/batch_estimation.py")


def _covariance_cases():
    rng = np.random.default_rng(2120)
    for i in range(N_CASES):
        n = int(rng.integers(1, 7))
        A = rng.standard_normal((n, n))
        kind = i % 3
        if kind == 0:
            P = A @ A.T + 0.1 * np.eye(n)
        elif kind == 1 and n > 1:
            B = rng.standard_normal((n, n - 1))
            P = B @ B.T
        else:
            P = A @ A.T
        yield rng.standard_normal(n), (P + P.T) / 2


def _run(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except np.linalg.LinAlgError as exc:
        return exc


@pytest.mark.parametrize("name", ["merwe", "julier"])
def test_sigma_points_match_v2111(old_unscented, name):
    old = getattr(old_unscented, f"sigma_points_{name}")
    new = getattr(unscented, f"sigma_points_{name}")
    outcomes = {"ok": 0, "raised": 0}
    for x, P in _covariance_cases():
        a, b = _run(old, x, P), _run(new, x, P)
        if isinstance(a, Exception):
            assert isinstance(b, np.linalg.LinAlgError)
            outcomes["raised"] += 1
            continue
        np.testing.assert_array_equal(a.points, b.points)
        np.testing.assert_array_equal(a.Wm, b.Wm)
        np.testing.assert_array_equal(a.Wc, b.Wc)
        outcomes["ok"] += 1
    assert outcomes["ok"] > 100


def test_unscented_has_no_private_copy_of_eig_constants():
    from pytcl.dynamic_estimation.kalman import matrix_utils

    assert unscented._EIG_FLOOR_REL is matrix_utils._EIG_FLOOR_REL
    assert unscented._EIG_NEGATIVE_REL_TOL is matrix_utils._EIG_NEGATIVE_REL_TOL


def _jacobian_cases():
    rng = np.random.default_rng(2121)
    for _ in range(N_CASES):
        n = int(rng.integers(1, 6))
        m = int(rng.integers(1, 5))
        W = rng.standard_normal((m, n))
        x = rng.standard_normal(n)
        dx = float(10 ** rng.uniform(-9, -4))
        yield (lambda v, W=W: np.tanh(W @ v) + (W @ v) ** 2), x, dx


def test_numerical_jacobian_matches_v2111(old_extended):
    for f, x, dx in _jacobian_cases():
        old = old_extended.numerical_jacobian(f, x, dx)
        np.testing.assert_array_equal(extended.numerical_jacobian(f, x, dx), old)
        np.testing.assert_array_equal(jac_numerical_jacobian(f, x, dx), old)


def test_extended_numerical_jacobian_keeps_keyword_f():
    J = extended.numerical_jacobian(f=lambda v: v**2, x=np.array([2.0, 3.0]))
    np.testing.assert_allclose(J, np.diag([4.0, 6.0]), atol=1e-6)


def _matrix_cases():
    rng = np.random.default_rng(2122)
    for i in range(N_CASES):
        m, n = (int(k) for k in rng.integers(1, 7, size=2))
        if i % 2:
            yield rng.integers(-9, 10, size=(m, n))
        else:
            yield rng.standard_normal((m, n))


def test_vec_unvec_match_v2111(old_special):
    for A in _matrix_cases():
        m, n = A.shape
        v_old = old_special.vec(A)
        v_new = special_matrices.vec(A)
        assert v_new.shape == (m * n,)
        assert v_new.dtype == np.float64
        np.testing.assert_array_equal(v_new, v_old)
        np.testing.assert_array_equal(
            special_matrices.unvec(v_new, m, n), old_special.unvec(v_old, m, n)
        )
        np.testing.assert_array_equal(special_matrices.unvec(v_new, m, n), A)


def test_vec_shape_contracts_differ_by_module():
    A = np.arange(6).reshape(2, 3)
    assert special_matrices.vec(A).shape == (6,)
    assert array_utils.vec(A).shape == (6, 1)


def test_block_diag_matches_v2111(old_special):
    rng = np.random.default_rng(2123)
    for i in range(N_CASES):
        k = int(rng.integers(1, 5))
        blocks = []
        for _ in range(k):
            shape = tuple(int(s) for s in rng.integers(1, 5, size=2))
            blocks.append(
                rng.integers(-9, 10, size=shape)
                if i % 2
                else rng.standard_normal(shape)
            )
        old = old_special.block_diag(*blocks)
        new = special_matrices.block_diag(*blocks)
        assert new.dtype == np.float64
        np.testing.assert_array_equal(new, old)


def test_array_utils_block_diag_keeps_integer_dtype():
    out = array_utils.block_diag(np.array([[1, 2]]), np.array([[3]]))
    assert np.issubdtype(out.dtype, np.integer)


def test_batch_estimation_block_diag_replaced_by_array_utils(old_batch):
    assert not hasattr(batch_estimation, "_block_diag")
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    rng = np.random.default_rng(2124)
    hs = [
        (lambda k: lambda x: (np.linalg.matrix_power(F, k) @ x)[:1] ** 1)(k)
        for k in range(4)
    ]
    for _ in range(20):
        truth = rng.standard_normal(2) + 3.0
        z = np.column_stack([hs[k](truth) for k in range(4)])
        z = z + 0.01 * rng.standard_normal(z.shape)
        x0 = truth + 0.3 * rng.standard_normal(2)
        R = 0.01 * np.eye(1)
        for fn in ("batch_ls_nonlin_meas_nonlin_dyn", "batch_ls_nonlin_meas_lin_dyn"):
            args = (x0, z, hs, R) if fn.endswith("nonlin_dyn") else (x0, z, hs, F, R, 0)
            old = getattr(old_batch, fn)(*args)
            new = getattr(batch_estimation, fn)(*args)
            np.testing.assert_array_equal(new.x, old.x)
            np.testing.assert_array_equal(new.P, old.P)


def test_check_symmetric_raises_on_asymmetric_matrix():
    with pytest.raises(ValidationError, match="symmetric"):
        check_symmetric([[1.0, 2.0], [0.5, 1.0]])
    with pytest.raises(ValidationError, match="square"):
        check_symmetric(np.ones((2, 3)))


def test_check_symmetric_returns_input_unsymmetrized():
    mat = np.array([[1.0, 2.0 + 1e-11], [2.0, 1.0]])
    out = check_symmetric(mat)
    np.testing.assert_array_equal(out, mat)
    assert out[0, 1] != out[1, 0]


def test_core_ensure_symmetric_is_deprecated_and_still_validates():
    with pytest.warns(PytclDeprecationWarning, match="check_symmetric"):
        with pytest.raises(ValidationError, match="symmetric"):
            ensure_symmetric([[1.0, 2.0], [0.5, 1.0]])
    with pytest.warns(PytclDeprecationWarning, match="matrix_utils.ensure_symmetric"):
        out = ensure_symmetric([[1.0, 2.0 + 1e-11], [2.0, 1.0]])
    np.testing.assert_array_equal(out, out.T)


def test_validators_do_not_emit_the_deprecation():
    from pytcl.core.validation import (
        ensure_positive_definite,
        ensure_positive_semidefinite,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", PytclDeprecationWarning)
        ensure_positive_definite(np.eye(2))
        ensure_positive_semidefinite(np.eye(2))
