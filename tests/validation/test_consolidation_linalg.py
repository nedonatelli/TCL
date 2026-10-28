"""Linear-algebra helpers consolidated to one body each (v2.12.0 Task 6.1).

Each delegation is pinned against the code shipped in v2.11.1: the old and
new paths must agree bit for bit on 200 seeded inputs.
"""

import importlib
import warnings

import numpy as np
import pytest

from pytcl.coordinate_systems.jacobians.jacobians import (
    numerical_jacobian as jac_numerical_jacobian,
)
from pytcl.core import array_utils
from pytcl.core.exceptions import PytclDeprecationWarning, ValidationError
from pytcl.core.validation import check_symmetric, ensure_symmetric
from pytcl.dynamic_estimation import (
    batch_estimation,
    batch_smoothers,
)
from pytcl.dynamic_estimation.kalman import extended, unscented
from pytcl.mathematical_functions.basic_matrix import special_matrices
from tests.validation._v2111 import shipped_module

information_filter = importlib.import_module(
    "pytcl.dynamic_estimation.information_filter"
)

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
    assert not hasattr(unscented, "_EIG_FLOOR_REL")
    assert not hasattr(unscented, "_EIG_NEGATIVE_REL_TOL")


def _jacobian_cases():
    rng = np.random.default_rng(2121)
    for _ in range(N_CASES):
        n = int(rng.integers(1, 6))
        m = int(rng.integers(1, 5))
        W = rng.standard_normal((m, n))
        x = rng.standard_normal(n)
        dx = float(10 ** rng.uniform(-9, -4))
        yield (lambda v, W=W: np.tanh(W @ v) + (W @ v) ** 2), x, dx


@pytest.fixture(scope="module")
def old_jacobians():
    return shipped_module("pytcl/coordinate_systems/jacobians/jacobians.py")


def _recording(f, seen):
    def wrapped(v):
        seen.append(np.shape(v))
        return f(v)

    return wrapped


def _shaped(x, kind):
    if kind == "col":
        return x.reshape(-1, 1)
    if kind == "int":
        return np.round(x * 3).astype(np.int64)
    return x


@pytest.mark.parametrize("kind", ["1d", "col", "int"])
def test_extended_numerical_jacobian_matches_v2111(old_extended, kind):
    for f, x, dx in _jacobian_cases():
        x = _shaped(x, kind)
        s_old, s_new = [], []
        old = old_extended.numerical_jacobian(_recording(f, s_old), x, dx)
        new = extended.numerical_jacobian(_recording(f, s_new), x, dx)
        np.testing.assert_array_equal(new, old)
        assert s_new == s_old


@pytest.mark.parametrize("kind", ["1d", "col", "int"])
def test_jacobians_numerical_jacobian_matches_v2111(old_jacobians, kind):
    for f, x, dx in _jacobian_cases():
        x = _shaped(x, kind)
        s_old, s_new = [], []
        g = (lambda v, f=f: f(np.ravel(v))) if kind == "col" else f
        try:
            old = old_jacobians.numerical_jacobian(_recording(g, s_old), x, dx)
        except ValueError:
            with pytest.raises(ValueError):
                jac_numerical_jacobian(_recording(g, s_new), x, dx)
            continue
        new = jac_numerical_jacobian(_recording(g, s_new), x, dx)
        np.testing.assert_array_equal(new, old)
        assert s_new == s_old


def test_jacobians_numerical_jacobian_passes_x_unflattened():
    seen = []
    jac_numerical_jacobian(_recording(lambda v: v.ravel() ** 2, seen), np.ones((2, 1)))
    assert set(seen) == {(2, 1)}


def test_sigma_point_fallback_message_names_matrix():
    P = np.diag([1.0, -1.0])
    for fn in (unscented.sigma_points_merwe, unscented.sigma_points_julier):
        with pytest.raises(
            np.linalg.LinAlgError, match="matrix is not positive semi-definite"
        ):
            fn(np.zeros(2), P)


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


@pytest.fixture(scope="module")
def old_information():
    return shipped_module("pytcl/dynamic_estimation/information_filter.py")


@pytest.fixture(scope="module")
def old_smoothers():
    return shipped_module("pytcl/dynamic_estimation/batch_smoothers.py")


_F2 = np.array([[1.0, 0.1], [0.0, 1.0]])
_H2 = np.array([[1.0, 0.0], [0.5, 1.0]])


def _as_shape(fun, shape):
    return lambda x: np.asarray(fun(np.ravel(x))).reshape(shape)


def _same_outcome(old_call, new_call):
    try:
        old = old_call()
    except (ValueError, np.linalg.LinAlgError) as exc:
        with pytest.raises(type(exc)):
            new_call()
        return None, None
    return old, new_call()


@pytest.mark.parametrize("shape", [(-1,), (-1, 1)])
def test_esrif_predict_matches_v2111_for_flat_and_column_f(old_information, shape):
    rng = np.random.default_rng(2125)
    f = _as_shape(lambda x: _F2 @ x + 0.01 * x**2, shape)
    for _ in range(20):
        x0 = rng.standard_normal(2)
        R0 = np.eye(2) + 0.1 * np.triu(rng.standard_normal((2, 2)))
        args = (R0 @ x0, R0, f, None, 0.1 * np.eye(2))
        old = old_information.esrif_predict(*args)
        new = information_filter.esrif_predict(*args)
        np.testing.assert_array_equal(new.y_sqrt, old.y_sqrt)
        np.testing.assert_array_equal(new.R, old.R)


@pytest.mark.parametrize("shape", [(-1,), (-1, 1)])
def test_esrif_update_matches_v2111_for_flat_and_column_h(old_information, shape):
    rng = np.random.default_rng(2126)
    h = _as_shape(lambda x: _H2[:1] @ x + 0.01 * x[:1] ** 2, shape)
    for _ in range(20):
        x0 = rng.standard_normal(2)
        R0 = np.eye(2) + 0.1 * np.triu(rng.standard_normal((2, 2)))
        z = _H2[:1] @ x0 + 0.05 * rng.standard_normal(1)
        args = (R0 @ x0, R0, z, 0.2 * np.eye(1), h)
        old = old_information.esrif_update(*args)
        new = information_filter.esrif_update(*args)
        np.testing.assert_array_equal(new.y_sqrt, old.y_sqrt)
        np.testing.assert_array_equal(new.R, old.R)


@pytest.mark.parametrize("shape", [(-1,), (-1, 1)])
def test_batch_estimation_default_jacobian_matches_v2111(old_batch, shape):
    rng = np.random.default_rng(2127)
    hs = [
        _as_shape(
            (lambda k: lambda x: np.array([x[0] + k * x[1], x[1] ** 2]))(k), shape
        )
        for k in range(4)
    ]
    for _ in range(10):
        truth = rng.standard_normal(2) + 3.0
        z = np.column_stack([np.ravel(hs[k](truth)) for k in range(4)])
        x0 = truth + 0.3 * rng.standard_normal(2)
        args = (x0, z, hs, 0.01 * np.eye(2))
        old, new = _same_outcome(
            lambda: old_batch.batch_ls_nonlin_meas_nonlin_dyn(*args),
            lambda: batch_estimation.batch_ls_nonlin_meas_nonlin_dyn(*args),
        )
        if old is not None:
            np.testing.assert_array_equal(new.x, old.x)
            np.testing.assert_array_equal(new.P, old.P)


@pytest.mark.parametrize("shape", [(-1,), (-1, 1)])
def test_ekalman_batch_smoother_default_jacobian_matches_v2111(old_smoothers, shape):
    rng = np.random.default_rng(2128)
    f = _as_shape(lambda x: _F2 @ x, shape)
    h = _as_shape(lambda x: x[:1] ** 1, shape)
    truth = [np.array([1.0, 0.5])]
    for _ in range(5):
        truth.append(_F2 @ truth[-1])
    z = np.array([[t[0] + 0.01 * rng.standard_normal() for t in truth]])
    args = (np.array([1.0, 0.4]), np.eye(2), z, h, None, f, None)
    args += (0.01 * np.eye(1), 0.001 * np.eye(2))
    old, new = _same_outcome(
        lambda: old_smoothers.ekalman_batch_smoother(*args),
        lambda: batch_smoothers.ekalman_batch_smoother(*args),
    )
    if old is not None:
        np.testing.assert_array_equal(new.x, old.x)


def _psd_cases():
    rng = np.random.default_rng(2129)
    for i in range(N_CASES):
        n = int(rng.integers(1, 6))
        A = rng.standard_normal((n, n))
        P = A @ A.T + (0.1 if i % 2 else 0.0) * np.eye(n)
        if i % 4 == 0 and n > 1:
            P = A[:, :-1] @ A[:, :-1].T
        yield n, P, rng


def test_ckf_predict_update_match_v2111_on_pd_and_singular_psd(old_unscented):
    for n, P, rng in _psd_cases():
        x = rng.standard_normal(n)
        Q = 0.1 * np.eye(n)
        R = 0.2 * np.eye(n)
        z = rng.standard_normal(n)
        f = lambda v: np.cos(v) + 0.5 * v  # noqa: E731
        old_p = old_unscented.ckf_predict(x, P, f, Q)
        new_p = unscented.ckf_predict(x, P, f, Q)
        np.testing.assert_array_equal(new_p.x, old_p.x)
        np.testing.assert_array_equal(new_p.P, old_p.P)
        old_u = old_unscented.ckf_update(x, P, z, f, R)
        new_u = unscented.ckf_update(x, P, z, f, R)
        np.testing.assert_array_equal(new_u.x, old_u.x)
        np.testing.assert_array_equal(new_u.P, old_u.P)


def test_ckf_not_psd_message_is_the_shared_one():
    P = np.diag([1.0, -1.0])
    with pytest.raises(
        np.linalg.LinAlgError, match="matrix is not positive semi-definite"
    ):
        unscented.ckf_predict(np.zeros(2), P, lambda v: v, np.eye(2))
    with pytest.raises(
        np.linalg.LinAlgError, match="matrix is not positive semi-definite"
    ):
        unscented.ckf_update(np.zeros(2), P, np.zeros(2), lambda v: v, np.eye(2))


def test_ckf_not_psd_still_raises_in_v2111(old_unscented):
    P = np.diag([1.0, -1.0])
    with pytest.raises(np.linalg.LinAlgError, match="covariance is not positive"):
        old_unscented.ckf_predict(np.zeros(2), P, lambda v: v, np.eye(2))
