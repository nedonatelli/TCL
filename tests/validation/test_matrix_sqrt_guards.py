"""Non-PSD guard tests for the eigh matrix-square-root fallback.

Six call sites fall back from ``np.linalg.cholesky`` to ``np.linalg.eigh``
on a near-singular covariance and used to clamp every negative eigenvalue
to an absolute ``1e-10`` with no check on how negative it was -- so
``diag(1, -1)`` (a variance of -1, an impossible covariance) came back as
an ordinary-looking ``diag(1, 1e-10)`` with no warning:

- ``pytcl.dynamic_estimation.kalman.matrix_utils.compute_matrix_sqrt``
- ``pytcl.dynamic_estimation.kalman.unscented``: ``sigma_points_merwe``,
  ``sigma_points_julier``, ``ckf_predict``, ``ckf_update`` (and
  transitively ``ukf_predict``, which calls ``sigma_points_merwe``)
- ``pytcl.dynamic_estimation.kalman.constrained.ConstrainedEKF``'s
  covariance projection

Each site now raises ``np.linalg.LinAlgError`` when the most negative
eigenvalue from the eigh fallback is more than a measured-and-margined
relative tolerance below zero, and continues to clamp silently below
that tolerance -- genuine roundoff must still pass through unchanged.
"""

import warnings

import numpy as np
import pytest

from pytcl.dynamic_estimation.kalman.constrained import (
    ConstrainedEKF,
    ConstraintFunction,
)
from pytcl.dynamic_estimation.kalman.matrix_utils import compute_matrix_sqrt
from pytcl.dynamic_estimation.kalman.unscented import (
    ckf_predict,
    ckf_update,
    sigma_points_julier,
    sigma_points_merwe,
    ukf_predict,
)


def _identity_fn(state: np.ndarray) -> np.ndarray:
    return state


def _invoke(fn, x: np.ndarray, P: np.ndarray):
    """Call each guarded entry point with equivalent, otherwise-benign args."""
    n = len(x)
    if fn is compute_matrix_sqrt:
        return fn(P)
    if fn in (sigma_points_merwe, sigma_points_julier):
        return fn(x, P)
    if fn in (ukf_predict, ckf_predict):
        return fn(x, P, _identity_fn, np.zeros((n, n)))
    if fn is ckf_update:
        return fn(x, P, np.zeros(n), _identity_fn, np.eye(n))
    raise AssertionError(f"no invocation registered for {fn!r}")


GUARDED_ENTRY_POINTS = [
    sigma_points_merwe,
    sigma_points_julier,
    ukf_predict,
    ckf_predict,
    ckf_update,
    compute_matrix_sqrt,
]


@pytest.mark.parametrize("fn", GUARDED_ENTRY_POINTS)
def test_non_psd_covariance_raises_rather_than_clamping(fn):
    """diag(1, -1) states a variance of -1 -- never roundoff, always invalid."""
    P = np.diag([1.0, -1.0])
    with pytest.raises(np.linalg.LinAlgError, match="not positive semi-definite"):
        _invoke(fn, np.zeros(2), P)


@pytest.mark.parametrize("fn", GUARDED_ENTRY_POINTS)
@pytest.mark.parametrize("eig", [-1e-18, -1e-14])
def test_roundoff_scale_negatives_are_still_clamped_silently(fn, eig):
    """The fix must not reject legitimate eigh roundoff."""
    P = np.diag([1.0, eig])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = _invoke(fn, np.zeros(2), P)
    if fn is compute_matrix_sqrt:
        S = result
    elif fn in (sigma_points_merwe, sigma_points_julier):
        S = result.points
    else:
        S = result.P
    assert np.all(np.isfinite(S))


def test_compute_matrix_sqrt_eigenvalue_named_in_message():
    """The message must name the actual offending eigenvalue, not a generic one."""
    P = np.diag([1.0, -1.0])
    with pytest.raises(np.linalg.LinAlgError, match=r"-1\.000000e\+00"):
        compute_matrix_sqrt(P)


def test_use_eigh_fallback_false_still_raises_the_original_way():
    """The new guard must not bypass use_eigh_fallback=False."""
    P = np.diag([1.0, -1.0])
    with pytest.raises(np.linalg.LinAlgError):
        compute_matrix_sqrt(P, use_eigh_fallback=False)


# Three state dimensions, with the equality constraint touching only
# dim 0: the projection's rank-1 update (P - P G^T (G P G^T)^-1 G P) only
# changes row/column 0, so dim 2's eigenvalue survives into the post-
# projection eigh call at its original scale relative to dim 1's "1" --
# reproducing the same diag(1, eig) shape the other five sites are
# tested with, rather than having the constraint's own projection erase
# the scale being tested.
def _build_constrained_ekf() -> ConstrainedEKF:
    ekf = ConstrainedEKF()
    ekf.add_constraint(
        ConstraintFunction(
            g=lambda x: np.array([x[0]]),
            G=lambda x: np.array([[1.0, 0.0, 0.0]]),
            constraint_type="equality",
        )
    )
    return ekf


_CONSTRAINED_X = np.array([5.0, 1.0, 1.0])


def test_constrained_ekf_projection_raises_on_non_psd_covariance():
    ekf = _build_constrained_ekf()
    P = np.diag([1.0, 1.0, -1.0])
    with pytest.raises(np.linalg.LinAlgError, match="not positive semi-definite"):
        ekf._project_onto_constraints(_CONSTRAINED_X, P)


@pytest.mark.parametrize("eig", [-1e-18, -1e-14])
def test_constrained_ekf_projection_still_clamps_roundoff_scale_negatives(eig):
    ekf = _build_constrained_ekf()
    P = np.diag([1.0, 1.0, eig])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _, P_proj = ekf._project_onto_constraints(_CONSTRAINED_X, P)
    assert np.all(np.isfinite(P_proj))
