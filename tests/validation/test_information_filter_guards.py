"""Guard test for the information filter's singular-F prediction defect.

``information_filter``'s singular-Y branch (unknown/partially unknown
initial state) predicts via ``M = F^-T Y F^-1``, which requires F
invertible. Before this fix, a singular F raised ``LinAlgError`` internally
and the exception was swallowed with ``pass``, leaving ``y``/``Y``
unchanged -- the filter silently skipped prediction and reported the prior
as if it had propagated, with zero warning.

Reproduced exactly as measured: with ``F = [[1, 1], [0, 0]]``, an unknown
initial state (``Y0 = 0``), and three position-only measurements, ``Y``
went ``diag(1, 0) -> diag(2, 0) -> diag(3, 0)`` -- indistinguishable from
*skipping prediction entirely*, not from running with ``F = I``: the
same scenario with ``F = I`` (and the same ``Q``) actually damps to
``1.0, 1.909, 2.603`` (measured), since the Q update still runs there.

The brief's own Step 1 sketch calls ``information_filter_predict`` directly
with a *non-singular* ``Y = eye(2)``; that call takes a different branch
(``pytcl.dynamic_estimation.kalman.linear.information_filter_predict``,
used only when Y is already full rank) which never inverts F at all and,
for that combination of F/Y/Q, returns a legitimate, non-garbage answer
today (measured: no exception, ``Y_pred = diag(1/3, 1)``) -- there is
nothing to fix on that path, and forcing it to raise would reject a
scenario that currently produces a correct result. The actual defect only
manifests through ``information_filter``'s singular-Y branch, so the tests
below drive it through that function instead, using the brief's own F and
measurement scenario.
"""

from pathlib import Path

import numpy as np
import pytest

from pytcl.dynamic_estimation.information_filter import (
    information_filter,
    srif_update,
)
from pytcl.dynamic_estimation.kalman.linear import information_filter_predict

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "matlab"


class TestInformationFilterSingularF:
    """information_filter must not silently skip prediction on a singular F."""

    def test_singular_f_with_singular_y_raises(self):
        """F=[[1,1],[0,0]] starting from an unknown state must raise, not
        silently return the prior as if it had been predicted."""
        n = 2
        y0 = np.zeros(n)
        Y0 = np.zeros((n, n))
        F = np.array([[1.0, 1.0], [0.0, 0.0]])
        Q = np.eye(n) * 0.1
        H = np.array([[1.0, 0.0]])
        R = np.array([[1.0]])
        measurements = [np.array([1.0]), np.array([2.0]), np.array([3.0])]

        with pytest.raises(np.linalg.LinAlgError, match="singular"):
            information_filter(y0, Y0, measurements, F, Q, H, R)

    def test_regular_f_with_singular_y_still_propagates_information(self):
        """The defect's fix must not disturb the already-correct path: a
        regular F must still let velocity information appear even when Y0
        starts singular (unknown state)."""
        y0 = np.zeros(2)
        Y0 = np.diag([1.0, 0.0])
        F = np.array([[1.0, 1.0], [0.0, 1.0]])
        Q = np.zeros((2, 2))
        H = np.array([[1.0, 0.0]])
        R = np.array([[1.0]])

        result = information_filter(y0, Y0, [None], F, Q, H, R)

        assert result.Y_filt[0][1, 1] > 0.0, "velocity information was never created"


class TestSRIFPredictSingularCovariance:
    """srif_predict's SVD fallback must not return nan/inf as an ordinary
    result when the predicted covariance is genuinely singular.

    Before this fix, a singular ``P_pred`` reaching the SVD fallback
    (Cholesky already failed) divided by a zero singular value with only
    numpy's own generic ``RuntimeWarning: divide by zero`` as a signal,
    and returned ``r_pred``/``R_pred`` containing ``nan``/``inf`` as an
    ordinary result. ``np.linalg.qr`` tolerates ``nan``/``inf`` without
    raising, so ``srif_update`` ran to completion on that garbage;
    ``srif_filter`` then failed several lines later, in its own
    state-form conversion (``np.linalg.matrix_rank(Y)``, an SVD -- not a
    QR call), with the misleading ``LinAlgError: SVD did not converge``
    -- misleading because the SVD in ``srif_predict`` itself converged
    fine; it was the *result* that was degenerate.
    """

    def _singular_case(self):
        n = 2
        P0 = np.eye(n)
        R0 = np.linalg.cholesky(np.linalg.inv(P0)).T
        x0 = np.array([1.0, 0.5])
        r0 = R0 @ x0
        F = np.array([[1.0, 1.0], [0.0, 0.0]])
        Q = np.zeros((2, 2))
        return r0, R0, F, Q

    def test_singular_predicted_covariance_raises(self):
        from pytcl.dynamic_estimation.information_filter import srif_predict

        r0, R0, F, Q = self._singular_case()

        with pytest.raises(np.linalg.LinAlgError, match="singular"):
            srif_predict(r0, R0, F, Q)

    def test_srif_filter_raises_at_the_predict_step_not_later(self):
        """Confirms the fix moves the failure to its true cause instead of
        letting srif_filter fail downstream with a misleading QR error."""
        from pytcl.dynamic_estimation.information_filter import srif_filter

        r0, R0, F, Q = self._singular_case()
        H = np.array([[1.0, 0.0]])
        R_meas = np.array([[1.0]])

        with pytest.raises(np.linalg.LinAlgError, match="srif_predict"):
            srif_filter(r0, R0, [np.array([1.0])], F, Q, H, R_meas)


class TestSRIFPredictNearSingularCovariance:
    def test_near_singular_predicted_covariance_raises(self):
        """Defect test: F = [[1, 1], [0, 1e-18]] gave R_pred entries ~1e18,
        finite, with no warning; the exact-zero test never fired."""
        from pytcl.dynamic_estimation.information_filter import srif_predict

        F = np.array([[1.0, 1.0], [0.0, 1e-18]])
        with pytest.raises(np.linalg.LinAlgError, match="singular"):
            srif_predict(np.zeros(2), np.eye(2), F, np.zeros((2, 2)))

    def test_ill_conditioned_but_representable_covariance_still_works(self):
        from pytcl.dynamic_estimation.information_filter import srif_predict

        F = np.diag([1.0, 1e-5])  # P_pred condition number 1e10
        _, R_pred = srif_predict(np.zeros(2), np.eye(2), F, np.zeros((2, 2)))
        np.testing.assert_allclose(R_pred.T @ R_pred, np.diag([1.0, 1e10]), rtol=1e-6)


# Inputs mirrored from scripts/matlab_capture/capture_info_filter_pred.m.
_F2 = np.array([[1.0, 1.0], [0.0, 1.0]])
_Y2 = np.diag([1.0, 0.0])
_y2 = np.array([0.5, 0.0])
_F4 = np.array(
    [
        [1.0, 0.0, 0.5, 0.0],
        [0.0, 1.0, 0.0, 0.5],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
_A4 = np.array([[1.0, 0.2], [0.3, 1.5], [0.1, 0.0], [0.0, 0.4]])
_Y4 = _A4 @ _A4.T
_y4 = _Y4 @ np.array([1.0, -2.0, 0.5, 0.3])
_g4 = np.array([0.125, 0.125, 0.5, 0.5])

_INFO_PRED_CASES = {
    "q0": (_y2, _Y2, _F2, np.zeros((2, 2))),
    "q1e-6": (_y2, _Y2, _F2, 1e-6 * np.eye(2)),
    "q1e-12": (_y2, _Y2, _F2, 1e-12 * np.eye(2)),
    "q1e-15": (_y2, _Y2, _F2, 1e-15 * np.eye(2)),
    "q1e-2": (_y2, _Y2, _F2, 1e-2 * np.eye(2)),
    "qdiag": (_y2, _Y2, _F2, np.diag([0.0, 1e-2])),
    "y0": (np.zeros(2), np.zeros((2, 2)), _F2, 1e-2 * np.eye(2)),
    "4state": (_y4, _Y4, _F4, 0.1 * np.outer(_g4, _g4)),
}


class TestSingularYPredictionMatchesMatlab:
    """The singular-Y prediction is infoFilterDiscPred.m:47-58 (MATLAB TCL
    a9acd8f). v2.11.1 inverted ``Q + 1e-9 (tr(Q)/n + 1) I`` instead, which
    put a uniform 2.0e-9 relative error on every case here, exact ``Q = 0``
    included.
    """

    @pytest.mark.parametrize("name", list(_INFO_PRED_CASES))
    def test_one_prediction_matches_matlab(self, name):
        path = FIXTURE_DIR / f"info_pred_{name}.csv"
        assert path.exists(), f"committed MATLAB fixture missing: {path.name}"
        ref = np.loadtxt(path, delimiter=",", ndmin=2)
        y, Y, F, Q = _INFO_PRED_CASES[name]
        H = np.zeros((1, len(y)))
        res = information_filter(y, Y, [None], F, Q, H, np.eye(1))
        np.testing.assert_allclose(res.y_filt[0], ref[0], rtol=1e-12, atol=0)
        np.testing.assert_allclose(res.Y_filt[0], ref[1:], rtol=1e-12, atol=0)

    @pytest.mark.parametrize("seed", range(5))
    def test_full_rank_y_agrees_with_covariance_route(self, seed):
        """PROPERTY: for full-rank Y the MATLAB form and
        ``information_filter_predict`` (inv(Y), F P F' + Q, inv) are the
        same prediction."""
        from pytcl.dynamic_estimation.information_filter import (
            _information_predict_singular,
        )

        rng = np.random.default_rng(seed)
        n = 4
        A = rng.standard_normal((n, n))
        Y = A @ A.T + n * np.eye(n)
        y = rng.standard_normal(n)
        F = np.eye(n) + 0.3 * rng.standard_normal((n, n))
        B = rng.standard_normal((n, n))
        Q = 0.1 * B @ B.T
        y_m, Y_m = _information_predict_singular(y, Y, F, Q)
        y_c, Y_c = information_filter_predict(y, Y, F, Q)
        np.testing.assert_allclose(y_m, y_c, rtol=1e-12)
        np.testing.assert_allclose(Y_m, Y_c, rtol=1e-12)


def _V2111_srif_update(*args):
    from tests.validation._v2111 import shipped_module

    return shipped_module("pytcl/dynamic_estimation/information_filter.py").srif_update(
        *args
    )


def _case(R_meas):
    """Zero prior, so the posterior information is the measurement's own
    ``R_meas^-1`` and scales exactly with ``R_meas``."""
    return np.zeros(2), np.zeros((2, 2)), np.array([0.3, -0.7]), np.eye(2), R_meas


def _pd_case():
    R0 = np.linalg.cholesky(np.linalg.inv(2.0 * np.eye(2))).T
    return (
        R0 @ np.array([1.0, 0.5]),
        R0,
        np.array([1.2, 0.4]),
        np.array([[1.0, 0.0], [0.5, 1.0]]),
        np.diag([0.5, 0.3]),
    )


def _info(result):
    _, R_upd = result
    return R_upd.T @ R_upd


def _bitwise_equal(a, b):
    return all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))


class TestSRIFUpdateMeasurementCovariance:
    """srif_update's non-PD fallback added an absolute ``1e-10 I`` to
    ``R_meas``: a zero variance became 1e10 information without a word,
    ``-1e-12`` was accepted as ``9.9e-11``, and a covariance below ``1e-10``
    in scale was swamped by the jitter. The fallback now floors eigenvalues
    relative to the largest (``matrix_utils._EIG_FLOOR_REL``) and rejects a
    zero or negative variance.
    """

    @pytest.mark.parametrize("r_meas", [0.0, -1e-12, -1e-8])
    def test_zero_or_negative_measurement_variance_raises(self, r_meas):
        with pytest.raises(np.linalg.LinAlgError, match="variance"):
            srif_update(
                np.zeros(1), np.eye(1), np.zeros(1), np.eye(1), np.array([[r_meas]])
            )

    def test_semidefinite_measurement_covariance_is_floored_relatively(self):
        R_meas = np.diag([1e-14, 0.0])
        a = srif_update(*_case(R_meas))
        b = srif_update(*_case(R_meas * 1e6))
        np.testing.assert_allclose(_info(a), _info(b) * 1e6, rtol=1e-9, atol=0)

    @pytest.mark.parametrize("scale", [1.0, 3.0, 1e-6, 0.1, 2.0, 7.0, 1e-14])
    def test_correlated_singular_covariance_is_floored_along_its_null_space(
        self, scale
    ):
        """REFERENCE: ``R_meas = s [[1, 1], [1, 1]]`` has eigenvalue ``2 s``
        along ``u = (1, 1)/sqrt(2)`` and ``0`` along ``w = (1, -1)/sqrt(2)``.
        The zero is floored at ``1e-10 * 2 s``, so with a zero prior and
        ``H = I`` the posterior information is ``V diag(1/floored) V'`` =
        ``u u' / (2 s) + w w' / (2e-10 s)``. v2.11.1's ``+ 1e-10 I`` jitter
        gave ``1e10`` along ``w`` whatever ``s`` was. At s = 1, 3 and 1e-6
        Cholesky fails; at 0.1, 2, 7 and 1e-14 it succeeds on a roundoff
        pivot, and v2.11.1 and v2.12.0 before the roundoff gate returned
        its unfloored information (largest eigenvalue 3.95e18, 5.64e15,
        1.19e15 and 2.24e30)."""
        u = np.array([1.0, 1.0]) / np.sqrt(2.0)
        w = np.array([1.0, -1.0]) / np.sqrt(2.0)
        floored = np.array([2.0 * scale, 1e-10 * 2.0 * scale])
        V = np.column_stack([u, w])
        want = V @ np.diag(1.0 / floored) @ V.T
        got = _info(srif_update(*_case(scale * np.ones((2, 2)))))
        np.testing.assert_allclose(got, want, rtol=1e-9, atol=0)
        assert u @ got @ u == pytest.approx(1.0 / floored[0], rel=1e-5, abs=0)
        bound = 1.0 / (1e-10 * floored[0])
        assert np.linalg.eigvalsh(got).max() <= bound * (1 + 1e-9)

    @pytest.mark.parametrize("seed", range(40))
    def test_random_singular_covariance_is_floored(self, seed):
        """PROPERTY: a rank-deficient R_meas never yields more information
        than the floor allows, ``1 / (1e-10 * max eig)``, whether or not
        Cholesky succeeds on it."""
        rng = np.random.default_rng(300 + seed)
        m = int(rng.integers(2, 7))
        B = rng.standard_normal((m, int(rng.integers(1, m))))
        R_meas = 10.0 ** rng.uniform(-14, 6) * (B @ B.T)
        _, R_upd = srif_update(
            np.zeros(m), np.zeros((m, m)), np.zeros(m), np.eye(m), R_meas
        )
        bound = 1.0 / (1e-10 * np.linalg.eigvalsh(R_meas)[-1])
        assert np.linalg.eigvalsh(R_upd.T @ R_upd)[-1] <= bound * (1 + 1e-8)

    def test_diagonal_positive_definite_path_is_unchanged(self):
        """Bit-identical to v2.11.1 for diagonal R_meas, where inv(L) and
        inv(L).T coincide. Non-diagonal R_meas changed on purpose; see
        TestSRIFUpdateCorrelatedMeasurement."""
        assert _bitwise_equal(srif_update(*_pd_case()), _V2111_srif_update(*_pd_case()))

    @pytest.mark.parametrize(
        "R_meas",
        [
            np.diag([1e4, 1e-8]),
            np.diag([1.0, 5e-11]),
            np.diag([1.0, 1e-12]),
            np.diag([3.0, 3e-14]),
            np.diag([1.0, 1e-20]),
        ],
        ids=["1e4,1e-8", "1,5e-11", "cond1e12", "cond1e14", "cond1e20"],
    )
    def test_ill_conditioned_diagonal_is_bit_identical(self, R_meas):
        """A positive-definite R_meas keeps its information however
        ill-conditioned. The reverted first fix gated Cholesky on a 1e-10
        eigenvalue ratio, and ``diag(1e4, 1e-8)`` lost 100x of its
        second-component information."""
        args = (*_pd_case()[:4], R_meas)
        assert _bitwise_equal(srif_update(*args), _V2111_srif_update(*args))

    @pytest.mark.parametrize("seed", range(20))
    def test_random_diagonal_positive_definite_is_bit_identical(self, seed):
        rng = np.random.default_rng(200 + seed)
        n, m = 4, 3
        A = rng.standard_normal((n, n))
        R0 = np.triu(A) + n * np.eye(n)
        args = (
            rng.standard_normal(n),
            R0,
            rng.standard_normal(m),
            rng.standard_normal((m, n)),
            np.diag(10.0 ** rng.uniform(-10, 10, m)),
        )
        assert _bitwise_equal(srif_update(*args), _V2111_srif_update(*args))


def _rotated(cond, theta=0.3):
    c, s = np.cos(theta), np.sin(theta)
    V = np.array([[c, -s], [s, c]])
    return V @ np.diag([1.0, 1.0 / cond]) @ V.T


def _v2111_on_prewhitened(r, R, z, H, R_meas):
    """v2.11.1's QR stage fed the MATLAB whitener ``inv(L)``
    (sqrtInfoFilterUpdate.m:67-68) and ``R_meas = I``, which v2.11.1 passes
    through exactly. v2.11.1 itself whitened a correlated R_meas with
    ``inv(L).T`` (corrected in v2.12.0), so it is not the reference there."""
    W = np.linalg.inv(np.linalg.cholesky(R_meas))
    return _V2111_srif_update(r, R, W @ z, W @ H, np.eye(len(R_meas)))


class TestSRIFUpdateCorrelatedPositiveDefinite:
    """A correlated positive-definite R_meas stays on the Cholesky path,
    bitwise, up to a condition number near ``1 / (m * eps)``; only a
    smallest eigenvalue at roundoff scale routes it to the floor."""

    @pytest.mark.parametrize("cond", [1e8, 1e12, 1e14, 1e15])
    def test_rotated_ill_conditioned_is_bit_identical(self, cond):
        args = (*_pd_case()[:4], _rotated(cond))
        assert _bitwise_equal(srif_update(*args), _v2111_on_prewhitened(*args))

    @pytest.mark.parametrize("seed", range(20))
    def test_random_positive_definite_is_bit_identical(self, seed):
        rng = np.random.default_rng(400 + seed)
        n, m = 4, int(rng.integers(2, 7))
        A = rng.standard_normal((n, n))
        Q, _ = np.linalg.qr(rng.standard_normal((m, m)))
        R_meas = (Q * 10.0 ** rng.uniform(-14, 0, m)) @ Q.T
        R_meas = (R_meas + R_meas.T) / 2
        args = (
            rng.standard_normal(n),
            np.triu(A) + n * np.eye(n),
            rng.standard_normal(m),
            rng.standard_normal((m, n)),
            R_meas,
        )
        assert _bitwise_equal(srif_update(*args), _v2111_on_prewhitened(*args))


# Inputs mirrored from scripts/matlab_capture/capture_srif_update.m.
_SRIF_CASES = {
    "2state": (
        2.0 * np.eye(2),
        np.array([1.0, 0.5]),
        np.array([1.2, 0.4]),
        np.array([[2.0, 1.0], [1.0, 1.0]]),
        np.eye(2),
    ),
    "3state": (
        np.array([[1.0, 0.2, 0.1], [0.2, 2.0, 0.3], [0.1, 0.3, 0.5]]),
        np.array([1.0, -0.4, 0.3]),
        np.array([1.6, -0.7]),
        np.array([[0.5, 0.3], [0.3, 0.4]]),
        np.array([[1.0, 0.0, 0.5], [0.0, 1.0, -0.2]]),
    ),
}


def _srif_prior(P0, x0):
    R0 = np.linalg.cholesky(np.linalg.inv(P0)).T
    return R0 @ x0, R0


class TestSRIFUpdateCorrelatedMeasurement:
    """srif_update whitened with ``inv(L).T`` for ``R_meas = L L'``, so the
    measurement information was ``inv(L' L)``, not ``inv(R_meas)``, for any
    non-diagonal ``R_meas``. MATLAB's sqrtInfoFilterUpdate.m:67-68 whitens
    with ``SR\\H``, i.e. ``inv(L)``.
    """

    @pytest.mark.parametrize("name", list(_SRIF_CASES))
    def test_matches_matlab(self, name):
        path = FIXTURE_DIR / f"srif_update_{name}.csv"
        assert path.exists(), f"committed MATLAB fixture missing: {path.name}"
        ref = np.loadtxt(path, delimiter=",", ndmin=2)
        P0, x0, z, R_meas, H = _SRIF_CASES[name]
        r_upd, R_upd = srif_update(*_srif_prior(P0, x0), z, H, R_meas)
        np.testing.assert_allclose(
            np.linalg.solve(R_upd, r_upd), ref[0], rtol=1e-12, atol=0
        )
        np.testing.assert_allclose(R_upd.T @ R_upd, ref[1:], rtol=1e-12, atol=0)

    @pytest.mark.parametrize("seed", range(5))
    def test_posterior_equals_information_form_update(self, seed):
        """REFERENCE: Y = Y0 + H' R^-1 H, y = y0 + H' R^-1 z, formed with
        ``np.linalg.inv(R_meas)``."""
        rng = np.random.default_rng(seed)
        n, m = 3, 2
        A = rng.standard_normal((n, n))
        P0 = A @ A.T + np.eye(n)
        x0 = rng.standard_normal(n)
        H = rng.standard_normal((m, n))
        z = rng.standard_normal(m)
        B = rng.standard_normal((m, m))
        R_meas = B @ B.T + 0.1 * np.eye(m)
        r_upd, R_upd = srif_update(*_srif_prior(P0, x0), z, H, R_meas)
        R_inv = np.linalg.inv(R_meas)
        Y0 = np.linalg.inv(P0)
        Y = Y0 + H.T @ R_inv @ H
        y = Y0 @ x0 + H.T @ R_inv @ z
        np.testing.assert_allclose(R_upd.T @ R_upd, Y, rtol=1e-12)
        np.testing.assert_allclose(R_upd.T @ r_upd, y, rtol=1e-12)

    @pytest.mark.parametrize("seed", range(5))
    def test_whitener_carries_inverse_measurement_covariance(self, seed):
        """With a zero prior and H = I the posterior information is the
        whitener's W.T @ W, which must be R_meas^-1."""
        rng = np.random.default_rng(100 + seed)
        m = 4
        B = rng.standard_normal((m, m))
        R_meas = B @ B.T + 0.1 * np.eye(m)
        _, R_upd = srif_update(
            np.zeros(m), np.zeros((m, m)), np.zeros(m), np.eye(m), R_meas
        )
        np.testing.assert_allclose(R_upd.T @ R_upd, np.linalg.inv(R_meas), rtol=1e-12)
