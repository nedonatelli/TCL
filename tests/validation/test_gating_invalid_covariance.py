"""Invalid innovation covariances are rejected by every gating entry point.

v2.11.1 indexed S by the innovation's length without checking S's shape (a
length-7 innovation against a 6x6 S returned -5.45, a length-3 one 0.40) and
returned a finite number for an indefinite S. A squared distance is never
negative and is only defined for a positive-definite S. Valid inputs keep
their v2.11.1 bits.
"""

import importlib

import numpy as np
import pytest

from pytcl.assignment_algorithms import gating
from tests.validation._v2111 import shipped_module

jpda = importlib.import_module("pytcl.assignment_algorithms.jpda")

SIZES = [1, 2, 3, 5, 10, 12]
INVALID_SIZES = [2, 3, 5, 12]


@pytest.fixture(scope="module")
def old_gating():
    return shipped_module("pytcl/assignment_algorithms/gating.py")


def _spd(n, seed):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, n))
    return a @ a.T + n * np.eye(n)


def _indefinite(n, seed):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    eig = np.linspace(1.0, 3.0, n)
    eig[-1] = -0.5
    return (q * eig) @ q.T


def _negative_definite(n, seed):
    return -_spd(n, seed)


INVALID = [
    pytest.param(maker, n, id=f"{maker.__name__.strip('_')}-{n}")
    for maker in (_indefinite, _negative_definite)
    for n in INVALID_SIZES
]


class TestValidInputsKeepV2111Bits:
    @pytest.mark.parametrize("n", SIZES)
    def test_mahalanobis_distance(self, old_gating, n):
        S = _spd(n, 100 + n)
        for seed in range(5):
            nu = np.random.default_rng(seed).standard_normal(n)
            assert gating.mahalanobis_distance(
                nu, S
            ) == old_gating.mahalanobis_distance(nu, S)

    @pytest.mark.parametrize("n", SIZES)
    def test_ellipsoidal_gate(self, old_gating, n):
        S = _spd(n, 200 + n)
        nu = np.random.default_rng(n).standard_normal(n)
        for threshold in (0.5, 5.0, 50.0):
            assert gating.ellipsoidal_gate(
                nu, S, threshold
            ) == old_gating.ellipsoidal_gate(nu, S, threshold)

    @pytest.mark.parametrize("n", SIZES)
    @pytest.mark.parametrize("gate_type", ["ellipsoidal", "rectangular"])
    def test_gate_measurements(self, old_gating, n, gate_type):
        S = _spd(n, 300 + n)
        rng = np.random.default_rng(n)
        z_pred = rng.standard_normal(n)
        Z = z_pred + rng.standard_normal((20, n))
        threshold = 20.0 if gate_type == "ellipsoidal" else 3.0
        new_idx, new_d = gating.gate_measurements(z_pred, S, Z, threshold, gate_type)
        old_idx, old_d = old_gating.gate_measurements(
            z_pred, S, Z, threshold, gate_type
        )
        assert new_idx.tolist() == old_idx.tolist()
        assert new_d.tolist() == old_d.tolist()

    @pytest.mark.parametrize("n", SIZES)
    def test_mahalanobis_batch(self, old_gating, n):
        S_inv = np.linalg.inv(_spd(n, 400 + n))
        Z = np.random.default_rng(n).standard_normal((15, n))
        new = np.empty(15)
        old = np.empty(15)
        gating.mahalanobis_batch(Z, S_inv, new)
        old_gating.mahalanobis_batch(Z, S_inv, old)
        assert new.tolist() == old.tolist()


class TestDefectExistedInV2111:
    def test_length_7_against_6x6_raises(self):
        S = _spd(6, 1)
        nu = np.random.default_rng(101).standard_normal(7)
        with pytest.raises(ValueError, match=r"\(7,\).*\(6, 6\)"):
            gating.mahalanobis_distance(nu, S)

    def test_length_3_against_6x6_returned_a_number(self, old_gating):
        S = _spd(6, 1)
        nu = np.random.default_rng(101).standard_normal(7)[:3]
        assert old_gating.mahalanobis_distance(nu, S) == pytest.approx(
            0.6724271595640299, abs=0
        )
        with pytest.raises(ValueError, match=r"\(3,\).*\(6, 6\)"):
            gating.mahalanobis_distance(nu, S)

    def test_indefinite_returned_0_667(self, old_gating):
        nu = np.array([1.0, 1.0])
        S = np.array([[1.0, 2.0], [2.0, 1.0]])
        assert old_gating.mahalanobis_distance(nu, S) == pytest.approx(2 / 3, abs=0)
        with pytest.raises(np.linalg.LinAlgError, match="positive definite"):
            gating.mahalanobis_distance(nu, S)


class TestShapeMismatch:
    @pytest.mark.parametrize(
        "length", [7, 3, 5, 12], ids=["7-vs-6x6", "3-vs-6x6", "5-vs-6x6", "12-vs-6x6"]
    )
    def test_mahalanobis_distance_names_both_shapes(self, length):
        S = _spd(6, 1)
        with pytest.raises(ValueError, match=rf"\({length},\).*\(6, 6\)"):
            gating.mahalanobis_distance(np.ones(length), S)

    def test_two_dimensional_innovation(self):
        with pytest.raises(ValueError, match=r"\(3, 2\).*\(2, 2\)"):
            gating.mahalanobis_distance(np.ones((3, 2)), np.eye(2))

    def test_non_square_covariance(self):
        with pytest.raises(ValueError, match=r"\(2,\).*\(2, 3\)"):
            gating.mahalanobis_distance(np.ones(2), np.ones((2, 3)))

    def test_ellipsoidal_gate(self):
        with pytest.raises(ValueError, match=r"\(7,\).*\(6, 6\)"):
            gating.ellipsoidal_gate(np.ones(7), _spd(6, 1), 10.0)

    @pytest.mark.parametrize("gate_type", ["ellipsoidal", "rectangular"])
    def test_gate_measurements(self, gate_type):
        with pytest.raises(ValueError, match=r"\(7,\).*\(6, 6\)"):
            gating.gate_measurements(
                np.zeros(7), _spd(6, 1), np.ones((4, 7)), 10.0, gate_type
            )

    @pytest.mark.parametrize("gate_type", ["ellipsoidal", "rectangular"])
    def test_gate_measurements_column_broadcast(self, gate_type):
        with pytest.raises(ValueError, match=r"\(4, 1\).*\(3,\)"):
            gating.gate_measurements(
                np.zeros(3), np.eye(3), np.ones((4, 1)), 10.0, gate_type
            )

    def test_mahalanobis_batch(self):
        with pytest.raises(ValueError, match=r"\(4, 7\).*\(6, 6\)"):
            gating.mahalanobis_batch(np.ones((4, 7)), np.eye(6), np.empty(4))

    def test_mahalanobis_batch_output(self):
        with pytest.raises(ValueError, match=r"\(3,\)"):
            gating.mahalanobis_batch(np.ones((4, 2)), np.eye(2), np.empty(3))


class TestNotPositiveDefinite:
    @pytest.mark.parametrize("maker, n", INVALID)
    def test_mahalanobis_distance(self, maker, n):
        S = maker(n, 7)
        nu = np.random.default_rng(n).standard_normal(n)
        with pytest.raises(np.linalg.LinAlgError, match="positive definite"):
            gating.mahalanobis_distance(nu, S)

    def test_measured_indefinite_case(self):
        S = np.array([[1.0, 2.0], [2.0, 1.0]])
        with pytest.raises(np.linalg.LinAlgError, match="positive definite"):
            gating.mahalanobis_distance(np.array([1.0, 0.5]), S)

    def test_zero_leading_pivot(self):
        S = np.array([[0.0, 1.0], [1.0, 2.0]])
        with pytest.raises(np.linalg.LinAlgError):
            gating.mahalanobis_distance(np.array([1.0, 0.5]), S)

    def test_one_dimensional_negative(self):
        with pytest.raises(np.linalg.LinAlgError, match="positive definite"):
            gating.mahalanobis_distance(np.array([1.0]), np.array([[-2.0]]))

    @pytest.mark.parametrize("maker, n", INVALID)
    def test_ellipsoidal_gate(self, maker, n):
        with pytest.raises(np.linalg.LinAlgError, match="positive definite"):
            gating.ellipsoidal_gate(np.ones(n), maker(n, 7), 10.0)

    @pytest.mark.parametrize("gate_type", ["ellipsoidal", "rectangular"])
    @pytest.mark.parametrize("maker, n", INVALID)
    def test_gate_measurements(self, maker, n, gate_type):
        with pytest.raises(np.linalg.LinAlgError, match="positive definite"):
            gating.gate_measurements(
                np.zeros(n), maker(n, 7), np.ones((4, n)), 10.0, gate_type
            )

    @pytest.mark.parametrize("maker, n", INVALID)
    def test_mahalanobis_batch(self, maker, n):
        S_inv = np.linalg.inv(maker(n, 7))
        with pytest.raises(np.linalg.LinAlgError, match="positive definite"):
            gating.mahalanobis_batch(np.ones((4, n)), S_inv, np.empty(4))


class TestSingularUnchanged:
    @pytest.mark.parametrize("n", INVALID_SIZES)
    def test_mahalanobis_distance(self, old_gating, n):
        S = np.ones((n, n))
        nu = np.ones(n)
        with pytest.raises(np.linalg.LinAlgError):
            old_gating.mahalanobis_distance(nu, S)
        with pytest.raises(np.linalg.LinAlgError):
            gating.mahalanobis_distance(nu, S)

    @pytest.mark.parametrize("n", INVALID_SIZES)
    def test_message_is_the_lapack_one(self, old_gating, n):
        S = np.ones((n, n))
        nu = np.ones(n)
        with pytest.raises(np.linalg.LinAlgError) as old:
            old_gating.mahalanobis_distance(nu, S)
        with pytest.raises(np.linalg.LinAlgError) as new:
            gating.mahalanobis_distance(nu, S)
        assert str(new.value) == str(old.value)


def _asym(n):
    S = np.eye(n)
    S[0, 1] = 3.0 if n != 2 else 0.0
    if n == 2:
        S[1, 0] = 3.0
    return S


class TestNonSymmetric:
    @pytest.mark.parametrize(
        "S",
        [
            np.array([[1.0, 0.0], [3.0, 1.0]]),
            np.array([[1.0, 3.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            np.array([[1.0, 0.0, 0.0], [3.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            _asym(5),
            _asym(12),
        ],
        ids=["2x2", "3x3-upper", "3x3-lower", "5x5", "12x12"],
    )
    def test_negative_form_is_rejected(self, old_gating, S):
        n = S.shape[0]
        nu = np.zeros(n)
        nu[:2] = 1.0
        assert old_gating.mahalanobis_distance(nu, S) < 0
        with pytest.raises(np.linalg.LinAlgError, match="positive definite"):
            gating.mahalanobis_distance(nu, S)
        with pytest.raises(np.linalg.LinAlgError, match="positive definite"):
            gating.gate_measurements(np.zeros(n), S, np.ones((2, n)), 10.0)

    def test_literal_examples(self, old_gating):
        assert old_gating.mahalanobis_distance(
            np.array([1.0, 1.0]), np.array([[1.0, 0.0], [3.0, 1.0]])
        ) == pytest.approx(-1.0, abs=0)


class TestIllConditioned:
    @pytest.mark.parametrize("cond", [1e4, 1e8, 1e11, 1e12, 1e13, 1e14, 1e16])
    @pytest.mark.parametrize("n", [2, 3, 4, 6])
    def test_one_verdict_for_distance_and_gate(self, n, cond):
        rng = np.random.default_rng(n)
        q, _ = np.linalg.qr(rng.standard_normal((n, n)))
        S = (q * np.geomspace(1.0, 1.0 / cond, n)) @ q.T
        S = (S + S.T) / 2

        def raises(fn):
            try:
                fn()
            except np.linalg.LinAlgError as exc:
                assert "ill-conditioned" in str(exc) or "Singular" in str(exc)
                return True
            return False

        d = raises(lambda: gating.mahalanobis_distance(np.ones(n), S))
        g = raises(
            lambda: gating.gate_measurements(np.zeros(n), S, np.ones((2, n)), 1e30)
        )
        assert d == g


class TestScaleFree:
    @pytest.mark.parametrize(
        "n, scale",
        [(2, 1e-170), (2, 1e150), (3, 1e-110), (3, 1e-120), (3, 1e150), (5, 1e-120)],
    )
    def test_all_entry_points_agree_with_v2111(self, old_gating, n, scale):
        S = scale * np.eye(n)
        nu = np.ones(n)
        expected = old_gating.mahalanobis_distance(nu, S)
        assert gating.mahalanobis_distance(nu, S) == expected
        _, dists = gating.gate_measurements(np.zeros(n), S, nu[None, :], 1e300)
        assert dists.tolist() == [expected]
        out = np.empty(1)
        gating.mahalanobis_batch(nu[None, :], np.linalg.inv(S), out)
        old_out = np.empty(1)
        old_gating.mahalanobis_batch(nu[None, :], np.linalg.inv(S), old_out)
        assert out.tolist() == old_out.tolist()


class TestJpdaFallback:
    def test_batch_rejection_falls_back_to_per_pair(self, monkeypatch):
        tracks_x = [np.array([0.0, 0.0, 1.0, 1.0])]
        P = [np.diag([1.0, 1.0, 0.5, 0.5])]
        H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        R = 0.1 * np.eye(2)
        Z = np.array([[0.2, 0.1], [1.0, -0.5], [4.0, 4.0]])
        expected = jpda.compute_likelihood_matrix(tracks_x, P, Z, H, R, 0.9, 9.21)

        def reject(*args):
            raise np.linalg.LinAlgError("rejected")

        monkeypatch.setattr(jpda, "mahalanobis_batch", reject)
        got = jpda.compute_likelihood_matrix(tracks_x, P, Z, H, R, 0.9, 9.21)
        assert got[1].tolist() == expected[1].tolist()
        # per-pair solve and batched inverse round differently, so not bitwise
        assert got[0] == pytest.approx(expected[0], rel=1e-12, abs=0)


class TestNaNStillPropagates:
    @pytest.mark.parametrize("n", [1, 2, 3, 5, 12])
    def test_nan_covariance_returns_nan_like_v2111(self, old_gating, n):
        S = _spd(n, 9)
        S[0, 0] = np.nan
        with np.errstate(all="ignore"):
            old = old_gating.mahalanobis_distance(np.ones(n), S)
            new = gating.mahalanobis_distance(np.ones(n), S)
        assert np.isnan(old)
        assert np.isnan(new)
