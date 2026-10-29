"""Invalid innovation covariances are rejected by every gating entry point.

v2.11.1 indexed S by the innovation's length without checking S's shape (a
length-7 innovation against a 6x6 S returned -5.45, a length-3 one 0.40) and
returned a finite number for an indefinite S. A squared distance is never
negative and is only defined for a positive-definite S. Valid inputs keep
their v2.11.1 bits.
"""

import numpy as np
import pytest

from pytcl.assignment_algorithms import gating
from tests.validation._v2111 import shipped_module

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


class TestNaNStillPropagates:
    @pytest.mark.parametrize("n", [2, 3, 5, 12])
    def test_nan_covariance(self, n):
        S = _spd(n, 9)
        S[0, 0] = np.nan
        with np.errstate(invalid="ignore"):
            try:
                result = gating.mahalanobis_distance(np.ones(n), S)
            except np.linalg.LinAlgError as exc:
                assert "positive definite" not in str(exc)
            else:
                assert np.isnan(result)
