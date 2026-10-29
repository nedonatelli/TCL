"""
Tests for Debye functions.

Oracles (none of them call pytcl):

- D_n(x): mpmath Gauss-Legendre quadrature at 30 digits of the defining
  integral ``n / x^n * int_0^x t^n / (e^t - 1) dt``, split into subintervals
  because the integrand peaks near t = n. For x < 1 the substitution t = x*s
  is used instead, since ``t^n`` varies by many orders of magnitude over
  [0, x] and a plain subdivision loses digits there. Gauss-Legendre rather
  than tanh-sinh: tanh-sinh measured 7e-9 off a 40-term Bernoulli series for
  n = 10, x = 1e-3.
- Heat capacity: the direct phonon-mode integral
  ``3/x^3 * int_0^x t^4 e^t / (e^t - 1)^2 dt``, not the by-parts form
  ``4 D_3 - 3x/(e^x - 1)`` that the library uses.
- Entropy: the direct mode integral
  ``3/x^3 * int_0^x t^2 [t/(e^t - 1) - ln(1 - e^-t)] dt``.
- Closed-form limits: Debye T^3 law ``C/(3Nk) = (4 pi^4 / 5) (T/theta)^3`` and
  its entropy ``S/(3Nk) = (4 pi^4 / 15) (T/theta)^3`` at low T; the
  high-temperature expansion ``C/(3Nk) = 1 - x^2/20 + x^4/560 - x^6/18144``.
"""

import functools

import mpmath as mp
import numpy as np
import pytest

from pytcl.mathematical_functions.special_functions.debye import (
    debye,
    debye_1,
    debye_2,
    debye_3,
    debye_4,
    debye_entropy,
    debye_heat_capacity,
)


def _grid(x: mp.mpf, n: int) -> list:
    """Quadrature breakpoints in t covering [0, x] and the peak near t = n."""
    top = min(x, 8 * n + 100)
    m = max(16, int(top / 4) + 1)
    pts = [top * mp.mpf(i) / m for i in range(m + 1)]
    if x > top:
        pts.append(x)
    return pts


@functools.lru_cache(maxsize=None)
def _debye_oracle(n: int, x: float) -> mp.mpf:
    """D_n(x) at 30 digits from the defining integral."""
    with mp.workdps(30):
        xx = mp.mpf(x)
        pts = _grid(xx, n)
        if x < 1:
            integral = mp.quad(
                lambda u: u**n * xx / mp.expm1(xx * u),
                [p / xx for p in pts],
                method="gauss-legendre",
            )
            return n * integral
        integral = mp.quad(lambda t: t**n / mp.expm1(t), pts, method="gauss-legendre")
        return n * integral / xx**n


@functools.lru_cache(maxsize=None)
def _heat_capacity_oracle(x: float) -> mp.mpf:
    """C_V / (3 N k_B) at Theta_D / T = x from the phonon-mode integral."""
    with mp.workdps(30):
        xx = mp.mpf(x)

        def mode(t):
            return t**4 * mp.exp(t) / mp.expm1(t) ** 2

        return 3 * mp.quad(mode, _grid(xx, 3), method="gauss-legendre") / xx**3


@functools.lru_cache(maxsize=None)
def _entropy_oracle(x: float) -> mp.mpf:
    """S / (3 N k_B) at Theta_D / T = x from the phonon-mode integral."""
    with mp.workdps(30):
        xx = mp.mpf(x)

        def mode(t):
            return t**2 * (t / mp.expm1(t) - mp.log(-mp.expm1(-t)))

        return 3 * mp.quad(mode, _grid(xx, 3), method="gauss-legendre") / xx**3


def _x_switch(n: int) -> float:
    """Large-x regime boundary of the implementation: n + 2 sqrt(n) + 5."""
    return n + 2.0 * np.sqrt(n) + 5.0


_ORDERS = (1, 2, 3, 4, 6, 10, 12, 20, 30, 50)
_XS = (1e-3, 0.5, 1.0, 2.0, 5.0, 17.0, 40.0, 150.0, 700.0)
_BOUNDARY_ORDERS = (1, 4, 6, 8, 9, 10)
_BOUNDARY_XS = (0.99, 1.0, 1.01, 1.9, 1.99, 2.0, 10.0)
_SWITCH_POINTS = [(n, _x_switch(n) + d) for n in (3, 12, 30) for d in (-0.01, 0.01)]


def _debye_value(n, x):
    return float(debye(n, x)[0])


# =============================================================================
# D_n(x) against the quadrature oracle
# =============================================================================


class TestDebyeValues:
    """D_n(x) against 30-digit quadrature, rel 1e-13."""

    @pytest.mark.parametrize("n", _ORDERS)
    @pytest.mark.parametrize("x", _XS)
    def test_grid(self, n, x):
        assert _debye_value(n, x) == pytest.approx(
            float(_debye_oracle(n, x)), rel=1e-13, abs=0
        )

    @pytest.mark.parametrize("n", _BOUNDARY_ORDERS)
    @pytest.mark.parametrize("x", _BOUNDARY_XS)
    def test_small_x_branch_boundary(self, n, x):
        """x in [0.99, 2.0] straddles the former x0 = 1 and x0 = 2 switches."""
        assert _debye_value(n, x) == pytest.approx(
            float(_debye_oracle(n, x)), rel=1e-13, abs=0
        )

    @pytest.mark.parametrize(("n", "x"), _SWITCH_POINTS)
    def test_large_x_regime_boundary(self, n, x):
        assert _debye_value(n, x) == pytest.approx(
            float(_debye_oracle(n, x)), rel=1e-13, abs=0
        )

    @pytest.mark.parametrize("n", (1, 3, 12, 50))
    def test_array_input_matches_oracle_elementwise(self, n):
        x = np.array([0.5, 2.0, 17.0, 150.0])
        got = debye(n, x)
        expected = [float(_debye_oracle(n, float(v))) for v in x]
        assert got.shape == (4,)
        np.testing.assert_allclose(got, expected, rtol=1e-13, atol=0)

    @pytest.mark.parametrize("n", range(1, 7))
    def test_value_at_zero_is_one(self, n):
        np.testing.assert_array_equal(debye(n, 0.0), [1.0])

    @pytest.mark.parametrize("n", (1, 3, 12))
    def test_value_at_infinity_is_zero(self, n):
        np.testing.assert_array_equal(debye(n, np.inf), [0.0])

    @pytest.mark.parametrize(
        ("fn", "n"), [(debye_1, 1), (debye_2, 2), (debye_3, 3), (debye_4, 4)]
    )
    def test_convenience_functions_match_oracle(self, fn, n):
        xs = [0.5, 1.5, 5.0, 40.0]
        np.testing.assert_allclose(
            fn(xs),
            [float(_debye_oracle(n, v)) for v in xs],
            rtol=1e-13,
            atol=0,
        )

    def test_list_input(self):
        np.testing.assert_allclose(
            debye_1([0.5, 1.0, 2.0]),
            [float(_debye_oracle(1, v)) for v in (0.5, 1.0, 2.0)],
            rtol=1e-13,
            atol=0,
        )

    def test_low_x_expansion_leading_terms(self):
        """D_n(x) = 1 - n x / (2 (n + 1)) + n x^2 / (12 (n + 2)) + O(x^4)."""
        x = 1e-4
        for n in range(1, 5):
            expected = 1 - n * x / (2 * (n + 1)) + n * x**2 / (12 * (n + 2))
            assert _debye_value(n, x) == pytest.approx(expected, rel=1e-14, abs=0)

    def test_large_x_asymptote(self):
        """D_n(x) -> n * n! * zeta(n + 1) / x^n; the correction is O(e^-x)."""
        x = 150.0
        # zeta(4) = pi^4 / 90
        expected = 3 * 6 * (np.pi**4 / 90) / x**3
        assert _debye_value(3, x) == pytest.approx(expected, rel=1e-13, abs=0)


class TestDebyeErrors:
    def test_invalid_order(self):
        with pytest.raises(ValueError, match="Order n must be >= 1"):
            debye(0, 1.0)

    def test_non_integer_order(self):
        with pytest.raises(ValueError, match="Order n must be >= 1"):
            debye(1.5, 1.0)

    @pytest.mark.parametrize("x", (-1e-9, -1.0, -50.0))
    def test_negative_x_rejected(self, x):
        with pytest.raises(ValueError, match="x must be >= 0"):
            debye(3, x)

    def test_negative_element_in_array_rejected(self):
        with pytest.raises(ValueError, match="x must be >= 0"):
            debye(3, np.array([1.0, -1.0]))


# =============================================================================
# Heat capacity
# =============================================================================


class TestDebyeHeatCapacity:
    """Debye-model C_V / (3 N k_B)."""

    @pytest.mark.parametrize(
        ("temperature", "theta"),
        [
            (300.0, 428.0),
            (200.0, 300.0),
            (1000.0, 300.0),
            (5000.0, 100.0),
            (50.0, 400.0),
            (20.0, 400.0),
            (10.0, 400.0),
        ],
    )
    def test_matches_phonon_mode_integral(self, temperature, theta):
        x = theta / temperature
        got = debye_heat_capacity(temperature, theta)
        assert got[0] == pytest.approx(
            float(_heat_capacity_oracle(x)), rel=1e-12, abs=0
        )

    def test_array_input(self):
        theta = 300.0
        T = np.array([60.0, 150.0, 300.0, 600.0])
        expected = [float(_heat_capacity_oracle(theta / t)) for t in T]
        np.testing.assert_allclose(
            debye_heat_capacity(T, theta), expected, rtol=1e-12, atol=0
        )

    def test_low_temperature_t_cubed_law(self):
        """C/(3Nk) = (4 pi^4 / 5) (T/theta)^3 up to O(e^-x)."""
        theta, T = 1000.0, 10.0
        expected = 4 * np.pi**4 / 5 * (T / theta) ** 3
        assert debye_heat_capacity(T, theta)[0] == pytest.approx(
            expected, rel=1e-13, abs=0
        )

    def test_high_temperature_expansion(self):
        """C/(3Nk) = 1 - x^2/20 + x^4/560 - x^6/18144 + O(x^8)."""
        x = 0.05
        expected = 1 - x**2 / 20 + x**4 / 560 - x**6 / 18144
        assert debye_heat_capacity(1.0 / x, 1.0)[0] == pytest.approx(
            expected, rel=1e-13, abs=0
        )

    def test_invalid_temperature(self):
        with pytest.raises(ValueError, match="Temperature must be positive"):
            debye_heat_capacity(-100.0, 300.0)

    def test_invalid_debye_temperature(self):
        with pytest.raises(ValueError, match="Debye temperature must be positive"):
            debye_heat_capacity(300.0, -100.0)


# =============================================================================
# Entropy
# =============================================================================


class TestDebyeEntropy:
    """Debye-model S / (3 N k_B)."""

    @pytest.mark.parametrize(
        ("temperature", "theta"),
        [
            (300.0, 300.0),
            (100.0, 300.0),
            (1000.0, 300.0),
            (50.0, 400.0),
            (20.0, 400.0),
            (10.0, 400.0),
        ],
    )
    def test_matches_phonon_mode_integral(self, temperature, theta):
        x = theta / temperature
        got = debye_entropy(temperature, theta)
        assert got[0] == pytest.approx(float(_entropy_oracle(x)), rel=1e-12, abs=0)

    def test_array_input(self):
        theta = 300.0
        T = np.array([100.0, 200.0, 300.0, 400.0])
        expected = [float(_entropy_oracle(theta / t)) for t in T]
        np.testing.assert_allclose(
            debye_entropy(T, theta), expected, rtol=1e-12, atol=0
        )

    def test_low_temperature_t_cubed_law(self):
        """S/(3Nk) = (4 pi^4 / 15) (T/theta)^3 up to O(e^-x)."""
        theta, T = 1000.0, 10.0
        expected = 4 * np.pi**4 / 15 * (T / theta) ** 3
        assert debye_entropy(T, theta)[0] == pytest.approx(expected, rel=1e-13, abs=0)

    def test_invalid_temperature(self):
        with pytest.raises(ValueError, match="Temperature must be positive"):
            debye_entropy(-100.0, 300.0)

    def test_invalid_debye_temperature(self):
        with pytest.raises(ValueError, match="Debye temperature must be positive"):
            debye_entropy(300.0, -100.0)
