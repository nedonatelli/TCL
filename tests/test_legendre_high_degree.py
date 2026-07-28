"""
Ultra-high-degree spherical harmonics (EGM2008 and beyond).

Regression tests for gh-16. Two distinct defects are covered.

**The scaled Legendre routine was scaled along the wrong axis.**
``associated_legendre_scaled`` applied a per-*degree* factor
``10**(-280 n / n_max)``, but the quantity that underflows is ``u**m`` with
``u = sin(theta)`` -- a per-*order* effect. The addition-theorem norm
``sum_m Pbar_nm**2 == 2n+1`` was off by a factor of 14 at degree 1000 and by
1e199 at degree 2000. It now follows Holmes & Featherstone (2002) and recurses
on ``Pbar_nm / u**m``, in which ``u`` cancels from every recursion.

**The ordinary synthesis becomes unstable at exactly EGM2008's degree.**
On the reference sphere at colatitude 30 degrees, ``spherical_harmonic_sum``
returns 3.6e20 where the potential is 6.25e7 -- twelve orders of magnitude
out. It is stable through n_max=1600 and breaks at 2190.
``spherical_harmonic_sum_high_degree`` applies ``u**m`` progressively via
Horner's scheme, never forming the underflowing factor, and stays correct.

Reference values below were generated with mpmath at 60 digits and are
hard-coded so the suite carries no mpmath dependency.
"""

import numpy as np
import pytest

from pytcl.gravity.spherical_harmonics import (
    associated_legendre,
    associated_legendre_scaled,
    spherical_harmonic_sum,
    spherical_harmonic_sum_high_degree,
)

R_EARTH = 6.378137e6
GM_EARTH = 3.986004418e14

# Fully normalized (geodesy 4pi) values at x = cos(45 deg), from mpmath dps=60.
MPMATH_REFERENCE = {
    (10, 3): -1.978527585e00,
    (50, 25): 2.227570466e00,
    (88, 52): -2.646966404e-06,
    (120, 80): 2.092618292e00,
    (200, 150): 1.062293360e-01,
}


def addition_theorem_ratio(n_max, colat_deg):
    """sum_m Pbar_nm^2 / (2n+1), evaluated in log space to avoid underflow."""
    x = np.cos(np.radians(colat_deg))
    P_scaled, scale_exp = associated_legendre_scaled(n_max, n_max, x)
    with np.errstate(divide="ignore"):
        log_p = np.log10(np.abs(P_scaled[n_max, : n_max + 1])) + scale_exp[: n_max + 1]
    total = float(np.sum(10.0 ** (2 * np.clip(log_p, -400, 400))))
    return total / (2 * n_max + 1)


class TestScaledLegendreAccuracy:
    @pytest.mark.parametrize("n_max", [200, 500, 1000])
    @pytest.mark.parametrize("colat", [20.0, 45.0, 80.0])
    def test_addition_theorem_holds(self, n_max, colat):
        """The invariant that was off by 14x at n=1000 and 1e199 at n=2000."""
        ratio = addition_theorem_ratio(n_max, colat)
        assert ratio == pytest.approx(1.0, abs=1e-8), (
            f"addition theorem violated at n_max={n_max}, colat={colat}: {ratio}"
        )

    @pytest.mark.slow
    def test_addition_theorem_at_egm2008_degree(self):
        for colat in (20.0, 45.0):
            ratio = addition_theorem_ratio(2190, colat)
            assert ratio == pytest.approx(1.0, abs=1e-8)

    @pytest.mark.parametrize("n,m", sorted(MPMATH_REFERENCE))
    def test_matches_high_precision_reference(self, n, m):
        """Scaled values reconstruct to the mpmath result."""
        x = np.cos(np.radians(45.0))
        u = np.sqrt(1 - x * x)
        P_scaled, _ = associated_legendre_scaled(200, 200, x)
        value = P_scaled[n, m] * (u**m) * 1e280
        expected = MPMATH_REFERENCE[(n, m)]
        assert value == pytest.approx(expected, rel=1e-7)

    def test_agrees_with_direct_routine(self):
        """Where the ordinary routine is reliable, both must agree."""
        x = np.cos(np.radians(30.0))
        u = np.sqrt(1 - x * x)
        n_max = 80
        direct = associated_legendre(n_max, n_max, x, normalized=True)
        scaled, _ = associated_legendre_scaled(n_max, n_max, x)
        for m in range(0, n_max + 1, 10):
            factor = (u**m) * 1e280
            if not np.isfinite(factor) or factor == 0.0:
                continue
            for n in range(m, n_max + 1, 10):
                if abs(direct[n, m]) < 1e-250:
                    continue
                assert scaled[n, m] * factor == pytest.approx(direct[n, m], rel=1e-4)

    def test_reconstruction_exponent_is_per_order(self):
        """scale_exp is indexed by order and reconstructs correctly."""
        x = np.cos(np.radians(60.0))
        n_max = 50
        direct = associated_legendre(n_max, n_max, x, normalized=True)
        scaled, scale_exp = associated_legendre_scaled(n_max, n_max, x)
        assert scale_exp.shape == (n_max + 1,)
        for m in (0, 5, 20):
            recon = scaled[n_max, m] * 10.0 ** scale_exp[m]
            assert recon == pytest.approx(direct[n_max, m], rel=1e-6)

    def test_rejects_invalid_input(self):
        with pytest.raises(ValueError):
            associated_legendre_scaled(10, 20, 0.5)
        with pytest.raises(ValueError):
            associated_legendre_scaled(10, 10, 1.5)


class TestHighDegreeSynthesis:
    @staticmethod
    def _coefficients(n_max, seed=2):
        rng = np.random.default_rng(seed)
        idx = np.arange(n_max + 1)
        scale = np.where(idx > 0, 1e-5 / np.maximum(idx, 1) ** 2, 0.0)
        C = np.tril(rng.standard_normal((n_max + 1, n_max + 1)) * scale[:, None])
        S = np.tril(rng.standard_normal((n_max + 1, n_max + 1)) * scale[:, None])
        C[0, 0] = 1.0
        S[:, 0] = 0.0
        return C, S

    @pytest.mark.parametrize("colat", [20.0, 45.0, 70.0])
    def test_matches_standard_where_standard_is_valid(self, colat):
        n_max = 120
        C, S = self._coefficients(n_max)
        lat = np.radians(90 - colat)
        r = R_EARTH + 1e5
        want = spherical_harmonic_sum(lat, 0.3, r, C, S, R_EARTH, GM_EARTH, n_max)
        got = spherical_harmonic_sum_high_degree(
            lat, 0.3, r, C, S, R_EARTH, GM_EARTH, n_max
        )
        assert got[0] == pytest.approx(want[0], rel=1e-10)  # potential
        assert got[1] == pytest.approx(want[1], rel=1e-10)  # dV/dr
        assert got[2] == pytest.approx(want[2], rel=1e-6)  # dV/dlat

    def test_monopole_recovers_point_mass_potential(self):
        n_max = 60
        C = np.zeros((n_max + 1, n_max + 1))
        S = np.zeros((n_max + 1, n_max + 1))
        C[0, 0] = 1.0
        r = R_EARTH + 2e5
        V, dV_r, _ = spherical_harmonic_sum_high_degree(
            0.4, 0.3, r, C, S, R_EARTH, GM_EARTH, n_max
        )
        assert V == pytest.approx(GM_EARTH / r, rel=1e-12)
        assert dV_r == pytest.approx(-GM_EARTH / r**2, rel=1e-12)

    @pytest.mark.slow
    def test_stable_at_egm2008_degree_where_standard_breaks(self):
        """colat=30 deg on the reference sphere: standard returns 3.6e20."""
        n_max = 2190
        C, S = self._coefficients(n_max)
        lat = np.radians(90 - 30.0)
        V, _, _ = spherical_harmonic_sum_high_degree(
            lat, 0.3, R_EARTH, C, S, R_EARTH, GM_EARTH, n_max
        )
        # Dominated by the monopole; must stay near GM/R rather than diverging.
        assert V == pytest.approx(GM_EARTH / R_EARTH, rel=1e-3)
        assert np.isfinite(V)

    @pytest.mark.slow
    def test_consistent_across_colatitude_at_high_degree(self):
        """A smooth field must not jump between neighboring colatitudes."""
        n_max = 2190
        C, S = self._coefficients(n_max)
        values = [
            spherical_harmonic_sum_high_degree(
                np.radians(90 - c), 0.3, R_EARTH, C, S, R_EARTH, GM_EARTH, n_max
            )[0]
            for c in (28.0, 30.0, 32.0)
        ]
        spread = (max(values) - min(values)) / abs(np.mean(values))
        assert spread < 1e-3, f"discontinuity across colatitude: {values}"
