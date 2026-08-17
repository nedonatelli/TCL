"""Correctness audit tests for special functions, quadrature, and interpolation.

Reference values marked "mpmath" were computed with mpmath at 50 significant
digits. Values marked "quad" are computed in-test from defining integrals
using scipy. Property tests verify mathematical identities (recurrences,
polynomial exactness, node reproduction) rather than point values.
"""

import numpy as np
import pytest
import scipy.special as sp
from scipy.integrate import quad as scipy_quad
from scipy.stats import ncx2

import pytcl.mathematical_functions.interpolation as interp
import pytcl.mathematical_functions.numerical_integration as ni
import pytcl.mathematical_functions.special_functions as sf


class TestBesselReference:
    """Bessel family against mpmath 50-digit references."""

    def test_bessel_values(self):
        assert np.isclose(sf.besselj(2.5, 3.7), 0.45685188411295336, rtol=1e-14)
        assert np.isclose(sf.bessely(2.5, 3.7), -0.096504219513778383, rtol=1e-13)
        assert np.isclose(sf.besseli(2.5, 3.7), 3.414958395937987, rtol=1e-14)
        assert np.isclose(sf.besselk(2.5, 3.7), 0.032700514975185734, rtol=1e-14)

    def test_bessel_recurrence(self):
        # J_{n-1}(x) + J_{n+1}(x) = (2n/x) J_n(x)
        for n in [1, 2, 5]:
            for x in [0.5, 2.0, 7.3]:
                lhs = sf.besselj(n - 1, x) + sf.besselj(n + 1, x)
                rhs = 2 * n / x * sf.besselj(n, x)
                assert np.isclose(lhs, rhs, rtol=1e-12, atol=1e-14)

    def test_hankel_decomposition(self):
        for x in [0.5, 2.0, 5.0]:
            h1 = sf.besselh(1, 1, x)
            h2 = sf.besselh(1, 2, x)
            assert np.isclose(h1.real, sf.besselj(1, x), rtol=1e-13)
            assert np.isclose(h1.imag, sf.bessely(1, x), rtol=1e-13)
            assert np.isclose(h2.imag, -sf.bessely(1, x), rtol=1e-13)

    def test_spherical_bessel_analytic(self):
        x = 1.7
        assert np.isclose(sf.spherical_jn(0, x), np.sin(x) / x, rtol=1e-14)
        assert np.isclose(sf.spherical_yn(0, x), -np.cos(x) / x, rtol=1e-14)
        assert np.isclose(sf.spherical_in(0, x), np.sinh(x) / x, rtol=1e-14)
        assert np.isclose(sf.spherical_kn(0, x), np.pi / 2 * np.exp(-x) / x, rtol=1e-14)
        # j1(x) = sin(x)/x^2 - cos(x)/x
        assert np.isclose(
            sf.spherical_jn(1, x), np.sin(x) / x**2 - np.cos(x) / x, rtol=1e-13
        )

    def test_airy_wronskian(self):
        # Ai(x) Bi'(x) - Ai'(x) Bi(x) = 1/pi
        for x in [-2.0, 0.0, 1.5]:
            Ai, Aip, Bi, Bip = sf.airy(x)
            assert np.isclose(Ai * Bip - Aip * Bi, 1 / np.pi, rtol=1e-13)

    def test_struve_values(self):
        assert np.isclose(sf.struve_h(0, 3.7), 0.26935588631419573, rtol=1e-12)
        assert np.isclose(sf.struve_h(1.5, 2.1), 0.49739660479540864, rtol=1e-12)
        assert np.isclose(sf.struve_l(0, 3.7), 8.5526930983360051, rtol=1e-12)
        assert np.isclose(sf.struve_l(1.5, 2.1), 0.81188452520456483, rtol=1e-12)

    def test_kelvin_values(self):
        refs = {
            1.0: (
                0.98438178121308688,
                0.24956604003665972,
                0.28670620872831605,
                -0.4949946365187199,
            ),
            3.0: (
                -0.22138024959869389,
                1.9375867852660428,
                -0.067029233303798698,
                -0.051121884045986781,
            ),
            8.0: (
                20.973955610730256,
                -35.016725164881512,
                0.0014858340685189625,
                0.00036958395612595959,
            ),
        }
        for x, (ber, bei, ker, kei) in refs.items():
            got = sf.kelvin(x)
            assert np.isclose(got[0], ber, rtol=1e-12)
            assert np.isclose(got[1], bei, rtol=1e-12)
            assert np.isclose(got[2], ker, rtol=1e-11)
            assert np.isclose(got[3], kei, rtol=1e-11)

    def test_bessel_zeros_are_zeros(self):
        for zi in sf.bessel_zeros(0, 5, kind="j"):
            assert abs(sf.besselj(0, zi)) < 1e-10
        for zi in sf.bessel_zeros(2, 3, kind="y"):
            assert abs(sf.bessely(2, zi)) < 1e-10
        for zi in sf.bessel_zeros(1, 3, kind="jp"):
            assert abs(sf.bessel_deriv(1, zi, kind="j")) < 1e-10

    def test_bessel_ratio_and_deriv(self):
        x = 2.3
        assert np.isclose(
            sf.bessel_ratio(1, x, "j"),
            sf.besselj(2, x) / sf.besselj(1, x),
            rtol=1e-13,
        )
        assert np.isclose(
            sf.bessel_ratio(1, x, "i"),
            sf.besseli(2, x) / sf.besseli(1, x),
            rtol=1e-13,
        )
        # dJ_0/dx = -J_1
        assert np.isclose(sf.bessel_deriv(0, x, "j"), -sf.besselj(1, x), rtol=1e-13)
        # dI_0/dx = I_1
        assert np.isclose(sf.bessel_deriv(0, x, "i"), sf.besseli(1, x), rtol=1e-13)
        # dK_0/dx = -K_1
        assert np.isclose(sf.bessel_deriv(0, x, "k"), -sf.besselk(1, x), rtol=1e-13)


def _mp_bessel_ratio(n, x, kind):
    """Reference ratio B_{n+1}(x)/B_n(x) via mpmath at 50 significant digits."""
    import mpmath

    with mpmath.mp.workdps(50):
        fn = mpmath.besselj if kind == "j" else mpmath.besseli
        return float(fn(n + 1, mpmath.mpf(x)) / fn(n, mpmath.mpf(x)))


class TestBesselRatioStability:
    """bessel_ratio must not underflow to nan for large order (audit item B.2).

    The plain sp.jv(n+1, x) / sp.jv(n, x) quotient underflows both terms to
    0.0 at (n=170, x=1.0), returning nan. The continued-fraction evaluation
    must return the correct finite ratio there and across the (n, x) grid.
    """

    GRID_N = [0, 1, 5, 20, 80, 170, 400]
    GRID_X = [0.5, 1.0, 10.0, 50.0, 100.0]

    def test_large_order_underflow_case(self):
        got = float(sf.bessel_ratio(170, 1.0, "j"))
        assert np.isfinite(got)
        ref = _mp_bessel_ratio(170, 1.0, "j")
        assert np.isclose(got, ref, rtol=1e-12)

    @pytest.mark.parametrize("kind", ["j", "i"])
    def test_grid_vs_mpmath(self, kind):
        # Measured worst case over this grid: 9.6e-15 ('j', at n=80 x=100)
        # and 1.5e-15 ('i', at n=5 x=100); rtol gives ~10x headroom.
        for n in self.GRID_N:
            for x in self.GRID_X:
                ref = _mp_bessel_ratio(n, x, kind)
                got = float(sf.bessel_ratio(n, x, kind))
                assert np.isclose(got, ref, rtol=1e-13), (
                    f"kind={kind} n={n} x={x}: got {got}, ref {ref}"
                )

    def test_healthy_region_matches_plain_quotient(self):
        # Where the old plain sp.jv quotient is finite and well-conditioned,
        # the CF result must agree with it to 1e-12.
        for n in [0, 1, 5, 20]:
            for x in [0.5, 2.3, 10.0, 50.0]:
                for kind, fn in [("j", sp.jv), ("i", sp.iv)]:
                    den = fn(n, x)
                    num = fn(n + 1, x)
                    if not (np.isfinite(num) and np.isfinite(den)):
                        continue
                    if abs(den) < 1e-8:
                        continue  # near a zero of J_n: plain quotient ill-conditioned
                    assert np.isclose(
                        float(sf.bessel_ratio(n, x, kind)), num / den, rtol=1e-12
                    )

    def test_near_zero_of_denominator(self):
        # x near the first zero of J_0 (2.404825557695773...): the ratio is
        # huge but well-defined; the CF must track mpmath. Conditioning of
        # the ratio w.r.t. x degrades as 1/|J_0(x)|, so the achievable
        # relative tolerance loosens as x approaches the zero. Measured:
        # 5.1e-15, 2.0e-12, 9.7e-9 respectively; rtol gives ~10x headroom.
        cases = [
            (2.404, 1e-13),
            (2.40482, 1e-10),
            (2.4048255576, 1e-7),
        ]
        for x, rtol in cases:
            ref = _mp_bessel_ratio(0, x, "j")
            got = float(sf.bessel_ratio(0, x, "j"))
            assert np.isclose(got, ref, rtol=rtol), f"x={x}: got {got}, ref {ref}"

    def test_x_zero_limit(self):
        # lim_{x->0} J_{n+1}(x)/J_n(x) = 0 for every order n >= 0.
        for n in [0, 1, 5, 170]:
            for kind in ["j", "i"]:
                assert float(sf.bessel_ratio(n, 0.0, kind)) == 0.0

    def test_array_broadcasting(self):
        x = np.array([0.5, 1.0, 10.0])
        out = sf.bessel_ratio(5, x, "j")
        assert out.shape == x.shape
        for xi, oi in zip(x, out):
            assert np.isclose(oi, _mp_bessel_ratio(5, float(xi), "j"), rtol=1e-13)


class TestGammaFamily:
    def test_analytic_values(self):
        assert np.isclose(sf.gamma(0.5), np.sqrt(np.pi), rtol=1e-14)
        assert np.isclose(sf.gamma(5), 24.0, rtol=1e-14)
        # Reflection: Gamma(x) Gamma(1-x) = pi / sin(pi x)
        x = 0.3
        assert np.isclose(
            sf.gamma(x) * sf.gamma(1 - x), np.pi / np.sin(np.pi * x), rtol=1e-13
        )
        assert np.isclose(sf.gammaln(100), sp.gammaln(100), rtol=1e-15)

    def test_incomplete_gamma_complement_and_inverse(self):
        for a, x in [(0.5, 0.3), (2.0, 1.0), (5.0, 7.0)]:
            assert np.isclose(sf.gammainc(a, x) + sf.gammaincc(a, x), 1.0, rtol=1e-13)
            assert np.isclose(
                sf.gammaincinv(a, float(sf.gammainc(a, x))), x, rtol=1e-10
            )
        assert np.isclose(sf.gammainc(1, 1), 1 - np.exp(-1), rtol=1e-14)

    def test_digamma_polygamma(self):
        euler_gamma = 0.57721566490153286
        assert np.isclose(sf.digamma(1), -euler_gamma, rtol=1e-13)
        assert np.isclose(sf.polygamma(1, 1), np.pi**2 / 6, rtol=1e-13)
        # psi(x+1) = psi(x) + 1/x
        x = 2.7
        assert np.isclose(sf.digamma(x + 1), sf.digamma(x) + 1 / x, rtol=1e-13)

    def test_beta_identities(self):
        for a, b in [(0.5, 0.5), (2, 3), (7.5, 0.2)]:
            assert np.isclose(
                sf.beta(a, b), sf.gamma(a) * sf.gamma(b) / sf.gamma(a + b), rtol=1e-12
            )
            assert np.isclose(sf.beta(a, b), sf.beta(b, a), rtol=1e-14)
            assert np.isclose(sf.betaln(a, b), np.log(sf.beta(a, b)), rtol=1e-12)
        assert np.isclose(sf.beta(0.5, 0.5), np.pi, rtol=1e-14)

    def test_incomplete_beta(self):
        assert np.isclose(sf.betainc(1, 1, 0.4), 0.4, rtol=1e-14)
        for a, b, x in [(2, 3, 0.4), (0.5, 0.5, 0.7)]:
            # I_x(a,b) = 1 - I_{1-x}(b,a)
            assert np.isclose(
                sf.betainc(a, b, x), 1 - sf.betainc(b, a, 1 - x), rtol=1e-13
            )
            assert np.isclose(
                sf.betaincinv(a, b, float(sf.betainc(a, b, x))), x, rtol=1e-10
            )

    def test_combinatorics(self):
        assert sf.factorial(10) == 3628800
        assert sf.factorial2(9) == 945
        assert sf.factorial2(10) == 3840
        assert sf.comb(10, 4) == 210
        assert sf.comb(5, 2, repetition=True) == 15
        assert sf.perm(10, 4) == 5040


class TestErrorFunctions:
    def test_limits_and_symmetry(self):
        assert sf.erf(np.inf) == 1.0
        assert sf.erf(0) == 0.0
        assert np.isclose(sf.erf(1.3), -sf.erf(-1.3), rtol=1e-15)
        assert np.isclose(sf.erf(2.0) + sf.erfc(2.0), 1.0, rtol=1e-14)

    def test_scaled_and_imaginary(self):
        x = 1.7
        assert np.isclose(sf.erfcx(x), np.exp(x**2) * sf.erfc(x), rtol=1e-13)
        # dawsn(x) = sqrt(pi)/2 exp(-x^2) erfi(x)
        assert np.isclose(
            sf.dawsn(x),
            np.sqrt(np.pi) / 2 * np.exp(-(x**2)) * sf.erfi(x),
            rtol=1e-13,
        )

    def test_inverses(self):
        for y in [0.1, 0.5, 0.99]:
            assert np.isclose(sf.erf(sf.erfinv(y)), y, rtol=1e-12)
            assert np.isclose(sf.erfc(sf.erfcinv(y)), y, rtol=1e-12)

    def test_fresnel_and_wofz(self):
        # Fresnel integrals converge to 1/2
        S, C = sf.fresnel(50.0)
        assert np.isclose(S, 0.5, atol=0.01)
        assert np.isclose(C, 0.5, atol=0.01)
        S1, C1 = sf.fresnel(1.0)
        assert np.isclose(S1, 0.43825914739035476, rtol=1e-12)  # mpmath
        assert np.isclose(C1, 0.77989340037682282, rtol=1e-12)  # mpmath
        assert sf.wofz(0) == 1.0 + 0.0j
        # w(z) = exp(-z^2) erfc(-iz)
        z = 0.7 + 0.4j
        ref = np.exp(-(z**2)) * sp.erfc(-1j * z)
        assert np.isclose(sf.wofz(z), ref, rtol=1e-12)

    def test_voigt_limits(self):
        # gamma=0: Gaussian pdf
        assert np.isclose(
            sf.voigt_profile(0.5, 1.3, 0),
            np.exp(-(0.5**2) / (2 * 1.3**2)) / (1.3 * np.sqrt(2 * np.pi)),
            rtol=1e-10,
        )
        # sigma->0: Lorentzian pdf
        assert np.isclose(
            sf.voigt_profile(0.5, 1e-12, 0.7),
            0.7 / np.pi / (0.5**2 + 0.7**2),
            rtol=1e-6,
        )


class TestElliptic:
    def test_complete_values(self):
        assert np.isclose(sf.ellipk(0), np.pi / 2, rtol=1e-14)
        assert np.isclose(sf.ellipe(0), np.pi / 2, rtol=1e-14)
        assert sf.ellipe(1) == 1.0
        assert np.isclose(sf.ellipk(0.5), 1.8540746773013719, rtol=1e-14)  # mpmath
        assert np.isclose(sf.ellipkm1(1e-5), sf.ellipk(1 - 1e-5), rtol=1e-7)

    def test_legendre_relation(self):
        # E(m) K(1-m) + E(1-m) K(m) - K(m) K(1-m) = pi/2
        m = 0.3
        lhs = (
            sf.ellipe(m) * sf.ellipk(1 - m)
            + sf.ellipe(1 - m) * sf.ellipk(m)
            - sf.ellipk(m) * sf.ellipk(1 - m)
        )
        assert np.isclose(lhs, np.pi / 2, rtol=1e-13)

    def test_incomplete_reduce_to_complete(self):
        m = 0.6
        assert np.isclose(sf.ellipkinc(np.pi / 2, m), sf.ellipk(m), rtol=1e-13)
        assert np.isclose(sf.ellipeinc(np.pi / 2, m), sf.ellipe(m), rtol=1e-13)

    def test_carlson_identities(self):
        # K(m) = R_F(0, 1-m, 1); E(m) = 2 R_G(0, 1-m, 1)
        m = 0.7
        assert np.isclose(sf.elliprf(0, 1 - m, 1), sf.ellipk(m), rtol=1e-13)
        assert np.isclose(2 * sf.elliprg(0, 1 - m, 1), sf.ellipe(m), rtol=1e-13)
        # R_D cyclic sum: R_D(x,y,z)+R_D(y,z,x)+R_D(z,x,y) = 3/sqrt(xyz)
        x, y, z = 1.0, 2.0, 3.0
        total = sf.elliprd(x, y, z) + sf.elliprd(y, z, x) + sf.elliprd(z, x, y)
        assert np.isclose(total, 3 / np.sqrt(x * y * z), rtol=1e-13)
        # Degenerate cases
        assert np.isclose(sf.elliprc(1, 1), 1.0, rtol=1e-14)
        assert np.isclose(sf.elliprj(2, 2, 2, 2), 2 ** (-1.5), rtol=1e-13)
        # R_C(x, y) = arctan(sqrt((y-x)/x))/sqrt(y-x) for x < y
        assert np.isclose(
            sf.elliprc(1, 3), np.arctan(np.sqrt(2.0)) / np.sqrt(2.0), rtol=1e-13
        )
        # R_C(x, y) = arctanh(sqrt((x-y)/x))/sqrt(x-y) for x > y
        assert np.isclose(
            sf.elliprc(3, 1),
            np.arctanh(np.sqrt(2.0 / 3.0)) / np.sqrt(2.0),
            rtol=1e-13,
        )


class TestMarcumQ:
    # (m, a, b, Q_m(a,b)) from mpmath 50-digit integration
    REFS = [
        (1, 1.0, 2.0, 0.26901206003591),
        (1, 3.0, 4.0, 0.1965121893884076),
        (2, 2.0, 3.0, 0.3526978960496345),
        (5, 2.0, 8.0, 3.373623154936504e-7),
        (1, 0.5, 0.1, 0.9955971538791816),
        (3, 4.0, 1.0, 0.9999881559948034),
    ]

    def test_reference_values(self):
        for m, a, b, ref in self.REFS:
            assert np.isclose(sf.marcum_q(a, b, m), ref, rtol=1e-9), (m, a, b)
            assert np.isclose(sf.marcum_q1(a, b), sf.marcum_q(a, b, 1), rtol=1e-14)

    def test_edge_cases(self):
        assert sf.marcum_q(3.0, 0.0) == 1.0
        assert np.isclose(sf.marcum_q(0.0, 2.0, 2), sf.gammaincc(2, 2.0), rtol=1e-12)
        # Mixed edge cases in one array
        out = sf.marcum_q(np.array([0.0, 1.0, 2.0]), np.array([0.0, 0.0, 3.0]))
        assert out[0] == 1.0 and out[1] == 1.0
        assert np.isclose(out[2], ncx2.sf(9.0, 2, 4.0), rtol=1e-12)

    def test_log_marcum_q(self):
        for m, a, b, ref in self.REFS:
            assert np.isclose(sf.log_marcum_q(a, b, m), np.log(ref), rtol=1e-7)
        # Deep tail agrees with ncx2.logsf
        assert np.isclose(
            sf.log_marcum_q(1.0, 8.0), ncx2.logsf(64.0, 2, 1.0), rtol=1e-12
        )

    def test_log_marcum_q_broadcasting(self):
        # Regression: scalar a with vector b raised IndexError when any Q
        # was below the small-q switchover
        b = np.array([1.0, 5.0, 8.0, 12.0])
        got = sf.log_marcum_q(1.0, b)
        ref = ncx2.logsf(b**2, 2, 1.0)
        assert np.allclose(got, ref, rtol=1e-10)
        got2 = sf.log_marcum_q(np.array([1.0, 2.0]), np.array([8.0, 12.0]))
        ref2 = ncx2.logsf(np.array([64.0, 144.0]), 2, np.array([1.0, 4.0]))
        assert np.allclose(got2, ref2, rtol=1e-10)

    def test_inverse_roundtrip(self):
        for a, q in [(3.0, 0.5), (1.0, 0.01), (0.5, 0.99)]:
            b = sf.marcum_q_inv(a, q)
            assert np.isclose(sf.marcum_q(a, float(b)), q, rtol=1e-8)

    def test_nuttall_complement(self):
        assert np.isclose(sf.nuttall_q(2, 2), 1 - sf.marcum_q(2, 2), rtol=1e-13)


class TestSwerling:
    """Swerling Pd against direct integration of the defining models."""

    PFA = 1e-6

    @staticmethod
    def _threshold(n):
        return 2 * sp.gammainccinv(n, TestSwerling.PFA)

    def _sw_scan_to_scan_ref(self, xbar, n, dof4):
        # Common fluctuation across pulses: integrate ncx2 tail over the
        # target power pdf (exponential for SW1, chi-4 for SW3)
        T = self._threshold(n)
        if dof4:
            pdf = lambda x: 4 * x / xbar**2 * np.exp(-2 * x / xbar)  # noqa: E731
        else:
            pdf = lambda x: np.exp(-x / xbar) / xbar  # noqa: E731
        val, _ = scipy_quad(
            lambda x: pdf(x) * ncx2.sf(T, 2 * n, 2 * n * x), 0, np.inf, limit=300
        )
        return val

    @pytest.mark.parametrize("snr_db,n", [(10, 1), (10, 10), (13, 1), (5, 10), (8, 3)])
    def test_swerling1(self, snr_db, n):
        xbar = 10 ** (snr_db / 10)
        ref = self._sw_scan_to_scan_ref(xbar, n, dof4=False)
        got = float(np.asarray(sf.swerling_detection_probability(xbar, self.PFA, n, 1)))
        assert np.isclose(got, ref, rtol=1e-6), (snr_db, n)

    @pytest.mark.parametrize("snr_db,n", [(10, 1), (10, 10), (5, 10), (8, 3)])
    def test_swerling2(self, snr_db, n):
        xbar = 10 ** (snr_db / 10)
        # Pulse-to-pulse Rayleigh: integrated sum is Gamma(n, 1 + xbar)
        ref = sp.gammaincc(n, self._threshold(n) / (2 * (1 + xbar)))
        got = float(np.asarray(sf.swerling_detection_probability(xbar, self.PFA, n, 2)))
        assert np.isclose(got, ref, rtol=1e-10), (snr_db, n)

    @pytest.mark.parametrize("snr_db,n", [(10, 1), (10, 10), (13, 1), (5, 10), (8, 3)])
    def test_swerling3(self, snr_db, n):
        xbar = 10 ** (snr_db / 10)
        ref = self._sw_scan_to_scan_ref(xbar, n, dof4=True)
        got = float(np.asarray(sf.swerling_detection_probability(xbar, self.PFA, n, 3)))
        assert np.isclose(got, ref, rtol=1e-6), (snr_db, n)

    def test_swerling4_matches_swerling3_single_pulse(self):
        # With one pulse, scan-to-scan and pulse-to-pulse chi-4 coincide
        for snr_db in [5, 10, 13]:
            xbar = 10 ** (snr_db / 10)
            g3 = float(
                np.asarray(sf.swerling_detection_probability(xbar, self.PFA, 1, 3))
            )
            g4 = float(
                np.asarray(sf.swerling_detection_probability(xbar, self.PFA, 1, 4))
            )
            assert np.isclose(g3, g4, rtol=1e-10)

    def test_swerling4_monte_carlo(self):
        # MC references (4e6 samples): (snr_db, n, pd)
        refs = [(10, 10, 0.999959), (5, 10, 0.781953), (8, 3, 0.558722)]
        for snr_db, n, ref in refs:
            xbar = 10 ** (snr_db / 10)
            got = float(
                np.asarray(sf.swerling_detection_probability(xbar, self.PFA, n, 4))
            )
            assert np.isclose(got, ref, atol=2e-3), (snr_db, n)

    def test_swerling0_is_marcum(self):
        xbar, n = 10.0, 4
        T = self._threshold(n)
        got = float(np.asarray(sf.swerling_detection_probability(xbar, self.PFA, n, 0)))
        assert np.isclose(got, ncx2.sf(T, 2 * n, 2 * n * xbar), rtol=1e-8)

    def test_swerling_array_input(self):
        snr = np.array([1.0, 10.0, 100.0])
        for case in range(5):
            pd = np.asarray(sf.swerling_detection_probability(snr, self.PFA, 5, case))
            assert pd.shape == (3,)
            assert np.all((pd >= 0) & (pd <= 1 + 1e-12))
            assert np.all(np.diff(pd) > 0)  # Pd increases with SNR


class TestLambertW:
    def test_defining_equation(self):
        for z in [0.5, 2.0, -0.2, 100.0]:
            w = complex(sf.lambert_w(z))
            assert abs(w * np.exp(w) - z) < 1e-10 * max(1.0, abs(z))
        # Lower branch
        w = complex(sf.lambert_w(-0.2, k=-1))
        assert abs(w * np.exp(w) + 0.2) < 1e-12
        # Complex branch
        w = complex(sf.lambert_w(1.0 + 2.0j, k=1))
        assert abs(w * np.exp(w) - (1.0 + 2.0j)) < 1e-10

    def test_real_branch(self):
        assert np.isclose(
            float(sf.lambert_w_real(1.0)), 0.56714329040978384, rtol=1e-14
        )
        assert np.isclose(
            float(sf.lambert_w_real(-0.2, branch=-1)), -2.5426413577735265, rtol=1e-12
        )
        with pytest.raises(ValueError):
            sf.lambert_w_real(-1.0)
        with pytest.raises(ValueError):
            sf.lambert_w_real(0.5, branch=-1)

    def test_omega_constant(self):
        omega = sf.omega_constant()
        assert np.isclose(omega * np.exp(omega), 1.0, rtol=1e-14)

    def test_wright_omega_defining_equation(self):
        for z in [0.0, 1.0, 10.0, -5.0]:
            w = complex(sf.wright_omega(z))
            assert abs(w + np.log(w) - z) < 1e-12 * max(1.0, abs(z))

    def test_wright_omega_large_argument(self):
        # Regression: W(exp(z)) overflowed to inf for z > ~709
        for z in [710.0, 800.0]:
            w = float(sf.wright_omega(z).real)
            assert np.isfinite(w)
            assert abs(w + np.log(w) - z) < 1e-10 * z

    def test_wright_omega_branch(self):
        # Regression: for |Im(z)| > pi the principal-branch W(e^z) is wrong
        for z in [-2 + 3.5j, 1 - 3.5j]:
            w = complex(sf.wright_omega(np.complex128(z)))
            assert np.isclose(w, complex(sp.wrightomega(z)), rtol=1e-12)

    def test_equation_solvers(self):
        x = complex(sf.solve_exponential_equation(2.0, 3.0, 5.0))
        assert abs(2 * x * np.exp(3 * x) - 5) < 1e-10
        s = complex(sf.time_delay_equation(1.0, 2.0))
        assert abs(s + 1.0 * np.exp(-s * 2.0)) < 1e-10


class TestDebye:
    # (n, x, D_n(x)) from mpmath 50-digit integration of the definition
    REFS = [
        (1, 0.099, 0.97552222332122431),
        (1, 0.5, 0.88192715679060553),
        (1, 1.0, 0.77750463411224828),
        (1, 50.0, 0.032898681336964529),
        (2, 0.099, 0.96740833053574437),
        (2, 25.0, 0.0076931641501345618),
        (2, 50.0, 0.0019232910450553509),
        (3, 0.099, 0.96336499283204186),
        (3, 1.0, 0.67441556407781468),
        (3, 2.5, 0.35413603481042394),
        (3, 10.0, 0.01929576569034549),
        (4, 0.5, 0.81384569172034043),
        (4, 5.0, 0.091471377664481165),
        (6, 0.7, 0.73042684171485539),
        (6, 12.0, 0.0013925446582644944),
    ]

    def test_reference_values(self):
        for n, x, ref in self.REFS:
            got = float(sf.debye(n, x)[0])
            assert np.isclose(got, ref, rtol=1e-12), (n, x, got, ref)

    def test_special_points(self):
        assert float(sf.debye(3, 0.0)[0]) == 1.0
        for n in [1, 2, 3, 4]:
            # Large-x asymptote: D_n(x) -> n * n! * zeta(n+1) / x^n
            x = 300.0
            ref = n * sp.factorial(n) * sp.zeta(n + 1) / x**n
            assert np.isclose(float(sf.debye(n, x)[0]), ref, rtol=1e-12)

    def test_series_boundary_continuity(self):
        # Small-x Bernoulli series and complement series must agree at x = 1
        for n in [1, 2, 3, 6]:
            below = float(sf.debye(n, 1.0 - 1e-9)[0])
            above = float(sf.debye(n, 1.0 + 1e-9)[0])
            assert np.isclose(below, above, rtol=1e-8)

    def test_order_shortcuts(self):
        x = np.array([0.3, 1.5, 7.0])
        assert np.allclose(sf.debye_1(x), sf.debye(1, x), rtol=1e-14)
        assert np.allclose(sf.debye_2(x), sf.debye(2, x), rtol=1e-14)
        assert np.allclose(sf.debye_3(x), sf.debye(3, x), rtol=1e-14)
        assert np.allclose(sf.debye_4(x), sf.debye(4, x), rtol=1e-14)

    def test_heat_capacity(self):
        # (T, theta_D, C_V/(3 N k_B)) from mpmath: 4 D_3(x) - 3x/(e^x - 1)
        refs = [
            (300, 428, 0.90518982126177901),
            (100, 428, 0.46236330229150362),
            (30, 428, 0.026799506676871435),
            (1000, 428, 0.99090038508239358),
        ]
        for T, theta, ref in refs:
            got = float(sf.debye_heat_capacity(T, theta)[0])
            assert np.isclose(got, ref, rtol=1e-10), (T, theta, got, ref)

    def test_heat_capacity_limits(self):
        # Classical limit
        assert np.isclose(float(sf.debye_heat_capacity(1e6, 428)[0]), 1.0, rtol=1e-6)
        # Low-T Debye T^3 law: C_V/(3NkB) -> (4 pi^4 / 5) (T/theta)^3
        T, theta = 5.0, 428.0
        ref = 4 * np.pi**4 / 5 * (T / theta) ** 3
        assert np.isclose(float(sf.debye_heat_capacity(T, theta)[0]), ref, rtol=1e-4)

    def test_entropy(self):
        # (T, theta_D, S/(3 N k_B)) from mpmath: (4/3) D_3(x) - ln(1 - e^-x)
        refs = [
            (300, 428, 1.0271018234305032),
            (100, 428, 0.22813870240932097),
            (30, 428, 0.0089428921071415439),
            (1000, 428, 2.1865300924514556),
        ]
        for T, theta, ref in refs:
            got = float(sf.debye_entropy(T, theta)[0])
            assert np.isclose(got, ref, rtol=1e-10), (T, theta, got, ref)

    def test_thermodynamic_consistency(self):
        # C_V = T * dS/dT must hold numerically
        theta = 428.0
        for T in [50.0, 300.0, 900.0]:
            dT = T * 1e-6
            dS = (
                float(sf.debye_entropy(T + dT, theta)[0])
                - float(sf.debye_entropy(T - dT, theta)[0])
            ) / (2 * dT)
            cv = float(sf.debye_heat_capacity(T, theta)[0])
            assert np.isclose(T * dS, cv, rtol=1e-5), T


class TestHypergeometric:
    def test_analytic_identities(self):
        z = 0.7
        assert np.isclose(sf.hyp1f1(1.5, 1.5, z), np.exp(z), rtol=1e-13)
        assert np.isclose(sf.hyp2f1(1, 1, 2, -z), np.log(1 + z) / z, rtol=1e-13)
        assert np.isclose(sf.hyp2f1(0.5, 0.5, 1.5, z**2), np.arcsin(z) / z, rtol=1e-13)
        # 0F1(n+1; -x^2/4) relation to J_n
        x = 1.9
        assert np.isclose(
            sf.hyp0f1(2, -(x**2) / 4),
            sf.besselj(1, x) * sf.gamma(2) / (x / 2),
            rtol=1e-12,
        )

    def test_reference_values(self):
        # mpmath 50-digit
        assert np.isclose(sf.hyp0f1(1.5, -2.3), 0.035682393374045927, rtol=1e-12)
        assert np.isclose(sf.hyp1f1(1.2, 3.4, -7.0), 0.21008171135284206, rtol=1e-12)
        assert np.isclose(sf.hyp2f1(2, 3, 4.5, -0.8), 0.4411802873860462, rtol=1e-12)
        assert np.isclose(sf.hyperu(1.2, 0.7, 3.3), 0.16294920165884404, rtol=1e-12)

    def test_regularized(self):
        a, b, z = 0.5, 1.5, 1.0
        assert np.isclose(
            sf.hyp1f1_regularized(a, b, z),
            sf.hyp1f1(a, b, z) / sf.gamma(b),
            rtol=1e-13,
        )

    def test_pochhammer(self):
        assert sf.pochhammer(3, 4) == 360
        assert sf.pochhammer(1, 5) == 120
        assert np.isclose(
            sf.pochhammer(0.5, 2.5),
            sf.gamma(3.0) / sf.gamma(0.5),
            rtol=1e-13,
        )
        assert sf.falling_factorial(5, 3) == 60
        # (a)_n falling = (-1)^n (-a)_n rising
        assert np.isclose(
            sf.falling_factorial(2.5, 2), sf.pochhammer(-2.5, 2), rtol=1e-13
        )

    def test_generalized_hypergeometric(self):
        # Dispatch cases reduce to scipy implementations
        assert np.isclose(
            float(np.asarray(sf.generalized_hypergeometric([1], [2], 1.0))),
            np.e - 1,
            rtol=1e-12,
        )
        # General cases against mpmath 50-digit references
        assert np.isclose(
            float(np.asarray(sf.generalized_hypergeometric([1, 2, 3], [4, 5], 0.5))),
            1.1898747542564229,
            rtol=1e-12,
        )
        assert np.isclose(
            float(
                np.asarray(
                    sf.generalized_hypergeometric([0.5, 1.5, 2.5], [3.5, 4.5], 0.9)
                )
            ),
            1.1478539933504661,
            rtol=1e-11,
        )
        assert np.isclose(
            float(np.asarray(sf.generalized_hypergeometric([2, 3], [4, 5, 6], 2.0))),
            1.105947952301559,
            rtol=1e-12,
        )


class TestQuadratureRules:
    def test_gauss_legendre_polynomial_exactness(self):
        for n in [1, 3, 5, 8]:
            x, w = ni.gauss_legendre(n)
            for d in range(2 * n):
                exact = 2 / (d + 1) if d % 2 == 0 else 0.0
                assert np.isclose(np.sum(w * x**d), exact, atol=1e-12), (n, d)

    def test_gauss_hermite_polynomial_exactness(self):
        for n in [2, 5, 10]:
            x, w = ni.gauss_hermite(n)
            for d in range(2 * n):
                exact = sp.gamma((d + 1) / 2) if d % 2 == 0 else 0.0
                assert np.isclose(np.sum(w * x**d), exact, atol=1e-10), (n, d)

    def test_gauss_laguerre_polynomial_exactness(self):
        for n in [2, 5, 8]:
            x, w = ni.gauss_laguerre(n)
            for d in range(2 * n):
                exact = sp.factorial(d)
                assert np.isclose(np.sum(w * x**d), exact, rtol=1e-9), (n, d)

    def test_gauss_chebyshev_polynomial_exactness(self):
        for n in [3, 6]:
            x1, w1 = ni.gauss_chebyshev(n, kind=1)
            x2, w2 = ni.gauss_chebyshev(n, kind=2)
            for d in range(0, 2 * n, 2):
                # int x^d / sqrt(1-x^2) dx = pi (d-1)!!/d!!
                exact1 = (
                    np.pi
                    * sp.gamma((d + 1) / 2)
                    / (sp.gamma(d / 2 + 1) * np.sqrt(np.pi))
                )
                assert np.isclose(np.sum(w1 * x1**d), exact1, atol=1e-12), (n, d)
                # int x^d sqrt(1-x^2) dx = exact1 - int x^{d+2}/sqrt: use
                # closed form pi/2 * (d-1)!!/(d+2)!!
                exact2 = (
                    np.pi
                    * sp.gamma((d + 1) / 2)
                    / (2 * sp.gamma(d / 2 + 2) * np.sqrt(np.pi))
                )
                assert np.isclose(np.sum(w2 * x2**d), exact2, atol=1e-12), (n, d)
            # Odd moments vanish
            assert np.isclose(np.sum(w1 * x1), 0.0, atol=1e-13)
            assert np.isclose(np.sum(w2 * x2), 0.0, atol=1e-13)

    def test_adaptive_integrators(self):
        assert np.isclose(ni.quad(np.sin, 0, np.pi)[0], 2.0, rtol=1e-9)
        assert np.isclose(
            ni.dblquad(lambda y, x: x * y, 0, 1, lambda x: 0, lambda x: 1)[0],
            0.25,
            rtol=1e-9,
        )
        assert np.isclose(
            ni.tplquad(
                lambda z, y, x: x * y * z,
                0,
                1,
                lambda x: 0,
                lambda x: 1,
                lambda x, y: 0,
                lambda x, y: 1,
            )[0],
            0.125,
            rtol=1e-9,
        )

    def test_fixed_quad_degree(self):
        # n-point Gauss-Legendre is exact for degree 2n-1
        result, _ = ni.fixed_quad(lambda x: x**9, 0, 1, n=5)
        assert np.isclose(result, 0.1, rtol=1e-13)

    def test_romberg(self):
        assert np.isclose(ni.romberg(np.exp, 0, 1), np.e - 1, rtol=1e-9)
        assert np.isclose(ni.romberg(np.sin, 0, np.pi, tol=1e-12), 2.0, rtol=1e-10)

    def test_sampled_integrators(self):
        x = np.linspace(0, np.pi, 201)
        assert np.isclose(ni.simpson(np.sin(x), x), 2.0, rtol=1e-7)
        assert np.isclose(ni.trapezoid(np.sin(x), x), 2.0, rtol=1e-4)

    def test_cubature_gauss_hermite_moments(self):
        pts, wts = ni.cubature_gauss_hermite(2, 4)
        assert pts.shape == (16, 2)
        # Weight normalization: int e^{-|x|^2} dx = pi in 2D
        assert np.isclose(np.sum(wts), np.pi, rtol=1e-12)
        # int x^2 y^2 e^{-x^2-y^2} = Gamma(3/2)^2
        got = np.sum(wts * pts[:, 0] ** 2 * pts[:, 1] ** 2)
        assert np.isclose(got, sp.gamma(1.5) ** 2, rtol=1e-12)

    def test_spherical_cubature_moments(self):
        pts, wts = ni.spherical_cubature(3)
        assert np.isclose(np.sum(wts), 1.0, rtol=1e-14)
        assert np.allclose(wts @ pts, 0.0, atol=1e-14)
        # Second moment must equal identity (exactness to degree 3)
        assert np.allclose((pts.T * wts) @ pts, np.eye(3), atol=1e-12)

    def test_unscented_transform_moments(self):
        for n_dim, alpha in [(2, 1.0), (3, 0.9), (5, 1e-3)]:
            pts, wm, wc = ni.unscented_transform_points(n_dim, alpha=alpha)
            assert np.isclose(np.sum(wm), 1.0, atol=1e-10)
            assert np.allclose(wm @ pts, 0.0, atol=1e-10)
            # Covariance weights reproduce the identity covariance
            assert np.allclose((pts.T * wc) @ pts, np.eye(n_dim), atol=1e-8)


class TestInterpolation:
    def test_interp1d_nodes_and_accuracy(self):
        x = np.linspace(0, 2 * np.pi, 15)
        y = np.sin(x)
        f = interp.interp1d(x, y, kind="cubic")
        assert np.allclose(f(x), y, atol=1e-12)
        assert np.isclose(float(f(1.0)), np.sin(1.0), atol=1e-3)

    def test_linear_interp_exact_on_linear(self):
        xp = np.array([0.0, 1.0, 3.0, 7.0])
        fp = 2 * xp + 1
        xq = np.array([0.5, 2.0, 6.5])
        assert np.allclose(interp.linear_interp(xq, xp, fp), 2 * xq + 1, atol=1e-14)
        assert np.isclose(
            float(interp.linear_interp(2.5, [1, 2, 3], [1, 4, 9])), 6.5, atol=1e-14
        )

    def test_cubic_spline_nodes_and_convergence(self):
        x = np.linspace(0, 1, 10)
        cs = interp.cubic_spline(x, np.exp(x))
        assert np.allclose(cs(x), np.exp(x), atol=1e-13)
        # 4th-order convergence on smooth data
        errs = []
        for npts in [10, 20, 40]:
            xx = np.linspace(0, 1, npts)
            cs2 = interp.cubic_spline(xx, np.exp(xx))
            fine = np.linspace(0, 1, 999)
            errs.append(np.max(np.abs(cs2(fine) - np.exp(fine))))
        order = np.log2(errs[0] / errs[1])
        assert order > 3.0

    def test_pchip_nodes_and_monotonicity(self):
        x = np.arange(6.0)
        y = np.array([0.0, 1.0, 1.2, 3.0, 3.1, 4.0])
        p = interp.pchip(x, y)
        assert np.allclose(p(x), y, atol=1e-13)
        fine = np.linspace(0, 5, 501)
        assert np.all(np.diff(p(fine)) >= -1e-12)

    def test_akima_nodes(self):
        x = np.linspace(0, 2 * np.pi, 15)
        y = np.sin(x)
        a = interp.akima(x, y)
        assert np.allclose(a(x), y, atol=1e-13)

    def test_interp2d_multilinear_exact(self):
        gx = np.linspace(0, 4, 5)
        gy = np.linspace(0, 4, 5)
        z = np.add.outer(2 * gx, 3 * gy) + 1
        f = interp.interp2d(gx, gy, z)
        got = f([[1.3, 2.7]])
        assert np.isclose(float(got[0]), 2 * 1.3 + 3 * 2.7 + 1, atol=1e-12)
        # Cubic reproduces the bilinear product x*y
        fc = interp.interp2d(gx, gy, np.outer(gx, gy), kind="cubic")
        assert np.isclose(float(fc([[2.5, 3.5]])[0]), 2.5 * 3.5, atol=1e-9)

    def test_interp3d_trilinear_exact(self):
        gx = np.linspace(0, 4, 5)
        gy = np.linspace(0, 4, 5)
        gz = np.linspace(0, 2, 3)
        vals = 1 + 2 * gx[:, None, None] + 3 * gy[None, :, None] + 4 * gz[None, None, :]
        f = interp.interp3d(gx, gy, gz, vals)
        got = f([[1.5, 2.5, 0.5]])
        assert np.isclose(float(got[0]), 1 + 2 * 1.5 + 3 * 2.5 + 4 * 0.5, atol=1e-12)

    def test_rbf_reproduces_nodes(self):
        rng = np.random.default_rng(0)
        pts = rng.uniform(0, 1, (30, 2))
        vals = np.sin(3 * pts[:, 0]) + pts[:, 1] ** 2
        rbf = interp.rbf_interpolate(pts, vals)
        assert np.allclose(rbf(pts), vals, atol=1e-8)

    def test_barycentric_polynomial_exact(self):
        xb = np.cos(np.linspace(0, np.pi, 8))
        pb = interp.barycentric(xb, xb**5 - 2 * xb**3 + xb)
        t = np.linspace(-1, 1, 50)
        assert np.allclose(pb(t), t**5 - 2 * t**3 + t, atol=1e-11)

    def test_krogh_hermite_data(self):
        k = interp.krogh(np.array([0.0, 0.0, 1.0, 1.0]), np.array([1.0, 0.0, 2.0, 1.0]))
        assert np.isclose(float(k(0.0)), 1.0, atol=1e-13)
        assert np.isclose(float(k(1.0)), 2.0, atol=1e-13)

    def test_spherical_interp_reproduces_nodes(self):
        rng = np.random.default_rng(1)
        lat = rng.uniform(-1.4, 1.4, 40)
        lon = rng.uniform(-np.pi, np.pi, 40)
        v = np.sin(lat) * np.cos(lon)
        si = interp.spherical_interp(lat, lon, v)
        xyz = np.column_stack(
            [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
        )
        assert np.allclose(si(xyz), v, atol=1e-8)
