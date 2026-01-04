"""
Tests for Special Mathematical Functions.

Tests cover:
- Marcum Q function
- Lambert W function
- Debye functions
- Hypergeometric functions
- Advanced Bessel functions (ratios, derivatives, zeros, Struve, Kelvin)
"""

import numpy as np

# =============================================================================
# Marcum Q Function Tests
# =============================================================================


class TestMarcumQ:
    """Tests for Marcum Q function."""

    def test_marcum_q_at_zero(self):
        """Test Q_m(a, 0) = 1."""
        from pytcl.mathematical_functions.special_functions import marcum_q

        assert np.isclose(marcum_q(3, 0), 1.0)
        assert np.isclose(marcum_q(0, 0), 1.0)

    def test_marcum_q_with_zero_a(self):
        """Test Q_m(0, b) uses incomplete gamma."""
        from scipy.special import gammaincc

        from pytcl.mathematical_functions.special_functions import marcum_q

        b = 2.0
        m = 2
        expected = gammaincc(m, 0.5 * b**2)
        result = marcum_q(0, b, m=m)
        assert np.isclose(result, expected, rtol=1e-10)

    def test_marcum_q_standard(self):
        """Test standard Marcum Q values."""
        from pytcl.mathematical_functions.special_functions import marcum_q

        # Known value from tables
        result = marcum_q(3, 4)
        assert 0.1 < result < 0.3  # Reasonable range

    def test_marcum_q1(self):
        """Test marcum_q1 convenience function."""
        from pytcl.mathematical_functions.special_functions import marcum_q, marcum_q1

        a, b = 2.0, 3.0
        assert np.isclose(marcum_q1(a, b), marcum_q(a, b, m=1))

    def test_log_marcum_q(self):
        """Test log Marcum Q for precision."""
        from pytcl.mathematical_functions.special_functions import (
            log_marcum_q,
            marcum_q,
        )

        a, b = 1.0, 2.0
        result = log_marcum_q(a, b)
        expected = np.log(marcum_q(a, b))
        assert np.isclose(result, expected, rtol=1e-6)

    def test_marcum_q_inv(self):
        """Test inverse Marcum Q function."""
        from pytcl.mathematical_functions.special_functions import (
            marcum_q,
            marcum_q_inv,
        )

        a = 3.0
        q = 0.5
        b = marcum_q_inv(a, q)
        recovered = marcum_q(a, b)
        assert np.isclose(recovered, q, rtol=1e-6)

    def test_nuttall_q(self):
        """Test Nuttall Q (complementary Marcum Q)."""
        from pytcl.mathematical_functions.special_functions import marcum_q, nuttall_q

        a, b = 2.0, 3.0
        assert np.isclose(nuttall_q(a, b), 1 - marcum_q(a, b))

    def test_swerling_case_0(self):
        """Test Swerling case 0 (non-fluctuating)."""
        from pytcl.mathematical_functions.special_functions import (
            swerling_detection_probability,
        )

        snr = 10  # Linear, not dB
        pfa = 1e-6
        pd = swerling_detection_probability(snr, pfa, n_pulses=1, swerling_case=0)
        assert 0 < pd < 1


# =============================================================================
# Lambert W Function Tests
# =============================================================================


class TestLambertW:
    """Tests for Lambert W function."""

    def test_lambert_w_at_zero(self):
        """Test W(0) = 0."""
        from pytcl.mathematical_functions.special_functions import lambert_w

        result = lambert_w(0)
        assert np.isclose(np.real(result), 0.0)

    def test_lambert_w_at_e(self):
        """Test W(e) = 1."""
        from pytcl.mathematical_functions.special_functions import lambert_w

        result = lambert_w(np.e)
        assert np.isclose(np.real(result), 1.0)

    def test_lambert_w_branch_point(self):
        """Test W near branch point W(-1/e) ≈ -1."""
        from pytcl.mathematical_functions.special_functions import lambert_w

        # Slightly above branch point to avoid numerical issues
        result = lambert_w(-np.exp(-1) + 1e-10)
        assert np.isclose(np.real(result), -1.0, atol=0.01)

    def test_lambert_w_definition(self):
        """Test W(x) * exp(W(x)) = x."""
        from pytcl.mathematical_functions.special_functions import lambert_w

        x = 2.5
        w = lambert_w(x)
        recovered = w * np.exp(w)
        assert np.isclose(np.real(recovered), x)

    def test_lambert_w_real(self):
        """Test real-valued Lambert W."""
        from pytcl.mathematical_functions.special_functions import lambert_w_real

        result = lambert_w_real(1)
        assert isinstance(result, np.floating)
        assert np.isclose(result, 0.5671432904097838)

    def test_omega_constant(self):
        """Test omega constant."""
        from pytcl.mathematical_functions.special_functions import omega_constant

        omega = omega_constant()
        # Omega * exp(Omega) = 1
        assert np.isclose(omega * np.exp(omega), 1.0)
        assert np.isclose(omega, 0.5671432904097838)

    def test_wright_omega(self):
        """Test Wright omega function."""
        from pytcl.mathematical_functions.special_functions import wright_omega

        # omega(z) + log(omega(z)) = z
        z = 2.0
        w = wright_omega(z)
        recovered = w + np.log(w)
        assert np.isclose(np.real(recovered), z, rtol=1e-6)

    def test_solve_exponential_equation(self):
        """Test solving a*x*exp(b*x) = c."""
        from pytcl.mathematical_functions.special_functions import (
            solve_exponential_equation,
        )

        # x * exp(x) = e implies x = 1
        x = solve_exponential_equation(1, 1, np.e)
        assert np.isclose(np.real(x), 1.0)

    def test_time_delay_equation(self):
        """Test delay differential equation characteristic equation."""
        from pytcl.mathematical_functions.special_functions import time_delay_equation

        a, tau = 1.0, 1.0
        s = time_delay_equation(a, tau)
        # s + a*exp(-s*tau) should be approximately 0
        residual = s + a * np.exp(-s * tau)
        assert np.abs(residual) < 1e-10


# =============================================================================
# Debye Function Tests
# =============================================================================


class TestDebye:
    """Tests for Debye functions."""

    def test_debye_at_zero(self):
        """Test D_n(0) = 1."""
        from pytcl.mathematical_functions.special_functions import debye

        assert np.isclose(debye(1, np.array([0.0]))[0], 1.0)
        assert np.isclose(debye(3, np.array([0.0]))[0], 1.0)

    def test_debye_order_3(self):
        """Test D_3 for known values."""
        from pytcl.mathematical_functions.special_functions import debye_3

        # D_3(1) ≈ 0.674
        result = debye_3(np.array([1.0]))
        assert np.isclose(result[0], 0.674, atol=0.01)

    def test_debye_decreasing(self):
        """Test Debye function is decreasing for x > 0."""
        from pytcl.mathematical_functions.special_functions import debye_3

        x = np.array([0.5, 1.0, 2.0, 5.0])
        y = debye_3(x)
        assert np.all(np.diff(y) < 0)

    def test_debye_heat_capacity_limits(self):
        """Test heat capacity limits."""
        from pytcl.mathematical_functions.special_functions import debye_heat_capacity

        # High T: approaches 1 (classical limit)
        high_t = debye_heat_capacity(np.array([1000.0]), 100)
        assert np.isclose(high_t[0], 1.0, atol=0.05)

        # Low T: much smaller
        low_t = debye_heat_capacity(np.array([10.0]), 428)
        assert low_t[0] < 0.1

    def test_debye_entropy(self):
        """Test Debye entropy."""
        from pytcl.mathematical_functions.special_functions import debye_entropy

        # Entropy should be positive for positive temperatures
        s = debye_entropy(np.array([300.0]), 428)
        assert s[0] > 0


# =============================================================================
# Hypergeometric Function Tests
# =============================================================================


class TestHypergeometric:
    """Tests for hypergeometric functions."""

    def test_hyp0f1_bessel_relation(self):
        """Test 0F1 relation to Bessel functions."""
        from scipy.special import gamma as gamma_func
        from scipy.special import jv

        from pytcl.mathematical_functions.special_functions import hyp0f1

        # J_n(x) = (x/2)^n / Gamma(n+1) * 0F1(n+1; -x^2/4)
        n, x = 2, 3.0
        hyp_val = hyp0f1(n + 1, -(x**2) / 4)
        jn_from_hyp = (x / 2) ** n / gamma_func(n + 1) * hyp_val
        jn_direct = jv(n, x)
        assert np.isclose(jn_from_hyp, jn_direct, rtol=1e-10)

    def test_hyp1f1_exponential(self):
        """Test 1F1(a; a; z) = exp(z)."""
        from pytcl.mathematical_functions.special_functions import hyp1f1

        z = 2.0
        result = hyp1f1(1.5, 1.5, z)
        assert np.isclose(result, np.exp(z))

    def test_hyp1f1_at_zero(self):
        """Test 1F1(a; b; 0) = 1."""
        from pytcl.mathematical_functions.special_functions import hyp1f1

        assert np.isclose(hyp1f1(2, 3, 0), 1.0)

    def test_hyp2f1_log(self):
        """Test 2F1(1, 1; 2; z) = -log(1-z)/z."""
        from pytcl.mathematical_functions.special_functions import hyp2f1

        z = 0.5
        result = hyp2f1(1, 1, 2, z)
        expected = -np.log(1 - z) / z
        assert np.isclose(result, expected)

    def test_hyp2f1_power(self):
        """Test 2F1(a, b; b; z) = (1-z)^(-a)."""
        from pytcl.mathematical_functions.special_functions import hyp2f1

        a, b, z = 2.0, 3.0, 0.3
        result = hyp2f1(a, b, b, z)
        expected = (1 - z) ** (-a)
        assert np.isclose(result, expected)

    def test_hyperu(self):
        """Test Tricomi function U."""
        from pytcl.mathematical_functions.special_functions import hyperu

        # U(a, b, z) for simple cases
        result = hyperu(1, 1, 1)
        assert result > 0  # Should be positive for positive args

    def test_pochhammer(self):
        """Test Pochhammer symbol (rising factorial)."""
        from pytcl.mathematical_functions.special_functions import pochhammer

        # (1)_n = n!
        assert np.isclose(pochhammer(1, 5), 120)  # 5!

        # (a)_0 = 1
        assert np.isclose(pochhammer(3.5, 0), 1)

        # (3)_4 = 3*4*5*6 = 360
        assert np.isclose(pochhammer(3, 4), 360)

    def test_falling_factorial(self):
        """Test falling factorial."""
        from pytcl.mathematical_functions.special_functions import falling_factorial

        # (5)_3 falling = 5*4*3 = 60
        assert np.isclose(falling_factorial(5, 3), 60)


# =============================================================================
# Advanced Bessel Function Tests
# =============================================================================


class TestAdvancedBessel:
    """Tests for advanced Bessel functions."""

    def test_bessel_ratio(self):
        """Test Bessel function ratio."""
        from scipy.special import jv

        from pytcl.mathematical_functions.special_functions import bessel_ratio

        n, x = 1, 2.0
        ratio = bessel_ratio(n, x, kind="j")
        expected = jv(n + 1, x) / jv(n, x)
        assert np.isclose(ratio, expected)

    def test_bessel_ratio_modified(self):
        """Test modified Bessel function ratio."""
        from scipy.special import iv

        from pytcl.mathematical_functions.special_functions import bessel_ratio

        n, x = 0, 1.0
        ratio = bessel_ratio(n, x, kind="i")
        expected = iv(n + 1, x) / iv(n, x)
        assert np.isclose(ratio, expected)

    def test_bessel_deriv_j0(self):
        """Test J_0'(x) = -J_1(x)."""
        from scipy.special import jv

        from pytcl.mathematical_functions.special_functions import bessel_deriv

        x = 2.0
        deriv = bessel_deriv(0, x, kind="j")
        expected = -jv(1, x)
        assert np.isclose(deriv, expected)

    def test_bessel_deriv_kinds(self):
        """Test derivatives for all Bessel kinds."""
        from pytcl.mathematical_functions.special_functions import bessel_deriv

        x = 1.5
        # All should return finite values
        for kind in ["j", "y", "i", "k"]:
            result = bessel_deriv(1, x, kind=kind)
            assert np.isfinite(result)

    def test_struve_h(self):
        """Test Struve function H_n."""
        from pytcl.mathematical_functions.special_functions import struve_h

        # H_0(0) = 0
        assert np.isclose(struve_h(0, 0), 0)

        # H_n should be finite for positive x
        result = struve_h(0, 1)
        assert np.isfinite(result)

    def test_struve_l(self):
        """Test modified Struve function L_n."""
        from pytcl.mathematical_functions.special_functions import struve_l

        result = struve_l(0, 1)
        assert np.isfinite(result)

    def test_bessel_zeros(self):
        """Test zeros of Bessel functions."""
        from pytcl.mathematical_functions.special_functions import bessel_zeros, besselj

        # First 3 zeros of J_0
        zeros = bessel_zeros(0, 3, kind="j")
        assert len(zeros) == 3

        # Verify they are actually zeros
        for z in zeros:
            assert np.abs(besselj(0, z)) < 1e-10

        # Known approximate values
        assert np.isclose(zeros[0], 2.4048, atol=0.001)
        assert np.isclose(zeros[1], 5.5201, atol=0.001)

    def test_bessel_zeros_derivative(self):
        """Test zeros of Bessel function derivatives."""
        from pytcl.mathematical_functions.special_functions import (
            bessel_deriv,
            bessel_zeros,
        )

        zeros = bessel_zeros(0, 2, kind="jp")
        assert len(zeros) == 2

        # Verify they are zeros of the derivative
        for z in zeros:
            assert np.abs(bessel_deriv(0, z, kind="j")) < 1e-10

    def test_kelvin(self):
        """Test Kelvin functions."""
        from pytcl.mathematical_functions.special_functions import kelvin

        ber, bei, ker, kei = kelvin(1.0)

        # ber and bei should be finite for positive x
        assert np.isfinite(ber)
        assert np.isfinite(bei)
        # ker and kei should also be finite for x > 0
        assert np.isfinite(ker)
        assert np.isfinite(kei)

        # ber(1) ≈ 0.9844, bei(1) ≈ 0.2496 (known values)
        assert np.isclose(ber, 0.9844, atol=0.001)
        assert np.isclose(bei, 0.2496, atol=0.001)


# =============================================================================
# Integration Tests
# =============================================================================


class TestSpecialFunctionsIntegration:
    """Integration tests combining multiple special functions."""

    def test_marcum_q_with_bessel(self):
        """Marcum Q involves modified Bessel functions internally."""
        from pytcl.mathematical_functions.special_functions import marcum_q

        # Just verify it works for arrays
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([2.0, 3.0, 4.0])
        result = marcum_q(a, b)
        assert result.shape == (3,)
        assert np.all((result >= 0) & (result <= 1))

    def test_hypergeometric_special_cases(self):
        """Test hypergeometric function special cases."""
        from pytcl.mathematical_functions.special_functions import hyp2f1

        # Array inputs
        z = np.array([0.1, 0.2, 0.3])
        result = hyp2f1(1, 1, 2, z)
        expected = -np.log(1 - z) / z
        assert np.allclose(result, expected)

    def test_all_exports(self):
        """Test all functions are properly exported."""
        from pytcl.mathematical_functions.special_functions import (  # Marcum Q
            bessel_deriv,
            bessel_ratio,
            bessel_zeros,
            debye,
            debye_1,
            debye_2,
            debye_3,
            debye_4,
            debye_entropy,
            debye_heat_capacity,
            falling_factorial,
            generalized_hypergeometric,
            hyp0f1,
            hyp1f1,
            hyp1f1_regularized,
            hyp2f1,
            hyperu,
            kelvin,
            lambert_w,
            lambert_w_real,
            log_marcum_q,
            marcum_q,
            marcum_q1,
            marcum_q_inv,
            nuttall_q,
            omega_constant,
            pochhammer,
            solve_exponential_equation,
            struve_h,
            struve_l,
            swerling_detection_probability,
            time_delay_equation,
            wright_omega,
        )

        # All should be callable
        funcs = [
            marcum_q,
            marcum_q1,
            marcum_q_inv,
            log_marcum_q,
            nuttall_q,
            swerling_detection_probability,
            lambert_w,
            lambert_w_real,
            wright_omega,
            omega_constant,
            solve_exponential_equation,
            time_delay_equation,
            debye,
            debye_1,
            debye_2,
            debye_3,
            debye_4,
            debye_entropy,
            debye_heat_capacity,
            hyp0f1,
            hyp1f1,
            hyp1f1_regularized,
            hyp2f1,
            hyperu,
            generalized_hypergeometric,
            pochhammer,
            falling_factorial,
            bessel_zeros,
            bessel_deriv,
            bessel_ratio,
            struve_h,
            struve_l,
            kelvin,
        ]
        for func in funcs:
            assert callable(func)
