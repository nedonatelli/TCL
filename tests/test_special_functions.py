"""
Tests for Special Mathematical Functions.

Tests cover:
- Lambert W function
- Advanced Bessel functions (ratios, derivatives, zeros, Struve, Kelvin)
- Integration tests combining multiple special functions
"""

import numpy as np

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
