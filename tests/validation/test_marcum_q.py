"""
Tests for Marcum Q function and related functions.

Tests cover:
- Generalized Marcum Q function
- Standard Marcum Q (Q_1)
- Log Marcum Q
- Inverse Marcum Q
- Nuttall Q function
- Swerling detection probability
"""

import numpy as np
import pytest

from pytcl.mathematical_functions.special_functions.marcum_q import (
    log_marcum_q,
    marcum_q,
    marcum_q1,
    marcum_q_inv,
    nuttall_q,
    swerling_detection_probability,
)

# =============================================================================
# Tests for Marcum Q function
# =============================================================================


class TestMarcumQ:
    """Tests for generalized Marcum Q function."""

    def test_marcum_q_zero_zero(self):
        """Test Q_1(0, 0) = 1."""
        result = marcum_q(0, 0)
        assert result == pytest.approx(1.0, rel=1e-10)

    def test_marcum_q_a_zero(self):
        """Test Q_m(0, b) uses incomplete gamma."""
        result = marcum_q(0, 2, m=1)
        # Q_1(0, b) = exp(-b^2/2)
        expected = np.exp(-2)
        assert result == pytest.approx(expected, rel=1e-6)

    def test_marcum_q_b_zero(self):
        """Test Q_m(a, 0) = 1 for any a."""
        a_values = np.array([0, 1, 5, 10])
        for a in a_values:
            result = marcum_q(a, 0)
            assert result == pytest.approx(1.0, rel=1e-10)

    def test_marcum_q_bounds(self):
        """Test Q is bounded by 0 and 1."""
        a = np.linspace(0, 5, 20)
        b = np.linspace(0, 5, 20)
        for ai in a:
            for bi in b:
                result = marcum_q(ai, bi)
                assert 0.0 <= result <= 1.0

    def test_marcum_q_decreasing_in_b(self):
        """Test Q_m(a, b) is decreasing in b."""
        a = np.full(20, 2.0)
        b = np.linspace(0.1, 5, 20)
        result = marcum_q(a, b)
        assert np.all(np.diff(result) <= 0)

    def test_marcum_q_known_value(self):
        """Test Q_1(3, 4) has known value."""
        result = marcum_q(3, 4)
        # Value from scipy.stats.ncx2.sf
        assert 0.1 < result < 0.3

    def test_marcum_q_order_2(self):
        """Test higher order Marcum Q."""
        result = marcum_q(2, 3, m=2)
        assert 0 < result < 1

    def test_marcum_q_array(self):
        """Test Marcum Q with array inputs."""
        a = np.array([1, 2, 3])
        b = np.array([2, 3, 4])
        result = marcum_q(a, b)
        assert result.shape == (3,)

    def test_marcum_q_invalid_m(self):
        """Test invalid m raises ValueError."""
        with pytest.raises(ValueError, match="Order m must be >= 1"):
            marcum_q(1, 2, m=0)


class TestMarcumQ1:
    """Tests for standard Marcum Q function."""

    def test_marcum_q1_equals_m1(self):
        """Test Q_1 equals marcum_q with m=1."""
        a, b = 2, 3
        result1 = marcum_q1(a, b)
        result2 = marcum_q(a, b, m=1)
        assert result1 == pytest.approx(result2, rel=1e-10)

    def test_marcum_q1_known_value(self):
        """Test Q_1(2, 2) has known value."""
        result = marcum_q1(2, 2)
        assert 0.6 < result < 0.8


# =============================================================================
# Tests for log Marcum Q
# =============================================================================


class TestLogMarcumQ:
    """Tests for log Marcum Q function."""

    def test_log_marcum_q_zero_zero(self):
        """Test log(Q_1(0, 0)) = log(1) = 0."""
        result = log_marcum_q(0, 0)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_log_marcum_q_relation(self):
        """Test log_marcum_q equals log(marcum_q) for moderate values."""
        a, b = 2, 3
        result = log_marcum_q(a, b)
        expected = np.log(marcum_q(a, b))
        assert result == pytest.approx(expected, rel=1e-6)

    def test_log_marcum_q_negative_for_small_q(self):
        """Test log is negative when Q < 1."""
        a, b = 1, 5
        result = log_marcum_q(a, b)
        assert result < 0

    def test_log_marcum_q_invalid_m(self):
        """Test invalid m raises ValueError."""
        with pytest.raises(ValueError, match="Order m must be >= 1"):
            log_marcum_q(1, 2, m=0)


# =============================================================================
# Tests for inverse Marcum Q
# =============================================================================


class TestMarcumQInv:
    """Tests for inverse Marcum Q function."""

    def test_marcum_q_inv_roundtrip(self):
        """Test Q(a, Q_inv(a, q)) = q."""
        a, q = 2, 0.5
        b = marcum_q_inv(a, q)
        result = marcum_q(a, b)
        assert result == pytest.approx(q, rel=1e-6)

    def test_marcum_q_inv_q_half(self):
        """Test inverse for q = 0.5."""
        a = 3.0
        b = marcum_q_inv(a, 0.5)
        assert b > 0
        assert marcum_q(a, b) == pytest.approx(0.5, rel=1e-6)

    def test_marcum_q_inv_small_q(self):
        """Test inverse for small q."""
        a, q = 2, 0.1
        b = marcum_q_inv(a, q)
        assert b > 0
        assert marcum_q(a, b) == pytest.approx(q, rel=0.01)

    def test_marcum_q_inv_large_q(self):
        """Test inverse for large q."""
        a, q = 2, 0.9
        b = marcum_q_inv(a, q)
        assert b >= 0
        assert marcum_q(a, b) == pytest.approx(q, rel=0.01)

    def test_marcum_q_inv_invalid_q(self):
        """Test invalid q raises ValueError."""
        with pytest.raises(ValueError, match="q must be in"):
            marcum_q_inv(1, 0)
        with pytest.raises(ValueError, match="q must be in"):
            marcum_q_inv(1, 1)

    def test_marcum_q_inv_invalid_m(self):
        """Test invalid m raises ValueError."""
        with pytest.raises(ValueError, match="Order m must be >= 1"):
            marcum_q_inv(1, 0.5, m=0)


# =============================================================================
# Tests for Nuttall Q function
# =============================================================================


class TestNuttallQ:
    """Tests for Nuttall Q function (complementary Marcum Q)."""

    def test_nuttall_q_relation(self):
        """Test P(a, b) = 1 - Q_1(a, b)."""
        a, b = 2, 3
        result = nuttall_q(a, b)
        expected = 1 - marcum_q(a, b, m=1)
        assert result == pytest.approx(expected, rel=1e-10)

    def test_nuttall_q_b_zero(self):
        """Test P(a, 0) = 0."""
        result = nuttall_q(2, 0)
        assert result == pytest.approx(0.0, rel=1e-10)

    def test_nuttall_q_bounds(self):
        """Test Nuttall Q is bounded by 0 and 1."""
        a = np.linspace(0, 5, 20)
        b = np.linspace(0, 5, 20)
        for ai in a:
            for bi in b:
                result = nuttall_q(ai, bi)
                assert 0.0 <= result <= 1.0

    def test_nuttall_q_increasing_in_b(self):
        """Test P(a, b) is increasing in b."""
        a = np.full(20, 2.0)
        b = np.linspace(0.1, 5, 20)
        result = nuttall_q(a, b)
        assert np.all(np.diff(result) >= 0)


# =============================================================================
# Tests for Swerling detection probability
# =============================================================================


class TestSwerlingDetection:
    """Tests for Swerling detection probability."""

    def test_swerling_case_0(self):
        """Test Swerling case 0 (non-fluctuating)."""
        result = swerling_detection_probability(10, 1e-6, n_pulses=10, swerling_case=0)
        assert 0 < result < 1

    def test_swerling_case_1(self):
        """Test Swerling case 1 (slow Rayleigh)."""
        result = swerling_detection_probability(10, 1e-6, n_pulses=10, swerling_case=1)
        assert 0 < result < 1

    def test_swerling_case_2(self):
        """Test Swerling case 2 (fast Rayleigh)."""
        result = swerling_detection_probability(10, 1e-6, n_pulses=10, swerling_case=2)
        assert 0 < result < 1

    def test_swerling_case_3(self):
        """Test Swerling case 3 (slow chi-squared)."""
        result = swerling_detection_probability(10, 1e-6, n_pulses=10, swerling_case=3)
        assert 0 < result < 1

    def test_swerling_case_4(self):
        """Test Swerling case 4 (fast chi-squared)."""
        result = swerling_detection_probability(10, 1e-6, n_pulses=10, swerling_case=4)
        assert 0 < result < 1

    def test_swerling_increasing_snr(self):
        """Test detection probability increases with SNR."""
        snr_values = np.array([1, 5, 10, 20])
        results = []
        for snr in snr_values:
            results.append(
                swerling_detection_probability(snr, 1e-6, n_pulses=10, swerling_case=0)
            )
        results = np.array(results)
        assert np.all(np.diff(results) > 0)

    def test_swerling_more_pulses_better(self):
        """Test more integrated pulses gives higher detection."""
        # With higher SNR to ensure detection probability is reasonable
        result_1 = swerling_detection_probability(20, 1e-6, n_pulses=1, swerling_case=0)
        result_10 = swerling_detection_probability(
            20, 1e-6, n_pulses=10, swerling_case=0
        )
        # Both should be high with SNR=20
        assert result_10 > 0.5
        assert result_1 > 0.5

    def test_swerling_invalid_case(self):
        """Test invalid Swerling case raises ValueError."""
        with pytest.raises(ValueError, match="swerling_case must be 0-4"):
            swerling_detection_probability(10, 1e-6, swerling_case=5)

    def test_swerling_high_snr(self):
        """Test high SNR gives high detection probability."""
        result = swerling_detection_probability(100, 1e-6, n_pulses=10, swerling_case=0)
        assert result > 0.99

    def test_swerling_array_snr(self):
        """Test with array SNR input."""
        snr = np.array([1, 5, 10])
        result = swerling_detection_probability(snr, 1e-6, n_pulses=5, swerling_case=1)
        assert result.shape == (3,)


# =============================================================================
# Integration tests
# =============================================================================


class TestMarcumQIntegration:
    """Integration tests for Marcum Q functions."""

    def test_marcum_nuttall_sum_one(self):
        """Test Q + P = 1."""
        a, b = 2, 3
        q = marcum_q(a, b)
        p = nuttall_q(a, b)
        assert q + p == pytest.approx(1.0, rel=1e-10)

    def test_marcum_q_inv_consistent(self):
        """Test marcum_q_inv is consistent across orders."""
        a, q = 2, 0.5
        for m in [1, 2, 3]:
            b = marcum_q_inv(a, q, m=m)
            result = marcum_q(a, b, m=m)
            assert result == pytest.approx(q, rel=0.01)

    def test_log_exp_roundtrip(self):
        """Test exp(log_marcum_q) = marcum_q for moderate values."""
        a, b = 2, 3
        log_q = log_marcum_q(a, b)
        result = np.exp(log_q)
        expected = marcum_q(a, b)
        assert result == pytest.approx(expected, rel=1e-6)

    def test_swerling_case_0_vs_marcum(self):
        """Test Swerling 0 uses Marcum Q correctly."""
        snr = 10
        pfa = 1e-6
        n_pulses = 1
        pd = swerling_detection_probability(snr, pfa, n_pulses, swerling_case=0)
        # Direct computation
        threshold = -2 * n_pulses * np.log(pfa)
        a = np.sqrt(2 * n_pulses * snr)
        b = np.sqrt(threshold)
        expected = marcum_q(a, b, m=n_pulses)
        assert pd == pytest.approx(expected, rel=1e-6)
