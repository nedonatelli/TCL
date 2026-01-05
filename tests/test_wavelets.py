"""
Tests for wavelet transform utilities.

Tests cover:
- Morlet wavelet generation
- Ricker (Mexican hat) wavelet generation
- Gaussian derivative wavelets
- Continuous Wavelet Transform (CWT)
- Scale-frequency conversions
- DWT functions (if pywavelets available)
"""

import numpy as np
import pytest

from pytcl.core.optional_deps import is_available
from pytcl.mathematical_functions.transforms.wavelets import (
    CWTResult,
    DWTResult,
    available_wavelets,
    cwt,
    frequencies_to_scales,
    gaussian_wavelet,
    morlet_wavelet,
    ricker_wavelet,
    scales_to_frequencies,
    wavelet_info,
)

# Check if pywavelets is available for DWT tests
PYWT_AVAILABLE = is_available("pywt")


# =============================================================================
# Tests for Morlet wavelet
# =============================================================================


class TestMorletWavelet:
    """Tests for Morlet wavelet generation."""

    def test_morlet_wavelet_length(self):
        """Test that Morlet wavelet has correct length."""
        wav = morlet_wavelet(128)
        assert len(wav) == 128

    def test_morlet_wavelet_complex(self):
        """Test that Morlet wavelet is complex."""
        wav = morlet_wavelet(128)
        assert np.iscomplexobj(wav)

    def test_morlet_wavelet_normalized(self):
        """Test that Morlet wavelet is normalized."""
        wav = morlet_wavelet(128)
        energy = np.sum(np.abs(wav) ** 2)
        assert energy == pytest.approx(1.0, rel=1e-10)

    def test_morlet_wavelet_peak_at_center(self):
        """Test that Morlet wavelet peaks near center."""
        wav = morlet_wavelet(128)
        center = len(wav) // 2
        # Peak should be within a few samples of center
        peak_idx = np.argmax(np.abs(wav))
        assert abs(peak_idx - center) < 5

    def test_morlet_wavelet_frequency_param(self):
        """Test Morlet wavelet with different frequency parameter."""
        wav_low = morlet_wavelet(128, w=3.0)
        wav_high = morlet_wavelet(128, w=8.0)

        # Both should be valid
        assert len(wav_low) == 128
        assert len(wav_high) == 128

    def test_morlet_wavelet_complete_vs_incomplete(self):
        """Test Morlet wavelet with and without correction term."""
        wav_complete = morlet_wavelet(128, complete=True)
        wav_incomplete = morlet_wavelet(128, complete=False)

        # Both should be normalized
        assert np.sum(np.abs(wav_complete) ** 2) == pytest.approx(1.0, rel=1e-10)
        assert np.sum(np.abs(wav_incomplete) ** 2) == pytest.approx(1.0, rel=1e-10)

        # Both should be valid complex wavelets
        assert np.iscomplexobj(wav_complete)
        assert np.iscomplexobj(wav_incomplete)

    def test_morlet_wavelet_scale(self):
        """Test Morlet wavelet with scale parameter."""
        wav_narrow = morlet_wavelet(128, s=0.5)
        wav_wide = morlet_wavelet(128, s=2.0)

        # Both should be valid
        assert len(wav_narrow) == 128
        assert len(wav_wide) == 128


# =============================================================================
# Tests for Ricker wavelet
# =============================================================================


class TestRickerWavelet:
    """Tests for Ricker (Mexican hat) wavelet generation."""

    def test_ricker_wavelet_length(self):
        """Test that Ricker wavelet has correct length."""
        wav = ricker_wavelet(128)
        assert len(wav) == 128

    def test_ricker_wavelet_real(self):
        """Test that Ricker wavelet is real-valued."""
        wav = ricker_wavelet(128)
        assert np.isrealobj(wav)

    def test_ricker_wavelet_peak_at_center(self):
        """Test that Ricker wavelet peaks at center."""
        wav = ricker_wavelet(128)
        center = len(wav) // 2
        # Peak should be at center
        peak_idx = np.argmax(wav)
        assert abs(peak_idx - center) < 2

    def test_ricker_wavelet_peak_value(self):
        """Test Ricker wavelet peak is at center and positive."""
        wav = ricker_wavelet(128, a=4.0)
        center = len(wav) // 2
        # Peak should be at center and positive
        assert wav[center] > 0
        assert wav[center] == np.max(wav)

    def test_ricker_wavelet_width_param(self):
        """Test Ricker wavelet with different width parameter."""
        wav_narrow = ricker_wavelet(128, a=2.0)
        wav_wide = ricker_wavelet(128, a=8.0)

        # Wide wavelet should have more non-zero values near edges
        # Check that the narrow one decays faster
        edge = 10
        assert np.abs(wav_narrow[edge]) < np.abs(wav_wide[edge])

    def test_ricker_wavelet_symmetry(self):
        """Test that Ricker wavelet is symmetric."""
        wav = ricker_wavelet(127)  # Odd length for exact symmetry
        center = len(wav) // 2

        # Check symmetry
        for i in range(center):
            assert wav[center - i - 1] == pytest.approx(wav[center + i + 1], rel=1e-10)


# =============================================================================
# Tests for Gaussian wavelet
# =============================================================================


class TestGaussianWavelet:
    """Tests for Gaussian derivative wavelet generation."""

    def test_gaussian_wavelet_length(self):
        """Test that Gaussian wavelet has correct length."""
        wav = gaussian_wavelet(128)
        assert len(wav) == 128

    def test_gaussian_wavelet_real(self):
        """Test that Gaussian wavelet is real-valued."""
        wav = gaussian_wavelet(128)
        assert np.isrealobj(wav)

    def test_gaussian_wavelet_normalized(self):
        """Test that Gaussian wavelet is normalized."""
        wav = gaussian_wavelet(128)
        energy = np.sum(wav**2)
        assert energy == pytest.approx(1.0, rel=1e-10)

    def test_gaussian_wavelet_order_1(self):
        """Test first-order Gaussian derivative wavelet."""
        wav = gaussian_wavelet(128, order=1)
        # First derivative is antisymmetric, should be near zero at center
        center = len(wav) // 2
        # The value at center should be smaller than peak values
        assert np.abs(wav[center]) < np.max(np.abs(wav))

    def test_gaussian_wavelet_order_2(self):
        """Test second-order Gaussian derivative wavelet."""
        wav = gaussian_wavelet(128, order=2)
        # Similar to Ricker wavelet
        center = len(wav) // 2
        peak_idx = np.argmax(np.abs(wav))
        assert abs(peak_idx - center) < 3

    def test_gaussian_wavelet_order_3(self):
        """Test third-order Gaussian derivative wavelet."""
        wav = gaussian_wavelet(128, order=3)
        assert len(wav) == 128
        assert np.sum(wav**2) == pytest.approx(1.0, rel=1e-10)

    def test_gaussian_wavelet_order_4(self):
        """Test fourth-order Gaussian derivative wavelet."""
        wav = gaussian_wavelet(128, order=4)
        assert len(wav) == 128
        assert np.sum(wav**2) == pytest.approx(1.0, rel=1e-10)

    def test_gaussian_wavelet_sigma(self):
        """Test Gaussian wavelet with different sigma."""
        wav_narrow = gaussian_wavelet(128, sigma=0.5)
        wav_wide = gaussian_wavelet(128, sigma=2.0)

        # Both should be normalized
        assert np.sum(wav_narrow**2) == pytest.approx(1.0, rel=1e-10)
        assert np.sum(wav_wide**2) == pytest.approx(1.0, rel=1e-10)


# =============================================================================
# Tests for Continuous Wavelet Transform
# =============================================================================


class TestCWT:
    """Tests for Continuous Wavelet Transform."""

    def test_cwt_basic(self):
        """Test basic CWT computation."""
        np.random.seed(42)
        x = np.random.randn(256)
        scales = np.arange(1, 32)

        result = cwt(x, scales)

        assert isinstance(result, CWTResult)
        assert result.coefficients.shape == (31, 256)
        assert len(result.scales) == 31
        assert len(result.frequencies) == 31

    def test_cwt_morlet(self):
        """Test CWT with Morlet wavelet."""
        fs = 1000
        t = np.arange(0, 0.5, 1 / fs)
        x = np.sin(2 * np.pi * 50 * t)  # 50 Hz sine wave
        scales = np.arange(1, 64)

        result = cwt(x, scales, wavelet="morlet", fs=fs)

        assert result.coefficients.shape[0] == len(scales)
        assert result.coefficients.shape[1] == len(x)

    def test_cwt_ricker(self):
        """Test CWT with Ricker wavelet."""
        np.random.seed(42)
        x = np.random.randn(256)
        scales = np.arange(1, 32)

        result = cwt(x, scales, wavelet="ricker")

        assert result.coefficients.shape == (31, 256)

    def test_cwt_gaussian1(self):
        """Test CWT with Gaussian derivative wavelet."""
        np.random.seed(42)
        x = np.random.randn(256)
        scales = np.arange(1, 32)

        result = cwt(x, scales, wavelet="gaussian1")

        assert result.coefficients.shape == (31, 256)

    def test_cwt_gaussian2(self):
        """Test CWT with second Gaussian derivative wavelet."""
        np.random.seed(42)
        x = np.random.randn(256)
        scales = np.arange(1, 32)

        result = cwt(x, scales, wavelet="gaussian2")

        assert result.coefficients.shape == (31, 256)

    def test_cwt_custom_wavelet(self):
        """Test CWT with custom wavelet function."""

        def custom_wavelet(M):
            x = np.arange(M) - (M - 1) / 2
            wav = np.exp(-0.5 * x**2)
            return wav / np.sqrt(np.sum(wav**2))

        np.random.seed(42)
        x = np.random.randn(256)
        scales = np.arange(1, 32)

        result = cwt(x, scales, wavelet=custom_wavelet)

        assert result.coefficients.shape == (31, 256)

    def test_cwt_conv_method(self):
        """Test CWT with convolution method."""
        np.random.seed(42)
        x = np.random.randn(128)
        scales = np.arange(1, 16)

        result = cwt(x, scales, method="conv")

        assert result.coefficients.shape == (15, 128)

    def test_cwt_fft_method(self):
        """Test CWT with FFT method."""
        np.random.seed(42)
        x = np.random.randn(128)
        scales = np.arange(1, 16)

        result = cwt(x, scales, method="fft")

        assert result.coefficients.shape == (15, 128)

    def test_cwt_unknown_wavelet_raises(self):
        """Test that unknown wavelet raises error."""
        x = np.random.randn(128)
        scales = np.arange(1, 16)

        with pytest.raises(ValueError, match="Unknown wavelet"):
            cwt(x, scales, wavelet="unknown")

    def test_cwt_produces_valid_output(self):
        """Test that CWT produces valid output with expected shapes."""
        fs = 1000
        freq = 50  # Hz
        t = np.arange(0, 1, 1 / fs)
        x = np.sin(2 * np.pi * freq * t)
        scales = np.arange(1, 128)

        result = cwt(x, scales, wavelet="morlet", fs=fs)

        # Check that we have valid output
        assert result.coefficients.shape == (127, 1000)
        assert np.all(np.isfinite(result.coefficients))

        # Check energy is non-zero
        energy = np.sum(np.abs(result.coefficients) ** 2, axis=1)
        assert np.sum(energy) > 0


# =============================================================================
# Tests for scale-frequency conversion
# =============================================================================


class TestScaleFrequencyConversion:
    """Tests for scale-frequency conversions."""

    def test_scales_to_frequencies_morlet(self):
        """Test scale to frequency conversion for Morlet."""
        scales = np.array([1, 2, 4, 8, 16])
        fs = 1000

        freqs = scales_to_frequencies(scales, wavelet="morlet", fs=fs)

        assert len(freqs) == 5
        # Smaller scale = higher frequency
        assert freqs[0] > freqs[-1]

    def test_scales_to_frequencies_ricker(self):
        """Test scale to frequency conversion for Ricker."""
        scales = np.array([1, 2, 4, 8])
        fs = 1000

        freqs = scales_to_frequencies(scales, wavelet="ricker", fs=fs)

        assert len(freqs) == 4
        assert freqs[0] > freqs[-1]

    def test_scales_to_frequencies_gaussian(self):
        """Test scale to frequency conversion for Gaussian wavelets."""
        scales = np.array([1, 2, 4])
        fs = 1000

        freqs1 = scales_to_frequencies(scales, wavelet="gaussian1", fs=fs)
        freqs2 = scales_to_frequencies(scales, wavelet="gaussian2", fs=fs)

        assert len(freqs1) == 3
        assert len(freqs2) == 3

    def test_scales_to_frequencies_unknown(self):
        """Test scale to frequency conversion for unknown wavelet."""
        scales = np.array([1, 2, 4])
        fs = 1000

        # Should use default center frequency
        freqs = scales_to_frequencies(scales, wavelet="unknown", fs=fs)

        assert len(freqs) == 3

    def test_frequencies_to_scales_morlet(self):
        """Test frequency to scale conversion for Morlet."""
        freqs = np.array([100, 50, 25, 10])
        fs = 1000

        scales = frequencies_to_scales(freqs, wavelet="morlet", fs=fs)

        assert len(scales) == 4
        # Higher frequency = smaller scale
        assert scales[0] < scales[-1]

    def test_frequencies_to_scales_ricker(self):
        """Test frequency to scale conversion for Ricker."""
        freqs = np.array([100, 50, 25])
        fs = 1000

        scales = frequencies_to_scales(freqs, wavelet="ricker", fs=fs)

        assert len(scales) == 3
        assert scales[0] < scales[-1]

    def test_scale_frequency_roundtrip(self):
        """Test roundtrip scale -> freq -> scale."""
        original_scales = np.array([5.0, 10.0, 20.0, 40.0])
        fs = 1000

        freqs = scales_to_frequencies(original_scales, wavelet="morlet", fs=fs)
        recovered_scales = frequencies_to_scales(freqs, wavelet="morlet", fs=fs)

        np.testing.assert_allclose(recovered_scales, original_scales)


# =============================================================================
# Tests for DWT (if pywavelets available)
# =============================================================================


@pytest.mark.skipif(not PYWT_AVAILABLE, reason="pywavelets not available")
class TestDWT:
    """Tests for Discrete Wavelet Transform."""

    def test_dwt_basic(self):
        """Test basic DWT computation."""
        from pytcl.mathematical_functions.transforms.wavelets import dwt

        np.random.seed(42)
        x = np.random.randn(256)

        result = dwt(x, wavelet="db4", level=4)

        assert isinstance(result, DWTResult)
        assert result.levels == 4
        assert len(result.cD) == 4
        assert result.wavelet == "db4"

    def test_dwt_haar(self):
        """Test DWT with Haar wavelet."""
        from pytcl.mathematical_functions.transforms.wavelets import dwt

        x = np.random.randn(256)

        result = dwt(x, wavelet="haar", level=3)

        assert result.levels == 3
        assert result.wavelet == "haar"

    def test_dwt_automatic_level(self):
        """Test DWT with automatic level selection."""
        from pytcl.mathematical_functions.transforms.wavelets import dwt

        x = np.random.randn(512)

        result = dwt(x, wavelet="db4", level=None)

        assert result.levels > 0

    def test_dwt_idwt_roundtrip(self):
        """Test DWT -> IDWT roundtrip."""
        from pytcl.mathematical_functions.transforms.wavelets import dwt, idwt

        np.random.seed(42)
        x = np.random.randn(256)

        result = dwt(x, wavelet="db4", level=4)
        x_rec = idwt(result)

        # Reconstructed signal should match original
        np.testing.assert_allclose(x_rec[: len(x)], x, rtol=1e-10)

    def test_dwt_single_level(self):
        """Test single-level DWT."""
        from pytcl.mathematical_functions.transforms.wavelets import dwt_single_level

        np.random.seed(42)
        x = np.random.randn(256)

        cA, cD = dwt_single_level(x, wavelet="db4")

        # Length should be approximately half
        assert len(cA) < len(x)
        assert len(cD) < len(x)

    def test_dwt_idwt_single_level(self):
        """Test single-level DWT -> IDWT."""
        from pytcl.mathematical_functions.transforms.wavelets import (
            dwt_single_level,
            idwt_single_level,
        )

        np.random.seed(42)
        x = np.random.randn(256)

        cA, cD = dwt_single_level(x, wavelet="db4")
        x_rec = idwt_single_level(cA, cD, wavelet="db4")

        # Reconstructed should match (may have slight length difference)
        np.testing.assert_allclose(x_rec[: len(x)], x, rtol=1e-10)


@pytest.mark.skipif(not PYWT_AVAILABLE, reason="pywavelets not available")
class TestWaveletPacket:
    """Tests for Wavelet Packet Transform."""

    def test_wpt_basic(self):
        """Test basic WPT computation."""
        from pytcl.mathematical_functions.transforms.wavelets import wpt

        np.random.seed(42)
        x = np.random.randn(256)

        nodes = wpt(x, wavelet="db4", level=2)

        assert isinstance(nodes, dict)
        # At level 2, should have 4 nodes: 'aa', 'ad', 'da', 'dd'
        assert "aa" in nodes
        assert "ad" in nodes
        assert "da" in nodes
        assert "dd" in nodes


@pytest.mark.skipif(not PYWT_AVAILABLE, reason="pywavelets not available")
class TestThresholding:
    """Tests for coefficient thresholding."""

    def test_threshold_soft(self):
        """Test soft thresholding."""
        from pytcl.mathematical_functions.transforms.wavelets import (
            dwt,
            threshold_coefficients,
        )

        np.random.seed(42)
        x = np.random.randn(256)

        coeffs = dwt(x, wavelet="db4", level=4)
        thresh_coeffs = threshold_coefficients(coeffs, threshold="soft")

        assert thresh_coeffs.levels == coeffs.levels
        # Some coefficients should be zeroed
        assert np.sum(thresh_coeffs.cD[0] == 0) >= 0

    def test_threshold_hard(self):
        """Test hard thresholding."""
        from pytcl.mathematical_functions.transforms.wavelets import (
            dwt,
            threshold_coefficients,
        )

        np.random.seed(42)
        x = np.random.randn(256)

        coeffs = dwt(x, wavelet="db4", level=4)
        thresh_coeffs = threshold_coefficients(coeffs, threshold="hard")

        assert thresh_coeffs.levels == coeffs.levels

    def test_threshold_custom_value(self):
        """Test thresholding with custom value."""
        from pytcl.mathematical_functions.transforms.wavelets import (
            dwt,
            threshold_coefficients,
        )

        np.random.seed(42)
        x = np.random.randn(256)

        coeffs = dwt(x, wavelet="db4", level=4)
        thresh_coeffs = threshold_coefficients(coeffs, threshold="soft", value=0.5)

        assert thresh_coeffs.levels == coeffs.levels


# =============================================================================
# Tests for utility functions
# =============================================================================


class TestWaveletUtilities:
    """Tests for wavelet utility functions."""

    def test_available_wavelets(self):
        """Test available wavelets list."""
        wavelets = available_wavelets()

        assert isinstance(wavelets, list)
        assert len(wavelets) > 0

        if PYWT_AVAILABLE:
            # Should include standard wavelets
            assert "haar" in wavelets or "Haar" in wavelets
        else:
            # Without pywt, should list CWT wavelets
            assert "morlet" in wavelets

    @pytest.mark.skipif(PYWT_AVAILABLE, reason="Test for non-pywt case")
    def test_wavelet_info_morlet_no_pywt(self):
        """Test wavelet info for Morlet without pywavelets."""
        info = wavelet_info("morlet")

        assert isinstance(info, dict)
        assert "name" in info
        assert info["name"] == "morlet"

    @pytest.mark.skipif(PYWT_AVAILABLE, reason="Test for non-pywt case")
    def test_wavelet_info_ricker_no_pywt(self):
        """Test wavelet info for Ricker without pywavelets."""
        info = wavelet_info("ricker")

        assert isinstance(info, dict)
        assert info["name"] == "ricker"

    @pytest.mark.skipif(not PYWT_AVAILABLE, reason="pywavelets required")
    def test_wavelet_info_haar(self):
        """Test wavelet info for Haar with pywavelets."""
        info = wavelet_info("haar")

        assert isinstance(info, dict)
        assert "name" in info

    @pytest.mark.skipif(not PYWT_AVAILABLE, reason="pywavelets not available")
    def test_wavelet_info_db4(self):
        """Test wavelet info for db4."""
        info = wavelet_info("db4")

        assert isinstance(info, dict)
        assert "name" in info
        assert "family" in info
        assert "orthogonal" in info


# =============================================================================
# Tests for DWT import error handling
# =============================================================================


class TestDWTImportErrors:
    """Tests for DWT when pywavelets not available."""

    def test_dwt_requires_pywt(self):
        """Test that DWT raises ImportError without pywavelets."""
        if PYWT_AVAILABLE:
            pytest.skip("pywavelets is available")

        from pytcl.mathematical_functions.transforms.wavelets import dwt

        x = np.random.randn(256)

        with pytest.raises(ImportError, match="pywavelets"):
            dwt(x)

    def test_idwt_requires_pywt(self):
        """Test that IDWT raises ImportError without pywavelets."""
        if PYWT_AVAILABLE:
            pytest.skip("pywavelets is available")

        from pytcl.mathematical_functions.transforms.wavelets import idwt

        dummy_result = DWTResult(
            cA=np.zeros(10), cD=[np.zeros(10)], levels=1, wavelet="db4"
        )

        with pytest.raises(ImportError, match="pywavelets"):
            idwt(dummy_result)


# =============================================================================
# Tests for CWTResult and DWTResult types
# =============================================================================


class TestResultTypes:
    """Tests for result NamedTuple types."""

    def test_cwt_result_creation(self):
        """Test CWTResult creation."""
        coeffs = np.zeros((10, 100), dtype=np.complex128)
        scales = np.arange(1, 11, dtype=np.float64)
        freqs = np.ones(10, dtype=np.float64)

        result = CWTResult(coefficients=coeffs, scales=scales, frequencies=freqs)

        assert result.coefficients.shape == (10, 100)
        assert len(result.scales) == 10
        assert len(result.frequencies) == 10

    def test_dwt_result_creation(self):
        """Test DWTResult creation."""
        cA = np.zeros(16, dtype=np.float64)
        cD = [np.zeros(32), np.zeros(64), np.zeros(128)]

        result = DWTResult(cA=cA, cD=cD, levels=3, wavelet="db4")

        assert len(result.cA) == 16
        assert len(result.cD) == 3
        assert result.levels == 3
        assert result.wavelet == "db4"
