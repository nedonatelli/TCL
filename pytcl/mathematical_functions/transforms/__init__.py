"""
Transform utilities.

This module provides signal transforms for time-frequency analysis:
- Fourier transforms (FFT, RFFT, 2D FFT)
- Short-Time Fourier Transform (STFT) and spectrograms
- Wavelet transforms (CWT and DWT)
"""

from pytcl.mathematical_functions.transforms.fourier import (
    CoherenceResult,
    CrossSpectrum,
    PowerSpectrum,
    coherence,
    cross_spectrum,
    fft,
    fft2,
    fftshift,
    frequency_axis,
    ifft,
    ifft2,
    ifftshift,
    irfft,
    magnitude_spectrum,
    periodogram,
    phase_spectrum,
    power_spectrum,
    rfft,
    rfft_frequency_axis,
)
from pytcl.mathematical_functions.transforms.stft import (
    Spectrogram,
    STFTResult,
    get_window,
    istft,
    mel_spectrogram,
    reassigned_spectrogram,
    spectrogram,
    stft,
    window_bandwidth,
)
from pytcl.mathematical_functions.transforms.wavelets import (
    PYWT_AVAILABLE,
    CWTResult,
    DWTResult,
    available_wavelets,
    cwt,
    dwt,
    dwt_single_level,
    frequencies_to_scales,
    gaussian_wavelet,
    idwt,
    idwt_single_level,
    morlet_wavelet,
    ricker_wavelet,
    scales_to_frequencies,
    threshold_coefficients,
    wavelet_info,
    wpt,
)

__all__ = [
    # Fourier transform types
    "PowerSpectrum",
    "CrossSpectrum",
    "CoherenceResult",
    # Core FFT functions
    "fft",
    "ifft",
    "rfft",
    "irfft",
    "fft2",
    "ifft2",
    "fftshift",
    "ifftshift",
    # Frequency axis
    "frequency_axis",
    "rfft_frequency_axis",
    # Spectral analysis
    "power_spectrum",
    "cross_spectrum",
    "coherence",
    "periodogram",
    "magnitude_spectrum",
    "phase_spectrum",
    # STFT types
    "STFTResult",
    "Spectrogram",
    # STFT functions
    "stft",
    "istft",
    "spectrogram",
    "get_window",
    "window_bandwidth",
    # Advanced STFT
    "reassigned_spectrogram",
    "mel_spectrogram",
    # Wavelet types
    "CWTResult",
    "DWTResult",
    # Wavelet functions
    "morlet_wavelet",
    "ricker_wavelet",
    "gaussian_wavelet",
    # CWT
    "cwt",
    "scales_to_frequencies",
    "frequencies_to_scales",
    # DWT
    "dwt",
    "idwt",
    "dwt_single_level",
    "idwt_single_level",
    "wpt",
    # Wavelet utilities
    "available_wavelets",
    "wavelet_info",
    "threshold_coefficients",
    "PYWT_AVAILABLE",
]
