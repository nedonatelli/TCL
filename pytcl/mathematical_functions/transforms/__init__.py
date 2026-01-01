"""
Transform utilities.

This module provides signal transforms for time-frequency analysis:
- Fourier transforms (FFT, RFFT, 2D FFT)
- Short-Time Fourier Transform (STFT) and spectrograms
- Wavelet transforms (CWT and DWT)
"""

from pytcl.mathematical_functions.transforms.fourier import CoherenceResult
from pytcl.mathematical_functions.transforms.fourier import CrossSpectrum
from pytcl.mathematical_functions.transforms.fourier import PowerSpectrum
from pytcl.mathematical_functions.transforms.fourier import coherence
from pytcl.mathematical_functions.transforms.fourier import cross_spectrum
from pytcl.mathematical_functions.transforms.fourier import fft
from pytcl.mathematical_functions.transforms.fourier import fft2
from pytcl.mathematical_functions.transforms.fourier import fftshift
from pytcl.mathematical_functions.transforms.fourier import frequency_axis
from pytcl.mathematical_functions.transforms.fourier import ifft
from pytcl.mathematical_functions.transforms.fourier import ifft2
from pytcl.mathematical_functions.transforms.fourier import ifftshift
from pytcl.mathematical_functions.transforms.fourier import irfft
from pytcl.mathematical_functions.transforms.fourier import magnitude_spectrum
from pytcl.mathematical_functions.transforms.fourier import periodogram
from pytcl.mathematical_functions.transforms.fourier import phase_spectrum
from pytcl.mathematical_functions.transforms.fourier import power_spectrum
from pytcl.mathematical_functions.transforms.fourier import rfft
from pytcl.mathematical_functions.transforms.fourier import rfft_frequency_axis
from pytcl.mathematical_functions.transforms.stft import Spectrogram
from pytcl.mathematical_functions.transforms.stft import STFTResult
from pytcl.mathematical_functions.transforms.stft import get_window
from pytcl.mathematical_functions.transforms.stft import istft
from pytcl.mathematical_functions.transforms.stft import mel_spectrogram
from pytcl.mathematical_functions.transforms.stft import reassigned_spectrogram
from pytcl.mathematical_functions.transforms.stft import spectrogram
from pytcl.mathematical_functions.transforms.stft import stft
from pytcl.mathematical_functions.transforms.stft import window_bandwidth
from pytcl.mathematical_functions.transforms.wavelets import PYWT_AVAILABLE
from pytcl.mathematical_functions.transforms.wavelets import CWTResult
from pytcl.mathematical_functions.transforms.wavelets import DWTResult
from pytcl.mathematical_functions.transforms.wavelets import available_wavelets
from pytcl.mathematical_functions.transforms.wavelets import cwt
from pytcl.mathematical_functions.transforms.wavelets import dwt
from pytcl.mathematical_functions.transforms.wavelets import dwt_single_level
from pytcl.mathematical_functions.transforms.wavelets import frequencies_to_scales
from pytcl.mathematical_functions.transforms.wavelets import gaussian_wavelet
from pytcl.mathematical_functions.transforms.wavelets import idwt
from pytcl.mathematical_functions.transforms.wavelets import idwt_single_level
from pytcl.mathematical_functions.transforms.wavelets import morlet_wavelet
from pytcl.mathematical_functions.transforms.wavelets import ricker_wavelet
from pytcl.mathematical_functions.transforms.wavelets import scales_to_frequencies
from pytcl.mathematical_functions.transforms.wavelets import threshold_coefficients
from pytcl.mathematical_functions.transforms.wavelets import wavelet_info
from pytcl.mathematical_functions.transforms.wavelets import wpt

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
