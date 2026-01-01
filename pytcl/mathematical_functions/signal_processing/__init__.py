"""
Signal processing utilities.

This module provides signal processing functions for target tracking and
radar applications, including:
- Digital filter design (IIR and FIR)
- Matched filtering for signal detection
- CFAR (Constant False Alarm Rate) detection algorithms
"""

from pytcl.mathematical_functions.signal_processing.detection import CFARResult
from pytcl.mathematical_functions.signal_processing.detection import CFARResult2D
from pytcl.mathematical_functions.signal_processing.detection import cfar_2d
from pytcl.mathematical_functions.signal_processing.detection import cfar_ca
from pytcl.mathematical_functions.signal_processing.detection import cfar_go
from pytcl.mathematical_functions.signal_processing.detection import cfar_os
from pytcl.mathematical_functions.signal_processing.detection import cfar_so
from pytcl.mathematical_functions.signal_processing.detection import cluster_detections
from pytcl.mathematical_functions.signal_processing.detection import (
    detection_probability,
)
from pytcl.mathematical_functions.signal_processing.detection import snr_loss
from pytcl.mathematical_functions.signal_processing.detection import threshold_factor
from pytcl.mathematical_functions.signal_processing.filters import FilterCoefficients
from pytcl.mathematical_functions.signal_processing.filters import FrequencyResponse
from pytcl.mathematical_functions.signal_processing.filters import apply_filter
from pytcl.mathematical_functions.signal_processing.filters import bessel_design
from pytcl.mathematical_functions.signal_processing.filters import butter_design
from pytcl.mathematical_functions.signal_processing.filters import cheby1_design
from pytcl.mathematical_functions.signal_processing.filters import cheby2_design
from pytcl.mathematical_functions.signal_processing.filters import ellip_design
from pytcl.mathematical_functions.signal_processing.filters import filter_order
from pytcl.mathematical_functions.signal_processing.filters import filtfilt
from pytcl.mathematical_functions.signal_processing.filters import fir_design
from pytcl.mathematical_functions.signal_processing.filters import fir_design_remez
from pytcl.mathematical_functions.signal_processing.filters import frequency_response
from pytcl.mathematical_functions.signal_processing.filters import group_delay
from pytcl.mathematical_functions.signal_processing.filters import sos_to_zpk
from pytcl.mathematical_functions.signal_processing.filters import zpk_to_sos
from pytcl.mathematical_functions.signal_processing.matched_filter import (
    MatchedFilterResult,
)
from pytcl.mathematical_functions.signal_processing.matched_filter import (
    PulseCompressionResult,
)
from pytcl.mathematical_functions.signal_processing.matched_filter import (
    ambiguity_function,
)
from pytcl.mathematical_functions.signal_processing.matched_filter import (
    cross_ambiguity,
)
from pytcl.mathematical_functions.signal_processing.matched_filter import (
    generate_lfm_chirp,
)
from pytcl.mathematical_functions.signal_processing.matched_filter import (
    generate_nlfm_chirp,
)
from pytcl.mathematical_functions.signal_processing.matched_filter import matched_filter
from pytcl.mathematical_functions.signal_processing.matched_filter import (
    matched_filter_frequency,
)
from pytcl.mathematical_functions.signal_processing.matched_filter import optimal_filter
from pytcl.mathematical_functions.signal_processing.matched_filter import (
    pulse_compression,
)

__all__ = [
    # Filter design types
    "FilterCoefficients",
    "FrequencyResponse",
    # IIR filter design
    "butter_design",
    "cheby1_design",
    "cheby2_design",
    "ellip_design",
    "bessel_design",
    # FIR filter design
    "fir_design",
    "fir_design_remez",
    # Filter application
    "apply_filter",
    "filtfilt",
    # Filter analysis
    "frequency_response",
    "group_delay",
    "filter_order",
    "sos_to_zpk",
    "zpk_to_sos",
    # Matched filter types
    "MatchedFilterResult",
    "PulseCompressionResult",
    # Matched filtering
    "matched_filter",
    "matched_filter_frequency",
    "optimal_filter",
    "pulse_compression",
    # Chirp generation
    "generate_lfm_chirp",
    "generate_nlfm_chirp",
    # Ambiguity function
    "ambiguity_function",
    "cross_ambiguity",
    # CFAR types
    "CFARResult",
    "CFARResult2D",
    # CFAR algorithms
    "cfar_ca",
    "cfar_go",
    "cfar_so",
    "cfar_os",
    "cfar_2d",
    # CFAR utilities
    "threshold_factor",
    "detection_probability",
    "cluster_detections",
    "snr_loss",
]
