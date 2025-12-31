"""
Signal processing utilities.

This module provides signal processing functions for target tracking and
radar applications, including:
- Digital filter design (IIR and FIR)
- Matched filtering for signal detection
- CFAR (Constant False Alarm Rate) detection algorithms
"""

from pytcl.mathematical_functions.signal_processing.detection import (
    CFARResult,
    CFARResult2D,
    cfar_2d,
    cfar_ca,
    cfar_go,
    cfar_os,
    cfar_so,
    cluster_detections,
    detection_probability,
    snr_loss,
    threshold_factor,
)
from pytcl.mathematical_functions.signal_processing.filters import (
    FilterCoefficients,
    FrequencyResponse,
    apply_filter,
    bessel_design,
    butter_design,
    cheby1_design,
    cheby2_design,
    ellip_design,
    filter_order,
    filtfilt,
    fir_design,
    fir_design_remez,
    frequency_response,
    group_delay,
    sos_to_zpk,
    zpk_to_sos,
)
from pytcl.mathematical_functions.signal_processing.matched_filter import (
    MatchedFilterResult,
    PulseCompressionResult,
    ambiguity_function,
    cross_ambiguity,
    generate_lfm_chirp,
    generate_nlfm_chirp,
    matched_filter,
    matched_filter_frequency,
    optimal_filter,
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
