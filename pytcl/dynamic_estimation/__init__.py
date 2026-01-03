"""
Dynamic state estimation algorithms.

This module provides filtering and smoothing algorithms for state estimation:
- Kalman filter family (KF, EKF, UKF, CKF, etc.)
- Square-root Kalman filters (numerically stable)
- Interacting Multiple Model (IMM) estimator
- Particle filters (bootstrap, auxiliary, regularized)
- Smoothers (RTS, fixed-lag, fixed-interval, two-filter)
- Information filters (standard and square-root)
- Batch estimation methods
"""

# Import submodules for easy access
from pytcl.dynamic_estimation import kalman, particle_filters

# Gaussian Sum Filter
from pytcl.dynamic_estimation.gaussian_sum_filter import (
    GaussianComponent,
    GaussianSumFilter,
    gaussian_sum_filter_predict,
    gaussian_sum_filter_update,
)

# IMM estimator
from pytcl.dynamic_estimation.imm import (
    IMMEstimator,
    IMMPrediction,
    IMMState,
    IMMUpdate,
    imm_predict,
    imm_predict_update,
    imm_update,
)

# Information filter
from pytcl.dynamic_estimation.information_filter import (
    InformationFilterResult,
    InformationState,
    SRIFResult,
    SRIFState,
    fuse_information,
    information_filter,
    information_to_state,
    srif_filter,
    srif_predict,
    srif_update,
    state_to_information,
)

# Square-root Kalman filters
# Cubature Kalman filter
# Unscented Kalman filter
# Extended Kalman filter
# Linear Kalman filter
from pytcl.dynamic_estimation.kalman import (
    KalmanPrediction,
    KalmanState,
    KalmanUpdate,
    SigmaPoints,
    SRKalmanPrediction,
    SRKalmanState,
    SRKalmanUpdate,
    UDState,
    ckf_predict,
    ckf_spherical_cubature_points,
    ckf_update,
    ekf_predict,
    ekf_predict_auto,
    ekf_update,
    ekf_update_auto,
    information_filter_predict,
    information_filter_update,
    iterated_ekf_update,
    kf_predict,
    kf_predict_update,
    kf_smooth,
    kf_update,
    numerical_jacobian,
    sigma_points_julier,
    sigma_points_merwe,
    sr_ukf_predict,
    sr_ukf_update,
    srkf_predict,
    srkf_predict_update,
    srkf_update,
    ud_factorize,
    ud_predict,
    ud_reconstruct,
    ud_update,
    ukf_predict,
    ukf_update,
    unscented_transform,
)

# Particle filters
from pytcl.dynamic_estimation.particle_filters import (
    ParticleState,
    bootstrap_pf_predict,
    bootstrap_pf_step,
    bootstrap_pf_update,
    effective_sample_size,
    gaussian_likelihood,
    initialize_particles,
    particle_covariance,
    particle_mean,
    resample_multinomial,
    resample_residual,
    resample_systematic,
)

# Rao-Blackwellized Particle Filter
from pytcl.dynamic_estimation.rbpf import (
    RBPFFilter,
    RBPFParticle,
    rbpf_predict,
    rbpf_update,
)

# Smoothers
from pytcl.dynamic_estimation.smoothers import (
    FixedLagResult,
    RTSResult,
    SmoothedState,
    fixed_interval_smoother,
    fixed_lag_smoother,
    rts_smoother,
    rts_smoother_single_step,
    two_filter_smoother,
)

# Re-export commonly used functions at the top level


__all__ = [
    # Submodules
    "kalman",
    "particle_filters",
    # Smoothers
    "SmoothedState",
    "RTSResult",
    "FixedLagResult",
    "rts_smoother",
    "fixed_lag_smoother",
    "fixed_interval_smoother",
    "two_filter_smoother",
    "rts_smoother_single_step",
    # Information filter
    "InformationState",
    "InformationFilterResult",
    "SRIFState",
    "SRIFResult",
    "information_to_state",
    "state_to_information",
    "information_filter",
    "srif_predict",
    "srif_update",
    "srif_filter",
    "fuse_information",
    # Linear KF
    "KalmanState",
    "KalmanPrediction",
    "KalmanUpdate",
    "kf_predict",
    "kf_update",
    "kf_predict_update",
    "kf_smooth",
    "information_filter_predict",
    "information_filter_update",
    # EKF
    "ekf_predict",
    "ekf_update",
    "numerical_jacobian",
    "ekf_predict_auto",
    "ekf_update_auto",
    "iterated_ekf_update",
    # UKF
    "SigmaPoints",
    "sigma_points_merwe",
    "sigma_points_julier",
    "unscented_transform",
    "ukf_predict",
    "ukf_update",
    # CKF
    "ckf_spherical_cubature_points",
    "ckf_predict",
    "ckf_update",
    # Square-root KF
    "SRKalmanState",
    "SRKalmanPrediction",
    "SRKalmanUpdate",
    "srkf_predict",
    "srkf_update",
    "srkf_predict_update",
    "UDState",
    "ud_factorize",
    "ud_reconstruct",
    "ud_predict",
    "ud_update",
    "sr_ukf_predict",
    "sr_ukf_update",
    # IMM
    "IMMState",
    "IMMPrediction",
    "IMMUpdate",
    "imm_predict",
    "imm_update",
    "imm_predict_update",
    "IMMEstimator",
    # Particle filters
    # Gaussian Sum Filter
    "GaussianComponent",
    "GaussianSumFilter",
    "gaussian_sum_filter_predict",
    "gaussian_sum_filter_update",
    # Rao-Blackwellized Particle Filter
    "RBPFParticle",
    "RBPFFilter",
    "rbpf_predict",
    "rbpf_update",
    "ParticleState",
    "resample_multinomial",
    "resample_systematic",
    "resample_residual",
    "effective_sample_size",
    "bootstrap_pf_predict",
    "bootstrap_pf_update",
    "gaussian_likelihood",
    "bootstrap_pf_step",
    "particle_mean",
    "particle_covariance",
    "initialize_particles",
]
