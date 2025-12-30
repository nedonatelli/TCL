"""
Dynamic state estimation algorithms.

This module provides filtering and smoothing algorithms for state estimation:
- Kalman filter family (KF, EKF, UKF, CKF, etc.)
- Particle filters (bootstrap, auxiliary, regularized)
- Batch estimation methods
"""

# Import submodules for easy access
from tcl.dynamic_estimation import kalman
from tcl.dynamic_estimation import particle_filters

# Re-export commonly used functions at the top level

# Linear Kalman filter
from tcl.dynamic_estimation.kalman import (
    KalmanState,
    KalmanPrediction,
    KalmanUpdate,
    kf_predict,
    kf_update,
    kf_predict_update,
    kf_smooth,
    information_filter_predict,
    information_filter_update,
)

# Extended Kalman filter
from tcl.dynamic_estimation.kalman import (
    ekf_predict,
    ekf_update,
    numerical_jacobian,
    ekf_predict_auto,
    ekf_update_auto,
    iterated_ekf_update,
)

# Unscented Kalman filter
from tcl.dynamic_estimation.kalman import (
    SigmaPoints,
    sigma_points_merwe,
    sigma_points_julier,
    unscented_transform,
    ukf_predict,
    ukf_update,
)

# Cubature Kalman filter
from tcl.dynamic_estimation.kalman import (
    ckf_spherical_cubature_points,
    ckf_predict,
    ckf_update,
)

# Particle filters
from tcl.dynamic_estimation.particle_filters import (
    ParticleState,
    resample_multinomial,
    resample_systematic,
    resample_residual,
    effective_sample_size,
    bootstrap_pf_predict,
    bootstrap_pf_update,
    gaussian_likelihood,
    bootstrap_pf_step,
    particle_mean,
    particle_covariance,
    initialize_particles,
)

__all__ = [
    # Submodules
    "kalman",
    "particle_filters",
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
    # Particle filters
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
