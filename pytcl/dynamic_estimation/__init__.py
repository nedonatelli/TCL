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
from pytcl.dynamic_estimation import kalman
from pytcl.dynamic_estimation import particle_filters

# IMM estimator
from pytcl.dynamic_estimation.imm import IMMEstimator
from pytcl.dynamic_estimation.imm import IMMPrediction
from pytcl.dynamic_estimation.imm import IMMState
from pytcl.dynamic_estimation.imm import IMMUpdate
from pytcl.dynamic_estimation.imm import imm_predict
from pytcl.dynamic_estimation.imm import imm_predict_update
from pytcl.dynamic_estimation.imm import imm_update

# Information filter
from pytcl.dynamic_estimation.information_filter import InformationFilterResult
from pytcl.dynamic_estimation.information_filter import InformationState
from pytcl.dynamic_estimation.information_filter import SRIFResult
from pytcl.dynamic_estimation.information_filter import SRIFState
from pytcl.dynamic_estimation.information_filter import fuse_information
from pytcl.dynamic_estimation.information_filter import information_filter
from pytcl.dynamic_estimation.information_filter import information_to_state
from pytcl.dynamic_estimation.information_filter import srif_filter
from pytcl.dynamic_estimation.information_filter import srif_predict
from pytcl.dynamic_estimation.information_filter import srif_update
from pytcl.dynamic_estimation.information_filter import state_to_information

# Square-root Kalman filters
# Cubature Kalman filter
# Unscented Kalman filter
# Extended Kalman filter
# Linear Kalman filter
from pytcl.dynamic_estimation.kalman import KalmanPrediction
from pytcl.dynamic_estimation.kalman import KalmanState
from pytcl.dynamic_estimation.kalman import KalmanUpdate
from pytcl.dynamic_estimation.kalman import SigmaPoints
from pytcl.dynamic_estimation.kalman import SRKalmanPrediction
from pytcl.dynamic_estimation.kalman import SRKalmanState
from pytcl.dynamic_estimation.kalman import SRKalmanUpdate
from pytcl.dynamic_estimation.kalman import UDState
from pytcl.dynamic_estimation.kalman import ckf_predict
from pytcl.dynamic_estimation.kalman import ckf_spherical_cubature_points
from pytcl.dynamic_estimation.kalman import ckf_update
from pytcl.dynamic_estimation.kalman import ekf_predict
from pytcl.dynamic_estimation.kalman import ekf_predict_auto
from pytcl.dynamic_estimation.kalman import ekf_update
from pytcl.dynamic_estimation.kalman import ekf_update_auto
from pytcl.dynamic_estimation.kalman import information_filter_predict
from pytcl.dynamic_estimation.kalman import information_filter_update
from pytcl.dynamic_estimation.kalman import iterated_ekf_update
from pytcl.dynamic_estimation.kalman import kf_predict
from pytcl.dynamic_estimation.kalman import kf_predict_update
from pytcl.dynamic_estimation.kalman import kf_smooth
from pytcl.dynamic_estimation.kalman import kf_update
from pytcl.dynamic_estimation.kalman import numerical_jacobian
from pytcl.dynamic_estimation.kalman import sigma_points_julier
from pytcl.dynamic_estimation.kalman import sigma_points_merwe
from pytcl.dynamic_estimation.kalman import sr_ukf_predict
from pytcl.dynamic_estimation.kalman import sr_ukf_update
from pytcl.dynamic_estimation.kalman import srkf_predict
from pytcl.dynamic_estimation.kalman import srkf_predict_update
from pytcl.dynamic_estimation.kalman import srkf_update
from pytcl.dynamic_estimation.kalman import ud_factorize
from pytcl.dynamic_estimation.kalman import ud_predict
from pytcl.dynamic_estimation.kalman import ud_reconstruct
from pytcl.dynamic_estimation.kalman import ud_update
from pytcl.dynamic_estimation.kalman import ukf_predict
from pytcl.dynamic_estimation.kalman import ukf_update
from pytcl.dynamic_estimation.kalman import unscented_transform

# Particle filters
from pytcl.dynamic_estimation.particle_filters import ParticleState
from pytcl.dynamic_estimation.particle_filters import bootstrap_pf_predict
from pytcl.dynamic_estimation.particle_filters import bootstrap_pf_step
from pytcl.dynamic_estimation.particle_filters import bootstrap_pf_update
from pytcl.dynamic_estimation.particle_filters import effective_sample_size
from pytcl.dynamic_estimation.particle_filters import gaussian_likelihood
from pytcl.dynamic_estimation.particle_filters import initialize_particles
from pytcl.dynamic_estimation.particle_filters import particle_covariance
from pytcl.dynamic_estimation.particle_filters import particle_mean
from pytcl.dynamic_estimation.particle_filters import resample_multinomial
from pytcl.dynamic_estimation.particle_filters import resample_residual
from pytcl.dynamic_estimation.particle_filters import resample_systematic

# Smoothers
from pytcl.dynamic_estimation.smoothers import FixedLagResult
from pytcl.dynamic_estimation.smoothers import RTSResult
from pytcl.dynamic_estimation.smoothers import SmoothedState
from pytcl.dynamic_estimation.smoothers import fixed_interval_smoother
from pytcl.dynamic_estimation.smoothers import fixed_lag_smoother
from pytcl.dynamic_estimation.smoothers import rts_smoother
from pytcl.dynamic_estimation.smoothers import rts_smoother_single_step
from pytcl.dynamic_estimation.smoothers import two_filter_smoother

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
