"""
Kalman filter family implementations.

This module provides:
- Linear Kalman filter (predict, update, smoothing)
- Extended Kalman filter (EKF)
- Unscented Kalman filter (UKF)
- Cubature Kalman filter (CKF)
- Information filter
"""

from tcl.dynamic_estimation.kalman.linear import (
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

from tcl.dynamic_estimation.kalman.extended import (
    ekf_predict,
    ekf_update,
    numerical_jacobian,
    ekf_predict_auto,
    ekf_update_auto,
    iterated_ekf_update,
)

from tcl.dynamic_estimation.kalman.unscented import (
    SigmaPoints,
    sigma_points_merwe,
    sigma_points_julier,
    unscented_transform,
    ukf_predict,
    ukf_update,
    ckf_spherical_cubature_points,
    ckf_predict,
    ckf_update,
)

__all__ = [
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
]
