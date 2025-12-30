"""
Kalman filter family implementations.

This module provides:
- Linear Kalman filter (predict, update, smoothing)
- Extended Kalman filter (EKF)
- Unscented Kalman filter (UKF)
- Cubature Kalman filter (CKF)
- Information filter
- Square-root Kalman filters (numerically stable)
- U-D factorization filter (Bierman's method)
"""

from pytcl.dynamic_estimation.kalman.linear import (
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

from pytcl.dynamic_estimation.kalman.extended import (
    ekf_predict,
    ekf_update,
    numerical_jacobian,
    ekf_predict_auto,
    ekf_update_auto,
    iterated_ekf_update,
)

from pytcl.dynamic_estimation.kalman.unscented import (
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

from pytcl.dynamic_estimation.kalman.square_root import (
    SRKalmanState,
    SRKalmanPrediction,
    SRKalmanUpdate,
    cholesky_update,
    qr_update,
    srkf_predict,
    srkf_update,
    srkf_predict_update,
    UDState,
    ud_factorize,
    ud_reconstruct,
    ud_predict,
    ud_update_scalar,
    ud_update,
    sr_ukf_predict,
    sr_ukf_update,
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
    # Square-root KF
    "SRKalmanState",
    "SRKalmanPrediction",
    "SRKalmanUpdate",
    "cholesky_update",
    "qr_update",
    "srkf_predict",
    "srkf_update",
    "srkf_predict_update",
    # U-D factorization
    "UDState",
    "ud_factorize",
    "ud_reconstruct",
    "ud_predict",
    "ud_update_scalar",
    "ud_update",
    # Square-root UKF
    "sr_ukf_predict",
    "sr_ukf_update",
]
