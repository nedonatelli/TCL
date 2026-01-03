"""
Kalman filter family implementations.

This module provides:
- Linear Kalman filter (predict, update, smoothing)
- Extended Kalman filter (EKF)
- Constrained Extended Kalman filter (CEKF)
- Unscented Kalman filter (UKF)
- Cubature Kalman filter (CKF)
- Information filter
- Square-root Kalman filters (numerically stable)
- U-D factorization filter (Bierman's method)
- H-infinity filter (robust filtering)
"""

from pytcl.dynamic_estimation.kalman.constrained import (
    ConstrainedEKF,
    ConstraintFunction,
    constrained_ekf_predict,
    constrained_ekf_update,
)
from pytcl.dynamic_estimation.kalman.extended import (
    ekf_predict,
    ekf_predict_auto,
    ekf_update,
    ekf_update_auto,
    iterated_ekf_update,
    numerical_jacobian,
)
from pytcl.dynamic_estimation.kalman.h_infinity import (
    HInfinityPrediction,
    HInfinityUpdate,
    extended_hinf_update,
    find_min_gamma,
    hinf_predict,
    hinf_predict_update,
    hinf_update,
)
from pytcl.dynamic_estimation.kalman.linear import (
    KalmanPrediction,
    KalmanState,
    KalmanUpdate,
    information_filter_predict,
    information_filter_update,
    kf_predict,
    kf_predict_update,
    kf_smooth,
    kf_update,
)
from pytcl.dynamic_estimation.kalman.square_root import (
    SRKalmanPrediction,
    SRKalmanState,
    SRKalmanUpdate,
    UDState,
    cholesky_update,
    qr_update,
    sr_ukf_predict,
    sr_ukf_update,
    srkf_predict,
    srkf_predict_update,
    srkf_update,
    ud_factorize,
    ud_predict,
    ud_reconstruct,
    ud_update,
    ud_update_scalar,
)
from pytcl.dynamic_estimation.kalman.unscented import (
    SigmaPoints,
    ckf_predict,
    ckf_spherical_cubature_points,
    ckf_update,
    sigma_points_julier,
    sigma_points_merwe,
    ukf_predict,
    ukf_update,
    unscented_transform,
)

__all__ = [
    # Constrained EKF
    "ConstraintFunction",
    "ConstrainedEKF",
    "constrained_ekf_predict",
    "constrained_ekf_update",
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
    # H-infinity filter
    "HInfinityPrediction",
    "HInfinityUpdate",
    "hinf_predict",
    "hinf_update",
    "hinf_predict_update",
    "extended_hinf_update",
    "find_min_gamma",
]
