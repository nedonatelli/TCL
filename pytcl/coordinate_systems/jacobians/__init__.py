"""
Jacobian matrices for coordinate transformations.

This module provides:
- Jacobians for spherical/Cartesian transformations
- Jacobians for polar transformations
- Jacobians for r-u-v direction cosines
- Jacobians for ECEF/ENU/NED transformations
- Jacobians for geodetic transformations
- Covariance transformation utilities
"""

from pytcl.coordinate_systems.jacobians.component_gradients import (
    pol_ang_gradient,
    range_gradient,
    range_rate_gradient,
    spher_ang_gradient,
    tdoa_gradient,
    u_gradient_2d,
    u_gradient_3d,
    uv_gradient,
)
from pytcl.coordinate_systems.jacobians.jacobians import (
    cross_covariance_transform,
    enu_jacobian,
    geodetic_jacobian,
    ned_jacobian,
    numerical_jacobian,
)
from pytcl.coordinate_systems.jacobians.measurement_jacobians import (
    calc_cart_rr_jacob,
    calc_polar_conv_jacob,
    calc_polar_jacob,
    calc_polar_rr_conv_jacob,
    calc_polar_rr_jacob,
    calc_ruv_conv_jacob,
    calc_ruv_jacob,
    calc_ruv_rr_conv_jacob,
    calc_ruv_rr_jacob,
    calc_spher_conv_jacob,
    calc_spher_inv_jacob,
    calc_spher_jacob,
    calc_spher_rr_jacob,
    norm_vec_jacob,
)

__all__ = [
    "range_gradient",
    "range_rate_gradient",
    "spher_ang_gradient",
    "pol_ang_gradient",
    "uv_gradient",
    "u_gradient_2d",
    "u_gradient_3d",
    "tdoa_gradient",
    "calc_spher_jacob",
    "calc_spher_inv_jacob",
    "calc_spher_rr_jacob",
    "calc_polar_jacob",
    "calc_polar_rr_jacob",
    "calc_ruv_jacob",
    "calc_ruv_rr_jacob",
    "calc_cart_rr_jacob",
    "calc_spher_conv_jacob",
    "calc_polar_conv_jacob",
    "calc_polar_rr_conv_jacob",
    "calc_ruv_conv_jacob",
    "calc_ruv_rr_conv_jacob",
    "norm_vec_jacob",
    "enu_jacobian",
    "ned_jacobian",
    "geodetic_jacobian",
    "cross_covariance_transform",
    "numerical_jacobian",
]
