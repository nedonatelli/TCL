"""
Hessians of measurement functions.

Ports of the MATLAB TCL's ``Coordinate_Systems/Hessians``: second
derivatives of bistatic range, spherical angles and direction cosines
with respect to Cartesian position, the composed spherical
measurement Hessians (direct, inverse and converted), and the
chain-rule/affine helpers. Layout mirrors MATLAB: a Hessian stack has
shape ``(n, n, num_out)`` with ``H[:, :, k]`` the Hessian of the k-th
output component.
"""

from pytcl.coordinate_systems.hessians.component_hessians import (
    range_hessian,
    spher_ang_hessian,
    u_hessian_2d,
    u_hessian_3d,
    uv_hessian,
)
from pytcl.coordinate_systems.hessians.cross_derivatives import (
    polar_u_2d_cross_grad,
    polar_u_2d_cross_hessian,
    spher_ang_uv_cross_grad,
    spher_ang_uv_cross_hessian,
    u_polar_2d_cross_grad,
    u_polar_2d_cross_hessian,
    uv_spher_ang_cross_grad,
    uv_spher_ang_cross_hessian,
)
from pytcl.coordinate_systems.hessians.measurement_hessians import (
    calc_spher_conv_hessian,
    calc_spher_hessian,
    calc_spher_inv_hessian,
    hessian_chain_rule,
    hessian_of_affine_trans_fun,
)

__all__ = [
    "range_hessian",
    "spher_ang_hessian",
    "uv_hessian",
    "u_hessian_2d",
    "u_hessian_3d",
    "calc_spher_hessian",
    "calc_spher_inv_hessian",
    "calc_spher_conv_hessian",
    "hessian_of_affine_trans_fun",
    "hessian_chain_rule",
    "uv_spher_ang_cross_grad",
    "spher_ang_uv_cross_grad",
    "u_polar_2d_cross_grad",
    "polar_u_2d_cross_grad",
    "uv_spher_ang_cross_hessian",
    "spher_ang_uv_cross_hessian",
    "u_polar_2d_cross_hessian",
    "polar_u_2d_cross_hessian",
]
