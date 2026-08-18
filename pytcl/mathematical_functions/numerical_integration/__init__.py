"""
Numerical integration (quadrature) methods.

This module provides:
- Gaussian quadrature rules (Legendre, Hermite, Laguerre, Chebyshev)
- Adaptive integration functions
- Multi-dimensional cubature rules for filtering (CKF, UKF)
"""

from pytcl.mathematical_functions.numerical_integration.cubature_points import (  # noqa: E501
    cubature_point_moments,
    fifth_order_cubature_points,
    fourteenth_order_cubature_points,
    genz_keister_points,
    second_order_cubature_points,
    seventh_order_cubature_points,
    smolyak_points,
    sphere_surface_to_gauss_points,
    spherical_radial_points,
    student_t_cubature_points,
    transform_cubature_points,
)
from pytcl.mathematical_functions.numerical_integration.quadrature import (  # noqa: E501
    cubature_gauss_hermite,
    dblquad,
    fixed_quad,
    gauss_chebyshev,
    gauss_hermite,
    gauss_laguerre,
    gauss_legendre,
    quad,
    romberg,
    simpson,
    spherical_cubature,
    tplquad,
    trapezoid,
    unscented_transform_points,
)
from pytcl.mathematical_functions.numerical_integration.region_cubature import (  # noqa: E501
    cube_cubature_points,
)

__all__ = [
    # 1D Quadrature rules
    "gauss_legendre",
    "gauss_hermite",
    "gauss_laguerre",
    "gauss_chebyshev",
    # Integration functions
    "quad",
    "dblquad",
    "tplquad",
    "fixed_quad",
    "romberg",
    "simpson",
    "trapezoid",
    # Multi-dimensional cubature
    "cubature_gauss_hermite",
    "spherical_cubature",
    "unscented_transform_points",
    "second_order_cubature_points",
    "fifth_order_cubature_points",
    "fourteenth_order_cubature_points",
    "genz_keister_points",
    "seventh_order_cubature_points",
    "smolyak_points",
    "sphere_surface_to_gauss_points",
    "spherical_radial_points",
    "student_t_cubature_points",
    "transform_cubature_points",
    "cubature_point_moments",
    # Region cubature (true-measure weights, NOT Gaussian/probability -- see
    # region_cubature.py's module docstring)
    "cube_cubature_points",
]
