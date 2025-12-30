"""
Numerical integration (quadrature) methods.

This module provides:
- Gaussian quadrature rules (Legendre, Hermite, Laguerre, Chebyshev)
- Adaptive integration functions
- Multi-dimensional cubature rules for filtering (CKF, UKF)
"""

from tracker_component_library.mathematical_functions.numerical_integration.quadrature import (  # noqa: E501
    gauss_legendre,
    gauss_hermite,
    gauss_laguerre,
    gauss_chebyshev,
    quad,
    dblquad,
    tplquad,
    fixed_quad,
    romberg,
    simpson,
    trapezoid,
    cubature_gauss_hermite,
    spherical_cubature,
    unscented_transform_points,
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
]
