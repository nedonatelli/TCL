"""
Numerical integration (quadrature) methods.

This module provides:
- Gaussian quadrature rules (Legendre, Hermite, Laguerre, Chebyshev)
- Adaptive integration functions
- Multi-dimensional cubature rules for filtering (CKF, UKF)
"""

from pytcl.mathematical_functions.numerical_integration.quadrature import (  # noqa: E501
    cubature_gauss_hermite,
)
from pytcl.mathematical_functions.numerical_integration.quadrature import dblquad
from pytcl.mathematical_functions.numerical_integration.quadrature import fixed_quad
from pytcl.mathematical_functions.numerical_integration.quadrature import (
    gauss_chebyshev,
)
from pytcl.mathematical_functions.numerical_integration.quadrature import gauss_hermite
from pytcl.mathematical_functions.numerical_integration.quadrature import gauss_laguerre
from pytcl.mathematical_functions.numerical_integration.quadrature import gauss_legendre
from pytcl.mathematical_functions.numerical_integration.quadrature import quad
from pytcl.mathematical_functions.numerical_integration.quadrature import romberg
from pytcl.mathematical_functions.numerical_integration.quadrature import simpson
from pytcl.mathematical_functions.numerical_integration.quadrature import (
    spherical_cubature,
)
from pytcl.mathematical_functions.numerical_integration.quadrature import tplquad
from pytcl.mathematical_functions.numerical_integration.quadrature import trapezoid
from pytcl.mathematical_functions.numerical_integration.quadrature import (
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
