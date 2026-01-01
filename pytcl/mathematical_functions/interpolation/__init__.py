"""
Interpolation methods.

This module provides:
- 1D interpolation (linear, spline, PCHIP, Akima)
- 2D/3D interpolation on regular grids
- RBF interpolation for scattered data
- Spherical interpolation
"""

from pytcl.mathematical_functions.interpolation.interpolation import akima  # noqa: E501
from pytcl.mathematical_functions.interpolation.interpolation import (
    barycentric,
    cubic_spline,
    interp1d,
    interp2d,
    interp3d,
    krogh,
    linear_interp,
    pchip,
    rbf_interpolate,
    spherical_interp,
)

__all__ = [
    "interp1d",
    "linear_interp",
    "cubic_spline",
    "pchip",
    "akima",
    "interp2d",
    "interp3d",
    "rbf_interpolate",
    "barycentric",
    "krogh",
    "spherical_interp",
]
