"""
Interpolation methods.

This module provides:
- 1D interpolation (linear, spline, PCHIP, Akima)
- 2D/3D interpolation on regular grids
- RBF interpolation for scattered data
- Spherical interpolation
"""

from pytcl.mathematical_functions.interpolation.interpolation import (  # noqa: E501
    interp1d,
    linear_interp,
    cubic_spline,
    pchip,
    akima,
    interp2d,
    interp3d,
    rbf_interpolate,
    barycentric,
    krogh,
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
