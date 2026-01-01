"""
Interpolation methods.

This module provides:
- 1D interpolation (linear, spline, PCHIP, Akima)
- 2D/3D interpolation on regular grids
- RBF interpolation for scattered data
- Spherical interpolation
"""

from pytcl.mathematical_functions.interpolation.interpolation import akima  # noqa: E501
from pytcl.mathematical_functions.interpolation.interpolation import barycentric
from pytcl.mathematical_functions.interpolation.interpolation import cubic_spline
from pytcl.mathematical_functions.interpolation.interpolation import interp1d
from pytcl.mathematical_functions.interpolation.interpolation import interp2d
from pytcl.mathematical_functions.interpolation.interpolation import interp3d
from pytcl.mathematical_functions.interpolation.interpolation import krogh
from pytcl.mathematical_functions.interpolation.interpolation import linear_interp
from pytcl.mathematical_functions.interpolation.interpolation import pchip
from pytcl.mathematical_functions.interpolation.interpolation import rbf_interpolate
from pytcl.mathematical_functions.interpolation.interpolation import spherical_interp

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
