"""
Mathematical functions and utilities.

This module contains a wide variety of mathematical functions including:
- Basic matrix operations
- Combinatorics (permutations, combinations)
- Continuous optimization
- Geometry primitives
- Interpolation methods
- Numerical integration
- Polynomials
- Signal processing
- Special functions
- Statistics and distributions
"""

# Import submodules for easy access
from tcl.mathematical_functions import basic_matrix
from tcl.mathematical_functions import special_functions
from tcl.mathematical_functions import statistics
from tcl.mathematical_functions import numerical_integration
from tcl.mathematical_functions import interpolation
from tcl.mathematical_functions import combinatorics
from tcl.mathematical_functions import geometry

# Re-export commonly used functions at the top level for convenience

# Basic matrix operations
from tcl.mathematical_functions.basic_matrix import (
    chol_semi_def,
    tria,
    tria_sqrt,
    pinv_truncated,
    matrix_sqrt,
    null_space,
    range_space,
    block_diag,
    kron,
    vec,
    unvec,
)

# Special functions
from tcl.mathematical_functions.special_functions import (
    gamma,
    gammaln,
    beta,
    betaln,
    erf,
    erfc,
    erfinv,
    besselj,
    bessely,
    besseli,
    besselk,
)

# Statistics
from tcl.mathematical_functions.statistics import (
    Gaussian,
    MultivariateGaussian,
    Uniform,
    ChiSquared,
    nees,
    nis,
    weighted_mean,
    weighted_cov,
    mad,
)

# Numerical integration
from tcl.mathematical_functions.numerical_integration import (
    gauss_legendre,
    gauss_hermite,
    quad,
    spherical_cubature,
    unscented_transform_points,
)

# Interpolation
from tcl.mathematical_functions.interpolation import (
    interp1d,
    linear_interp,
    cubic_spline,
    interp2d,
    rbf_interpolate,
)

# Combinatorics
from tcl.mathematical_functions.combinatorics import (
    factorial,
    n_choose_k,
    permutations,
    combinations,
    permutation_rank,
    permutation_unrank,
)

# Geometry
from tcl.mathematical_functions.geometry import (
    point_in_polygon,
    convex_hull,
    polygon_area,
    line_intersection,
    bounding_box,
)

__all__ = [
    # Submodules
    "basic_matrix",
    "special_functions",
    "statistics",
    "numerical_integration",
    "interpolation",
    "combinatorics",
    "geometry",
    # Basic matrix
    "chol_semi_def",
    "tria",
    "tria_sqrt",
    "pinv_truncated",
    "matrix_sqrt",
    "null_space",
    "range_space",
    "block_diag",
    "kron",
    "vec",
    "unvec",
    # Special functions
    "gamma",
    "gammaln",
    "beta",
    "betaln",
    "erf",
    "erfc",
    "erfinv",
    "besselj",
    "bessely",
    "besseli",
    "besselk",
    # Statistics
    "Gaussian",
    "MultivariateGaussian",
    "Uniform",
    "ChiSquared",
    "nees",
    "nis",
    "weighted_mean",
    "weighted_cov",
    "mad",
    # Numerical integration
    "gauss_legendre",
    "gauss_hermite",
    "quad",
    "spherical_cubature",
    "unscented_transform_points",
    # Interpolation
    "interp1d",
    "linear_interp",
    "cubic_spline",
    "interp2d",
    "rbf_interpolate",
    # Combinatorics
    "factorial",
    "n_choose_k",
    "permutations",
    "combinations",
    "permutation_rank",
    "permutation_unrank",
    # Geometry
    "point_in_polygon",
    "convex_hull",
    "polygon_area",
    "line_intersection",
    "bounding_box",
]
