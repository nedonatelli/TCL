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
from pytcl.mathematical_functions import (
    basic_matrix,
    combinatorics,
    geometry,
    interpolation,
    numerical_integration,
    signal_processing,
    special_functions,
    statistics,
    transforms,
)

# Basic matrix operations
from pytcl.mathematical_functions.basic_matrix import (
    block_diag,
    chol_semi_def,
    kron,
    matrix_sqrt,
    null_space,
    pinv_truncated,
    range_space,
    tria,
    tria_sqrt,
    unvec,
    vec,
)

# Combinatorics
from pytcl.mathematical_functions.combinatorics import (
    combinations,
    factorial,
    n_choose_k,
    permutation_rank,
    permutation_unrank,
    permutations,
)

# Geometry
from pytcl.mathematical_functions.geometry import (
    bounding_box,
    convex_hull,
    line_intersection,
    point_in_polygon,
    polygon_area,
)

# Interpolation
from pytcl.mathematical_functions.interpolation import (
    cubic_spline,
    interp1d,
    interp2d,
    linear_interp,
    rbf_interpolate,
)

# Numerical integration
from pytcl.mathematical_functions.numerical_integration import (
    gauss_hermite,
    gauss_legendre,
    quad,
    spherical_cubature,
    unscented_transform_points,
)

# Signal processing
from pytcl.mathematical_functions.signal_processing import (
    butter_design,
    cfar_ca,
    matched_filter,
)

# Special functions
from pytcl.mathematical_functions.special_functions import (
    besseli,
    besselj,
    besselk,
    bessely,
    beta,
    betaln,
    erf,
    erfc,
    erfinv,
    gamma,
    gammaln,
)

# Statistics
from pytcl.mathematical_functions.statistics import (
    ChiSquared,
    Gaussian,
    MultivariateGaussian,
    Uniform,
    mad,
    nees,
    nis,
    weighted_cov,
    weighted_mean,
)

# Transforms
from pytcl.mathematical_functions.transforms import (
    cwt,
    fft,
    ifft,
    power_spectrum,
    spectrogram,
    stft,
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
    "signal_processing",
    "transforms",
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
    # Signal processing
    "butter_design",
    "cfar_ca",
    "matched_filter",
    # Transforms
    "fft",
    "ifft",
    "stft",
    "spectrogram",
    "power_spectrum",
    "cwt",
]
