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
from pytcl.mathematical_functions import basic_matrix
from pytcl.mathematical_functions import combinatorics
from pytcl.mathematical_functions import geometry
from pytcl.mathematical_functions import interpolation
from pytcl.mathematical_functions import numerical_integration
from pytcl.mathematical_functions import signal_processing
from pytcl.mathematical_functions import special_functions
from pytcl.mathematical_functions import statistics
from pytcl.mathematical_functions import transforms

# Basic matrix operations
from pytcl.mathematical_functions.basic_matrix import block_diag
from pytcl.mathematical_functions.basic_matrix import chol_semi_def
from pytcl.mathematical_functions.basic_matrix import kron
from pytcl.mathematical_functions.basic_matrix import matrix_sqrt
from pytcl.mathematical_functions.basic_matrix import null_space
from pytcl.mathematical_functions.basic_matrix import pinv_truncated
from pytcl.mathematical_functions.basic_matrix import range_space
from pytcl.mathematical_functions.basic_matrix import tria
from pytcl.mathematical_functions.basic_matrix import tria_sqrt
from pytcl.mathematical_functions.basic_matrix import unvec
from pytcl.mathematical_functions.basic_matrix import vec

# Combinatorics
from pytcl.mathematical_functions.combinatorics import combinations
from pytcl.mathematical_functions.combinatorics import factorial
from pytcl.mathematical_functions.combinatorics import n_choose_k
from pytcl.mathematical_functions.combinatorics import permutation_rank
from pytcl.mathematical_functions.combinatorics import permutation_unrank
from pytcl.mathematical_functions.combinatorics import permutations

# Geometry
from pytcl.mathematical_functions.geometry import bounding_box
from pytcl.mathematical_functions.geometry import convex_hull
from pytcl.mathematical_functions.geometry import line_intersection
from pytcl.mathematical_functions.geometry import point_in_polygon
from pytcl.mathematical_functions.geometry import polygon_area

# Interpolation
from pytcl.mathematical_functions.interpolation import cubic_spline
from pytcl.mathematical_functions.interpolation import interp1d
from pytcl.mathematical_functions.interpolation import interp2d
from pytcl.mathematical_functions.interpolation import linear_interp
from pytcl.mathematical_functions.interpolation import rbf_interpolate

# Numerical integration
from pytcl.mathematical_functions.numerical_integration import gauss_hermite
from pytcl.mathematical_functions.numerical_integration import gauss_legendre
from pytcl.mathematical_functions.numerical_integration import quad
from pytcl.mathematical_functions.numerical_integration import spherical_cubature
from pytcl.mathematical_functions.numerical_integration import (
    unscented_transform_points,
)

# Signal processing
from pytcl.mathematical_functions.signal_processing import butter_design
from pytcl.mathematical_functions.signal_processing import cfar_ca
from pytcl.mathematical_functions.signal_processing import matched_filter

# Special functions
from pytcl.mathematical_functions.special_functions import besseli
from pytcl.mathematical_functions.special_functions import besselj
from pytcl.mathematical_functions.special_functions import besselk
from pytcl.mathematical_functions.special_functions import bessely
from pytcl.mathematical_functions.special_functions import beta
from pytcl.mathematical_functions.special_functions import betaln
from pytcl.mathematical_functions.special_functions import erf
from pytcl.mathematical_functions.special_functions import erfc
from pytcl.mathematical_functions.special_functions import erfinv
from pytcl.mathematical_functions.special_functions import gamma
from pytcl.mathematical_functions.special_functions import gammaln

# Statistics
from pytcl.mathematical_functions.statistics import ChiSquared
from pytcl.mathematical_functions.statistics import Gaussian
from pytcl.mathematical_functions.statistics import MultivariateGaussian
from pytcl.mathematical_functions.statistics import Uniform
from pytcl.mathematical_functions.statistics import mad
from pytcl.mathematical_functions.statistics import nees
from pytcl.mathematical_functions.statistics import nis
from pytcl.mathematical_functions.statistics import weighted_cov
from pytcl.mathematical_functions.statistics import weighted_mean

# Transforms
from pytcl.mathematical_functions.transforms import cwt
from pytcl.mathematical_functions.transforms import fft
from pytcl.mathematical_functions.transforms import ifft
from pytcl.mathematical_functions.transforms import power_spectrum
from pytcl.mathematical_functions.transforms import spectrogram
from pytcl.mathematical_functions.transforms import stft

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
