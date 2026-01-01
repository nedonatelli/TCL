"""
Statistics and probability distributions.

This module provides:
- Probability distribution classes with consistent APIs
- Descriptive statistics (mean, variance, correlation)
- Robust estimators (MAD, IQR)
- Filter consistency metrics (NEES, NIS)
"""

from pytcl.mathematical_functions.statistics.distributions import Beta
from pytcl.mathematical_functions.statistics.distributions import ChiSquared
from pytcl.mathematical_functions.statistics.distributions import Distribution
from pytcl.mathematical_functions.statistics.distributions import Exponential
from pytcl.mathematical_functions.statistics.distributions import Gamma
from pytcl.mathematical_functions.statistics.distributions import Gaussian
from pytcl.mathematical_functions.statistics.distributions import MultivariateGaussian
from pytcl.mathematical_functions.statistics.distributions import Poisson
from pytcl.mathematical_functions.statistics.distributions import StudentT
from pytcl.mathematical_functions.statistics.distributions import Uniform
from pytcl.mathematical_functions.statistics.distributions import VonMises
from pytcl.mathematical_functions.statistics.distributions import Wishart
from pytcl.mathematical_functions.statistics.estimators import iqr
from pytcl.mathematical_functions.statistics.estimators import kurtosis
from pytcl.mathematical_functions.statistics.estimators import mad
from pytcl.mathematical_functions.statistics.estimators import median
from pytcl.mathematical_functions.statistics.estimators import moment
from pytcl.mathematical_functions.statistics.estimators import nees
from pytcl.mathematical_functions.statistics.estimators import nis
from pytcl.mathematical_functions.statistics.estimators import sample_corr
from pytcl.mathematical_functions.statistics.estimators import sample_cov
from pytcl.mathematical_functions.statistics.estimators import sample_mean
from pytcl.mathematical_functions.statistics.estimators import sample_var
from pytcl.mathematical_functions.statistics.estimators import skewness
from pytcl.mathematical_functions.statistics.estimators import weighted_cov
from pytcl.mathematical_functions.statistics.estimators import weighted_mean
from pytcl.mathematical_functions.statistics.estimators import weighted_var

__all__ = [
    # Distributions
    "Distribution",
    "Gaussian",
    "MultivariateGaussian",
    "Uniform",
    "Exponential",
    "Gamma",
    "ChiSquared",
    "StudentT",
    "Beta",
    "Poisson",
    "VonMises",
    "Wishart",
    # Estimators
    "weighted_mean",
    "weighted_var",
    "weighted_cov",
    "sample_mean",
    "sample_var",
    "sample_cov",
    "sample_corr",
    "median",
    "mad",
    "iqr",
    "skewness",
    "kurtosis",
    "moment",
    "nees",
    "nis",
]
