"""
Statistics and probability distributions.

This module provides:
- Probability distribution classes with consistent APIs
- Descriptive statistics (mean, variance, correlation)
- Robust estimators (MAD, IQR)
- Filter consistency metrics (NEES, NIS)
"""

from pytcl.mathematical_functions.statistics.distributions import (
    Beta,
    ChiSquared,
    Distribution,
    Exponential,
    Gamma,
    Gaussian,
    MultivariateGaussian,
    Poisson,
    StudentT,
    Uniform,
    VonMises,
    Wishart,
)
from pytcl.mathematical_functions.statistics.estimators import (
    iqr,
    kurtosis,
    mad,
    median,
    moment,
    nees,
    nis,
    sample_corr,
    sample_cov,
    sample_mean,
    sample_var,
    skewness,
    weighted_cov,
    weighted_mean,
    weighted_var,
)

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
