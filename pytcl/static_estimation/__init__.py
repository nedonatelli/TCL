"""
Static estimation module.

This module provides methods for static (batch) estimation including
least squares variants and robust estimation techniques.
"""

from pytcl.static_estimation.least_squares import (
    LSResult,
    TLSResult,
    WLSResult,
    generalized_least_squares,
    ordinary_least_squares,
    recursive_least_squares,
    ridge_regression,
    total_least_squares,
    weighted_least_squares,
)
from pytcl.static_estimation.maximum_likelihood import (
    CRBResult,
    MLResult,
    aic,
    aicc,
    bic,
    cramer_rao_bound,
    cramer_rao_bound_biased,
    efficiency,
    fisher_information_exponential_family,
    fisher_information_gaussian,
    fisher_information_numerical,
    mle_gaussian,
    mle_newton_raphson,
    mle_scoring,
    observed_fisher_information,
)
from pytcl.static_estimation.robust import (
    RANSACResult,
    RobustResult,
    cauchy_weight,
    huber_regression,
    huber_rho,
    huber_weight,
    irls,
    mad,
    ransac,
    ransac_n_trials,
    tau_scale,
    tukey_regression,
    tukey_rho,
    tukey_weight,
)

__all__ = [
    # Least squares results
    "LSResult",
    "WLSResult",
    "TLSResult",
    # Least squares functions
    "ordinary_least_squares",
    "weighted_least_squares",
    "total_least_squares",
    "generalized_least_squares",
    "recursive_least_squares",
    "ridge_regression",
    # Robust results
    "RobustResult",
    "RANSACResult",
    # Weight functions
    "huber_weight",
    "huber_rho",
    "tukey_weight",
    "tukey_rho",
    "cauchy_weight",
    # Scale estimators
    "mad",
    "tau_scale",
    # Robust estimators
    "irls",
    "huber_regression",
    "tukey_regression",
    "ransac",
    "ransac_n_trials",
    # Maximum likelihood results
    "MLResult",
    "CRBResult",
    # Fisher information
    "fisher_information_numerical",
    "fisher_information_gaussian",
    "fisher_information_exponential_family",
    "observed_fisher_information",
    # Cramer-Rao bounds
    "cramer_rao_bound",
    "cramer_rao_bound_biased",
    "efficiency",
    # MLE algorithms
    "mle_newton_raphson",
    "mle_scoring",
    "mle_gaussian",
    # Information criteria
    "aic",
    "bic",
    "aicc",
]
