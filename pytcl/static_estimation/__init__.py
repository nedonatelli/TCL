"""
Static estimation module.

This module provides methods for static (batch) estimation including
least squares variants and robust estimation techniques.
"""

from pytcl.static_estimation.least_squares import LSResult
from pytcl.static_estimation.least_squares import TLSResult
from pytcl.static_estimation.least_squares import WLSResult
from pytcl.static_estimation.least_squares import generalized_least_squares
from pytcl.static_estimation.least_squares import ordinary_least_squares
from pytcl.static_estimation.least_squares import recursive_least_squares
from pytcl.static_estimation.least_squares import ridge_regression
from pytcl.static_estimation.least_squares import total_least_squares
from pytcl.static_estimation.least_squares import weighted_least_squares
from pytcl.static_estimation.maximum_likelihood import CRBResult
from pytcl.static_estimation.maximum_likelihood import MLResult
from pytcl.static_estimation.maximum_likelihood import aic
from pytcl.static_estimation.maximum_likelihood import aicc
from pytcl.static_estimation.maximum_likelihood import bic
from pytcl.static_estimation.maximum_likelihood import cramer_rao_bound
from pytcl.static_estimation.maximum_likelihood import cramer_rao_bound_biased
from pytcl.static_estimation.maximum_likelihood import efficiency
from pytcl.static_estimation.maximum_likelihood import (
    fisher_information_exponential_family,
)
from pytcl.static_estimation.maximum_likelihood import fisher_information_gaussian
from pytcl.static_estimation.maximum_likelihood import fisher_information_numerical
from pytcl.static_estimation.maximum_likelihood import mle_gaussian
from pytcl.static_estimation.maximum_likelihood import mle_newton_raphson
from pytcl.static_estimation.maximum_likelihood import mle_scoring
from pytcl.static_estimation.maximum_likelihood import observed_fisher_information
from pytcl.static_estimation.robust import RANSACResult
from pytcl.static_estimation.robust import RobustResult
from pytcl.static_estimation.robust import cauchy_weight
from pytcl.static_estimation.robust import huber_regression
from pytcl.static_estimation.robust import huber_rho
from pytcl.static_estimation.robust import huber_weight
from pytcl.static_estimation.robust import irls
from pytcl.static_estimation.robust import mad
from pytcl.static_estimation.robust import ransac
from pytcl.static_estimation.robust import ransac_n_trials
from pytcl.static_estimation.robust import tau_scale
from pytcl.static_estimation.robust import tukey_regression
from pytcl.static_estimation.robust import tukey_rho
from pytcl.static_estimation.robust import tukey_weight

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
