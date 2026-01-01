"""
Performance evaluation module.

This module provides metrics for evaluating tracking and estimation performance,
including:

- **Track metrics**: OSPA, MOTA/MOTP, track purity, fragmentation
- **Estimation metrics**: RMSE, NEES, NIS, consistency tests

Examples
--------
>>> from pytcl.performance_evaluation import ospa, rmse, nees
>>> import numpy as np

>>> # OSPA between two point sets
>>> X = [np.array([0, 0]), np.array([10, 10])]
>>> Y = [np.array([1, 0]), np.array([9, 11])]
>>> result = ospa(X, Y, c=100, p=2)
>>> print(f"OSPA: {result.ospa:.2f}")  # doctest: +SKIP
OSPA: 1.12

>>> # RMSE between true and estimated states
>>> true = np.array([[0, 0], [1, 1], [2, 2]])
>>> est = np.array([[0.1, -0.1], [1.1, 0.9], [2.0, 2.1]])
>>> print(f"RMSE: {rmse(true, est):.3f}")  # doctest: +SKIP
RMSE: 0.100
"""

# Estimation metrics
from pytcl.performance_evaluation.estimation_metrics import ConsistencyResult
from pytcl.performance_evaluation.estimation_metrics import average_nees
from pytcl.performance_evaluation.estimation_metrics import consistency_test
from pytcl.performance_evaluation.estimation_metrics import credibility_interval
from pytcl.performance_evaluation.estimation_metrics import estimation_error_bounds
from pytcl.performance_evaluation.estimation_metrics import monte_carlo_rmse
from pytcl.performance_evaluation.estimation_metrics import nees
from pytcl.performance_evaluation.estimation_metrics import nees_sequence
from pytcl.performance_evaluation.estimation_metrics import nis
from pytcl.performance_evaluation.estimation_metrics import nis_sequence
from pytcl.performance_evaluation.estimation_metrics import position_rmse
from pytcl.performance_evaluation.estimation_metrics import rmse
from pytcl.performance_evaluation.estimation_metrics import velocity_rmse

# Track metrics
from pytcl.performance_evaluation.track_metrics import MOTMetrics
from pytcl.performance_evaluation.track_metrics import OSPAResult
from pytcl.performance_evaluation.track_metrics import identity_switches
from pytcl.performance_evaluation.track_metrics import mot_metrics
from pytcl.performance_evaluation.track_metrics import ospa
from pytcl.performance_evaluation.track_metrics import ospa_over_time
from pytcl.performance_evaluation.track_metrics import track_fragmentation
from pytcl.performance_evaluation.track_metrics import track_purity

__all__ = [
    # Track metrics
    "OSPAResult",
    "MOTMetrics",
    "ospa",
    "ospa_over_time",
    "track_purity",
    "track_fragmentation",
    "identity_switches",
    "mot_metrics",
    # Estimation metrics
    "ConsistencyResult",
    "rmse",
    "position_rmse",
    "velocity_rmse",
    "nees",
    "nees_sequence",
    "average_nees",
    "nis",
    "nis_sequence",
    "consistency_test",
    "credibility_interval",
    "monte_carlo_rmse",
    "estimation_error_bounds",
]
