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
from pytcl.performance_evaluation.estimation_metrics import (
    ConsistencyResult,
    average_nees,
    consistency_test,
    credibility_interval,
    estimation_error_bounds,
    monte_carlo_rmse,
    nees,
    nees_sequence,
    nis,
    nis_sequence,
    position_rmse,
    rmse,
    velocity_rmse,
)

# Track metrics
from pytcl.performance_evaluation.track_metrics import (
    MOTMetrics,
    OSPAResult,
    identity_switches,
    mot_metrics,
    ospa,
    ospa_over_time,
    track_fragmentation,
    track_purity,
)

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
