"""
Plotting utilities for tracking and estimation visualization.

This module provides comprehensive plotting functions for:

- **Covariance ellipses**: 2D/3D uncertainty visualization
- **Tracks**: Trajectory and multi-target track plotting
- **Coordinates**: Coordinate system and rotation visualization
- **Metrics**: Performance metric visualization (RMSE, NEES, OSPA)

All plotting functions use Plotly for interactive visualizations.

Examples
--------
>>> from pytcl.plotting import covariance_ellipse_points, plot_tracking_result
>>> import numpy as np

>>> # Generate covariance ellipse points
>>> mean = [0, 0]
>>> cov = [[1, 0.5], [0.5, 2]]
>>> x, y = covariance_ellipse_points(mean, cov, n_std=2.0)

>>> # Plot tracking results
>>> true_states = np.random.randn(50, 4)
>>> estimates = true_states + 0.1 * np.random.randn(50, 4)
>>> fig = plot_tracking_result(true_states, estimates)  # doctest: +SKIP

Notes
-----
Plotly is required for all plotting functions. Install with:
    pip install plotly
"""

# Coordinate system visualization
from pytcl.plotting.coordinates import (
    plot_coordinate_axes_3d,
    plot_coordinate_transform,
    plot_euler_angles,
    plot_points_spherical,
    plot_quaternion_interpolation,
    plot_rotation_comparison,
    plot_spherical_grid,
)

# Covariance ellipse utilities
from pytcl.plotting.ellipses import (
    confidence_region_radius,
    covariance_ellipse_points,
    covariance_ellipsoid_points,
    ellipse_parameters,
    plot_covariance_ellipse,
    plot_covariance_ellipses,
    plot_covariance_ellipsoid,
)

# Performance metric visualization
from pytcl.plotting.metrics import (
    plot_cardinality_over_time,
    plot_consistency_summary,
    plot_error_histogram,
    plot_monte_carlo_rmse,
    plot_nees_sequence,
    plot_nis_sequence,
    plot_ospa_over_time,
    plot_rmse_over_time,
)

# Track and trajectory plotting
from pytcl.plotting.tracks import (
    create_animated_tracking,
    plot_estimation_comparison,
    plot_measurements_2d,
    plot_multi_target_tracks,
    plot_state_time_series,
    plot_tracking_result,
    plot_trajectory_2d,
    plot_trajectory_3d,
)

__all__ = [
    # Ellipses
    "covariance_ellipse_points",
    "covariance_ellipsoid_points",
    "ellipse_parameters",
    "confidence_region_radius",
    "plot_covariance_ellipse",
    "plot_covariance_ellipses",
    "plot_covariance_ellipsoid",
    # Tracks
    "plot_trajectory_2d",
    "plot_trajectory_3d",
    "plot_measurements_2d",
    "plot_tracking_result",
    "plot_multi_target_tracks",
    "plot_state_time_series",
    "plot_estimation_comparison",
    "create_animated_tracking",
    # Coordinates
    "plot_coordinate_axes_3d",
    "plot_rotation_comparison",
    "plot_euler_angles",
    "plot_quaternion_interpolation",
    "plot_spherical_grid",
    "plot_points_spherical",
    "plot_coordinate_transform",
    # Metrics
    "plot_rmse_over_time",
    "plot_nees_sequence",
    "plot_nis_sequence",
    "plot_ospa_over_time",
    "plot_cardinality_over_time",
    "plot_error_histogram",
    "plot_consistency_summary",
    "plot_monte_carlo_rmse",
]
