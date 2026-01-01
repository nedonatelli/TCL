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
from pytcl.plotting.coordinates import plot_coordinate_axes_3d
from pytcl.plotting.coordinates import plot_coordinate_transform
from pytcl.plotting.coordinates import plot_euler_angles
from pytcl.plotting.coordinates import plot_points_spherical
from pytcl.plotting.coordinates import plot_quaternion_interpolation
from pytcl.plotting.coordinates import plot_rotation_comparison
from pytcl.plotting.coordinates import plot_spherical_grid

# Covariance ellipse utilities
from pytcl.plotting.ellipses import confidence_region_radius
from pytcl.plotting.ellipses import covariance_ellipse_points
from pytcl.plotting.ellipses import covariance_ellipsoid_points
from pytcl.plotting.ellipses import ellipse_parameters
from pytcl.plotting.ellipses import plot_covariance_ellipse
from pytcl.plotting.ellipses import plot_covariance_ellipses
from pytcl.plotting.ellipses import plot_covariance_ellipsoid

# Performance metric visualization
from pytcl.plotting.metrics import plot_cardinality_over_time
from pytcl.plotting.metrics import plot_consistency_summary
from pytcl.plotting.metrics import plot_error_histogram
from pytcl.plotting.metrics import plot_monte_carlo_rmse
from pytcl.plotting.metrics import plot_nees_sequence
from pytcl.plotting.metrics import plot_nis_sequence
from pytcl.plotting.metrics import plot_ospa_over_time
from pytcl.plotting.metrics import plot_rmse_over_time

# Track and trajectory plotting
from pytcl.plotting.tracks import create_animated_tracking
from pytcl.plotting.tracks import plot_estimation_comparison
from pytcl.plotting.tracks import plot_measurements_2d
from pytcl.plotting.tracks import plot_multi_target_tracks
from pytcl.plotting.tracks import plot_state_time_series
from pytcl.plotting.tracks import plot_tracking_result
from pytcl.plotting.tracks import plot_trajectory_2d
from pytcl.plotting.tracks import plot_trajectory_3d

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
