# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] - 2025-12-30

### Added
- **Static Estimation Module** (`pytcl.static_estimation`):
  - **Least Squares** (`least_squares.py`):
    - `ordinary_least_squares` - SVD-based OLS with rank and singular value output
    - `weighted_least_squares` - WLS with weight matrix or diagonal weights
    - `total_least_squares` - TLS for errors-in-variables problems
    - `generalized_least_squares` - GLS for correlated errors
    - `recursive_least_squares` - RLS with forgetting factor for streaming data
    - `ridge_regression` - L2-regularized least squares
  - **Robust Estimation** (`robust.py`):
    - `huber_regression`, `tukey_regression` - M-estimators for robust regression
    - `irls` - Iteratively Reweighted Least Squares framework
    - `huber_weight`, `tukey_weight`, `cauchy_weight` - Weight functions
    - `huber_rho`, `tukey_rho` - Loss (rho) functions
    - `mad`, `tau_scale` - Robust scale estimators
    - `ransac` - RANSAC robust estimation with automatic threshold
    - `ransac_n_trials` - Compute required RANSAC iterations
- **Spatial Data Structures** (`pytcl.containers`):
  - **K-D Tree** (`kd_tree.py`):
    - `KDTree` - K-dimensional tree for O(log n) nearest neighbor queries
    - `BallTree` - Ball tree for high-dimensional nearest neighbor queries
    - `query` - Find k nearest neighbors
    - `query_radius` / `query_ball_point` - Range queries within radius
- 66 new tests for static estimation and spatial data structures

### Changed
- Test count increased from 508 to 574
- Source file count increased from 99 to 102

## [0.4.2] - 2025-12-30

### Fixed
- Fixed flake8 linting errors in test files (unused imports, lambda expressions)

## [0.4.1] - 2025-12-30

### Added
- **DBSCAN Clustering** (`pytcl.clustering.dbscan`):
  - `dbscan` - Density-based clustering algorithm
  - `dbscan_predict` - Predict clusters for new points
  - `compute_neighbors` - Efficient neighbor computation
- **Hierarchical (Agglomerative) Clustering** (`pytcl.clustering.hierarchical`):
  - `agglomerative_clustering` - Hierarchical clustering with 4 linkage methods
  - `cut_dendrogram` - Cut dendrogram at specified level
  - `fcluster` - scipy-compatible cluster extraction
  - Support for single, complete, average, and Ward linkage
- 22 new tests for DBSCAN and hierarchical clustering

### Changed
- Test count increased from 486 to 508
- Source file count increased from 97 to 99

## [0.4.0] - 2025-12-30

### Added
- **Gaussian Mixture Operations** (`pytcl.clustering.gaussian_mixture`):
  - `GaussianComponent`, `GaussianMixture` classes for mixture representation
  - `moment_match` - Compute moment-matched mean and covariance
  - `runnalls_merge_cost`, `west_merge_cost` - Merge cost functions
  - `merge_gaussians` - Merge two Gaussian components
  - `prune_mixture` - Remove low-weight components
  - `reduce_mixture_runnalls` - Runnalls' mixture reduction algorithm
  - `reduce_mixture_west` - West's mixture reduction algorithm
- **K-means Clustering** (`pytcl.clustering.kmeans`):
  - `kmeans` - K-means clustering with K-means++ initialization
  - `kmeans_plusplus_init` - K-means++ initialization
  - `assign_clusters`, `update_centers` - Core K-means operations
  - `kmeans_elbow` - Helper for elbow method analysis
- **Multiple Hypothesis Tracking (MHT)** (`pytcl.trackers.mht`):
  - `MHTTracker` - Track-oriented MHT with N-scan pruning
  - `MHTConfig` - Configuration for MHT parameters
  - `MHTResult` - Result container for MHT processing
  - `HypothesisTree` - Hypothesis tree management
  - `generate_joint_associations` - Enumerate valid associations
  - `n_scan_prune` - N-scan hypothesis pruning
  - `prune_hypotheses_by_probability` - Probability-based pruning
- 78 new tests for v0.4.0 features

### Changed
- Test count increased from 408 to 486
- Source file count increased from 93 to 97

## [0.3.1] - 2025-12-30

### Fixed
- Type annotations: Changed `callable` to `Callable` in `sr_ukf_predict` and `sr_ukf_update` to fix mypy errors

## [0.3.0] - 2025-12-30

### Added
- **Square-Root Kalman Filters** for improved numerical stability:
  - `srkf_predict`, `srkf_update` - Cholesky-based square-root KF
  - `sr_ukf_predict`, `sr_ukf_update` - Square-root UKF
  - `cholesky_update` - Efficient rank-1 Cholesky update/downdate
  - `qr_update` - QR-based covariance propagation
- **U-D Factorization Filter** (Bierman's method):
  - `ud_factorize`, `ud_reconstruct` - U-D decomposition utilities
  - `ud_predict`, `ud_update` - U-D filter predict/update
  - `ud_update_scalar` - Efficient scalar measurement update
- **Interacting Multiple Model (IMM) Estimator**:
  - `imm_predict`, `imm_update` - IMM filter functions
  - `IMMEstimator` class for stateful IMM filtering
  - Mode probability mixing and combination
- **Joint Probabilistic Data Association (JPDA)**:
  - `jpda`, `jpda_update` - JPDA association and update
  - `jpda_probabilities` - Compute association probabilities
  - Support for cluttered environments with detection probability
- Comprehensive documentation for new features:
  - User guides for square-root filters, IMM, and JPDA
  - API reference documentation
  - Data association user guide

### Changed
- Test count increased from 355 to 408
- Test coverage increased from 58% to 61%

## [0.2.2] - 2025-12-30

### Fixed
- Documentation: Updated pip install command to use correct package name `nrl-tracker`
- Documentation: Updated git clone URLs to point to correct repository

## [0.2.1] - 2025-12-30

### Fixed
- Documentation: Updated all Sphinx autodoc imports from `tracker_component_library` to `pytcl` for Read the Docs compatibility

## [0.2.0] - 2025-12-30

### Added
- New `pytcl.plotting` module with 30 visualization functions:
  - `ellipses.py`: Covariance ellipse/ellipsoid utilities (`covariance_ellipse_points`, `plot_covariance_ellipse`, etc.)
  - `tracks.py`: Trajectory visualization (`plot_trajectory_2d/3d`, `plot_tracking_result`, `create_animated_tracking`)
  - `coordinates.py`: Coordinate system visualization (`plot_coordinate_axes_3d`, `plot_euler_angles`, `plot_quaternion_interpolation`)
  - `metrics.py`: Performance metric plots (`plot_nees_sequence`, `plot_ospa_over_time`, `plot_error_histogram`)
- Interactive plotting examples:
  - `coordinate_visualization.py`: 3D rotation and coordinate system visualization
  - `filter_uncertainty_visualization.py`: Kalman filter covariance ellipse animations
- Comprehensive test suite with 170+ new tests
  - `test_coordinate_systems.py`: 53 tests for coordinate transforms and rotations
  - `test_dynamic_models.py`: 35 tests for state transition and process noise
  - `test_kalman_filters.py`: 33 tests for KF, EKF, UKF, CKF
  - `test_mathematical_functions.py`: 49 tests for matrix operations and geometry
  - `test_plotting.py`: 35 tests for plotting module

### Changed
- Test coverage increased from 29% to 58%

## [0.1.2] - 2025-12-30

### Added
- Comprehensive example scripts demonstrating library capabilities:
  - `coordinate_systems.py`: Coordinate transforms (spherical, geodetic, ENU/NED, rotations, quaternions)
  - `kalman_filter_comparison.py`: KF vs EKF vs UKF comparison with NEES/NIS metrics
  - `navigation_geodesy.py`: Geodetic conversions, distance calculations, waypoint navigation
  - `performance_evaluation.py`: OSPA metric, filter consistency testing, Monte Carlo evaluation

### Changed
- Switched visualization from matplotlib to plotly for interactive plots
- Updated all existing examples to use plotly

## [0.1.1] - 2025-12-29

### Added
- Read the Docs configuration
- Package prepared for PyPI publishing
- CI workflow using flake8 for linting

### Changed
- Renamed package from `tracker_component_library` to `tcl` for simpler imports
- Updated all imports across the codebase

## [0.1.0] - 2025-12-28

### Added
- Initial release of the Python port
- Core mathematical functions and utilities
- Coordinate system transformations (Cartesian, spherical, geodetic, ECEF, ENU/NED)
- Dynamic models (constant velocity, constant acceleration, coordinated turn)
- Kalman filters (linear, extended, unscented)
- Assignment algorithms (Hungarian, auction, GNN)
- Multi-target tracking with track management
- Navigation utilities (geodetic calculations, INS algorithms)
- Astronomical functions (ephemerides, celestial mechanics)
- Atmospheric models
- Performance evaluation metrics (OSPA, NEES, NIS)
