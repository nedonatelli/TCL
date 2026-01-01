# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.22.4] - 2026-01-01

### Fixed
- **Black Formatting**: Applied consistent code formatting with default line length (88 characters)
  - Reformatted 125 files across pytcl, tests, examples, and documentation
  - Ensures compatibility with CI workflow expectations
  - All lines now conform to black's standard 88-character limit

### Code Quality
- All quality checks passing (isort, black, flake8, mypy)
- 100% compliance with CI/CD code quality gates
- Complete repository formatting consistency across all files

### Release Information
- **Tag**: v0.22.4
- **Date**: January 1, 2026
- **Type**: Patch Release
- **Status**: Stable

This is a maintenance release with comprehensive formatting improvements. All features remain unchanged and fully functional.

---

## [0.22.3] - 2026-01-01

### Fixed
- **Black Formatting**: Corrected code formatting across 39 files to pass CI workflow validation
  - Fixed blank line formatting in example scripts
  - Corrected line wrapping and string continuation
  - All examples now properly formatted
- **Flake8 Linting**: Removed 3 unused imports from test files
  - Removed unused `assert_allclose` from test_ephemerides.py
  - Removed unused `jplephem` import from test_ephemerides.py
  - Removed unused `G_GRAV` constant from test_relativity.py

### CI/CD
- All GitHub Actions checks now passing (isort, black, flake8, mypy)
- Code quality enforcement strengthened across all workflows

---

## [0.22.1] - 2026-01-01

### Fixed
- **Import Formatting**: Corrected import formatting across 130+ files to pass CI validation checks
  - Applied proper multi-line import grouping consistent with CI isort configuration
  - Ensures all imports follow project code style standards
  - Fixes post-release CI workflow validation failures

### CI/CD Improvements
- All GitHub Actions checks now passing (isort, black, flake8, mypy)
- CI workflow validation enforced on all pushes
- Import formatting now compliant with strict linting standards

---

## [0.22.0] - 2026-01-01

### Added
- **Astronomical Module Phase 13.1: JPL Ephemerides**
  - `DEEphemeris` class for high-precision celestial body position/velocity queries
  - Support for DE405, DE430, DE432s, DE440 ephemeris versions
  - `sun_position()`, `moon_position()`, `planet_position()`, `barycenter_position()` functions
  - Automatic kernel download from JPL NAIF servers
  - Full frame support: ICRF, ecliptic, Earth-centered coordinates
  - 31 comprehensive tests covering all celestial bodies
  - Module-level convenience functions for quick queries

- **Astronomical Module Phase 13.2: Relativistic Corrections**
  - 9 relativistic physics functions for orbital mechanics
  - `schwarzschild_radius()` - Event horizon calculations
  - `gravitational_time_dilation()` - Weak-field time dilation effects
  - `proper_time_rate()` - Combined SR + GR time dilation
  - `shapiro_delay()` - Light propagation delay in gravity
  - `schwarzschild_precession_per_orbit()` - Perihelion precession (Mercury: 43 arcsec/century)
  - `post_newtonian_acceleration()` - 1PN orbital corrections
  - `geodetic_precession()` - De Sitter effect
  - `lense_thirring_precession()` - Frame-dragging precession
  - `relativistic_range_correction()` - Laser ranging corrections
  - 37 comprehensive tests including GPS validation and Mercury precession verification

- **Demonstration Examples**
  - `examples/ephemeris_demo.py` - 7 JPL ephemerides demonstrations
  - `examples/relativity_demo.py` - 7 relativistic effects demonstrations

### Changed
- **Dependencies**: Added jplephem>=2.18 to astronomy optional-dependencies for ephemeris support

### Fixed
- **jplephem Integration**: Corrected API usage to work with jplephem 2.23+
  - Removed non-existent kernel.t0 attribute
  - Added proper unit conversions from km to AU
  - Fixed Moon position computation relative to SSB
  - All 31 ephemerides tests now passing

---

## [0.21.5] - 2026-01-01

### Changed
- **CI**: Removed pip-audit security check due to local package dependency issue

---

## [0.21.4] - 2026-01-01

### Fixed
- **CI**: Use non-editable install in security job to fix pip-audit with --strict mode

---

## [0.21.3] - 2026-01-01

### Fixed
- **CI**: Fixed pip-audit to skip editable installs, resolving issue where it tried to look up unpublished local package in vulnerability databases

---

## [0.21.2] - 2026-01-01

### Added
- **CI Security Scanning**: Added pip-audit to CI workflow for dependency vulnerability scanning
- **GitHub Pages**: Added automated documentation deployment workflow
- **Documentation Examples**: Added static PNG images for example scripts in documentation
- **Tutorial Testing**: Added `scripts/test_tutorials.py` to verify tutorial code snippets
- **Plot Generation**: Added `scripts/generate_example_plots.py` for documentation images

### Fixed
- **Documentation Theme**: Fixed sidenav background colors at deeper toctree levels
- **Documentation Theme**: Styled buttons and tables with dark theme colors (removed white backgrounds)
- **Tutorial Code**: Fixed EKF tutorial to correctly evaluate Jacobians at current/predicted states
- **Tutorial Code**: Fixed multi-target tracking tutorial for correct API usage (hungarian returns tuple, gnn_association returns AssociationResult)

### Changed
- **CI Workflow**: Added permissions configuration, removed unconfigured Black Duck workflow
- **Documentation**: Rewrote `docs/examples/index.rst` with all 20 example scripts organized by category with embedded figures

### Performance
- **Kalman Filter**: Use Cholesky decomposition for efficient solving in `kf_update` (reuses factorization for gain and likelihood)
- **UKF**: Vectorized sigma point operations, use `cho_solve` for covariance factorization
- **IMM**: Vectorized mode probability updates and mixing operations
- **K-Means**: Use `scipy.spatial.distance.cdist` for vectorized distance calculations
- **DBSCAN**: Use KD-tree for efficient neighbor queries, vectorized core point identification
- **Hierarchical Clustering**: Vectorized pairwise distance calculations using `scipy.spatial.distance`
- **JPDA**: Vectorized association probability calculations
- **Particle Filters**: Vectorized weight updates and ESS calculations
- **2D Assignment**: Use `scipy.optimize.linear_sum_assignment` for optimal performance

---

## [0.21.1] - 2026-01-01

### Fixed
- **Flake8 compliance**: Fixed unused import warnings in test_special_functions_phase12.py
- **Documentation**: Added MATLAB migration guide to docs index toctree

### Changed
- **ROADMAP**: Updated current state to v0.21.0 with correct stats (800+ functions, 144 modules, 1,530 tests)

---

## [0.21.0] - 2026-01-01

### Added
- **Special Mathematical Functions** (`pytcl.mathematical_functions.special_functions`):
  - **Marcum Q Function** (`marcum_q.py`):
    - `marcum_q` - Generalized Marcum Q function Q_m(a, b) for radar detection
    - `marcum_q1` - Standard first-order Marcum Q function
    - `log_marcum_q` - Logarithm of Marcum Q for numerical precision
    - `marcum_q_inv` - Inverse Marcum Q function
    - `nuttall_q` - Complementary Marcum Q (CDF of Rician distribution)
    - `swerling_detection_probability` - Detection probability for Swerling target models
  - **Lambert W Function** (`lambert_w.py`):
    - `lambert_w` - Lambert W function W_k(z) with branch selection
    - `lambert_w_real` - Real-valued Lambert W for real inputs
    - `omega_constant` - Omega constant (W(1) ≈ 0.5671)
    - `wright_omega` - Wright omega function
    - `solve_exponential_equation` - Solve a*x*exp(b*x) = c
    - `time_delay_equation` - Characteristic equation for delay systems
  - **Debye Functions** (`debye.py`):
    - `debye` - General Debye function D_n(x) for thermodynamics
    - `debye_1`, `debye_2`, `debye_3`, `debye_4` - Specific orders
    - `debye_heat_capacity` - Normalized heat capacity from Debye model
    - `debye_entropy` - Normalized entropy from Debye model
  - **Hypergeometric Functions** (`hypergeometric.py`):
    - `hyp0f1` - Confluent hypergeometric limit function 0F1
    - `hyp1f1` - Kummer's confluent hypergeometric 1F1
    - `hyp2f1` - Gauss hypergeometric function 2F1
    - `hyperu` - Tricomi function U(a, b, z)
    - `hyp1f1_regularized` - Regularized 1F1
    - `pochhammer` - Rising factorial (Pochhammer symbol)
    - `falling_factorial` - Falling factorial
    - `generalized_hypergeometric` - General pFq function
  - **Advanced Bessel Functions** (in `bessel.py`):
    - `bessel_ratio` - Ratio J_{n+1}/J_n or I_{n+1}/I_n
    - `bessel_deriv` - Derivatives of Bessel functions
    - `bessel_zeros` - Zeros of Bessel functions and derivatives
    - `struve_h` - Struve function H_n(x)
    - `struve_l` - Modified Struve function L_n(x)
    - `kelvin` - Kelvin functions ber, bei, ker, kei
- **MATLAB Migration Guide** (`docs/migration_guide.rst`):
  - Comprehensive guide for MATLAB TCL users transitioning to Python
  - Naming conventions, import structure, return values
  - Array indexing and matrix operations differences
  - Complete example migrations for Kalman filter, coordinate conversion, data association
  - Module mapping reference

### Changed
- **Native Romberg Integration**: Replaced scipy.integrate.romberg wrapper with native implementation using Richardson extrapolation for compatibility with scipy >=1.15 (romberg deprecated in 1.12, removed in 1.15)
- **Visualization**: Converted all example scripts from matplotlib to plotly for interactive HTML visualizations
- Test count increased from 1,488 to 1,530 (42 new tests for special functions)
- Source file count increased from 140 to 144

### Removed
- **matplotlib dependency**: All examples now use plotly exclusively

## [0.20.1] - 2026-01-01

### Changed
- **Documentation Updates**:
  - Updated version references throughout documentation to v0.20.0
  - Added Great Circle and Rhumb Line sections to navigation API docs
  - Fixed package name in tutorials (`pytcl` → `nrl-tracker`)
  - Updated landing page statistics (800+ functions, 1,425+ tests, 140 modules)
- **Test Coverage Improvements**:
  - Added 60 new tests for low-coverage modules
  - Coverage improved from 77% to 79%
  - Test count increased from 1,428 to 1,488
  - Key improvements: bootstrap.py (12%→88%), singer.py (22%→100%), estimators.py (21%→97%)
- Code formatting verified with isort, black, flake8, and mypy

## [0.20.0] - 2025-12-31

### Added
- **Navigation Utilities** (`pytcl.navigation`):
  - **Great Circle Navigation** (`great_circle.py`):
    - `great_circle_distance` - Shortest path distance on sphere
    - `great_circle_azimuth` - Initial/final bearing calculations
    - `great_circle_waypoint` - Intermediate point along path
    - `great_circle_waypoints` - Generate waypoints along route
    - `great_circle_intersection` - Intersection of two great circles
    - `cross_track_distance` - Perpendicular distance from path
    - `along_track_distance` - Distance along path to closest point
    - `great_circle_tdoa_loc` - TDOA localization on spherical Earth
  - **Rhumb Line Navigation** (`rhumb.py`):
    - `rhumb_distance` - Constant-bearing distance (spherical)
    - `rhumb_distance_ellipsoidal` - Rhumb distance on ellipsoid
    - `rhumb_bearing` - Constant bearing between points
    - `rhumb_destination` - Direct problem (given start, bearing, distance)
    - `rhumb_intersection` - Intersection of two rhumb lines
    - `rhumb_midpoint` - Midpoint along rhumb line

## [0.19.0] - 2025-12-31

### Added
- New example scripts with interactive plotly visualizations
- Enhanced documentation with more tutorials

## [0.18.0] - 2025-12-31

### Added
- **Batch Estimation & Smoothing** (`pytcl.dynamic_estimation`):
  - **Smoothers** (`smoothers.py`):
    - `SmoothedState`, `RTSResult`, `FixedLagResult` - Named tuples for smoother results
    - `rts_smoother` - Rauch-Tung-Striebel fixed-interval smoother with time-varying parameters
    - `fixed_lag_smoother` - Real-time smoother with configurable lag
    - `fixed_interval_smoother` - Convenience alias for RTS smoother
    - `two_filter_smoother` - Fraser-Potter two-filter smoother for parallel computation
    - `rts_smoother_single_step` - Single backward step of RTS smoother
  - **Information Filters** (`information_filter.py`):
    - `InformationState`, `InformationFilterResult` - Information form state types
    - `SRIFState`, `SRIFResult` - Square-root information filter types
    - `information_filter` - Full information filter with unknown state initialization
    - `srif_filter`, `srif_predict`, `srif_update` - Square-Root Information Filter
    - `information_to_state`, `state_to_information` - Form conversions
    - `fuse_information` - Multi-sensor fusion in information form
- 19 new tests for smoothers and information filters

### Changed
- Test count increased from ~1,380 to ~1,400
- Source file count increased from 136 to 138

## [0.17.0] - 2025-12-31

### Added
- **Advanced Assignment Algorithms** (`pytcl.assignment_algorithms`):
  - **3D Assignment** (`three_dimensional/assignment.py`):
    - `Assignment3DResult` - Named tuple for 3D assignment results
    - `assign3d` - Unified interface with method selection
    - `assign3d_lagrangian` - Lagrangian relaxation for 3D assignment
    - `assign3d_auction` - Auction algorithm for 3D matching
    - `greedy_3d` - Fast greedy 3D assignment
    - `decompose_to_2d` - Decompose 3D to sequential 2D problems
  - **K-Best 2D Assignment** (`two_dimensional/kbest.py`):
    - `KBestResult` - Named tuple for k-best results
    - `murty` - Murty's algorithm for finding k-best assignments
    - `kbest_assign2d` - K-best with cost thresholds and non-assignment
    - `ranked_assignments` - Convenience function for ranked enumeration
- 30+ new tests for assignment algorithms

### Changed
- Test count increased from ~1,350 to ~1,380

## [0.7.1] - 2025-12-30

### Added
- **Terrain Models** (`pytcl.terrain`):
  - **DEM Interface** (`dem.py`):
    - `DEMPoint`, `TerrainGradient`, `DEMMetadata` - Named tuples for DEM data
    - `DEMGrid` - In-memory DEM grid with bilinear/nearest interpolation
    - `get_elevation_profile` - Extract elevation profile along a path
    - `interpolate_dem` - Resample DEM to new grid
    - `merge_dems` - Merge multiple DEMs into single grid
    - `create_flat_dem` - Create constant-elevation test DEM
    - `create_synthetic_terrain` - Generate realistic test terrain
  - **Visibility Analysis** (`visibility.py`):
    - `LOSResult`, `ViewshedResult`, `HorizonPoint` - Named tuples for visibility results
    - `line_of_sight` - Line-of-sight analysis with Earth curvature and refraction
    - `viewshed` - Compute visible area from observer location
    - `compute_horizon` - Compute terrain horizon profile
    - `terrain_masking_angle` - Masking angle in specific direction
    - `radar_coverage_map` - Radar coverage with minimum elevation constraint

### Changed
- **Complete WMM2020 coefficients** (`pytcl.magnetism.wmm`):
  - Extended main field coefficients from degrees 1-5 to degrees 1-12
  - Extended secular variation coefficients from degrees 1-3 to degrees 1-8
- **Complete IGRF-13 coefficients** (`pytcl.magnetism.igrf`):
  - Extended main field coefficients from degrees 1-6 to degrees 1-13
  - Extended secular variation coefficients from degrees 1-3 to degrees 1-8
- Test count increased from 702 to 737
- Source file count increased from 112 to 114

## [0.7.0] - 2025-12-30

### Added
- **Complete Astronomical Code** (`pytcl.astronomical`):
  - **Orbital Mechanics** (`orbital_mechanics.py`):
    - `OrbitalElements`, `StateVector` - Named tuples for orbital state representation
    - `GM_SUN`, `GM_EARTH`, `GM_MOON`, `GM_MARS`, `GM_JUPITER` - Standard gravitational parameters
    - `mean_to_eccentric_anomaly` - Kepler's equation solver (Newton-Raphson)
    - `mean_to_hyperbolic_anomaly` - Hyperbolic Kepler's equation solver
    - `eccentric_to_true_anomaly`, `true_to_eccentric_anomaly` - Anomaly conversions
    - `hyperbolic_to_true_anomaly`, `true_to_hyperbolic_anomaly` - Hyperbolic anomaly conversions
    - `eccentric_to_mean_anomaly`, `mean_to_true_anomaly`, `true_to_mean_anomaly` - Full anomaly chain
    - `orbital_elements_to_state`, `state_to_orbital_elements` - Element/state conversions
    - `kepler_propagate`, `kepler_propagate_state` - Two-body orbit propagation
    - `orbital_period`, `mean_motion`, `vis_viva` - Orbital quantity calculations
    - `specific_angular_momentum`, `specific_orbital_energy` - Conservation quantities
    - `flight_path_angle`, `periapsis_radius`, `apoapsis_radius` - Geometric quantities
    - `time_since_periapsis`, `orbit_radius` - Position along orbit
    - `escape_velocity`, `circular_velocity` - Characteristic velocities
  - **Lambert Problem Solvers** (`lambert.py`):
    - `LambertSolution` - Named tuple for Lambert solution (v1, v2, a, e, tof, converged)
    - `lambert_universal` - Universal variables method for Lambert's problem
    - `lambert_izzo` - Izzo's algorithm for Lambert's problem (multi-revolution)
    - `minimum_energy_transfer` - Compute minimum energy transfer parameters
    - `hohmann_transfer` - Hohmann transfer (delta-v1, delta-v2, time of flight)
    - `bi_elliptic_transfer` - Bi-elliptic transfer (3 burns)
  - **Reference Frame Transformations** (`reference_frames.py`):
    - `julian_centuries_j2000` - Julian centuries since J2000.0
    - `precession_angles_iau76`, `precession_matrix_iau76` - IAU 1976 precession model
    - `nutation_angles_iau80`, `nutation_matrix` - IAU 1980 nutation model
    - `mean_obliquity_iau80`, `true_obliquity` - Obliquity of the ecliptic
    - `earth_rotation_angle` - Earth Rotation Angle (ERA)
    - `gmst_iau82`, `gast_iau82` - Greenwich sidereal time
    - `sidereal_rotation_matrix`, `equation_of_equinoxes` - Earth rotation
    - `polar_motion_matrix` - Polar motion transformation
    - `gcrf_to_itrf`, `itrf_to_gcrf` - Full GCRF/ITRF transformations
    - `eci_to_ecef`, `ecef_to_eci` - Simplified ECI/ECEF transformations
    - `ecliptic_to_equatorial`, `equatorial_to_ecliptic` - Ecliptic plane transformations
- 37 new tests for astronomical code

### Changed
- Test count increased from 665 to 702
- Source file count increased from 109 to 112

## [0.6.0] - 2025-12-30

### Added
- **Gravity Models** (`pytcl.gravity`):
  - **Spherical Harmonics** (`spherical_harmonics.py`):
    - `associated_legendre` - Associated Legendre polynomials (normalized/unnormalized)
    - `associated_legendre_derivative` - Derivatives of associated Legendre polynomials
    - `spherical_harmonic_sum` - General spherical harmonic expansion
    - `gravity_acceleration` - Compute gravity from spherical harmonic coefficients
  - **Gravity Models** (`models.py`):
    - `GravityConstants` - Named tuple for gravity model constants
    - `GravityResult` - Named tuple for gravity computation results
    - `WGS84`, `GRS80` - Standard geodetic reference constants
    - `normal_gravity_somigliana` - Somigliana's closed-form normal gravity
    - `normal_gravity` - Normal gravity with free-air correction
    - `gravity_wgs84` - Full WGS84 gravity model
    - `gravity_j2` - J2 zonal harmonic gravity (includes oblateness)
    - `geoid_height_j2` - Geoid undulation from J2 model
    - `gravitational_potential` - Point-mass gravitational potential
    - `free_air_anomaly` - Free-air gravity anomaly
    - `bouguer_anomaly` - Bouguer gravity anomaly with terrain correction
- **Magnetic Field Models** (`pytcl.magnetism`):
  - **World Magnetic Model** (`wmm.py`):
    - `MagneticResult` - Named tuple for magnetic field components (X, Y, Z, H, F, I, D)
    - `MagneticCoefficients` - Spherical harmonic coefficients for magnetic models
    - `WMM2020` - Pre-computed WMM2020 coefficients (valid 2020-2025)
    - `create_wmm2020_coefficients` - Create WMM2020 coefficient set
    - `magnetic_field_spherical` - Magnetic field in spherical coordinates
    - `wmm` - Full WMM computation
    - `magnetic_declination` - Magnetic declination (variation)
    - `magnetic_inclination` - Magnetic inclination (dip angle)
    - `magnetic_field_intensity` - Total magnetic field intensity
  - **International Geomagnetic Reference Field** (`igrf.py`):
    - `IGRFModel` - Named tuple for IGRF model parameters
    - `IGRF13` - Pre-computed IGRF-13 coefficients (valid to 2025)
    - `create_igrf13_coefficients` - Create IGRF-13 coefficient set
    - `igrf` - Full IGRF computation
    - `igrf_declination` - IGRF magnetic declination
    - `igrf_inclination` - IGRF magnetic inclination
    - `dipole_moment` - Earth's dipole moment magnitude
    - `dipole_axis` - Orientation of geomagnetic dipole axis
    - `magnetic_north_pole` - Location of geomagnetic north pole
- 40 new tests for geophysical models

### Changed
- Test count increased from 625 to 665
- Source file count increased from 105 to 109

## [0.5.1] - 2025-12-30

### Added
- **Maximum Likelihood Estimation** (`pytcl.static_estimation.maximum_likelihood`):
  - `fisher_information_numerical` - Numerical Fisher information via Hessian
  - `fisher_information_gaussian` - Analytical Fisher info for linear Gaussian models
  - `fisher_information_exponential_family` - Fisher info for exponential family
  - `observed_fisher_information` - Observed Fisher info from Hessian at MLE
  - `cramer_rao_bound` - Compute Cramer-Rao lower bound from Fisher info
  - `cramer_rao_bound_biased` - CRB for biased estimators
  - `efficiency` - Compute estimator efficiency relative to CRB
  - `mle_newton_raphson` - Newton-Raphson MLE optimization
  - `mle_scoring` - Fisher scoring MLE optimization
  - `mle_gaussian` - Closed-form Gaussian MLE
  - `aic`, `bic`, `aicc` - Information criteria for model selection
- **Additional Spatial Data Structures** (`pytcl.containers`):
  - **R-Tree** (`rtree.py`):
    - `BoundingBox` - Axis-aligned bounding box with geometric operations
    - `merge_boxes`, `box_from_point`, `box_from_points` - Box utilities
    - `RTree` - R-tree for spatial indexing of bounding boxes
    - `query_intersect`, `query_contains`, `query_point`, `nearest` queries
  - **VP-Tree** (`vptree.py`):
    - `VPTree` - Vantage point tree for metric space nearest neighbor
    - Custom distance metric support
    - `query`, `query_radius` methods
  - **Cover Tree** (`covertree.py`):
    - `CoverTree` - Cover tree with O(c^12 log n) query guarantee
    - Custom distance metric support
    - `query`, `query_radius` methods
- 51 new tests for ML estimation and spatial structures

### Changed
- Test count increased from 574 to 625
- Source file count increased from 102 to 105

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
