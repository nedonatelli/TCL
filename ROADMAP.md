# TCL (Tracker Component Library) - Development Roadmap

## Current State (v0.16.0)

- **750+ functions** implemented across 133 Python files
- **1,300+ tests** with comprehensive coverage
- **10 comprehensive example scripts** covering all major library features
- **Core tracking functionality complete**: Kalman filters (KF, EKF, UKF, CKF), particle filters, coordinate systems, dynamic models, data association (GNN, JPDA, MHT), multi-target tracking
- **Gaussian mixture operations**: moment matching, Runnalls/West reduction algorithms
- **Complete clustering module**: K-means, DBSCAN, hierarchical clustering
- **Static estimation**: Least squares (OLS, WLS, TLS, GLS, RLS), robust M-estimators (Huber, Tukey), RANSAC, maximum likelihood estimation, Fisher information, Cramer-Rao bounds
- **Spatial data structures**: K-D tree, Ball tree, R-tree, VP-tree, Cover tree for efficient nearest neighbor queries
- **Tracking containers**: TrackList, MeasurementSet, ClusterSet for managing tracking data
- **Geophysical models**: Gravity (spherical harmonics, WGS84, J2, EGM96/EGM2008), Magnetism (WMM2020, IGRF-13, EMM, WMMHR)
- **Tidal effects**: Solid Earth tides, ocean tide loading, atmospheric pressure loading, pole tide
- **Terrain models**: DEM interface, GEBCO/Earth2014 loaders, line-of-sight, viewshed analysis
- **Map projections**: Mercator, Transverse Mercator, UTM, Stereographic, Lambert Conformal Conic, Azimuthal Equidistant
- **Astronomical code**: Orbital mechanics, Kepler propagation, Lambert problem, reference frame transformations
- **INS/Navigation**: Strapdown INS mechanization, coning/sculling corrections, alignment algorithms, error state model
- **INS/GNSS Integration**: Loosely-coupled and tightly-coupled integration, DOP computation, fault detection
- **Signal Processing**: Digital filter design (IIR/FIR), matched filtering, CFAR detection
- **Transforms**: FFT utilities, STFT/spectrogram, wavelet transforms (CWT, DWT)
- **Published on PyPI** as `nrl-tracker`

---

## Completed in v0.16.0

### Phase 8: Tracking Container Classes
- [x] **TrackList** container - Collection of tracks with filtering, querying, batch operations
- [x] **MeasurementSet** container - Time-indexed measurement collection with spatial queries
- [x] **ClusterSet** container - Track clustering with DBSCAN/K-means support
- [x] Properties for easy access: `confirmed`, `tentative`, `track_ids`, `times`, `sensors`
- [x] Spatial queries: `in_region`, `nearest_to`, `clusters_in_region`
- [x] Batch extraction: `states()`, `covariances()`, `positions()`, `values()`
- [x] Statistics: `TrackListStats`, `ClusterStats` with velocity coherence
- **Files**: `pytcl/containers/track_list.py`, `pytcl/containers/measurement_set.py`, `pytcl/containers/cluster_set.py`

---

## Completed in v0.15.0

### New Comprehensive Examples
- [x] **signal_processing.py** - Digital filters, matched filtering, CFAR detection, spectrum analysis
- [x] **transforms.py** - FFT, STFT, spectrograms, CWT/DWT wavelets
- [x] **ins_gnss_navigation.py** - INS mechanization, GNSS geometry, loosely-coupled integration
- [x] Fixed and verified all 10 example scripts run without errors

---

## Completed in v0.3.0

### Phase 1.1: Square-Root Filters (Numerical Stability)
- [x] Square-root Kalman filter (Cholesky-based) - `srkf_predict`, `srkf_update`
- [x] U-D factorization filter - `ud_factorize`, `ud_reconstruct`, `ud_predict`, `ud_update`
- [x] Square-root UKF - `sr_ukf_predict`, `sr_ukf_update`
- [x] Cholesky update/downdate - `cholesky_update`
- [x] QR-based covariance propagation - `qr_update`

### Phase 1.2: Joint Probabilistic Data Association (JPDA)
- [x] JPDA for multi-target tracking - `jpda`, `jpda_update`
- [x] Association probability computation - `jpda_probabilities`
- [x] Combined update with association probabilities
- [x] Support for cluttered environments with detection probability

### Phase 1.4: Interacting Multiple Model (IMM)
- [x] IMM estimator - `imm_predict`, `imm_update`
- [x] Model probability mixing - `mix_estimates`, `combine_estimates`
- [x] Markov transition matrix handling
- [x] `IMMEstimator` class for stateful filtering

---

## Completed in v0.4.0

### Phase 1.3: Multiple Hypothesis Tracking (MHT)
- [x] Hypothesis tree management - `HypothesisTree` class
- [x] N-scan pruning - `n_scan_prune`
- [x] Track-oriented MHT - `MHTTracker` class
- **Files**: `pytcl/trackers/mht.py`, `pytcl/trackers/hypothesis.py`

### Phase 2.1: Gaussian Mixture Operations
- [x] Gaussian mixture representation class - `GaussianMixture`, `GaussianComponent`
- [x] Mixture moment matching - `moment_match`
- [x] Runnalls' mixture reduction algorithm - `reduce_mixture_runnalls`
- [x] West's algorithm - `reduce_mixture_west`
- **Files**: `pytcl/clustering/gaussian_mixture.py`

### Phase 2.2: Clustering Algorithms
- [x] K-means clustering - `kmeans` with K-means++ initialization
- [x] DBSCAN - `dbscan`, `dbscan_predict`, `compute_neighbors`
- [x] Hierarchical clustering - `agglomerative_clustering`, `cut_dendrogram`, `fcluster`
- **Files**: `pytcl/clustering/kmeans.py`, `pytcl/clustering/dbscan.py`, `pytcl/clustering/hierarchical.py`

---

## Completed in v0.5.0

### Phase 3: Static Estimation
- [x] Ordinary least squares (SVD-based) - `ordinary_least_squares`
- [x] Weighted least squares - `weighted_least_squares`
- [x] Total least squares - `total_least_squares`
- [x] Generalized least squares - `generalized_least_squares`
- [x] Recursive least squares - `recursive_least_squares`
- [x] Ridge regression - `ridge_regression`
- [x] Robust M-estimators - `huber_regression`, `tukey_regression`, `irls`
- [x] RANSAC - `ransac`, `ransac_n_trials`
- [x] Scale estimators - `mad`, `tau_scale`
- **Files**: `pytcl/static_estimation/least_squares.py`, `pytcl/static_estimation/robust.py`

### Phase 4: Spatial Data Structures
- [x] K-D tree - `KDTree` with `query`, `query_radius`
- [x] Ball tree - `BallTree` with `query`
- **Files**: `pytcl/containers/kd_tree.py`

---

## Completed in v0.5.1

### Phase 3 (Completed): Static Estimation - Maximum Likelihood
- [x] ML estimator framework - `mle_newton_raphson`, `mle_scoring`, `mle_gaussian`
- [x] Fisher information computation - `fisher_information_numerical`, `fisher_information_gaussian`
- [x] Cramer-Rao bounds - `cramer_rao_bound`, `cramer_rao_bound_biased`, `efficiency`
- [x] Information criteria - `aic`, `bic`, `aicc`
- **Files**: `pytcl/static_estimation/maximum_likelihood.py`

### Phase 4 (Completed): Container Data Structures - Additional Spatial Structures
- [x] R-tree for bounding boxes - `RTree`, `BoundingBox`, `query_intersect`, `query_contains`
- [x] VP-tree (vantage point tree) - `VPTree` with custom metric support
- [x] Cover tree - `CoverTree` with O(c^12 log n) guarantee
- **Files**: `pytcl/containers/rtree.py`, `pytcl/containers/vptree.py`, `pytcl/containers/covertree.py`

---

## Completed in v0.6.0

### Phase 5.1: Gravity Models
- [x] Spherical harmonic evaluation - `associated_legendre`, `spherical_harmonic_sum`
- [x] WGS84/GRS80 gravity constants - `WGS84`, `GRS80`
- [x] Normal gravity (Somigliana) - `normal_gravity_somigliana`, `normal_gravity`
- [x] J2 gravity model - `gravity_j2`, `geoid_height_j2`
- [x] WGS84 gravity model - `gravity_wgs84`
- [x] Gravity anomalies - `free_air_anomaly`, `bouguer_anomaly`
- **Files**: `pytcl/gravity/spherical_harmonics.py`, `pytcl/gravity/models.py`

### Phase 5.2: Magnetic Field Models
- [x] World Magnetic Model (WMM2020) - `wmm`, `magnetic_declination`, `magnetic_inclination`
- [x] International Geomagnetic Reference Field (IGRF-13) - `igrf`, `igrf_declination`
- [x] Geomagnetic properties - `dipole_moment`, `dipole_axis`, `magnetic_north_pole`
- **Files**: `pytcl/magnetism/wmm.py`, `pytcl/magnetism/igrf.py`

---

## Completed in v0.7.0

### Phase 6.1: Celestial Mechanics
- [x] Two-body orbit propagation - `kepler_propagate`, `kepler_propagate_state`
- [x] Kepler's equation solvers - `mean_to_eccentric_anomaly`, `mean_to_hyperbolic_anomaly`
- [x] Orbital element conversions - `orbital_elements_to_state`, `state_to_orbital_elements`
- [x] Lambert problem solver - `lambert_universal`, `lambert_izzo`
- [x] Hohmann and bi-elliptic transfers - `hohmann_transfer`, `bi_elliptic_transfer`
- **Files**: `pytcl/astronomical/orbital_mechanics.py`, `pytcl/astronomical/lambert.py`

### Phase 6.2: Reference Frame Transformations
- [x] GCRF/ITRF conversions - `gcrf_to_itrf`, `itrf_to_gcrf`
- [x] Precession models (IAU 1976) - `precession_matrix_iau76`, `precession_angles_iau76`
- [x] Nutation models (IAU 1980) - `nutation_matrix`, `nutation_angles_iau80`
- [x] Earth rotation - `gmst_iau82`, `gast_iau82`, `earth_rotation_angle`
- [x] Polar motion corrections - `polar_motion_matrix`
- [x] Ecliptic/equatorial transformations - `ecliptic_to_equatorial`, `equatorial_to_ecliptic`
- **Files**: `pytcl/astronomical/reference_frames.py`

---

## Completed in v0.8.0

### Phase 5.3: Advanced Magnetic Models
- [x] Enhanced Magnetic Model (EMM2017) - `emm`, `emm_declination`, `emm_inclination`
- [x] High-resolution WMM (WMMHR) - `wmmhr`
- [x] Degree 790 spherical harmonic support
- **Files**: `pytcl/magnetism/emm.py`, `pytcl/magnetism/wmmhr.py`

### Phase 5.4: Terrain Models
- [x] Digital elevation model interface - `DEMGrid`, `DEMPoint`, `DEMMetadata`
- [x] GEBCO bathymetry/topography loader - `load_gebco`, `GEBCOMetadata`
- [x] Earth2014 terrain model loader - `load_earth2014`, `Earth2014Metadata`
- [x] Line-of-sight analysis - `line_of_sight`, `LOSResult`
- [x] Viewshed computation - `viewshed`, `ViewshedResult`
- [x] Horizon computation - `compute_horizon`, `HorizonPoint`
- [x] Terrain masking for radar - `terrain_masking_angle`, `radar_coverage_map`
- [x] Synthetic terrain generation - `create_synthetic_terrain`, `create_flat_dem`
- [x] Test data generators - `create_test_gebco_dem`, `create_test_earth2014_dem`
- **Files**: `pytcl/terrain/dem.py`, `pytcl/terrain/loaders.py`, `pytcl/terrain/visibility.py`

---

## Completed in v0.9.0

### Phase 5.5: Map Projections
- [x] Mercator projection - `mercator`, `mercator_inverse`
- [x] Transverse Mercator projection - `transverse_mercator`, `transverse_mercator_inverse`
- [x] UTM with zone handling - `geodetic2utm`, `utm2geodetic`, `utm_zone`, `utm_central_meridian`
- [x] Norway/Svalbard UTM zone exceptions
- [x] Stereographic projection (oblique and polar) - `stereographic`, `stereographic_inverse`, `polar_stereographic`
- [x] Lambert Conformal Conic - `lambert_conformal_conic`, `lambert_conformal_conic_inverse`
- [x] Azimuthal Equidistant - `azimuthal_equidistant`, `azimuthal_equidistant_inverse`
- [x] Batch UTM conversion - `geodetic2utm_batch`
- **Files**: `pytcl/coordinate_systems/projections/projections.py`

---

## Completed in v0.10.0

### Phase 5.6: Tidal Effects
- [x] Solid Earth tide displacement - `solid_earth_tide_displacement`, `TidalDisplacement`
- [x] Solid Earth tide gravity - `solid_earth_tide_gravity`, `TidalGravity`
- [x] Ocean tide loading - `ocean_tide_loading_displacement`, `OceanTideLoading`
- [x] Atmospheric pressure loading - `atmospheric_pressure_loading`
- [x] Pole tide effects - `pole_tide_displacement`
- [x] Combined tidal displacement - `total_tidal_displacement`
- [x] Tidal gravity correction - `tidal_gravity_correction`
- [x] Love/Shida numbers (IERS 2010) - `LOVE_H2`, `LOVE_K2`, `SHIDA_L2`
- [x] Fundamental astronomical arguments - `fundamental_arguments`
- [x] Moon/Sun position (low precision) - `moon_position_approximate`, `sun_position_approximate`
- **Files**: `pytcl/gravity/tides.py`

---

## Completed in v0.7.1

### Phase 5.7: EGM High-Degree Gravity Models
- [x] EGM96 model support (degree 360) - `load_egm_coefficients`, `EGMCoefficients`
- [x] EGM2008 model support (degree 2190) - `parse_egm_file`
- [x] Clenshaw summation for numerical stability - `clenshaw_potential`, `clenshaw_gravity`
- [x] Geoid height computation - `geoid_height`, `geoid_heights`
- [x] Gravity disturbance/anomaly - `gravity_disturbance`, `gravity_anomaly`
- [x] Deflection of vertical - `deflection_of_vertical`
- **Files**: `pytcl/gravity/egm.py`, `pytcl/gravity/clenshaw.py`

---

## Completed in v0.11.0

### Phase 6.3: INS Mechanization
- [x] INS state representation - `INSState`, `IMUData`, `INSErrorState`
- [x] Physical constants (WGS84) - `OMEGA_EARTH`, `GM_EARTH`, `A_EARTH`
- [x] Gravity computation - `normal_gravity`, `gravity_ned` (Somigliana formula)
- [x] Earth/transport rates - `earth_rate_ned`, `transport_rate_ned`, `radii_of_curvature`
- [x] Coning/sculling corrections - `coning_correction`, `sculling_correction`, `compensate_imu_data`
- [x] Attitude update - `update_quaternion`, `update_attitude_ned`, `skew_symmetric`
- [x] Strapdown mechanization (NED) - `mechanize_ins_ned`, `initialize_ins_state`
- [x] Alignment algorithms - `coarse_alignment`, `gyrocompass_alignment`
- [x] Error state model (15-state) - `ins_error_state_matrix`, `ins_process_noise_matrix`
- **Files**: `pytcl/navigation/ins.py`

---

## Completed in v0.12.0

### Phase 6.4: INS/GNSS Integration
- [x] GNSS measurement models - `GNSSMeasurement`, `SatelliteInfo`, `INSGNSSState`
- [x] Measurement matrices - `position_measurement_matrix`, `velocity_measurement_matrix`, `pseudorange_measurement_matrix`
- [x] Satellite geometry - `compute_line_of_sight`, `satellite_elevation_azimuth`, `compute_dop`
- [x] Loosely-coupled integration - `loose_coupled_predict`, `loose_coupled_update`, `loose_coupled_update_position`
- [x] Tightly-coupled integration - `tight_coupled_update`, `tight_coupled_pseudorange_innovation`
- [x] Fault detection - `gnss_outage_detection`
- **Files**: `pytcl/navigation/ins_gnss.py`

---

## Completed in v0.13.0

### Phase 7.1: Signal Processing
- [x] Digital filter design - `butter_design`, `cheby1_design`, `cheby2_design`, `ellip_design`, `bessel_design`
- [x] FIR filter design - `fir_design`, `fir_design_remez`
- [x] Filter application - `apply_filter`, `filtfilt`, `frequency_response`, `group_delay`
- [x] Matched filtering - `matched_filter`, `matched_filter_frequency`, `optimal_filter`
- [x] Pulse compression - `pulse_compression`, `generate_lfm_chirp`, `generate_nlfm_chirp`, `ambiguity_function`
- [x] CFAR detection - `cfar_ca`, `cfar_go`, `cfar_so`, `cfar_os`, `cfar_2d`
- [x] Detection utilities - `threshold_factor`, `detection_probability`, `cluster_detections`, `snr_loss`
- **Files**: `pytcl/mathematical_functions/signal_processing/filters.py`, `matched_filter.py`, `detection.py`

### Phase 7.2: Transforms
- [x] Fourier transforms - `fft`, `ifft`, `rfft`, `irfft`, `fft2`, `ifft2`, `fftshift`, `ifftshift`
- [x] Frequency analysis - `frequency_axis`, `rfft_frequency_axis`, `power_spectrum`, `periodogram`
- [x] Cross-spectral analysis - `cross_spectrum`, `coherence`, `magnitude_spectrum`, `phase_spectrum`
- [x] Short-time Fourier transform - `stft`, `istft`, `spectrogram`, `mel_spectrogram`
- [x] Window functions - `get_window`, `window_bandwidth`
- [x] Wavelet transforms - `cwt`, `dwt`, `idwt`, `dwt_single_level`
- [x] Wavelet functions - `morlet_wavelet`, `ricker_wavelet`, `gaussian_wavelet`, `scales_to_frequencies`
- **Files**: `pytcl/mathematical_functions/transforms/fourier.py`, `stft.py`, `wavelets.py`

---

## Phase 8: Performance & Infrastructure

### 8.1 Performance Optimization
- [x] Numba JIT for CFAR detection (CA, GO, SO, OS, 2D with parallel execution)
- [x] Numba JIT for ambiguity function computation (parallel Doppler-delay loop)
- [x] Numba JIT for batch Mahalanobis distance in data association
- [x] Numba JIT for rotation matrix utilities (inplace operations)
- [ ] Profile and optimize additional bottlenecks
- [ ] Consider Cython for hot spots

### 8.2 Documentation
- [x] User guides for square-root filters, IMM, and JPDA
- [x] API reference documentation for v0.3.0 features
- [x] Data association user guide
- [x] Complete API documentation for all modules (17 new API docs added)
- [x] Tutorials: Kalman filtering, nonlinear filtering, signal processing, radar detection, INS/GNSS, multi-target tracking
- [x] Example scripts: kalman_tracking.py, ukf_range_bearing.py, cfar_detection.py, coordinate_transforms.py, multi_target_tracking.py, spectral_analysis.py
- [x] Custom landing page with animated radar theme
- [x] Dark theme CSS for Sphinx RTD documentation

### 8.3 Testing
- [x] 1,255 tests (up from 1,109 in v0.14.0)
- [x] Test coverage boost: geodetic conversions, Jacobians, matrix decompositions, process noise, interpolation, statistics
- [ ] Increase test coverage to 80%+
- [ ] Add MATLAB validation tests for new functions
- [ ] Performance regression tests

---

## Priority Summary

| Priority | Focus Area | Key Deliverables | Status |
|----------|------------|------------------|--------|
| **P0** | Advanced Data Association | JPDA, MHT, IMM | ✅ Complete |
| **P1** | Clustering | Gaussian mixture, K-means, DBSCAN, hierarchical | ✅ Complete |
| **P2** | Static Estimation | Least squares, robust estimators, RANSAC | ✅ Complete |
| **P2.5** | Spatial Data Structures | K-D tree, Ball tree, R-tree, VP-tree, Cover tree | ✅ Complete |
| **P3** | Geophysical Models | Gravity (WGS84, J2), Magnetism (WMM, IGRF) | ✅ Complete |
| **P3.5** | Advanced Magnetic | EMM, WMMHR (degree 790) | ✅ Complete |
| **P3.6** | Terrain Models | DEM, GEBCO, Earth2014, visibility | ✅ Complete |
| **P3.7** | Map Projections | Mercator, UTM, Stereographic, LCC, AzEq | ✅ Complete |
| **P3.8** | Tidal Effects | Solid Earth, ocean loading, atmospheric | ✅ Complete |
| **P3.9** | Advanced Gravity | EGM96/2008, Clenshaw summation | ✅ Complete |
| **P4** | Astronomical | Orbit propagation, Lambert, reference frames | ✅ Complete |
| **P5** | INS/Navigation | Strapdown INS, coning/sculling, alignment | ✅ Complete |
| **P5.5** | INS/GNSS Integration | Loosely/tightly-coupled, DOP, fault detection | ✅ Complete |
| **P6** | Signal Processing & Transforms | Filters, matched filter, CFAR, FFT, STFT, wavelets | ✅ Complete |
| **P6.5** | Tracking Containers | TrackList, MeasurementSet, ClusterSet | ✅ Complete |
| **P7** | Infrastructure | Performance, docs, tests | In progress |

---

## Version Targets

| Version | Focus | Status |
|---------|-------|--------|
| **v0.3.0** | Square-root filters, JPDA, IMM estimator | Released 2025-12-30 |
| **v0.3.1** | Type annotation fix | Released 2025-12-30 |
| **v0.4.0** | Gaussian mixture reduction, K-means, MHT | Released 2025-12-30 |
| **v0.4.1** | DBSCAN, hierarchical clustering | Released 2025-12-30 |
| **v0.4.2** | Linting fixes | Released 2025-12-30 |
| **v0.5.0** | Static estimation, K-D/Ball trees | Released 2025-12-30 |
| **v0.5.1** | ML estimation, R-tree, VP-tree, Cover tree | Released 2025-12-30 |
| **v0.6.0** | Gravity and magnetic models (WGS84, WMM, IGRF) | Released 2025-12-30 |
| **v0.7.0** | Complete astronomical code (orbit propagation, Lambert, reference frames) | Released 2025-12-30 |
| **v0.7.1** | EGM96/EGM2008 gravity models with Clenshaw summation | Released 2025-12-30 |
| **v0.8.0** | EMM/WMMHR magnetic models, terrain (DEM, GEBCO, Earth2014, visibility) | Released 2025-12-30 |
| **v0.9.0** | Map projections (Mercator, UTM, Stereographic, LCC, Azimuthal Equidistant) | Released 2025-12-30 |
| **v0.10.0** | Tidal effects (solid Earth, ocean loading, atmospheric, pole tide) | Released 2025-12-30 |
| **v0.11.0** | INS mechanization and navigation | Released 2025-12-30 |
| **v0.12.0** | INS/GNSS integration (loosely/tightly-coupled, DOP, fault detection) | Released 2025-12-31 |
| **v0.13.0** | Signal processing & transforms (filters, CFAR, FFT, STFT, wavelets) | Released 2025-12-31 |
| **v0.13.1** | Numba JIT performance optimization for critical paths | Released 2025-12-31 |
| **v0.13.2** | Fix ricker_wavelet for cross-platform scipy compatibility | Released 2025-12-31 |
| **v0.14.0** | Documentation overhaul: tutorials, examples, landing page, dark theme | Released 2025-12-31 |
| **v0.14.1** | Test coverage boost, release process documentation | Released 2025-12-31 |
| **v0.14.2** | Add isort to release quality checks | Released 2025-12-31 |
| **v0.14.3** | Fix example scripts (API compatibility, type errors, linting) | Released 2025-12-31 |
| **v0.14.4** | Fix flake8 warnings in test files | Released 2025-12-31 |
| **v0.15.0** | New example scripts (signal processing, transforms, INS/GNSS) | Released 2025-12-31 |
| **v0.16.0** | Tracking containers (TrackList, MeasurementSet, ClusterSet) | Released 2025-12-31 |
| **v1.0.0** | Full feature parity, 80%+ test coverage | Planned |

---

## Contributing

Contributions are welcome! If you'd like to work on any of these features:

1. Open an issue to discuss your planned implementation
2. Fork the repository and create a feature branch
3. Follow the existing code style (black formatting, NumPy docstrings)
4. Add tests for new functionality
5. Submit a pull request

See the [original MATLAB library](https://github.com/USNavalResearchLaboratory/TrackerComponentLibrary) for reference implementations.
