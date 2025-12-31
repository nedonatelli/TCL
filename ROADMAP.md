# TCL (Tracker Component Library) - Development Roadmap

## Current State (v0.7.0)

- **550+ functions** implemented across 112 Python files
- **702 tests** with comprehensive coverage
- **Core tracking functionality complete**: Kalman filters (KF, EKF, UKF, CKF), particle filters, coordinate systems, dynamic models, data association (GNN, JPDA, MHT), multi-target tracking
- **Gaussian mixture operations**: moment matching, Runnalls/West reduction algorithms
- **Complete clustering module**: K-means, DBSCAN, hierarchical clustering
- **Static estimation**: Least squares (OLS, WLS, TLS, GLS, RLS), robust M-estimators (Huber, Tukey), RANSAC, maximum likelihood estimation, Fisher information, Cramer-Rao bounds
- **Spatial data structures**: K-D tree, Ball tree, R-tree, VP-tree, Cover tree for efficient nearest neighbor queries
- **Geophysical models**: Gravity (spherical harmonics, WGS84, J2), Magnetism (WMM2020, IGRF-13)
- **Astronomical code**: Orbital mechanics, Kepler propagation, Lambert problem, reference frame transformations
- **Published on PyPI** as `nrl-tracker`

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

## Phase 5 (Remaining): Geophysical Models

### 5.1 Advanced Gravity Models
- [ ] EGM96 model support (higher-degree harmonics)
- [ ] EGM2008 model support
- [ ] Tidal effects

### 5.2 Advanced Magnetic Models
- [ ] Enhanced Magnetic Model (EMM)
- [ ] Higher-degree coefficients (degree 12+)

### 5.3 Terrain Models
- [ ] Digital elevation model interface
- [ ] GEBCO integration
- [ ] Earth2014 support
- [ ] Terrain masking for visibility

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

## Phase 6 (Remaining): Advanced Navigation

### 6.3 INS Mechanization
- [ ] Complete strapdown INS
- [ ] Coning/sculling corrections
- [ ] Error state models
- [ ] INS/GNSS integration

---

## Phase 7: Signal Processing & Transforms

### 7.1 Signal Processing
- [ ] Digital filter design
- [ ] Matched filtering
- [ ] Detection algorithms (CFAR)

### 7.2 Transforms
- [ ] Discrete Fourier transform utilities
- [ ] Short-time Fourier transform
- [ ] Wavelet transforms

---

## Phase 8: Performance & Infrastructure

### 8.1 Performance Optimization
- [ ] Expand Numba JIT coverage to critical paths
- [ ] Profile and optimize bottlenecks
- [ ] Consider Cython for hot spots

### 8.2 Documentation
- [x] User guides for square-root filters, IMM, and JPDA
- [x] API reference documentation for v0.3.0 features
- [x] Data association user guide
- [ ] Complete API documentation for remaining modules
- [ ] Add tutorials and examples for new features

### 8.3 Testing
- [x] 408 tests (up from 355)
- [x] 61% coverage (up from 58%)
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
| **P3.5** | Advanced Geophysical | EGM96/2008, EMM, Terrain | Pending |
| **P4** | Astronomical | Orbit propagation, Lambert, reference frames | ✅ Complete |
| **P5** | INS/Navigation | Strapdown INS, coning/sculling, INS/GNSS | Pending |
| **P6** | Infrastructure | Performance, docs, tests | In progress |

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
| **v0.8.0** | INS mechanization and navigation | Planned |
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
