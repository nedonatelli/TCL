# TCL (Tracker Component Library) - Development Roadmap

## Current State (v0.5.0)

- **450+ functions** implemented across 102 Python files
- **574 tests** with comprehensive coverage
- **Core tracking functionality complete**: Kalman filters (KF, EKF, UKF, CKF), particle filters, coordinate systems, dynamic models, data association (GNN, JPDA, MHT), multi-target tracking
- **Gaussian mixture operations**: moment matching, Runnalls/West reduction algorithms
- **Complete clustering module**: K-means, DBSCAN, hierarchical clustering
- **Static estimation**: Least squares (OLS, WLS, TLS, GLS, RLS), robust M-estimators (Huber, Tukey), RANSAC
- **Spatial data structures**: K-D tree, Ball tree for efficient nearest neighbor queries
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

## Phase 3 (Remaining): Static Estimation

### 3.1 Maximum Likelihood Estimation
- [ ] ML estimator framework
- [ ] Fisher information computation
- [ ] Cramer-Rao bounds

---

## Phase 4 (Remaining): Container Data Structures

### 4.2 Additional Spatial Structures
- [ ] R-tree for bounding boxes
- [ ] VP-tree (vantage point tree)
- [ ] Cover tree

---

## Phase 5: Geophysical Models

### 5.1 Gravity Models
- [ ] Spherical harmonic evaluation
- [ ] EGM96 model support
- [ ] EGM2008 model support
- [ ] Tidal effects

### 5.2 Magnetic Field Models
- [ ] World Magnetic Model (WMM)
- [ ] International Geomagnetic Reference Field (IGRF)
- [ ] Enhanced Magnetic Model (EMM)

### 5.3 Terrain Models
- [ ] Digital elevation model interface
- [ ] GEBCO integration
- [ ] Earth2014 support
- [ ] Terrain masking for visibility

---

## Phase 6: Advanced Astronomical & Navigation

### 6.1 Celestial Mechanics
- [ ] Two-body orbit propagation
- [ ] Kepler's equation solvers
- [ ] Orbital element conversions
- [ ] Lambert problem solver

### 6.2 Reference Frame Transformations
- [ ] GCRF/ITRF conversions
- [ ] Precession/nutation models
- [ ] Earth orientation parameters (EOP)
- [ ] Polar motion corrections

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
| **P2.5** | Spatial Data Structures | K-D tree, Ball tree | ✅ Complete |
| **P3** | Geophysical Models | Gravity, magnetic, terrain | Pending |
| **P4** | Astronomical | Orbit propagation, reference frames | Pending |
| **P5** | Infrastructure | Performance, docs, tests | In progress |

---

## Version Targets

| Version | Focus | Status |
|---------|-------|--------|
| **v0.3.0** | Square-root filters, JPDA, IMM estimator | Released 2025-12-30 |
| **v0.3.1** | Type annotation fix | Released 2025-12-30 |
| **v0.4.0** | Gaussian mixture reduction, K-means, MHT | Released 2025-12-30 |
| **v0.4.1** | DBSCAN, hierarchical clustering | Released 2025-12-30 |
| **v0.4.2** | Linting fixes | Released 2025-12-30 |
| **v0.5.0** | Static estimation, spatial data structures | Released 2025-12-30 |
| **v0.6.0** | Gravity and magnetic models | Planned |
| **v0.7.0** | Complete astronomical code | Planned |
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
