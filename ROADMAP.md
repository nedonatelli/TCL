# TCL (Tracker Component Library) - Development Roadmap

## Current State (v0.3.1)

- **400+ functions** implemented across 90+ Python files
- **61% test coverage** with 408 tests
- **Core tracking functionality complete**: Kalman filters (KF, EKF, UKF, CKF), particle filters, coordinate systems, dynamic models, data association (GNN, JPDA), multi-target tracking
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

## Phase 1: Advanced Estimation & Data Association (Remaining)

### 1.3 Multiple Hypothesis Tracking (MHT)
- [ ] Hypothesis tree management
- [ ] N-scan pruning
- [ ] Track-oriented MHT

---

## Phase 2: Clustering & Mixture Reduction

### 2.1 Gaussian Mixture Operations
- [ ] Gaussian mixture representation class
- [ ] Mixture moment matching
- [ ] Runnalls' mixture reduction algorithm
- [ ] West's algorithm

### 2.2 Clustering Algorithms
- [ ] K-means clustering
- [ ] DBSCAN
- [ ] Hierarchical clustering for track fusion

---

## Phase 3: Static Estimation

### 3.1 Maximum Likelihood Estimation
- [ ] ML estimator framework
- [ ] Fisher information computation
- [ ] Cramer-Rao bounds

### 3.2 Least Squares & Robust Estimation
- [ ] Weighted least squares
- [ ] Total least squares
- [ ] Robust M-estimators (Huber, Tukey)
- [ ] RANSAC

---

## Phase 4: Container Data Structures

### 4.1 Spatial Search Structures
- [ ] k-d tree implementation
- [ ] Ball tree
- [ ] R-tree for bounding boxes

### 4.2 Metric Trees
- [ ] VP-tree (vantage point tree)
- [ ] Cover tree
- [ ] Efficient nearest neighbor queries

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
| **P0** | Advanced Data Association | JPDA, MHT, IMM | JPDA, IMM done; MHT pending |
| **P1** | Clustering | Gaussian mixture reduction | Pending |
| **P2** | Static Estimation | ML, robust estimators | Pending |
| **P3** | Geophysical Models | Gravity, magnetic, terrain | Pending |
| **P4** | Astronomical | Orbit propagation, reference frames | Pending |
| **P5** | Infrastructure | Performance, docs, tests | In progress |

---

## Version Targets

| Version | Focus | Status |
|---------|-------|--------|
| **v0.3.0** | Square-root filters, JPDA, IMM estimator | Released 2025-12-30 |
| **v0.3.1** | Type annotation fix | Released 2025-12-30 |
| **v0.4.0** | Clustering module, Gaussian mixture reduction, MHT | Planned |
| **v0.5.0** | Static estimation, spatial data structures | Planned |
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
