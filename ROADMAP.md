# TCL (Tracker Component Library) - Development Roadmap

**Current Version:** v1.15.0 (Released March 15, 2026)
**Test Suite:** 3,306 tests passing, 80% line coverage, 100% mypy --strict compliance
**Status:** Feature-complete MATLAB TCL parity achieved. All v2.0.0 development phases (1-8)
are complete; remaining work is release preparation (Phase 9).

This document covers **planned and future work only**. For what has already shipped, see
[CHANGELOG.md](CHANGELOG.md), the [GitHub releases](https://github.com/nedonatelli/TCL/releases),
and git history.

---

## Table of Contents

1. [v2.0.0 Release Plan](#v200-release-plan)
2. [v2.1 Roadmap](#v21-roadmap-post-v200)
3. [Performance Targets](#performance-targets)
4. [Known Issues & Planned Fixes](#known-issues--planned-fixes)
5. [Breaking Changes for v2.0.0](#breaking-changes-for-v200)
6. [Long-Term Vision (2027-2029)](#long-term-vision-2027-2029)
7. [Community Contribution Priorities](#community-contribution-priorities)
8. [Contributing](#contributing)

---

## v2.0.0 Release Plan

**Target:** October 2026
**Status:** Development complete (Phases 1-8 shipped incrementally in v1.8.0-v1.15.0).
Remaining work is packaging, validation, and the release pipeline.

All v2.0.0 feature work has landed on main: network flow optimization, API standardization,
exception hierarchy, optional dependency system, 9 Jupyter notebooks, dual-backend GPU
acceleration (CuPy + MLX), Numba JIT and caching, sparse assignment, and the full track
management stack (`TrackDatabaseManager`, `TrackHDF5Storage`, compat layer, migration tools,
workflow examples, benchmarks, and multi-sensor validation scenarios).

### Phase 9: Release Preparation & Packaging

#### 9.1 v2.0.0-alpha (August 2026)

- Final integration testing across all subsystems
- Verify Phase 8 quality gates: >1000 detections/sec SQL storage, <10ms track state updates,
  <100ms query latency, 5-10x HDF5 compression ratio
- Validate migration tools against real v1.x legacy code
- Alpha release notes and documentation

#### 9.2 v2.0.0-beta (September 2026)

- Beta release for community testing
- Publish performance benchmarks
- Finalize deprecation path (warnings in place for all APIs removed in v2.0.0)
- Extended GPU performance validation (CuPy on NVIDIA, MLX on Apple Silicon)
- Integrate community feedback

#### 9.3 v2.0.0-RC1 (September 2026)

- Release candidate cycle(s)
- Complete v1.x → v2.0.0 migration guide
- Finalize installation and upgrade instructions
- Freeze benchmarks and API

#### 9.4 v2.0.0 GA (October 2026)

Production release integrating everything shipped in the v1.8-v1.15 series plus release-cycle
polish:

- **Track Management:** SQL `TrackDatabaseManager` + HDF5 storage with full lifecycle support
- **Migration:** Tools and guides for v1.x → v2.0.0 transition, `pytcl.compat` layer
- **GPU Acceleration:** Full CuPy + MLX support for batch operations
- **Documentation:** 9 Jupyter notebooks, 20+ examples, comprehensive guides
- **Testing:** 3,306+ tests, 80%+ coverage, multi-sensor validation scenarios

### v2.0.0 Release Risks

| Risk | Impact | Likelihood | Mitigation |
|------|--------|-----------|-----------|
| Breaking API changes → user friction | High | Low | Deprecation path, migration guide, compat layer |
| GPU memory constraints on large batches | Medium | Medium | Auto-offload strategy, documentation |
| Jupyter notebook maintenance burden | Medium | Medium | CI validation with pytest-nbval |

---

## v2.1 Roadmap (Post-v2.0.0)

**Timeline:** Q1-Q3 2027 (6-9 months after v2.0.0 release)

### Enhanced GPU Support

#### RAPIDS Integration for Distributed Computing (Q1 2027)

- **cuML integration**: GPU-accelerated clustering (k-means, DBSCAN) and statistical functions
- **cuDF support**: DataFrames for high-dimensional tracking data
- **Multi-GPU orchestration**: Distributed Kalman filtering across GPU clusters
- **Performance goal**: 50-100x speedup on 1000+ target scenarios
- Modules affected: `pytcl.clustering`, `pytcl.containers` (GPU k-d trees via cuML),
  `pytcl.dynamic_estimation` (multi-GPU particle filters)

#### Intel oneAPI Backend (Q2 2027)

- oneAPI support for Intel Arc and Data Center GPU Max
- Automatic backend selection (CuPy, MLX, oneAPI, NumPy)
- Unified performance profiling across all backends

#### Quantization & Model Compression (Q2 2027)

- Mixed-precision Kalman filtering (float32/float16)
- Covariance matrix compression for edge devices
- 50-70% memory reduction with <1% accuracy loss

### Advanced Tracking Capabilities

#### Multi-Hypothesis Tracking Enhancements (Q1 2027)

- Bernoulli filter for track initiation/termination
- PHD/CPHD filters for clutter-heavy scenarios
- Generalized labeled multi-Bernoulli (GLMB) tracker
- Intensity function visualization

#### Nonlinear Manifold Estimation (Q2 2027)

- Constrained Kalman filtering for manifold-constrained state spaces
- Geodesic distance computation for rotation matrices
- Lie group integration for orientation tracking
- Applications: aircraft attitude, satellite orientation, marine headings

### Data Management & Interoperability

#### Format Support Expansion (Q1 2027)

- NetCDF4 backend for large geospatial datasets
- Parquet format for cloud-native tracking data
- Apache Arrow integration for inter-process communication
- ASDF (Advanced Scientific Data Format) for heterogeneous data

#### ROS 2 Integration, Optional Plugin (Q2 2027)

- ROS 2 node wrappers for core tracking functions
- Real-time message handling for /tf transforms
- RADAR/LiDAR/Camera sensor plugins

### Performance & Analytics

#### Comprehensive Benchmarking Suite (Q1 2027)

- Automated regression testing across hardware (CPU/GPU/Apple Silicon)
- Timeline profiling dashboard, memory allocation tracking
- Comparison with competing libraries (FilterPy, Stone Soup, etc.)

#### Advanced Diagnostic Tools (Q2 2027)

- Interactive filter health monitoring (covariance ellipses, innovation sequences)
- Automatic anomaly detection in tracking results
- Track quality metrics (completeness, purity, fragmentation)

### Documentation & Community

#### Case Study Library (Q2 2027)

- Air traffic control with 500+ simultaneous aircraft
- Autonomous vehicle fleet tracking and cooperative perception
- Maritime domain awareness with radar/AIS fusion
- Counter-UAS (C-UAS) multisensor tracking
- Space debris cataloging with optical/radar observations

Each case study: 200-400 lines of code, realistic datasets, benchmarked.

#### Extended API Documentation (Q1 2027)

- Video tutorials for major modules
- Interactive filter playground (web-based parameter tuning)
- Algorithm comparison visualizations
- Community-contributed examples (with vetting process)

---

## Performance Targets

Remaining optimization targets not yet met (tracked by the daily benchmark CI):

| Algorithm | Dataset | Current | Target |
|-----------|---------|---------|--------|
| JPDA (100 targets, 50 meas) | Mixed | 89 ms | <75 ms |
| Hungarian Assignment (500x500) | Dense | 45 ms | <30 ms |

### v2.0.0 Release Targets

- All CPU algorithms: sub-100ms for standard scenarios
- GPU: 10-15x speedup with batch processing (NVIDIA), 4-5x native (Apple Silicon)
- Memory usage: 50%+ reduction via sparse matrices
- Scalability: linear time for 1000+ targets with O(n log n) assignment

---

## Known Issues & Planned Fixes

### Critical (tracked in [#3](https://github.com/nedonatelli/TCL/issues/3))

| Issue | Module | Impact | Status |
|-------|--------|--------|--------|
| WMM/IGRF synthesis: wrong Legendre normalization, corrupted WMM2020 coefficient table above degree 4, no geodetic-to-geocentric conversion | `magnetism/` | Field values unreliable (declination ~180 deg off at NOAA test point); EMM/WMMHR needs accuracy audit | Open — fix before v2.0.0 |
| Dimensionally inconsistent relativity formulas (geodetic precession, Lense-Thirring, 1PN acceleration, range correction) | `astronomical/relativity.py` | Unphysical outputs | Open — fix before v2.0.0 |
| `ecef2sez` S-axis points north, contradicting standard SEZ (Vallado) | `coordinate_systems/conversions/geodetic.py` | Sign-flipped S components | Needs convention decision |

### High Priority

| Issue | Module | Impact | Status |
|-------|--------|--------|--------|
| Terrain loader signature mismatch | `terrain/loaders.py` | 13 tests skipped | Planned for v2.1 |
| Plotting array indexing bug ("too many indices for array") | `plotting/` | 2 tests skipped | Investigating |
| CuPy tests skip without NVIDIA GPU | `gpu/` | 11 tests skipped | By design |
| Dead modules: `assignment_algorithms/network_simplex.py`, `logging_config.py` | — | 0% coverage, referenced nowhere | Remove in v2.0.0 |

### Medium Priority

| Issue | Effect | Mitigation |
|-------|--------|-----------|
| EGM2008 data file (1.3GB) | Geoid height tests skipped without download | Optional download, fallback to EGM96 |
| pywavelets optional dep | Wavelet tests conditional | Falls back to signal processing module |
| Global Legendre cache | Thread safety for concurrent use | Use `clear_legendre_cache()` if needed |

### Limitations vs MATLAB TCL Original

| Feature | Gap | Plan |
|---------|-----|------|
| Distributed tracking | Not implemented | v2.1 with RAPIDS |
| Real-time C++ bindings | Not implemented | pybind11 if demand materializes |

### Performance Limitations

- **Very large arrays (>10GB)**: Out-of-core support not implemented (could use HDF5 chunking)
- **Thread safety**: Most functions thread-safe, but some caches are global (mitigated by clearing)
- **Real-time hard deadlines**: Python GC unpredictability (mitigated by profiling tools)

---

## Breaking Changes for v2.0.0

### API Changes Summary

**Function signature changes** (deprecated in v1.13.2, removed in v2.0.0):
```python
# OLD (deprecated)
kf_predict(x, P, F, Q)  # Returns (x_pred, P_pred)

# NEW (v2.0.0)
kf_predict(x, P, model)  # Returns FilterState(x, P, info)
```

**Module reorganization:**
- `pytcl.assignment_algorithms` → `pytcl.data_association.assignment`
- `pytcl.matrix_utilities` → internal only (use `pytcl.dynamic_estimation.kalman.matrix_utils`)
- `pytcl.special_functions` → `pytcl.mathematical_functions.special_functions`

**Exception handling:** all exceptions inherit from `pytcl.TCLError`. Replace generic
`ValueError` catches with specific exception types:
```python
# OLD
except ValueError as e:
    ...

# NEW
except pytcl.DimensionError as e:  # More specific
    ...
except pytcl.TCLError as e:  # Fallback
    ...
```

**GPU imports:**
```python
# OLD
from pytcl.gpu import batch_kf_predict

# NEW
from pytcl.gpu.kalman import batch_kf_predict
```

### Backward Compatibility Layer

Available in `pytcl.compat` for one major version (v2.0.0-v2.1.0):
- Old function signatures with intermediate transformation
- Deprecation warnings pointing to the new API
- Will be removed in v3.0.0

```python
from pytcl.compat import kf_predict as kf_predict_old

x_pred, P_pred = kf_predict_old(x, P, F, Q)
```

---

## Long-Term Vision (2027-2029)

### v2.x Series Direction

#### Core Tracking Evolution

1. **Real-time Particle Swarm Optimization tracking** (v2.2+)
   - Swarm intelligence for clutter-heavy tracking
   - Cooperative multi-target estimation
2. **Quantum-inspired algorithms** (v2.3+)
   - Quantum annealing backend for assignment (via D-Wave)
   - Quantum simulation of filter dynamics
3. **Adaptive learning tracking** (v2.4+)
   - Neural network-augmented Kalman filters
   - Transfer learning for new sensor types

#### Infrastructure Maturation

- **v2.0**: Consolidate GPU, data persistence, documentation
- **v2.1**: RAPIDS, distributed tracking, advanced diagnostics
- **v2.2**: Extended ecosystem (ROS 2, autonomous systems)
- **v2.3**: Emerging tech (quantum, federated learning)

#### Community & Ecosystem

- **2027**: 500+ GitHub stars, 50+ external contributions
- **2028**: Integration with major frameworks (CARLA, AirSim for autonomous systems)
- **2029**: Industrial adoption (defense contractors, automotive OEMs)

### v3.0.0 Vision (2030+)

- Full async/await support for real-time systems
- Native WebAssembly compilation for browser-based tracking
- Federated learning for multi-agent scenarios
- Quantum computing backend exploration

Estimated effort: 18-24 months post-v2.x stabilization.

---

## Community Contribution Priorities

### High-Impact Opportunities (Seeking Contributors)

#### 1. Documentation & Tutorials ⭐⭐⭐ (Low barrier to entry)

- Video tutorials for each major module (10-20 min each)
- Blog posts on advanced techniques (JPDA, MHT, particle filters)
- Real-world case studies with public datasets
- Glossary of tracking terminology

**Effort:** 20-40 hours per item | **Skills:** technical writing | **Impact:** high

#### 2. ROS 2 Integration Layer ⭐⭐⭐ (Medium barrier)

- ROS 2 node wrappers for core modules
- Sensor message adapters (RADAR, LiDAR, camera)
- Autonomous vehicle examples

**Effort:** 60-100 hours | **Skills:** ROS 2 experience | **Impact:** very high

#### 3. Additional GPU Backends ⭐⭐ (Medium-high barrier)

- Intel oneAPI backend (similar to CuPy/MLX structure)
- AMD ROCm support
- Vulkan compute for cross-platform GPU access

**Effort:** 80-150 hours per backend | **Skills:** GPU programming | **Impact:** high

#### 4. Performance Optimization ⭐⭐⭐ (High barrier)

- Profile existing algorithms for bottlenecks
- Implement missing Numba JIT targets
- SIMD optimization
- Caching opportunities analysis

**Effort:** 40-80 hours per element | **Skills:** profiling, numerical computing | **Impact:** medium-high

#### 5. Additional Assignment Algorithms ⭐⭐ (Medium-high barrier)

- Genetic algorithm-based assignment
- Ant colony optimization for dynamic assignment
- Deep learning-based cost matrix prediction

**Effort:** 50-100 hours per algorithm | **Skills:** algorithm implementation | **Impact:** medium

### Getting Started for Potential Contributors

**Repositories:**
- Python port (this repo): https://github.com/nedonatelli/TCL — published on PyPI as `nrl-tracker`
- MATLAB reference: https://github.com/USNavalResearchLaboratory/TrackerComponentLibrary

**Development setup:**
```bash
git clone https://github.com/nedonatelli/TCL.git
cd TCL
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

---

## Contributing

Contributions are welcome! If you'd like to work on any of these features:

1. Open an issue to discuss your planned implementation
2. Fork the repository and create a feature branch
3. Follow the existing code style (ruff formatting, NumPy docstrings)
4. Add tests for new functionality (aim for 80%+ coverage)
5. Submit a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines and the
[original MATLAB library](https://github.com/USNavalResearchLaboratory/TrackerComponentLibrary)
for reference implementations.

---

**Last Updated:** July 24, 2026
**Current Phase:** Phase 9 - Release Preparation
**Next Milestone:** v2.0.0-alpha (August 2026)
**v2.0.0 Target Release:** October 2026
**v2.1.0 Target Release:** Q1-Q3 2027 (RAPIDS, distributed tracking, advanced diagnostics)
**Long-term Horizon:** v3.0.0 planned for 2030+ (async/await, WASM, federated learning, quantum backends)
