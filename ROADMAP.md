# TCL (Tracker Component Library) - Development Roadmap

**Current Version:** v2.0.0 (Released August 2, 2026)
**Test Suite:** 5,925 tests passing, 100% mypy --strict compliance, 951/951 exported functions reached by a test with no standing exemptions
**Status:** Feature-complete MATLAB TCL parity achieved. v2.0.0 shipped 2 August 2026,
closing the v2 correctness audit.

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

**Superseded.** v2.0.0 shipped directly as GA on 2 August 2026 rather than through
the alpha/beta/RC cycle planned below. Recorded rather than deleted, because the
difference matters to anyone upgrading:

- **No deprecation cycle.** 9.2 planned "warnings in place for all APIs removed in
  v2.0.0". That was not done, and it conflicts with the project's standing rule
  against backwards-compatibility shims — a warning for a removed name requires
  keeping the name. The removals are hard breaks. `nuttall_q` is the one exception,
  kept as a warning alias because it is a rename rather than a removal.
- **Migration guide written, community testing skipped.** 9.3's v1.x-to-v2.0.0
  guide exists as `docs/migration_v1_to_v2.rst`. The beta feedback cycle in 9.2 did
  not happen.
- **Phase 8 quality gates measured.** Three of the four hold with margin; the
  compression figure does not and the target has been corrected to what the
  code achieves.

  | Target | Measured | |
  |---|---|---|
  | >1000 detections/sec SQL storage | 3575/sec single, 2134/sec batched | pass |
  | <10 ms track state update | 0.52 ms | pass |
  | <100 ms query latency | 5.06 ms, worst of ten query benchmarks | pass |
  | ~~5-10x~~ HDF5 compression | **4.3x** best case, **1.3x** realistic | corrected |

  The compression target was never met and nothing caught it, because
  `test_compression_ratio` asserts `>2.0` while this document claimed 5-10x —
  a test written to a weaker bar than the figure it was meant to defend. Its
  data is twenty smooth constant-velocity trajectories with **identity**
  covariance matrices, described in the docstring as "representative of real
  tracking data". They are not: identity matrices are mostly zeros and gzip
  removes them, which is where the 4.3x comes from. Substituting the full,
  varying, positive-definite covariances a filter actually produces gives
  **1.32x**.

  Measured on `d5b0add`, macOS arm64, gzip level 4.

The original plan follows.

#### 9.1 v2.0.0-alpha (August 2026)

- Final integration testing across all subsystems
- Verify Phase 8 quality gates: >1000 detections/sec SQL storage, <10ms track state updates,
  <100ms query latency, 5-10x HDF5 compression ratio *(the first three hold; the
  compression figure was wrong and is corrected above)*
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
- **Migration:** `docs/migration_v1_to_v2.rst` — hard breaks, no compat layer
- **GPU Acceleration:** Full CuPy + MLX support for batch operations
- **Documentation:** 9 Jupyter notebooks, 20+ examples, comprehensive guides
- **Testing:** 4,973+ tests, 80%+ coverage, multi-sensor validation scenarios

### v2.0.0 Release Risks

| Risk | Impact | Likelihood | Mitigation |
|------|--------|-----------|-----------|
| Breaking API changes → user friction | High | Low | Deprecation path, migration guide, compat layer |
| GPU memory constraints on large batches | Medium | Medium | Auto-offload strategy, documentation |
| Jupyter notebook maintenance burden | Medium | Medium | CI validation with pytest-nbval |

---

## v2.1 Roadmap (Post-v2.0.0)

### Measured backlog (from the MATLAB parity inventory)

`docs/matlab_parity_inventory.rst` was produced by walking every directory of
the MATLAB TCL repository; its Absent/Weak/Divergent verdicts are the
evidence-based candidate list, as distinct from the aspirational items below.
No dates are attached because none have been decided:

- **Cubature point library** — ~148 MATLAB files, a signature strength of the
  original; pytcl has Gauss-Hermite and spherical cubature only
- **Refraction suite** — entirely unported (astronomical refraction,
  standard-refraction ray tracing, refractivity models, humidity conversions)
- **Localization-style static estimators** — TDOA, Doppler-only init,
  direction-only; MATLAB's Static_Estimation content is almost all absent
- **Filter variants** — EnKF, ESRIF, QMC-Kalman, BLUE measurement updates,
  batch least squares, PCRLB/Riccati analysis tools
- **Time scales** — TDB/TCB/TCG, Besselian epochs, sidereal local time
- **Magnetic coordinate systems** — apex, quasi-dipole, centered-dipole
- **MOSPA/MMOSPA metrics**, AIS decoding, interval scheduling, polynomials
- **NRLMSISE-00 proper** — load the NOAA coefficient tables and retire the
  barometric approximation's caveats (gh-79), plus HWM winds

Session-identified, held deliberately out of 2.0.0:

- ADS-B tracking validation against live traffic (local branch
  `test/adsb-real-data-validation`; CC0 fixture and 10 tests ready)
- Satellite tracking validation (TLE/SGP4 prediction experiments, scratchpad)
- HDF5 compression to the once-claimed 5-10x (states-only chunking or a
  covariance transform; the honest measured figure today is 1.3-4.3x)


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

The v1.15.1-v1.18.0 correctness campaign closed every previously listed
critical issue -- magnetism synthesis, relativity formulas, SEZ convention, and
all five v2.0.0 blockers from [#9](https://github.com/nedonatelli/TCL/issues/9)
(MLX backend, SDP4 deep-space physics, Lagrangian bounds, Murty k-best,
high-degree Legendre) -- plus 72 further reference-verified bugs. Per-package
validation status is tracked in [AUDIT.md](AUDIT.md).

### Critical (must be resolved before v2.0.0)

Unit-level correctness is well covered. The remaining risk was **integration**:
the campaign's most serious findings were things individually correct but not
connected -- an advertised GPU backend never wired in, a high-degree Legendre
routine with zero callers, and three CI gates that could not fail.

v1.19.0 closed most of this. Each gate added found defects the previous layer
could not see: running the examples found four broken scripts, checking imports
found 92 dead references across 16 pages, and executing the pipeline found a
state-layout error on a page whose imports all resolved.

| Issue | Impact | Status |
|-------|--------|--------|
| No end-to-end pipeline test (measurements to tracks to persistence) | Cross-module seams unverified | **Resolved in v1.19.0** -- `tests/test_end_to_end_pipeline.py` spans conversion, gating, association, filtering, track management, persistence and scoring |
| The 30 example scripts are never executed in CI | One shipped fabricated filter output undetected until v1.15.1 | **Resolved in v1.19.0** -- dedicated `examples` job; exposed a `ConstrainedEKF` projection bug and four broken scripts |
| Notebook CI gate cannot fail (`\|\| echo` swallows the exit code) | 13 broken cells; `networkx` used but undeclared | **Resolved in v1.19.0** -- exit code no longer swallowed; the job also never installed plotly |
| Sphinx prose examples are not executed | 370 code blocks unverified | **Partial** -- every `pytcl` import in `docs/` is now checked (244/244 resolve) and the architecture and data-structures pages are executed by tests, but the remaining prose blocks are still not run |
| Orphaned public API | Exported symbols with no callers hid a 1e199 error | **Open** -- the autodoc restructure surfaced 21 submodules no page documented, but no systematic caller audit has been done |

### High Priority

| Issue | Module | Impact | Status |
|-------|--------|--------|--------|
| CuPy tests skip without NVIDIA GPU | `gpu/` | Validated on real hardware for 2.0.0 (RTX 5080); the manual `GPU` workflow re-runs it on demand | By design |
| API/contract cleanups from the audit ([#9](https://github.com/nedonatelli/TCL/issues/9)) | multiple | Complete — all audit issues closed by the 2.0.0 merge | Done |

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

There is none, deliberately. An earlier version of this section promised a
`pytcl.compat` module with old signatures and deprecation warnings, alongside
a code sample. That module was never built, the sample raised
`ModuleNotFoundError`, and the promise contradicted the project's standing
rule against compatibility shims for removed code. The 2.0.0 breaks are hard
breaks: `nuttall_q` (a rename, kept as a warning alias) is the sole
exception, and `docs/migration_v1_to_v2.rst` is the migration path.

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
