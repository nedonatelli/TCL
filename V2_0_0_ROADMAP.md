# v2.0.0 Comprehensive Release Strategy & Roadmap

**Release Target:** 18 months (Months 1-18, starting Q1 2026)  
**Status:** Planning Phase  
**Last Updated:** January 4, 2026

## Executive Summary

v2.0.0 targets architectural improvements, performance optimization, GPU acceleration, and documentation enhancement. The release focuses on resolving critical bottlenecks (network flow performance), standardizing APIs (spatial indexes, exceptions), expanding documentation and tutorials, implementing GPU acceleration for batch processing, and classifying modules by maturity level.

### Key Metrics
- **Lines of duplicate code to eliminate:** ~300 (Kalman filter consolidation)
- **Skipped tests to enable:** 13 (network flow tests)
- **Modules to standardize:** 7 (spatial index implementations)
- **New tutorials to create:** 8 (Jupyter notebooks)
- **GPU speedup target:** 5-10x (Kalman batch processing), 8-15x (particle filters)
- **Test coverage improvement:** +50 tests (from 2,057 → 2,100+)
- **Coverage target:** 80%+ line coverage (from current 76%)
- **Timeline:** 18 months with 8 major work streams

---

## Phase 1: Critical Fixes & Foundation (Months 1-3)

### 1.1 Network Flow Performance [BLOCKER]

**Problem:** `test_network_flow.py` has 13 skipped tests (lines 85-215) due to Bellman-Ford O(VE²) performance limitation.

**Skipped Tests:**
- `test_bellman_ford_2x2` - Small sparse graph timeout
- `test_bellman_ford_3x3` - Medium sparse graph timeout
- `test_bellman_ford_rectangular` - Rectangular sparse graph timeout
- `test_bellman_ford_performance` (x10) - Multiple problem sizes (5x5 to 20x20)

**Solution:** Implement network simplex algorithm with O(VE log V) complexity
- Create `pytcl/assignment_algorithms/network_flow_simplex.py` module
- Maintain API parity with current implementation
- Target performance: 50-100x faster for large problems (100+ nodes)
- Add 10+ new tests for simplex-specific cases (degeneracy handling, cycling prevention)

**Acceptance Criteria:**
- All 13 tests pass without skip markers
- Network simplex benchmark ≥50x faster than Bellman-Ford on 20x20 problems
- 100% backward compatibility with existing API

**Effort:** HIGH (3 weeks)  
**Owner:** To be assigned

---

### 1.2 Circular Imports Resolution

**Problem:** Late imports in `kalman/sr_ukf.py` (lines 453-459) use `# noqa: E402`

**Solution:** Refactor using TYPE_CHECKING blocks
- Use `from typing import TYPE_CHECKING` for forward references
- Move runtime imports to function-level scope where necessary
- Document dependency graph in architecture documentation

**Acceptance Criteria:**
- No `# noqa: E402` comments in codebase
- All circular imports documented
- IDE code intelligence works at module level

**Effort:** MEDIUM (2 weeks)

---

### 1.3 Empty Module Exports

**Problem:** Three modules have empty or incomplete `__all__` exports

**Solution:** Define clear public APIs or consolidate into misc

**Effort:** LOW (1 week)

---

### 1.4 Kalman Filter Code Consolidation

**Problem:** Four Kalman implementations have duplicate matrix operations (~300 lines)

**Solution:** Extract to `kalman/matrix_utils.py`

**Effort:** MEDIUM (2 weeks)

---

## Phase 2: API Standardization & Infrastructure (Months 2-4)

### 2.1 Spatial Index Interface Standardization

**Problem:** 7 spatial index implementations have inconsistent interfaces

**Solution:** Standardize to unified interface with `NeighborResult` NamedTuple

**Effort:** MEDIUM (3 weeks)

---

### 2.2 Custom Exception Hierarchy

**Problem:** Codebase uses 50+ generic ValueError/ImportError across modules

**Solution:** Define custom exception hierarchy in `pytcl.core.exceptions`

**Effort:** MEDIUM (2 weeks)

---

### 2.3 Optional Dependencies System

**Problem:** Plotly imports scattered across codebase with no consistent error handling

**Solution:** Implement `pytcl.optdeps` with decorators and installation groups

**Effort:** MEDIUM (2 weeks)

---

## Phase 3: Documentation Expansion & Module Graduation (Months 3-6)

### 3.1 Weak Module Docstring Expansion

**Problem:** Four modules have minimal docstrings (1 line each)

**Effort:** LOW (1 week)

---

### 3.2 Function-Level Documentation Examples

**Problem:** 20+ high-level functions lack docstring examples

**Effort:** LOW-MEDIUM (2 weeks)

---

### 3.3 Module Graduation System

**Problem:** 25 modules mixed maturity levels; unclear which are production-ready

**Effort:** MEDIUM (2 weeks)

---

## Phase 4: Jupyter Interactive Tutorials (Months 4-8)

### 4.1 Notebook Creation (8 notebooks)

**Location:** `docs/notebooks/` with `.gitattributes` for `nbstripout`

Eight comprehensive interactive notebooks covering:
1. Kalman Filters
2. Particle Filters
3. Multi-Target Tracking
4. Coordinate Systems
5. GPU Acceleration
6. Network Flow Solver
7. INS/GNSS Integration
8. Performance Optimization

**Effort:** HIGH (6 weeks)

---

### 4.2 Supporting Infrastructure

**Dataset Handling:** Create `examples/data/` directory  
**Binder Integration:** Set up cloud execution  
**CI Validation:** `pytest --nbval` integration

**Effort:** MEDIUM (2 weeks)

---

## Phase 5: GPU Acceleration Tier-1 Implementation (Months 6-10)

### 5.1 CuPy-Based Kalman Filters

**Implementations:**
- `CuPyKalmanFilter` - Linear KF with batch processing
- `CuPyExtendedKalmanFilter` - EKF with nonlinear models
- `CuPyUnscentedKalmanFilter` - UKF with sigma points
- `CuPySRKalmanFilter` - Square-root for numerical stability

**Performance Targets:** 5-10x speedup

**Effort:** HIGH (4 weeks)

---

### 5.2 GPU Particle Filters

**Implementations:**
- GPU-accelerated resampling
- Importance function evaluation
- Particle weight evolution

**Performance Targets:** 8-15x speedup

**Effort:** MEDIUM (3 weeks)

---

### 5.3 Matrix Utilities

**Utilities:**
- CuPy Cholesky/QR factorization
- Memory pooling for repeated allocations
- Auto-offload on memory exhaustion

**Effort:** MEDIUM (2 weeks)

---

## Phase 6: Test Expansion & Coverage Improvement (Months 7-12)

**Current Status:** 2,057 tests passing, 13 skipped, 76% line coverage

### 6.1 Network Flow Tests Re-enablement

**Current Status:** 13 tests skipped due to Bellman-Ford performance

**Resolution:** Implement network simplex algorithm in Phase 1
- Remove `@pytest.mark.skip` from all 13 tests
- Add 10+ new tests for simplex edge cases
- Benchmark new algorithm vs alternatives

**Target:** 100% pass rate on network flow tests

**Effort:** Already counted in Phase 1

---

### 6.2 Kalman Filter Variant Tests (20+ new tests)

**Low-Coverage Targets:**
- `kalman/sr_ukf.py` - Current: 6% → Target: 50%+
- `kalman/ud_filter.py` - Current: 11% → Target: 60%+
- `kalman/square_root.py` - Current: 19% → Target: 70%+

**Test Areas:**
- State prediction accuracy
- Covariance updates (positive definite maintenance)
- Numerical stability edge cases
- Large state dimension scenarios (100+ states)
- Singular covariance recovery

**Acceptance Criteria:**
- 20+ new tests added
- Coverage increase to 50%+ for each module
- All tests pass (no flaky tests)

**Effort:** MEDIUM (3 weeks)  
**Owner:** To be assigned

---

### 6.3 Advanced Filter Tests (15+ new tests)

**Low-Coverage Target:**
- `dynamic_estimation/imm.py` - Current: 21% → Target: 60%+

**Test Areas:**
- Mode probability transitions
- Model likelihood evaluation
- State estimate merging
- Mode-matched filtering accuracy
- Convergence behavior under different mode sequences

**Acceptance Criteria:**
- 15+ new tests for IMM filter
- Coverage increase to 60%+
- Real-world scenario tests (multimodal targets, mode switching)

**Effort:** MEDIUM (2 weeks)

---

### 6.4 Signal Processing Detection Tests (20+ new tests)

**Low-Coverage Target:**
- `signal_processing/detection.py` - Current: 34% → Target: 65%+

**Test Areas:**
- CFAR (Constant False Alarm Rate) detector variants
- Detection probability vs false alarm trade-off
- Receiver Operating Characteristic (ROC) curves
- Threshold selection algorithms
- Multi-hypothesis detection scenarios

**Acceptance Criteria:**
- 20+ new tests covering core detection functions
- Coverage increase to 65%+
- Performance benchmarks for detector implementations

**Effort:** MEDIUM (3 weeks)

---

### 6.5 Terrain Loader Error Path Tests (15+ new tests)

**Low-Coverage Target:**
- `terrain/loaders.py` - Current: 60% → Target: 80%+

**Test Areas:**
- Missing file handling
- Corrupted data detection
- Invalid format handling
- Coordinate system validation
- Out-of-bounds queries
- File permission errors

**Acceptance Criteria:**
- 15+ error path tests added
- Coverage increase to 80%+
- Proper exception types raised with informative messages

**Effort:** LOW-MEDIUM (2 weeks)

---

### 6.6 Signal Processing Filter Tests (10+ new tests)

**Low-Coverage Target:**
- `signal_processing/filters.py` - Current: 61% → Target: 75%+

**Test Areas:**
- Filter design edge cases
- Frequency response validation
- Phase distortion analysis
- Filter stability verification
- Low/high pass filter transitions

**Acceptance Criteria:**
- 10+ new tests added
- Coverage increase to 75%+

**Effort:** MEDIUM (2 weeks)

---

### 6.7 Integration & Property-Based Tests (10+ new tests)

**New Test Categories:**
- Multi-module workflows (tracking + coordinate transforms)
- End-to-end tracking pipeline
- Cross-module data flow validation
- Kalman filter invariants (positive definite covariance)
- Coordinate transform round-trip properties (transform → inverse → original)
- Assignment optimality verification

**Implementation:**
- Hypothesis property-based tests for algorithm invariants
- Integration test suite for common workflows
- Real-world scenario tests

**Acceptance Criteria:**
- 10+ property-based tests
- 5+ integration tests covering real workflows
- Edge cases discovered and fixed

**Effort:** MEDIUM (2 weeks)

---

### Phase 6 Summary

**Target:** Add 50+ new tests (2,057 → 2,100+)  
**Coverage Improvement:** 76% → 80%+

| Module | Current | Target | New Tests | Effort |
|--------|---------|--------|-----------|--------|
| Kalman filters (sr_ukf, ud, sr) | 6-19% | 50-70% | 20 | 3w |
| IMM filter | 21% | 60% | 15 | 2w |
| Signal detection | 34% | 65% | 20 | 3w |
| Signal filters | 61% | 75% | 10 | 2w |
| Terrain loaders | 60% | 80% | 15 | 2w |
| Network flow | N/A* | 100% | 13 | 1w |
| Integration/Property | - | - | 10 | 2w |
| **Total** | **76%** | **80%+** | **50+** | **15w** |

*Network flow: Re-enable 13 skipped tests after Phase 1 fixes

---

## Phase 7: Performance Optimization (Months 8-12)

### 7.1 JIT Compilation with Numba

**Target Functions:**
- Kalman filter sigma point computation
- JPDA measurement likelihood
- Particle filter weight computation

**Expected Speedup:** 20% per optimized function

**Effort:** MEDIUM (2 weeks)

---

### 7.2 Systematic Caching with lru_cache

**Target Functions:**
- Coordinate system Jacobians
- Legendre polynomials
- Transformation matrices

**Expected Speedup:** 25-40%

**Effort:** LOW-MEDIUM (2 weeks)

---

### 7.3 Sparse Matrix Support

**New Functionality:**
- Optional scipy.sparse support in `assignment_algorithms/nd_assignment.py`
- Sparse cost matrix representation for large problems

**Performance Target:** 50% memory reduction

**Effort:** MEDIUM (3 weeks)

---

## Phase 8: Release Preparation (Months 13-18)

### 8.1 v2.0-alpha (Month 12)

**Checklist:**
- ✅ Network flow simplex algorithm implemented and tested
- ✅ Kalman filter code consolidated
- ✅ Module graduation completed
- ✅ API standardization complete
- ✅ GPU Tier-1 working and benchmarked
- ✅ Test coverage 80%+

---

### 8.2 v2.0-beta (Month 14)

**Checklist:**
- ✅ All 8 Jupyter notebooks complete
- ✅ Documentation expansion complete
- ✅ 50+ new tests integrated
- ✅ GPU performance benchmarked
- ✅ Integration tests complete

---

### 8.3 v2.0-RC1 (Month 16)

**Checklist:**
- ✅ Migration guide complete
- ✅ Deprecation warnings in place
- ✅ Performance benchmarks documented
- ✅ Installation instructions updated

---

### 8.4 v2.0.0 (Month 18)

**Final Release** with all improvements integrated

---

## Success Metrics & KPIs

| Metric | Current | Target (v2.0) | Status |
|--------|---------|---------------|--------|
| Network flow tests skipped | 13 | 0 | ✓ |
| Kalman filter duplicate code | ~300 lines | 0 lines | ✓ |
| Spatial index implementations standardized | 0/7 | 7/7 | ✓ |
| Module docstring quality | 85% | 95%+ | ✓ |
| Jupyter tutorials | 0 | 8 | ✓ |
| GPU speedup (Kalman batch) | N/A | 5-10x | ✓ |
| Unit tests | 2057 | 2100+ | ✓ |
| Test coverage | 76% | 80%+ | ✓ |
| Test coverage (batch_estimation) | <50% | 80%+ | ✓ |
| Documentation quality | ~85% | 95%+ | ✓ |

---

## Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|-----------|-----------|
| Network simplex algorithm complexity | High | Low | Thorough testing, research phase |
| GPU memory constraints | Medium | Medium | Auto-offload strategy, documentation |
| Breaking API changes → user friction | High | Low | Deprecation path, migration guide |
| Skipped test complexity (13 tests) | High | Low | Phased implementation, benchmarking |
| Jupyter notebook maintenance | Medium | Medium | CI validation, doctest format |
| Test expansion timeline | Medium | Medium | Distribute across phases |

---

## Dependencies & Resources

### Technical Skills Required
- Numerical algorithms (network simplex, Kalman filters)
- GPU programming (CuPy, CUDA)
- Python profiling and optimization
- Documentation writing
- CI/CD infrastructure
- Test design and property-based testing

### External Dependencies
- CuPy 12.0+ (GPU support)
- Plotly 5.0+ (visualization)
- Numba (JIT compilation)
- Hypothesis (property-based testing)
- Jupyter ecosystem
- RAPIDS (future, v2.1)

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-04 | Planning | Initial comprehensive v2.0.0 roadmap with detailed Phase 6 test expansion plan |

---

**Last Updated:** January 4, 2026  
**Next Review:** Month 3 (after Phase 1-2 completion)  
**Target Release:** Month 18 (Q4 2027)
