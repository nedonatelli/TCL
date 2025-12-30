# TrackerComponentLibrary: MATLAB to Python Conversion Roadmap

## Executive Summary

This document outlines a structured approach to converting the U.S. Naval Research Laboratory's TrackerComponentLibrary from MATLAB to native Python. The library contains **22+ modules** spanning target tracking, coordinate systems, estimation algorithms, astronomical calculations, and more.

**Estimated Total Effort:** 12-24 months (1-2 FTE developers)  
**Recommended Approach:** Phased conversion with priority on core tracking functionality

---

## 1. Project Architecture

### 1.1 Proposed Python Package Structure

```
tracker_component_library/
├── __init__.py
├── core/                          # Foundation utilities
│   ├── __init__.py
│   ├── constants.py               # Physical constants
│   ├── validation.py              # Input validation helpers
│   └── array_utils.py             # MATLAB-like array operations
│
├── mathematical_functions/        # Mathematical_Functions/
│   ├── __init__.py
│   ├── basic_matrix/              # Matrix operations
│   ├── combinatorics/             # Permutations, combinations
│   ├── continuous_optimization/   # LBFGS, line search
│   ├── geometry/                  # Geometric primitives
│   ├── interpolation/             # Interpolation methods
│   ├── numerical_integration/     # Quadrature methods
│   ├── polynomials/               # Polynomial operations
│   ├── signal_processing/         # FFT, filtering
│   ├── special_functions/         # Bessel, gamma, etc.
│   ├── statistics/                # Distributions, estimation
│   └── transforms/                # Coordinate transforms
│
├── coordinate_systems/            # Coordinate_Systems/
│   ├── __init__.py
│   ├── conversions/               # Cartesian, spherical, etc.
│   ├── jacobians/                 # Jacobian calculations
│   ├── rotations/                 # Rotation matrices
│   └── projections/               # Map projections
│
├── dynamic_models/                # Dynamic_Models/
│   ├── __init__.py
│   ├── continuous_time/           # Continuous dynamics
│   ├── discrete_time/             # State transition matrices
│   └── process_noise/             # Process noise covariances
│
├── dynamic_estimation/            # Dynamic_Estimation/
│   ├── __init__.py
│   ├── kalman/                    # KF, EKF, UKF variants
│   ├── particle_filters/          # Sequential MC methods
│   ├── batch_estimation/          # Batch least squares
│   └── measurement_update/        # Update equations
│
├── static_estimation/             # Static_Estimation/
│   ├── __init__.py
│   ├── maximum_likelihood/        # ML estimation
│   ├── least_squares/             # LS variants
│   └── robust_estimation/         # M-estimators
│
├── assignment_algorithms/         # Assignment_Algorithms/
│   ├── __init__.py
│   ├── two_dimensional/           # 2D assignment (Hungarian)
│   ├── multi_dimensional/         # k-D assignment
│   └── auction/                   # Auction algorithms
│
├── clustering/                    # Clustering_and_Mixture_Reduction/
│   ├── __init__.py
│   ├── mixture_reduction/         # Gaussian mixture reduction
│   └── clustering/                # k-means, etc.
│
├── performance_evaluation/        # Performance_Evaluation/
│   ├── __init__.py
│   ├── track_metrics/             # OSPA, track purity
│   └── estimation_metrics/        # RMSE, NEES, etc.
│
├── astronomical/                  # Astronomical_Code/
│   ├── __init__.py
│   ├── ephemerides/               # JPL ephemeris reading
│   ├── time_systems/              # UTC, TAI, TT conversions
│   └── celestial_mechanics/       # Orbital mechanics
│
├── navigation/                    # Navigation/
│   ├── __init__.py
│   ├── geodetic/                  # Geodetic calculations
│   ├── inertial/                  # INS algorithms
│   └── gnss/                      # GPS/GNSS utilities
│
├── atmosphere/                    # Atmosphere_and_Refraction/
│   ├── __init__.py
│   ├── models/                    # Atmosphere models
│   └── refraction/                # Refraction corrections
│
├── gravity/                       # Gravity/
│   ├── __init__.py
│   ├── spherical_harmonics/       # Gravity models
│   └── tides/                     # Tidal effects
│
├── magnetism/                     # Magnetism/
│   ├── __init__.py
│   └── models/                    # WMM, IGRF, EMM
│
├── terrain/                       # Terrain/
│   ├── __init__.py
│   └── models/                    # EGM2008, Earth2014
│
├── transponders/                  # Transponders/
│   └── __init__.py
│
├── scheduling/                    # Scheduling/
│   └── __init__.py
│
├── containers/                    # Container_Classes/
│   ├── __init__.py
│   ├── metric_tree.py             # Metric tree data structure
│   └── kd_tree.py                 # k-d tree
│
├── physical_values/               # Physical_Values/
│   └── __init__.py
│
└── misc/                          # Misc/
    ├── __init__.py
    └── plotting/                  # Visualization utilities
```

### 1.2 Python Dependencies

| Category | Libraries |
|----------|-----------|
| **Core** | `numpy`, `scipy`, `numba` |
| **Optimization** | `scipy.optimize`, `cvxpy` |
| **Statistics** | `scipy.stats`, `statsmodels` |
| **Geodesy** | `pyproj`, `geographiclib` |
| **Astronomy** | `astropy`, `jplephem` |
| **Visualization** | `matplotlib`, `cartopy` |
| **Performance** | `numba`, `cython` (for C++ ports) |
| **Testing** | `pytest`, `hypothesis` |

---

## 2. Conversion Phases

### Phase 0: Foundation (Weeks 1-4)

**Goal:** Establish project infrastructure and core utilities.

| Task | Description | Effort |
|------|-------------|--------|
| Package setup | pyproject.toml, CI/CD, docs structure | 3 days |
| Core constants | Physical constants from Physical_Values/ | 2 days |
| Array utilities | MATLAB-like helpers (repmat, reshape edge cases) | 3 days |
| Validation module | Input validation decorators | 2 days |
| Testing framework | pytest setup with MATLAB comparison fixtures | 3 days |
| Documentation | Sphinx setup, contribution guidelines | 2 days |

**Deliverable:** Empty package structure with CI/CD and one working test.

---

### Phase 1: Mathematical Foundation (Weeks 5-12)

**Goal:** Convert fundamental mathematical functions that all other modules depend on.

#### 1.1 Basic Matrix Operations (Week 5)
- `pinv` variants, matrix decompositions
- Special matrix constructions (Vandermonde, Toeplitz)
- **Python:** Most exist in `numpy.linalg`, wrap with consistent API

#### 1.2 Special Functions (Week 6)
- Bessel functions, gamma functions, error functions
- Elliptic integrals
- **Python:** Wrap `scipy.special` with matching signatures

#### 1.3 Statistics & Distributions (Weeks 7-8)
- All distribution classes (`GaussianD`, `PoissonD`, `ChiSquaredSumD`, etc.)
- PDF, CDF, random sampling, moments
- **Python:** Create distribution classes wrapping `scipy.stats`

#### 1.4 Numerical Integration (Week 9)
- Gaussian quadrature (Legendre, Laguerre, Hermite)
- Cubature rules for multidimensional integration
- **Python:** Combine `scipy.integrate` with custom cubature

#### 1.5 Interpolation (Week 10)
- 1D, 2D, 3D interpolation
- Spherical interpolation
- **Python:** Wrap `scipy.interpolate`

#### 1.6 Combinatorics (Week 11)
- Permutation generation, ranking/unranking
- Partition functions
- Assignment problem utilities
- **Python:** Pure Python + `numba` for performance

#### 1.7 Geometry (Week 12)
- Point-in-polygon, convex hull
- Line/plane intersections
- **Python:** `scipy.spatial` + custom implementations

**Deliverable:** `mathematical_functions` subpackage with >90% test coverage.

---

### Phase 2: Coordinate Systems (Weeks 13-18)

**Goal:** Convert all coordinate conversion and transformation code.

#### 2.1 Basic Conversions (Weeks 13-14)
| From | To | Functions |
|------|----|-----------|
| Cartesian | Spherical | `Cart2Sphere`, `Sphere2Cart` |
| Cartesian | Polar | `Cart2Pol`, `Pol2Cart` |
| Geodetic | ECEF | `ellips2Cart`, `Cart2Ellipse` |
| ENU/NED | ECEF | Local tangent plane conversions |

#### 2.2 Rotation Representations (Week 15)
- Rotation matrices, quaternions, Euler angles
- Axis-angle, Rodrigues parameters
- Conversions between all representations

#### 2.3 Jacobians & Hessians (Week 16)
- `calcSpherJacob`, `calcPolarJacob`
- Coordinate transformation Jacobians
- **Note:** C++ implementations need Numba/Cython port

#### 2.4 Reference Frames (Weeks 17-18)
- ECEF, ECI, GCRF, ITRF conversions
- Precession, nutation, Earth rotation
- **Python:** Leverage `astropy.coordinates`

**Deliverable:** `coordinate_systems` subpackage fully functional.

---

### Phase 3: Dynamic Models (Weeks 19-24)

**Goal:** Convert state transition models used in tracking.

#### 3.1 Discrete-Time Models (Weeks 19-20)
| Model | Files | Description |
|-------|-------|-------------|
| Constant Velocity | `FPolyKal.m`, `QPolyKal.m` | F and Q matrices |
| Coordinated Turn | `FCoordTurn2D.m`, `FCoordTurn3D.m` | 2D/3D turn models |
| Singer | `FSinger.m`, `QSinger.m` | Acceleration model |
| Nearly Constant | Various | NCV, NCA models |

#### 3.2 Continuous-Time Models (Weeks 21-22)
- Continuous dynamics `a(x,t)`, `D(x,t)` functions
- Discretization methods (Van Loan, Taylor series)

#### 3.3 Process Noise (Weeks 23-24)
- Process noise covariance calculations
- Correlations between position/velocity/acceleration

**Deliverable:** `dynamic_models` subpackage with all major motion models.

---

### Phase 4: Estimation Algorithms (Weeks 25-36)

**Goal:** Convert the core filtering and estimation algorithms.

#### 4.1 Kalman Filter Family (Weeks 25-28)
| Algorithm | Priority | Notes |
|-----------|----------|-------|
| Linear KF | P0 | Standard Kalman filter |
| EKF | P0 | Extended Kalman |
| UKF | P0 | Unscented (sigma points) |
| Square-root variants | P1 | Numerical stability |
| Information filter | P1 | Inverse covariance form |
| Cubature KF | P2 | Spherical-radial cubature |

#### 4.2 Particle Filters (Weeks 29-31)
- Bootstrap filter
- Auxiliary particle filter
- Regularized particle filter
- Resampling schemes (multinomial, systematic, residual)

#### 4.3 Batch Estimation (Weeks 32-33)
- Gauss-Newton
- Levenberg-Marquardt
- **Python:** Wrap `scipy.optimize.least_squares`

#### 4.4 Static Estimation (Weeks 34-36)
- Maximum likelihood estimators
- Robust M-estimators
- BLUE (Best Linear Unbiased Estimator)

**Deliverable:** Full `dynamic_estimation` and `static_estimation` subpackages.

---

### Phase 5: Data Association & Assignment (Weeks 37-44)

**Goal:** Convert assignment algorithms critical for multi-target tracking.

#### 5.1 2D Assignment (Weeks 37-39)
| Algorithm | Complexity | Notes |
|-----------|------------|-------|
| Hungarian (Kuhn-Munkres) | O(n³) | Optimal |
| Auction | O(n³) | Parallel-friendly |
| JVC (Jonker-Volgenant-Castanon) | O(n³) | Fast in practice |
| Greedy | O(n²) | Suboptimal, fast |

#### 5.2 Multi-Dimensional Assignment (Weeks 40-42)
- S-D assignment (NP-hard approximations)
- Lagrangian relaxation
- m-best solutions

#### 5.3 Track-to-Measurement Association (Weeks 43-44)
- Gating (ellipsoidal, rectangular)
- GNN (Global Nearest Neighbor)
- JPDA (Joint Probabilistic Data Association)
- MHT foundations

**Deliverable:** `assignment_algorithms` subpackage.

---

### Phase 6: Specialized Domains (Weeks 45-60)

#### 6.1 Astronomical Code (Weeks 45-48)
- Time systems (UTC, TAI, TT, TDB)
- JPL ephemeris reading
- Star catalog access
- **Python:** Heavy use of `astropy`

#### 6.2 Navigation (Weeks 49-52)
- Geodetic problems (direct, inverse)
- INS mechanization
- GNSS positioning
- **Python:** Use `geographiclib`, `pyproj`

#### 6.3 Gravity & Magnetism (Weeks 53-56)
- Spherical harmonic evaluation
- EGM2008, WMM, IGRF models
- **Note:** Requires porting efficient C++ code

#### 6.4 Atmosphere & Terrain (Weeks 57-60)
- Standard atmosphere models
- Refraction corrections
- Terrain elevation queries

**Deliverable:** Domain-specific subpackages.

---

### Phase 7: Integration & Demo (Weeks 61-68)

#### 7.1 End-to-End Tracker (Weeks 61-64)
Convert `demo2DIntegratedDataAssociation.m`:
- Track initiation
- GNN-JIPDAF data association
- Track maintenance
- Track termination

#### 7.2 Sample Code (Weeks 65-66)
- Convert key examples from `Sample_Code/`
- Jupyter notebook tutorials

#### 7.3 Performance Optimization (Weeks 67-68)
- Profile critical paths
- Numba JIT compilation
- Cython for bottlenecks

**Deliverable:** Working end-to-end tracker demonstration.

---

## 3. Module Dependency Graph

```
                    ┌─────────────────────────┐
                    │   mathematical_functions │
                    │   (Phase 1)              │
                    └───────────┬─────────────┘
                                │
            ┌───────────────────┼───────────────────┐
            │                   │                   │
            ▼                   ▼                   ▼
    ┌───────────────┐   ┌───────────────┐   ┌───────────────┐
    │  coordinate   │   │   dynamic     │   │   static      │
    │  systems      │   │   models      │   │   estimation  │
    │  (Phase 2)    │   │   (Phase 3)   │   │   (Phase 4)   │
    └───────┬───────┘   └───────┬───────┘   └───────────────┘
            │                   │
            └─────────┬─────────┘
                      │
                      ▼
            ┌───────────────────┐
            │ dynamic_estimation │
            │ (Phase 4)          │
            └─────────┬─────────┘
                      │
                      ▼
            ┌───────────────────┐
            │    assignment     │
            │    algorithms     │
            │    (Phase 5)      │
            └─────────┬─────────┘
                      │
         ┌────────────┼────────────┐
         │            │            │
         ▼            ▼            ▼
    ┌─────────┐  ┌─────────┐  ┌─────────┐
    │ astro   │  │ nav     │  │ gravity │
    │ (Ph 6)  │  │ (Ph 6)  │  │ (Ph 6)  │
    └─────────┘  └─────────┘  └─────────┘
                      │
                      ▼
            ┌───────────────────┐
            │  End-to-End Demo  │
            │  (Phase 7)        │
            └───────────────────┘
```

---

## 4. C/C++ Code Strategy

The library contains significant C/C++ MEX code. Strategy for each category:

| C/C++ Code Type | Strategy | Tools |
|-----------------|----------|-------|
| **Performance-critical MATLAB replacements** | Port to Numba JIT | `@numba.jit(nopython=True)` |
| **Third-party libraries (GeographicLib)** | Use Python bindings | `geographiclib` package |
| **Third-party libraries (liblbfgs)** | Use scipy equivalent | `scipy.optimize.minimize(method='L-BFGS-B')` |
| **Complex algorithms (spherical harmonics)** | Cython wrapper or rewrite | Cython with numpy |
| **Simple utilities** | Pure Python | NumPy vectorization |

### Key C++ Files Requiring Special Attention

1. **`calcSpherJacob.cpp`** - Spherical Jacobians → Numba
2. **`directGeodeticProb.cpp`** - Geodetic calculations → `geographiclib`
3. **`indirectGeodeticProb.cpp`** - Inverse geodetic → `geographiclib`
4. **Spherical harmonic evaluation** - Performance critical → Cython
5. **Assignment algorithms** - May need Cython for large-scale problems

---

## 5. Testing Strategy

### 5.1 Validation Approach

```python
# Example test structure
import numpy as np
import pytest
from scipy.io import loadmat

class TestCart2Sphere:
    @pytest.fixture
    def matlab_reference(self):
        """Load pre-computed MATLAB reference values."""
        return loadmat('tests/fixtures/cart2sphere_reference.mat')
    
    def test_basic_conversion(self, matlab_reference):
        """Test against MATLAB reference implementation."""
        cart_points = matlab_reference['cart_input']
        expected_sphere = matlab_reference['sphere_output']
        
        result = cart2sphere(cart_points)
        
        np.testing.assert_allclose(result, expected_sphere, rtol=1e-12)
    
    def test_edge_cases(self):
        """Test singularities and edge cases."""
        # Origin
        assert cart2sphere([0, 0, 0]) == [0, 0, 0]
        # Poles
        # ... etc
```

### 5.2 Reference Data Generation

Create MATLAB script to generate test fixtures:

```matlab
% generate_test_fixtures.m
% Run in MATLAB with TrackerComponentLibrary in path

% Cart2Sphere tests
cart_input = randn(3, 1000);
sphere_output = Cart2Sphere(cart_input);
save('cart2sphere_reference.mat', 'cart_input', 'sphere_output');

% ... repeat for other functions
```

### 5.3 Continuous Integration

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.10', '3.11', '3.12']
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install dependencies
        run: pip install -e ".[dev]"
      - name: Run tests
        run: pytest --cov=tracker_component_library
```

---

## 6. Effort Estimates by Module

| Module | Files (est.) | LOC (est.) | Complexity | Weeks | Priority |
|--------|--------------|------------|------------|-------|----------|
| Mathematical Functions | 200+ | 30,000 | High | 8 | P0 |
| Coordinate Systems | 80+ | 15,000 | Medium | 6 | P0 |
| Dynamic Models | 50+ | 8,000 | Medium | 6 | P0 |
| Dynamic Estimation | 100+ | 20,000 | High | 12 | P0 |
| Assignment Algorithms | 40+ | 10,000 | High | 8 | P0 |
| Static Estimation | 30+ | 5,000 | Medium | 3 | P1 |
| Clustering | 20+ | 3,000 | Medium | 2 | P1 |
| Performance Evaluation | 15+ | 2,000 | Low | 2 | P1 |
| Astronomical Code | 60+ | 12,000 | High | 4 | P2 |
| Navigation | 40+ | 8,000 | Medium | 4 | P2 |
| Gravity | 30+ | 8,000 | High | 4 | P2 |
| Magnetism | 20+ | 5,000 | Medium | 2 | P2 |
| Atmosphere | 15+ | 3,000 | Low | 2 | P2 |
| Terrain | 10+ | 2,000 | Medium | 2 | P3 |
| Containers | 10+ | 2,000 | Low | 1 | P1 |
| Misc | 20+ | 4,000 | Low | 2 | P3 |

**Total: ~740+ files, ~137,000 LOC, 68 weeks**

---

## 7. Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Numerical precision differences | High | Extensive unit tests with tolerance bounds |
| C++ code without MATLAB equivalent | High | Prioritize understanding algorithm, reimplement |
| Undocumented MATLAB behavior | Medium | Create comprehensive test cases first |
| External data dependencies | Medium | Document data sources, provide download scripts |
| Performance regression | Medium | Benchmark critical paths, use Numba/Cython |
| API design conflicts with NumPy conventions | Low | Follow NumPy conventions, document MATLAB differences |

---

## 8. Quick Start: First 2 Weeks

If you want to begin immediately, here's a concrete starting point:

### Week 1: Setup
1. Create GitHub repo with package structure
2. Set up `pyproject.toml` with dependencies
3. Configure pytest, pre-commit hooks
4. Write `core/constants.py` with physical constants

### Week 2: First Module
1. Convert `Cart2Sphere.m` and `Sphere2Cart.m`
2. Convert `Cart2Pol.m` and `Pol2Cart.m`
3. Write comprehensive tests
4. Document with NumPy-style docstrings

This gives you a working package with real functionality to build on.

---

## 9. Existing Python Alternatives

Before converting everything, consider existing Python libraries that cover parts of the functionality:

| Domain | Python Library | Coverage |
|--------|----------------|----------|
| Kalman filtering | `filterpy`, `pykalman` | Good |
| Coordinate transforms | `pyproj`, `astropy` | Excellent |
| Astronomy/ephemerides | `astropy`, `jplephem` | Excellent |
| Geodesy | `geographiclib` | Excellent |
| Assignment problems | `scipy.optimize.linear_sum_assignment` | Basic |
| Spherical harmonics | `pyshtools` | Good |
| Magnetic models | `pyIGRF`, `wmm` | Good |

**Recommendation:** Use these as dependencies where possible rather than reimplementing.

---

## 10. Conclusion

This roadmap provides a structured 12-18 month path to convert the TrackerComponentLibrary to Python. The key principles are:

1. **Foundation first** - Mathematical functions enable everything else
2. **Test-driven** - Generate MATLAB reference data before converting
3. **Leverage existing libraries** - Don't reinvent scipy, astropy, etc.
4. **Performance-aware** - Use Numba/Cython for critical paths
5. **Incremental delivery** - Working code at each phase milestone

Would you like me to:
- **Start Phase 0** (create the package skeleton)?
- **Deep-dive on any specific module**?
- **Convert a sample function** as a proof of concept?
