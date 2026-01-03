# pyTCL Architecture Overview

This document provides a unified view of pyTCL's architecture, consolidating key design decisions and patterns.

## Core Design Principles

1. **NumPy-native API**: All functions accept and return NumPy arrays
2. **Numba acceleration**: Performance-critical paths use JIT compilation
3. **LRU caching**: Expensive computations are cached with quantized keys
4. **Lazy loading**: Large data files are loaded on first use
5. **Minimal dependencies**: Core functionality requires only NumPy and SciPy

## Module Organization

```
pytcl/
├── core/                     # Shared utilities and validation
├── dynamic_estimation/       # Kalman filters, particle filters, IMM
├── dynamic_models/           # Motion models (CV, CA, Singer, CT)
├── coordinate_systems/       # Spherical, geodetic, rotations
├── assignment_algorithms/    # Hungarian, JPDA, MHT, gating
├── navigation/               # INS, GNSS, geodesy
├── astronomical/             # Reference frames, ephemerides
├── gravity/                  # EGM, spherical harmonics
├── magnetism/                # WMM, IGRF, EMM
├── atmosphere/               # Standard atmosphere, ionosphere
├── mathematical_functions/   # Signal processing, special functions
└── clustering/               # DBSCAN, K-means
```

## Performance Architecture

### Caching Strategy (ADR-001)

pyTCL uses multi-level caching for computationally expensive operations:

#### Level 1: File Loading
- EGM coefficients: `@lru_cache(maxsize=4)`
- EMM coefficients: Instance-level persistence
- Ephemeris kernels: Lazy loading with file handle caching

#### Level 2: Expensive Computations
| Module | Function | Cache Size | Key Quantization |
|--------|----------|------------|------------------|
| gravity.spherical_harmonics | associated_legendre | 64 | 8 decimals |
| astronomical.reference_frames | precession_matrix | 128 | 0.001 days |
| navigation.great_circle | distance/azimuth | 256 | 10 decimals |
| navigation.geodesy | inverse/direct | 128 | 10 decimals |
| magnetism.wmm | magnetic_field_spherical | 1024 | 6 decimals |

#### Caching Guidelines

**When to cache:**
- Pure functions with deterministic output
- Computations with >100 µs execution time
- Functions called repeatedly with same/similar inputs

**When NOT to cache:**
- Functions with side effects
- State-dependent computations (Kalman filters)
- Memory-intensive results (large arrays)

### Lazy Loading (ADR-002)

Large datasets are loaded only when first accessed:

```python
# Pattern: Deferred data loading
class EGMModel:
    _coefficients: Optional[EGMCoefficients] = None

    @classmethod
    def _ensure_loaded(cls):
        if cls._coefficients is None:
            cls._coefficients = _load_coefficients()

    def geoid_height(self, lat, lon):
        self._ensure_loaded()  # Load on first call
        return _compute_geoid(lat, lon, self._coefficients)
```

**Benefits:**
- Fast initial import (<1 second for basic usage)
- Memory proportional to actual usage
- No API changes required

### Numba Acceleration

Performance-critical functions use Numba JIT compilation:

```python
from numba import njit

@njit(cache=True)
def _mahalanobis_distance_impl(x, mean, cov_inv):
    diff = x - mean
    return np.sqrt(diff @ cov_inv @ diff)
```

**Numba-optimized modules:**
- `assignment_algorithms.gating`: Mahalanobis distance
- `assignment_algorithms.hungarian`: Hungarian algorithm
- `coordinate_systems.rotations`: Rotation operations
- `mathematical_functions.signal_processing`: CFAR detection
- `clustering.dbscan`: Distance matrix computation

## Module Interdependencies

### Critical Performance Paths

**Track Processing:**
```
Observation → geodetic conversion → geodesy → Kalman filter → gating
```

**Gravity Computation:**
```
Position → EGM model → spherical harmonics (CACHED) → Clenshaw summation
                ↓
         (CACHED coefficient loading)
```

**Reference Frame Transformation:**
```
ICRF Position → reference_frames (CACHED) → ITRF Position
```

### Cache Management API

Each cached module provides management functions:

```python
# Navigation caching
from pytcl.navigation.great_circle import clear_great_circle_cache, get_cache_info
from pytcl.navigation.geodesy import clear_geodesy_cache, get_geodesy_cache_info

# Magnetism caching
from pytcl.magnetism import clear_magnetic_cache, get_magnetic_cache_info, configure_magnetic_cache

# Check statistics
print(get_cache_info())
print(get_magnetic_cache_info())

# Clear when needed
clear_great_circle_cache()
clear_magnetic_cache()

# Configure cache parameters
configure_magnetic_cache(maxsize=2048, precision={"lat": 8, "lon": 8})
```

## Data File Handling

### External Data Requirements

| Module | Data Type | Size | Source |
|--------|-----------|------|--------|
| gravity.egm | EGM96 coefficients | ~10 MB | NGA |
| gravity.egm | EGM2008 coefficients | ~455 MB | NGA |
| magnetism.emm | EMM coefficients | ~50 MB | NOAA |
| astronomical | JPL ephemerides | 100-300 MB | JPL |

### Data Directory Configuration

```python
from pytcl.magnetism import get_emm_data_dir
from pytcl.gravity import get_egm_data_dir

# Check configured paths
print(get_emm_data_dir())
print(get_egm_data_dir())

# Set custom paths via environment variables
# PYTCL_EMM_DATA=/path/to/emm
# PYTCL_EGM_DATA=/path/to/egm
```

## Logging Architecture

All modules use hierarchical loggers under `pytcl.*`:

```
pytcl
├── pytcl.navigation.great_circle
├── pytcl.navigation.geodesy
├── pytcl.astronomical.reference_frames
├── pytcl.gravity.egm
├── pytcl.gravity.spherical_harmonics
├── pytcl.magnetism.wmm
└── pytcl.magnetism.igrf
```

**Configuration:**
```python
import logging

# Enable debug logging for all pytcl modules
logging.getLogger("pytcl").setLevel(logging.DEBUG)

# Or specific modules
logging.getLogger("pytcl.magnetism").setLevel(logging.INFO)
```

## Error Handling

### Validation Pattern

Input validation is performed at module boundaries:

```python
def compute_something(lat, lon, altitude):
    # Validate at entry point
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)

    if np.any(np.abs(lat) > np.pi/2):
        raise ValueError("Latitude must be in [-π/2, π/2]")

    # Internal functions can skip validation
    return _compute_something_impl(lat, lon, altitude)
```

### Graceful Degradation

For optional features:

```python
try:
    from pytcl.gravity import egm2008
except ImportError:
    # Fall back to lower-resolution model
    from pytcl.gravity import egm96 as egm2008
    warnings.warn("EGM2008 coefficients not found, using EGM96")
```

## Testing Strategy

### Unit Tests
- Located in `tests/`
- Run with `pytest tests/ -v`
- Coverage target: 90%+

### Benchmarks
- Located in `benchmarks/`
- Light suite for PRs (Kalman, gating, rotations)
- Full suite for main branch
- SLOs defined in `.benchmarks/slos.json`

### Integration Tests
- End-to-end tracking scenarios
- Cross-module validation
- Real-world data processing

## Related Documents

- [Performance Dashboard](PERFORMANCE.md) - SLOs and benchmarking
- [Module Interdependencies](module-interdependencies.md) - Detailed dependency graph
- [ADR-001: Geophysical Caching](ADR-001-geophysical-caching.md) - Caching design
- [ADR-002: Lazy Loading](ADR-002-lazy-loading-architecture.md) - Lazy loading design

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.3.0 | 2026-01-02 | Added magnetism caching, Phase 16 complete |
| 1.2.0 | 2025-12-xx | Navigation caching, special functions |
| 1.1.0 | 2025-11-xx | Benchmarking infrastructure, SLOs |
| 1.0.0 | 2025-10-xx | Initial release |
