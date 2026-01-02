# pyTCL Module Interdependencies

This document maps the dependencies between pyTCL modules, with emphasis on caching layers and performance-critical paths.

## Module Dependency Graph

```
pytcl/
├── core/
│   ├── validation.py          # Used by: all modules (input validation)
│   └── types.py                # Used by: all modules (type definitions)
│
├── navigation/
│   ├── geodesy.py              # CACHED: inverse_geodetic, direct_geodetic
│   │   └── Uses: numpy, functools.lru_cache
│   │   └── Exports: WGS84, GRS80, SPHERE ellipsoids
│   │
│   ├── great_circle.py         # CACHED: great_circle_distance, great_circle_azimuth
│   │   └── Uses: numpy, functools.lru_cache
│   │   └── No internal dependencies
│   │
│   ├── ins.py                  # Inertial navigation
│   │   └── Uses: geodesy.py (coordinate conversions)
│   │
│   ├── ins_gnss.py             # INS/GNSS fusion
│   │   └── Uses: ins.py, geodesy.py
│   │   └── Uses: dynamic_estimation.kalman
│   │
│   └── rhumb.py                # Rhumb line navigation
│       └── Uses: geodesy.py (ellipsoid parameters)
│
├── astronomical/
│   ├── reference_frames.py     # CACHED: precession, nutation matrices
│   │   └── Uses: numpy, functools.lru_cache
│   │   └── Cache key: quantized Julian date
│   │
│   ├── ephemerides.py          # JPL ephemeris access
│   │   └── Uses: lazy file loading
│   │   └── Large memory footprint (100-300 MB kernels)
│   │
│   └── time_systems.py         # Time conversions
│       └── Uses: reference_frames.py
│       └── Pure functions (no caching needed)
│
├── gravity/
│   ├── spherical_harmonics.py  # CACHED: associated Legendre polynomials
│   │   └── Uses: numpy, functools.lru_cache
│   │   └── Cache key: (n_max, m_max, x_quantized, normalized)
│   │
│   ├── egm.py                  # EGM96/EGM2008 models
│   │   └── Uses: spherical_harmonics.py
│   │   └── CACHED: coefficient loading
│   │   └── Large memory: 10-455 MB coefficients
│   │
│   ├── clenshaw.py             # Clenshaw summation
│   │   └── Uses: numpy (Numba-optimized)
│   │   └── No caching (streaming computation)
│   │
│   └── models.py               # Gravity constants
│       └── Pure constants (no dependencies)
│
├── coordinate_systems/
│   ├── conversions/
│   │   ├── geodetic.py         # ECEF <-> Geodetic
│   │   │   └── Uses: numpy
│   │   │   └── No caching (array-vectorized)
│   │   │
│   │   └── spherical.py        # Spherical coordinates
│   │       └── Uses: numpy
│   │
│   └── rotations/
│       └── rotations.py        # Rotation utilities
│           └── Uses: numpy (Numba-optimized)
│           └── No caching (inexpensive operations)
│
├── dynamic_estimation/
│   ├── kalman/
│   │   ├── linear.py           # Linear Kalman filter
│   │   │   └── Uses: numpy
│   │   │   └── No caching (state-dependent)
│   │   │
│   │   └── extended.py         # Extended Kalman filter
│   │       └── Uses: numpy, linear.py
│   │
│   └── imm.py                  # Interacting Multiple Model
│       └── Uses: kalman/
│
├── assignment_algorithms/
│   ├── gating.py               # Mahalanobis gating
│   │   └── Uses: numpy (Numba-optimized)
│   │   └── No caching (data-dependent)
│   │
│   ├── hungarian.py            # Hungarian algorithm
│   │   └── Uses: numpy (Numba-optimized)
│   │
│   └── jpda.py                 # JPDA probabilities
│       └── Uses: gating.py
│
└── mathematical_functions/
    ├── signal_processing/
    │   └── detection.py        # CFAR detection
    │       └── Uses: numpy (Numba-optimized)
    │
    └── special_functions/
        └── hypergeometric.py   # Generalized hypergeometric
            └── Uses: numpy
            └── No caching (exact computation needed)
```

## Caching Architecture Summary

### Currently Cached Modules

| Module | Function | Cache Size | Purpose |
|--------|----------|------------|---------|
| `navigation.great_circle` | `great_circle_distance` | 256 | Angular distance |
| `navigation.great_circle` | `great_circle_azimuth` | 256 | Initial bearing |
| `navigation.geodesy` | `inverse_geodetic` | 128 | Vincenty inverse |
| `navigation.geodesy` | `direct_geodetic` | 128 | Vincenty direct |
| `astronomical.reference_frames` | `precession_matrix` | 128 | Precession computation |
| `astronomical.reference_frames` | `nutation_matrix` | 128 | Nutation computation |
| `gravity.spherical_harmonics` | `associated_legendre` | 64 | Legendre polynomials |
| `gravity.egm` | `_load_coefficients_cached` | 4 | EGM coefficient files |

### Cache Management Functions

Each cached module provides:
- `clear_<module>_cache()`: Clear all caches for the module
- `get_<module>_cache_info()`: Get cache statistics

Example:
```python
from pytcl.navigation.great_circle import clear_great_circle_cache, get_cache_info
from pytcl.navigation.geodesy import clear_geodesy_cache, get_geodesy_cache_info

# Check cache statistics
print(get_cache_info())
print(get_geodesy_cache_info())

# Clear caches when needed
clear_great_circle_cache()
clear_geodesy_cache()
```

## Performance-Critical Paths

### Path 1: Track Processing
```
Observation → coordinate_systems.geodetic → navigation.geodesy → dynamic_estimation.kalman
                                                                          ↓
                                                              assignment_algorithms.gating
```

### Path 2: Gravity Computation
```
Position → gravity.egm → gravity.spherical_harmonics (CACHED) → gravity.clenshaw
                ↓
           (CACHED coefficient loading)
```

### Path 3: Reference Frame Transformation
```
ICRF Position → astronomical.reference_frames (CACHED) → ITRF Position
                            ↓
           astronomical.time_systems (for epoch conversion)
```

### Path 4: Great Circle Navigation
```
Origin/Destination → navigation.great_circle (CACHED) → Waypoints
                              ↓
                    navigation.geodesy (CACHED for ellipsoidal)
```

## Logging Hierarchy

All modules use hierarchical loggers under `pytcl.*`:

```
pytcl
├── pytcl.navigation.great_circle
├── pytcl.navigation.geodesy
├── pytcl.astronomical.reference_frames
├── pytcl.gravity.egm
└── pytcl.gravity.spherical_harmonics
```

Configure logging:
```python
import logging
logging.getLogger("pytcl").setLevel(logging.DEBUG)
```

## Related Documentation

- [ADR-001: Geophysical Module Caching Strategy](ADR-001-geophysical-caching.md)
- [ADR-002: Lazy-Loading Architecture](ADR-002-lazy-loading-architecture.md)
