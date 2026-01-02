# ADR-002: Lazy-Loading Architecture for High-Resolution Data

## Status
Accepted

## Context

pyTCL includes several modules that require loading large datasets:

1. **Gravity Models** (`pytcl/gravity/`)
   - EGM96: ~10 MB coefficients (degree 360)
   - EGM2008: ~455 MB coefficients (degree 2190)
   - Load time: 0.5-15 seconds depending on model

2. **Navigation Models** (`pytcl/navigation/`)
   - Geodetic calculations: Vincenty iterations (10-100 per call)
   - Great circle: Trigonometric computations
   - Load time: Minimal, but repeated calls add up

3. **Astronomical Reference Frames** (`pytcl/astronomical/`)
   - Precession/nutation matrices: Complex matrix chain multiplications
   - Ephemeris kernels: 100-300 MB per kernel

4. **Terrain Models** (Future)
   - GEBCO: Multi-GB global bathymetry
   - Earth2014: High-resolution topography tiles

Current behavior loads all resources at import time, leading to:
- Slow initial import (~5-20 seconds for full library)
- High memory usage even when only using subset of features
- Poor user experience for simple use cases

## Decision

We will implement a lazy-loading architecture with the following patterns:

### Pattern 1: Module-Level Lazy Imports

For expensive modules, use lazy import wrappers:

```python
# In pytcl/__init__.py
def __getattr__(name):
    """Lazy-load submodules on first access."""
    if name == "gravity":
        from pytcl import gravity as _gravity
        globals()["gravity"] = _gravity
        return _gravity
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

### Pattern 2: Deferred Data Loading

For data-heavy modules, defer loading until first use:

```python
class EGMModel:
    """Earth Gravitational Model with lazy coefficient loading."""

    _coefficients: Optional[EGMCoefficients] = None

    @classmethod
    def _ensure_loaded(cls):
        """Load coefficients on first use."""
        if cls._coefficients is None:
            cls._coefficients = _load_coefficients()
            _logger.info(f"Loaded EGM coefficients: {cls._coefficients.n_max}")

    def geoid_height(self, lat: float, lon: float) -> float:
        """Compute geoid height (loads model on first call)."""
        self._ensure_loaded()
        return _compute_geoid(lat, lon, self._coefficients)
```

### Pattern 3: Session-Based Model Selection

Allow users to specify which models to load:

```python
from pytcl import configure

# Only load EGM96 (fast startup)
configure(gravity_model="egm96")

# Or defer loading until needed
configure(lazy_load=True)
```

### Pattern 4: LRU Caching for Computed Results

Cache expensive computations with quantized keys (see ADR-001):

```python
@lru_cache(maxsize=256)
def _gc_distance_cached(lat1_q, lon1_q, lat2_q, lon2_q) -> float:
    """Cached great circle distance computation."""
    # Computation...
```

### Implementation in Navigation Module

The navigation module implements Patterns 3 and 4:

1. **`pytcl/navigation/great_circle.py`**:
   - `_gc_distance_cached()`: Cached angular distance (256 entries)
   - `_gc_azimuth_cached()`: Cached azimuth (256 entries)
   - `_GC_CACHE_DECIMALS = 10`: ~0.01mm precision quantization
   - Cache management: `clear_great_circle_cache()`, `get_cache_info()`

2. **`pytcl/navigation/geodesy.py`**:
   - `_inverse_geodetic_cached()`: Cached Vincenty inverse (128 entries)
   - `_direct_geodetic_cached()`: Cached Vincenty direct (128 entries)
   - `_VINCENTY_CACHE_DECIMALS = 10`: ~0.01mm precision quantization
   - Cache management: `clear_geodesy_cache()`, `get_geodesy_cache_info()`

### Cache Configuration Guidelines

| Module | Function | Cache Size | Precision | Typical Use Case |
|--------|----------|------------|-----------|------------------|
| great_circle | distance | 256 | 10 decimals | Batch navigation |
| great_circle | azimuth | 256 | 10 decimals | Path planning |
| geodesy | inverse | 128 | 10 decimals | Survey calculations |
| geodesy | direct | 128 | 10 decimals | Waypoint generation |
| ref_frames | precession | 128 | 0.001 days | Epoch conversions |
| spherical_harmonics | legendre | 64 | 8 decimals | Gravity evaluation |

## Consequences

### Positive
- Fast initial import (<1 second for basic usage)
- Memory usage proportional to actual usage
- 5-10x speedup for repeated computations
- No API changes required for existing users

### Negative
- First call to lazy-loaded module incurs load time
- Cache memory overhead (~1-10 MB depending on usage)
- Slightly more complex module structure
- Quantization introduces negligible precision loss (~1e-10)

### Neutral
- Requires consistent logging to track load events
- Documentation must explain lazy-loading behavior
- Testing must cover both cached and uncached paths

## Module Interdependencies

The caching architecture creates the following dependency structure:

```
pytcl.navigation.great_circle
    └── Uses: functools.lru_cache, logging
    └── Cached: great_circle_distance, great_circle_azimuth

pytcl.navigation.geodesy
    └── Uses: functools.lru_cache, logging
    └── Cached: inverse_geodetic, direct_geodetic

pytcl.astronomical.reference_frames
    └── Uses: functools.lru_cache, logging
    └── Cached: precession_matrix, nutation_matrix

pytcl.gravity.spherical_harmonics
    └── Uses: functools.lru_cache, logging
    └── Cached: associated_legendre (via wrapper)
```

## Performance Impact

Based on local benchmarks:

| Operation | Uncached | Cached | Speedup |
|-----------|----------|--------|---------|
| great_circle_distance | 1.5 us | 0.3 us | 5x |
| inverse_geodetic | 8.2 us | 0.4 us | 20x |
| precession_matrix | 12.5 us | 0.5 us | 25x |
| associated_legendre (n=360) | 850 us | 1.2 us | 700x |

## References

- ADR-001: Geophysical Module Caching Strategy
- Python `functools.lru_cache` documentation
- Navigation module implementation: `pytcl/navigation/geodesy.py`, `pytcl/navigation/great_circle.py`
