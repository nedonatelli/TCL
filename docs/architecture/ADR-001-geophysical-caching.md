# ADR-001: Geophysical Module Caching Strategy

## Status
Accepted

## Context

The pyTCL library includes several geophysical modules that perform computationally expensive operations:

1. **Gravity Models** (`pytcl/gravity/`)
   - Associated Legendre polynomials: O(n_max^2) computation per evaluation
   - EGM coefficient loading: Large file I/O (10-455 MB)
   - Clenshaw summation: Iterative computation for high-degree models

2. **Magnetic Models** (`pytcl/magnetism/`)
   - Similar spherical harmonic evaluations to gravity
   - Coefficient loading from external files

3. **Astronomical Reference Frames** (`pytcl/astronomical/reference_frames.py`)
   - Precession/nutation matrix computation
   - Multiple matrix multiplications per transformation

4. **JPL Ephemerides** (`pytcl/astronomical/ephemerides.py`)
   - Chebyshev polynomial interpolation
   - Kernel file loading (100-300 MB)

Many of these operations are called repeatedly with the same or similar inputs during typical use cases (e.g., batch processing of observation data at the same epoch).

## Decision

We will implement a multi-level caching strategy using Python's `functools.lru_cache` decorator:

### Level 1: File Loading (Already Implemented)
- EGM coefficients: `@lru_cache(maxsize=4)` on `_load_coefficients_cached()`
- EMM coefficients: Similar pattern
- Ephemeris kernels: Lazy loading with instance-level persistence

### Level 2: Expensive Computations (To Implement)
- **Associated Legendre polynomials**: Cache by `(n_max, m_max, x_quantized, normalized)`
  - Quantize `x` to 6 decimal places to enable cache hits for nearby angles
  - `maxsize=64` (covers ~64 unique colatitude values)

- **Reference frame matrices**: Cache by `(jd_quantized,)`
  - Quantize JD to 0.001 days (~86 seconds) for precession/nutation
  - `maxsize=128` (covers typical batch processing windows)

### Level 3: Session-Based Caching (Future)
- For terrain data (GEBCO/Earth2014): Regional tile caching
- For trajectory processing: Path-based precomputation

### Caching Guidelines

1. **When to cache:**
   - Pure functions with deterministic output
   - Computations with >100 microsecond execution time
   - Functions called repeatedly with same inputs

2. **When NOT to cache:**
   - Functions with side effects
   - Functions where inputs rarely repeat
   - Memory-intensive results (large arrays)

3. **Cache invalidation:**
   - Use `lru_cache` for automatic LRU eviction
   - Provide `clear_cache()` functions for explicit invalidation
   - Document cache behavior in docstrings

### Implementation Pattern

```python
from functools import lru_cache

# For floating-point inputs, quantize to enable cache hits
def _quantize_float(x: float, decimals: int = 6) -> float:
    """Quantize float for cache key compatibility."""
    return round(x, decimals)

@lru_cache(maxsize=64)
def _associated_legendre_cached(
    n_max: int,
    m_max: int,
    x_quantized: float,
    normalized: bool,
) -> tuple:
    """Cached Legendre computation (returns tuple for hashability)."""
    # Implementation...
    return tuple(map(tuple, result))
```

## Consequences

### Positive
- 5-10x performance improvement for repeated evaluations
- Reduced memory allocation for repeated computations
- Consistent with existing caching in EGM module
- No API changes required

### Negative
- Memory overhead for cached results (~1-2 MB per cache entry for Legendre)
- Quantization introduces small precision loss (~1e-6)
- Cache invalidation complexity for mutable global state

### Neutral
- Requires documentation of cache behavior
- May need tuning of `maxsize` parameters

## Implementation Notes

1. Start with reference frames (highest impact, simplest implementation)
2. Add Legendre caching (requires array->tuple conversion)
3. Profile before/after to validate improvements
4. Add logging to track cache hit rates

## References

- Python `functools.lru_cache` documentation
- EGM module existing implementation: `pytcl/gravity/egm.py:293`
- Track B validation framework: `pytcl/core/validation.py`
