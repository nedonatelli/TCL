# Module Documentation Template

This template provides a standardized format for documenting pyTCL modules. Each module should have a corresponding documentation file following this structure.

---

## Template

```markdown
# {Module Name}

> **Module Path**: `pytcl.{subpackage}.{module}`
> **Status**: Production / Beta / Experimental
> **MATLAB Parity**: Full / Partial / Extended

## Overview

Brief description of the module's purpose and key functionality (2-3 sentences).

## Architecture

### Design Pattern
- [ ] Functional (stateless functions)
- [ ] Object-Oriented (classes with state)
- [ ] Hybrid (mix of both)

### Key Components

| Component | Type | Description |
|-----------|------|-------------|
| `function_name` | Function | Brief description |
| `ClassName` | Class | Brief description |

### Class Hierarchy (if applicable)

```
BaseClass
├── SubClassA
│   └── SpecializedA
└── SubClassB
```

### Algorithm Summary

| Algorithm | Complexity | Reference |
|-----------|------------|-----------|
| Algorithm name | O(n²) | [Paper/Book] Section X.Y |

## API Reference

### Functions

#### `function_name(param1, param2, **kwargs)`

Brief description.

**Parameters:**
- `param1` (type): Description. Units: [unit]. Range: [min, max].
- `param2` (type, optional): Description. Default: value.

**Returns:**
- `result` (type): Description. Shape: (n, m) for arrays.

**Raises:**
- `ValueError`: When param1 is out of range.

**Example:**
```python
from pytcl.module import function_name
result = function_name(1.0, 2.0)
```

### Classes

#### `ClassName`

Brief description.

**Attributes:**
- `attr1` (type): Description.

**Methods:**
- `method1(args)`: Brief description.

## Validation Contract

### Input Constraints

| Parameter | Type | Range | Units | Notes |
|-----------|------|-------|-------|-------|
| `param1` | float | [0, 2π) | radians | Normalized internally |
| `param2` | ndarray | shape (3,) | meters | Right-handed coords |

### Output Guarantees

| Output | Type | Range | Units | Notes |
|--------|------|-------|-------|-------|
| `result` | float | [0, ∞) | meters | Always non-negative |

### Domain Checks

- **Singularity handling**: How gimbal lock, division by zero, etc. are handled
- **Edge cases**: Behavior at boundaries (e.g., poles, date line)
- **Numerical stability**: Conditioning, precision considerations

## Logging Specification

### Log Messages

| Level | Event | Format |
|-------|-------|--------|
| DEBUG | Function entry | `"Entering {func} with param1={}"` |
| INFO | Major computation | `"Computing {algorithm} for n={} points"` |
| WARNING | Numerical issue | `"Near singularity detected, using fallback"` |

### Performance Markers

```python
from pytcl.logging_config import timed

@timed
def expensive_function(...):
    ...
```

| Marker | Description | Typical Duration |
|--------|-------------|------------------|
| `module.function` | Full function timing | 0.5-2.0 ms |
| `module.inner_loop` | Critical section | 0.01-0.1 ms |

## Performance Characteristics

### Computational Complexity

| Operation | Time | Space | Notes |
|-----------|------|-------|-------|
| `function1` | O(n) | O(1) | Linear scan |
| `function2` | O(n²) | O(n) | Distance matrix |

### Benchmark Results

See `.benchmarks/slos.json` for SLO thresholds.

| Benchmark | Mean | P99 | SLO |
|-----------|------|-----|-----|
| `test_function_small` | 0.05 ms | 0.15 ms | 0.10 ms |
| `test_function_large` | 5.0 ms | 15.0 ms | 10.0 ms |

### Optimization Notes

- **Numba JIT**: Which functions use `@njit`
- **Vectorization**: Which operations are vectorized
- **Caching**: Which results are cached (LRU, memoization)
- **Parallelization**: Thread/process parallelism if any

### Bottlenecks

Known performance bottlenecks and potential improvements:

1. **Matrix inversion in X**: Could use Cholesky for SPD matrices
2. **Loop in Y**: Candidate for Numba optimization

## Dependencies

### Internal

- `pytcl.core.math_utils`: Basic math utilities
- `pytcl.coordinate_systems`: Coordinate transforms

### External

- `numpy`: Array operations
- `scipy.linalg`: Matrix decompositions
- `numba`: JIT compilation (optional, for performance)

## Testing

### Test Coverage

| Test File | Coverage | Status |
|-----------|----------|--------|
| `test_module.py` | 85% | Passing |

### Test Categories

- **Unit tests**: Individual function correctness
- **Integration tests**: Multi-function workflows
- **MATLAB validation**: Comparison with reference implementation
- **Property tests**: Hypothesis-based fuzzing

### Known Limitations

- Limitation 1: Description and workaround
- Limitation 2: Description and planned fix

## References

1. Author, "Title", Journal, Year. DOI: xxx
2. MATLAB TCL: `Module/subModule.m`

## Changelog

| Version | Changes |
|---------|---------|
| v1.0.0 | Initial implementation |
| v1.1.0 | Added Numba optimization |
```

---

## Usage Guidelines

1. **Create documentation**: Copy this template to `docs/modules/{module_name}.md`
2. **Fill sections**: Complete all applicable sections (skip N/A sections)
3. **Update on changes**: Keep documentation in sync with code changes
4. **Link from API docs**: Reference from `docs/api/{subpackage}.rst`

## Section Importance

| Section | Required | Notes |
|---------|----------|-------|
| Overview | Yes | Always needed |
| Architecture | Yes | Core design info |
| API Reference | Yes | Primary reference |
| Validation Contract | Yes | Critical for users |
| Logging Specification | Recommended | For instrumented modules |
| Performance Characteristics | Recommended | For optimized modules |
| Dependencies | Yes | Always document |
| Testing | Recommended | Good practice |
| References | Recommended | Academic modules |
