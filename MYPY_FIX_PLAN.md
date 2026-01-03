# Mypy Type Error Fix Plan

**Current Status:** 409 errors in 62 files across 161 source files

## Error Categories (By Frequency)

| Category | Count | Effort |
|----------|-------|--------|
| `type-arg` - Missing type parameters for generic `NDArray` | 320 | HIGH |
| `no-untyped-def` - Functions missing return type annotations | 32 | MEDIUM |
| `untyped-decorator` - Decorators without type annotations | 28 | MEDIUM |
| `no-untyped-call` - Calls to untyped functions | 28 | MEDIUM |
| `builtins.bool` - Other type issues | 1 | LOW |

---

## Priority 1: NDArray Type Parameters (320 errors)

**Root Cause:** Using `NDArray` without `[np.float64]` or `[np.Any]` type parameters  
**Impact:** Affects numerical/array-heavy modules  
**Effort:** HIGH (requires systematic review)

### Top Files to Fix (descending by error count)
1. `pytcl/dynamic_estimation/rbpf.py` - 32 errors
2. `pytcl/mathematical_functions/signal_processing/detection.py` - 30 errors
3. `pytcl/assignment_algorithms/jpda.py` - 23 errors
4. `pytcl/mathematical_functions/signal_processing/filters.py` - 17 errors
5. `pytcl/dynamic_estimation/kalman/constrained.py` - 17 errors
6. `pytcl/navigation/great_circle.py` - 16 errors
7. `pytcl/dynamic_estimation/particle_filters/bootstrap.py` - 16 errors
8. `pytcl/clustering/hierarchical.py` - 15 errors
9. `pytcl/mathematical_functions/signal_processing/matched_filter.py` - 13 errors
10. `pytcl/mathematical_functions/geometry/geometry.py` - 13 errors

**Solution Pattern:**
```python
# Before
def process(data: NDArray) -> NDArray:
    ...

# After
def process(data: NDArray[np.float64]) -> NDArray[np.float64]:
    ...

# Or for generic/mixed types:
def process(data: NDArray[np.Any]) -> NDArray[np.Any]:
    ...
```

---

## Priority 2: Missing Function Return Type Annotations (32 errors)

**Root Cause:** Functions lack explicit return type hints  
**Impact:** Type inference disabled for downstream code  
**Effort:** MEDIUM (mostly add-only changes)

### Solution Pattern:
```python
# Before
def calculate(x: float, y: float):
    return x + y

# After
def calculate(x: float, y: float) -> float:
    return x + y

# For None-returning functions:
def setup() -> None:
    ...

# For complex returns:
from typing import Tuple
def split() -> Tuple[list[float], list[float]]:
    ...
```

---

## Priority 3: Untyped Decorators (28 errors)

**Root Cause:** Decorators from external libraries or custom decorators without type hints  
**Impact:** Suppresses type information through decorator boundary  
**Effort:** MEDIUM (may need decorator typing)

### Common Patterns to Fix:
1. **pytest fixtures** - Add `@overload` or `@pytest.fixture` type stubs
2. **Custom decorators** - Add `Callable` type hints
3. **External library decorators** - Use `# type: ignore` or type stubs

### Solution Pattern:
```python
# Before
@decorator
def my_function():
    pass

# After
from typing import TypeVar, Callable
T = TypeVar('T')

def typed_decorator(func: Callable[..., T]) -> Callable[..., T]:
    def wrapper(*args, **kwargs) -> T:
        return func(*args, **kwargs)
    return wrapper

@typed_decorator
def my_function() -> None:
    pass
```

---

## Priority 4: Untyped Function Calls (28 errors)

**Root Cause:** Calling functions without return type annotations or type stubs  
**Impact:** Type checking stops at call boundary  
**Effort:** LOW-MEDIUM (depends on target functions)

### Common Sources:
- Calls to functions without return type annotations
- Calls to external library functions without stubs
- Dynamic function calls

### Solution Pattern:
```python
# Option 1: Add type annotation to called function
def untyped_function():  # Problem
    ...

def untyped_function() -> int:  # Solution
    ...

# Option 2: Use type: ignore for external libraries
result = external_lib.call()  # type: ignore

# Option 3: Add type cast
from typing import cast
result = cast(int, untyped_call())
```

---

## Recommended Implementation Order

### Phase 1: Quick Wins (Week 1)
**Target:** 100+ errors fixed
1. Fix `builtins.bool` issue (1 error) - 5 minutes
2. Add return type annotations to missing-def functions (32 errors) - 2 hours
3. Start with smallest files in type-arg category

### Phase 2: Core Modules (Weeks 2-3)
**Target:** 200+ errors fixed
1. Focus on modules with <20 errors per file
2. Address untyped-call errors by adding stubs/annotations
3. Fix untyped-decorator issues

### Phase 3: Large Modules (Weeks 4-6)
**Target:** 350+ errors fixed
1. `pytcl/dynamic_estimation/rbpf.py` (32 errors)
2. `pytcl/mathematical_functions/signal_processing/detection.py` (30 errors)
3. `pytcl/assignment_algorithms/jpda.py` (23 errors)

### Phase 4: Remaining (Weeks 7+)
**Target:** 409 errors resolved
1. Final NDArray type parameters
2. Complex decorator typing
3. Type stub generation for external dependencies

---

## Tools & Utilities

### mypy Configuration
```bash
# Current strict mode check
python -m mypy pytcl/ --strict --ignore-missing-imports

# Check specific file
python -m mypy pytcl/path/to/file.py --strict

# Show error codes
python -m mypy pytcl/ --strict --show-error-codes --ignore-missing-imports

# Generate report
python -m mypy pytcl/ --strict --ignore-missing-imports --html reports/mypy
```

### Automated Fixes
```bash
# Some errors can be auto-fixed (though not all)
python -m mypy pytcl/ --strict --ignore-missing-imports --show-traceback
```

---

## File-by-File Breakdown

### High Priority (20+ errors)
- `rbpf.py` - 32 errors (NDArray + no-untyped-def)
- `detection.py` - 30 errors (NDArray heavy)
- `jpda.py` - 23 errors (NDArray + complex types)

### Medium Priority (10-20 errors)
- `filters.py` - 17 errors
- `constrained.py` - 17 errors
- `great_circle.py` - 16 errors
- `bootstrap.py` - 16 errors
- `hierarchical.py` - 15 errors
- `matched_filter.py` - 13 errors
- `geometry.py` - 13 errors
- `imm.py` - 13 errors
- `gating.py` - 13 errors

### Low Priority (5-10 errors)
- `maximum_likelihood.py` - 10 errors
- `stft.py` - 10 errors
- `rotations.py` - 10 errors
- `ephemerides.py` - 10 errors
- (And 40+ other files with 1-8 errors)

---

## Success Criteria

- [ ] Phase 1: Reduce to <300 errors
- [ ] Phase 2: Reduce to <200 errors
- [ ] Phase 3: Reduce to <100 errors
- [ ] Phase 4: **0 errors** - Full mypy strict compliance

---

## Related Improvements

While fixing type errors:
1. Update numpy type imports if needed (numpy 1.20+ has improved typing)
2. Consider TypeVar usage for generic functions
3. Add `from __future__ import annotations` for Python 3.10 compatibility
4. Generate `.pyi` stub files for complex modules

---

## Notes

- These are **not** functional bugs - the code works correctly
- Type annotations improve IDE support and catch real errors
- Gradual typing is acceptable - fix by priority
- Some errors may be in test code (lower priority)
- External library compatibility (scipy, numpy) may require stubs
