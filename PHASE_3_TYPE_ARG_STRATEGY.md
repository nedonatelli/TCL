# Phase 3: Type-Arg Error Resolution Strategy

## Current State (After Phase 2c)
- **Total Errors**: 354 (down from 409, 55 fixed in phases 1-2c)
- **Type-Arg Errors**: 325 (91.8% of remaining)
- **Untyped-Decorator Errors**: 28 (7.9%)
- **Comparison-Overlap Errors**: 1 (0.3%)
- **No-Untyped Errors**: 0 (CLEARED - major achievement!)

## Type-Arg Error Categories

### 1. NDArray/ndarray Type Parameters (~270 errors)
These are the most common type-arg errors:
```python
# WRONG (current):
y: NDArray  # Missing type parameter
x: ndarray

# CORRECT:
y: NDArray[Any]  # For unspecified dtype
x: ndarray[Any]
y: NDArray[np.float64]  # For specific dtype
```

**Challenge**: Choosing the right dtype parameter:
- `NDArray[Any]` - Safe, works anywhere, but less specific
- `NDArray[np.float64]` - More precise, better type checking
- `NDArray[np.floating[Any]]` - Generic float types

**High-Impact Files** (descending error count):
- `rbpf.py`: 32 errors
- `detection.py`: 23 errors
- `jpda.py`: 21 errors
- `filters.py`: 17 errors
- `bootstrap.py`: 14 errors
- `hierarchical.py`: 14 errors
- `constrained.py`: 13 errors
- `imm.py`: 13 errors
- `matched_filter.py`: 11 errors
- `maximum_likelihood.py`: 10 errors

### 2. Generic Container Type Parameters (~40 errors)
```python
# WRONG:
cost_matrix: dict
assignments: list
indices: set
bounds: tuple

# CORRECT:
cost_matrix: dict[str, float]
assignments: list[int]
indices: set[int]
bounds: tuple[int, int]
```

**Affected Files**:
- `assignment.py`: dict/list errors
- `network_flow.py`: list errors
- `jpda.py`: set/dict errors
- `great_circle.py`: dict errors
- `geodesy.py`: dict errors
- `reference_frames.py`: dict/tuple errors

### 3. Callable Type Parameters (~15 errors)
```python
# WRONG:
handler: Callable  # Untyped
process: Callable

# CORRECT:
handler: Callable[..., Any]  # Generic callable
process: Callable[[int, str], dict[str, float]]  # Specific signature
```

**Affected Files**:
- `gaussian_sum_filter.py`: Callable errors
- `kalman/h_infinity.py`: Callable errors
- `kalman/sr_ukf.py`: Callable errors

## Implementation Strategy - REVISED (Safer Approach)

### Phase 3 Implementation Plan

**CRITICAL SAFETY ISSUE IDENTIFIED**: 
Automated type-arg fixes require extremely careful import management. Simple regex replacements broke multiline imports and caused missing import errors. Manual verification is essential.

**SAFER PHASED APPROACH**:

#### Step 1: Manual Verification Per File (Required)
For each high-impact file:
1. Read file and identify NDArray/dict/list/etc. usage
2. Understand what types are actually stored (e.g., float64 arrays)
3. Check if file already imports `Any` or needs it
4. Manually fix imports first, then annotations
5. Run file-specific mypy check
6. Run related test suite
7. Commit atomically

#### Step 2: Pattern-Based Fixes (Lower Risk)
Common patterns that can be safely batch-applied:
- NDArray without [^[]] → NDArray[Any] (safest default)
- Callable without [^[]] → Callable[..., Any] (generic callables)
- Only in actual type annotation lines (skip imports, strings, comments)

#### Step 3: Conservative Defaults
- Use `[Any]` liberally rather than trying to infer specific dtypes
- Better to have `NDArray[Any]` than `NDArray` 
- Type safety improved even if not fully specific
- Defer detailed dtype specifications to Phase 4+ refactoring

### Manual Workflow for Each File

```python
# File: pytcl/dynamic_estimation/rbpf.py (32 errors)

# 1. Check current state:
#    - NamedTuple with NDArray fields
#    - Functions with NDArray parameters  
#    - Callable parameters with NDArray
#    
# 2. Verify imports:
#    - from typing import Any, Callable, NamedTuple
#    - from numpy.typing import NDArray
#
# 3. Apply fixes (safe patterns only):
#    - RBPFParticle.y: NDArray -> RBPFParticle.y: NDArray[Any]
#    - Functions: y0: NDArray -> y0: NDArray[Any]
#    - Callables: g: Callable[[NDArray], NDArray] -> Callable[[NDArray[Any]], NDArray[Any]]
#
# 4. Test:
#    - pytest pytcl/ -k "rbpf or RBPF"
#    - mypy pytcl/dynamic_estimation/rbpf.py --strict
#
# 5. Commit:
#    - git add pytcl/dynamic_estimation/rbpf.py
#    - git commit -m "refactor: Add type parameters to rbpf.py (32 errors fixed)"
```

### Priority for Manual Implementation

1. **Tier 1** (easiest, highest immediate ROI):
   - rbpf.py (32 errors) - Clear data flow, NDArray everywhere
   - detection.py (23 errors) - Signal processing, float arrays
   - jpda.py (21 errors) - Data association, clear patterns

2. **Tier 2** (moderate difficulty):
   - filters.py (17 errors)
   - bootstrap.py (14 errors)
   - hierarchical.py (14 errors)

3. **Tier 3** (variable complexity):
   - constrained.py, imm.py, matched_filter.py, etc.

4. **Tier 4** (defer for now):
   - Files with complex generic types or unclear usage patterns

## Rationale for Conservative Approach

1. **Type Safety vs. Complexity**: Using `[Any]` is still a significant improvement over bare types
2. **Correctness First**: Better to have safe defaults than incorrect specific types
3. **Automation Risk**: Regex-based fixes broke previously passing tests; manual approach safer
4. **Future Refactoring**: Can refine types later when better numpy stubs available
5. **Iteration Friendly**: Small focused commits easier to debug/revert if needed

## Next Immediate Steps

1. Start with rbpf.py (highest ROI, clear patterns)
2. Manually verify and apply fixes
3. Test thoroughly before committing
4. Document approach for subsequent files
5. Build reusable patterns as we go

## Success Criteria

- **Milestone 1**: <50 type-arg errors (81% reduction from current 325)
- **Milestone 2**: <25 type-arg errors (92% reduction)
- **Final**: <10 type-arg errors in core modules

## Alternative: Pragmatic Deferral

If type-arg fixes become too time-consuming, consider:
- Using `# type: ignore[type-arg]` on specific lines
- Focusing efforts on untyped-decorator errors instead (28 errors, possibly simpler)
- Accepting current error level as "acceptable" for non-critical code

Current metrics:
- 354 errors is **85% reduction** from v1.6 baseline
- No-untyped errors completely cleared (**100% success**)
- All tests passing (2,098/2,098)
- Production code fully functional

