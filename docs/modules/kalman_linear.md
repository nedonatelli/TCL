# Linear Kalman Filter

> **Module Path**: `pytcl.dynamic_estimation.kalman.linear`
> **Status**: Production
> **MATLAB Parity**: Full

## Overview

This module provides the standard linear Kalman filter for systems with linear dynamics and linear measurements with Gaussian noise. It includes prediction (time update) and update (measurement update) steps, as well as a combined predict-update function.

## Architecture

### Design Pattern
- [x] Functional (stateless functions)
- [ ] Object-Oriented (classes with state)
- [ ] Hybrid (mix of both)

### Key Components

| Component | Type | Description |
|-----------|------|-------------|
| `kf_predict` | Function | Time prediction step |
| `kf_update` | Function | Measurement update step |
| `kf_predict_update` | Function | Combined predict and update |
| `KalmanState` | NamedTuple | State container (x, P) |
| `KalmanPrediction` | NamedTuple | Prediction result (x, P) |
| `KalmanUpdate` | NamedTuple | Update result (x, P, y, S, K, likelihood) |

### Algorithm Summary

| Algorithm | Complexity | Reference |
|-----------|------------|-----------|
| Kalman Prediction | O(n²) | Bar-Shalom et al., "Estimation with Applications to Tracking", Ch. 5 |
| Kalman Update | O(n²m + m³) | Bar-Shalom et al., "Estimation with Applications to Tracking", Ch. 5 |
| Joseph Form Covariance | O(n²m) | Bierman, "Factorization Methods for Discrete Sequential Estimation" |

## API Reference

### Functions

#### `kf_predict(x, P, F, Q, B=None, u=None)`

Kalman filter prediction (time update) step.

**Parameters:**
- `x` (array_like): Current state estimate, shape (n,).
- `P` (array_like): Current state covariance, shape (n, n). Must be positive semi-definite.
- `F` (array_like): State transition matrix, shape (n, n).
- `Q` (array_like): Process noise covariance, shape (n, n). Must be positive semi-definite.
- `B` (array_like, optional): Control input matrix, shape (n, m).
- `u` (array_like, optional): Control input, shape (m,).

**Returns:**
- `KalmanPrediction`: Named tuple with:
  - `x` (ndarray): Predicted state, shape (n,).
  - `P` (ndarray): Predicted covariance, shape (n, n).

**Example:**
```python
import numpy as np
from pytcl.dynamic_estimation.kalman.linear import kf_predict

x = np.array([0.0, 1.0])  # position=0, velocity=1
P = np.eye(2) * 0.1
F = np.array([[1, 1], [0, 1]])  # CV model, T=1
Q = np.array([[0.25, 0.5], [0.5, 1.0]])

pred = kf_predict(x, P, F, Q)
print(pred.x)  # [1., 1.]
```

#### `kf_update(x, P, z, H, R)`

Kalman filter update (measurement update) step.

**Parameters:**
- `x` (array_like): Predicted state estimate, shape (n,).
- `P` (array_like): Predicted state covariance, shape (n, n).
- `z` (array_like): Measurement, shape (m,).
- `H` (array_like): Measurement matrix, shape (m, n).
- `R` (array_like): Measurement noise covariance, shape (m, m). Must be positive definite.

**Returns:**
- `KalmanUpdate`: Named tuple with:
  - `x` (ndarray): Updated state, shape (n,).
  - `P` (ndarray): Updated covariance, shape (n, n).
  - `y` (ndarray): Innovation (residual), shape (m,).
  - `S` (ndarray): Innovation covariance, shape (m, m).
  - `K` (ndarray): Kalman gain, shape (n, m).
  - `likelihood` (float): Measurement likelihood for data association.

**Example:**
```python
import numpy as np
from pytcl.dynamic_estimation.kalman.linear import kf_update

x = np.array([1.0, 1.0])
P = np.array([[0.35, 0.5], [0.5, 1.1]])
z = np.array([1.2])  # position measurement
H = np.array([[1, 0]])
R = np.array([[0.1]])

upd = kf_update(x, P, z, H, R)
print(upd.x)  # Updated state
print(upd.K)  # Kalman gain
```

### Data Classes

#### `KalmanState`

Container for Kalman filter state.

**Attributes:**
- `x` (ndarray): State estimate, shape (n,).
- `P` (ndarray): State covariance, shape (n, n).

#### `KalmanPrediction`

Result of prediction step.

**Attributes:**
- `x` (ndarray): Predicted state estimate.
- `P` (ndarray): Predicted state covariance.

#### `KalmanUpdate`

Result of update step.

**Attributes:**
- `x` (ndarray): Updated state estimate.
- `P` (ndarray): Updated state covariance.
- `y` (ndarray): Innovation (measurement residual).
- `S` (ndarray): Innovation covariance.
- `K` (ndarray): Kalman gain matrix.
- `likelihood` (float): Gaussian measurement likelihood.

## Validation Contract

### Input Constraints

| Parameter | Type | Shape | Constraints | Notes |
|-----------|------|-------|-------------|-------|
| `x` | array_like | (n,) | Real-valued | Flattened internally |
| `P` | array_like | (n, n) | Symmetric, PSD | Symmetry enforced |
| `F` | array_like | (n, n) | Full rank | Singular F causes issues |
| `Q` | array_like | (n, n) | Symmetric, PSD | Process noise |
| `z` | array_like | (m,) | Real-valued | Measurement vector |
| `H` | array_like | (m, n) | Full row rank | Measurement matrix |
| `R` | array_like | (m, m) | Symmetric, PD | Must be positive definite |

### Output Guarantees

| Output | Type | Shape | Constraints | Notes |
|--------|------|-------|-------------|-------|
| `x` | ndarray | (n,) | Real-valued | State estimate |
| `P` | ndarray | (n, n) | Symmetric, PSD | Covariance guaranteed symmetric |
| `y` | ndarray | (m,) | Real-valued | Innovation |
| `S` | ndarray | (m, m) | Symmetric, PD | Innovation covariance |
| `K` | ndarray | (n, m) | Real-valued | Kalman gain |
| `likelihood` | float | scalar | [0, ∞) | 0 if Cholesky fails |

### Domain Checks

- **Cholesky failure**: If innovation covariance S is not positive definite, falls back to `np.linalg.solve` and sets likelihood to 0.0
- **Symmetry enforcement**: Output covariances are explicitly symmetrized via `(P + P.T) / 2`
- **Numerical stability**: Uses Joseph form for covariance update to preserve positive semi-definiteness

## Logging Specification

### Performance Markers

The module is instrumented via the benchmarking framework. Key timing points:

| Marker | Description | Typical Duration |
|--------|-------------|------------------|
| `kf_predict` | Full prediction step | 3-6 μs (4-state) |
| `kf_update` | Full update step | 22-30 μs (4-state, 2-meas) |

### Log Integration

```python
from pytcl.logging_config import get_logger

logger = get_logger(__name__)
# No logging statements in hot path for performance
```

## Performance Characteristics

### Computational Complexity

| Operation | Time | Space | Notes |
|-----------|------|-------|-------|
| `kf_predict` | O(n²) | O(n²) | Matrix multiplication dominated |
| `kf_update` | O(n²m + m³) | O(n² + nm) | Cholesky is O(m³) |
| `kf_predict_update` | O(n²m + m³) | O(n² + nm) | Combined operation |

### Benchmark Results

From `.benchmarks/slos.json`:

| Benchmark | Mean (local) | SLO (CI) |
|-----------|--------------|----------|
| `kf_predict[4]` | ~3.5 μs | 60 μs |
| `kf_predict[6]` | ~3.8 μs | 75 μs |
| `kf_predict[9]` | ~5.2 μs | 105 μs |
| `kf_predict[12]` | ~5.5 μs | 150 μs |
| `kf_update[4-2]` | ~27 μs | 90 μs |
| `kf_update[6-3]` | ~28 μs | 120 μs |
| `10_cycles_state_4` | ~0.3 ms | 3 ms |
| `100_cycles_state_6` | ~3.5 ms | 30 ms |

### Optimization Notes

- **Cholesky decomposition**: Uses `scipy.linalg.cho_factor` and `cho_solve` for efficient symmetric system solving
- **No JIT compilation**: Pure NumPy/SciPy for maintainability; performance is adequate for typical use
- **Reused factorization**: Cholesky factor is computed once and reused for gain and likelihood
- **Joseph form update**: More computationally expensive but numerically stable

### Bottlenecks

1. **Cholesky decomposition**: O(m³) where m is measurement dimension
2. **Matrix multiplications**: O(n²m) for H @ P @ H.T
3. **Potential improvement**: Numba JIT could provide 2-5x speedup if needed

## Dependencies

### Internal

- None (self-contained module)

### External

- `numpy`: Array operations, linear algebra
- `scipy.linalg`: `cho_factor`, `cho_solve` for efficient Cholesky operations

## Testing

### Test Coverage

| Test File | Status |
|-----------|--------|
| `tests/dynamic_estimation/test_kalman_linear.py` | Passing |
| `benchmarks/test_kalman_bench.py` | Passing |

### Test Categories

- **Unit tests**: Individual function correctness
- **Cycle tests**: Multiple predict-update cycles
- **Numerical stability**: Edge cases with ill-conditioned matrices
- **MATLAB validation**: Verified against MATLAB TCL reference

### Known Limitations

- **Large state dimensions**: For n > 100, consider square-root or information filter variants
- **Non-positive-definite R**: Will cause Cholesky failure; use pseudo-inverse variant if needed

## References

1. Y. Bar-Shalom, X. R. Li, and T. Kirubarajan, "Estimation with Applications to Tracking and Navigation", Wiley, 2001.
2. R. E. Kalman, "A New Approach to Linear Filtering and Prediction Problems", ASME J. Basic Eng., 1960.
3. MATLAB TCL: `Dynamic Estimation/Measurement Update and Prediction/kalmanFilter.m`

## Changelog

| Version | Changes |
|---------|---------|
| v0.1.0 | Initial implementation |
| v1.0.0 | Added Cholesky-based update, Joseph form |
| v1.1.0 | Added benchmark coverage |
