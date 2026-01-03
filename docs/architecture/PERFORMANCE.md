# pyTCL Performance Dashboard

This document provides performance Service Level Objectives (SLOs) and benchmarking guidance for pyTCL modules.

## Performance Summary

| Category | Operation | Typical Latency | SLO Target |
|----------|-----------|-----------------|------------|
| Kalman Filter | predict (4-state) | 0.02ms | 0.06ms |
| Kalman Filter | update (4-state, 2-meas) | 0.03ms | 0.09ms |
| Gating | Mahalanobis (2D) | 0.005ms | 0.015ms |
| Rotations | euler2rotmat | 0.005ms | 0.015ms |
| JPDA | 5 tracks, 10 meas | 0.2ms | 0.6ms |
| CFAR | CA-CFAR (1000 samples) | 0.3ms | 0.9ms |
| Clustering | DBSCAN (100 points) | 1.7ms | 5.0ms |

## Kalman Filter Performance

The linear Kalman filter is one of the most frequently called modules. SLOs scale with state dimension.

### kf_predict

| State Dimension | Mean (ms) | P99 (ms) | Iterations |
|-----------------|-----------|----------|------------|
| 4 | 0.060 | 0.300 | 10,000 |
| 6 | 0.075 | 0.360 | 10,000 |
| 9 | 0.105 | 0.450 | 10,000 |
| 12 | 0.150 | 0.600 | 10,000 |

### kf_update

| Configuration | Mean (ms) | P99 (ms) | Iterations |
|---------------|-----------|----------|------------|
| 4-state, 2-meas | 0.090 | 0.360 | 10,000 |
| 6-state, 3-meas | 0.120 | 0.450 | 10,000 |
| 9-state, 3-meas | 0.150 | 0.540 | 10,000 |
| 12-state, 4-meas | 0.210 | 0.750 | 10,000 |

### Cycle Benchmarks

| Scenario | Mean (ms) | P99 (ms) | Iterations |
|----------|-----------|----------|------------|
| 10 cycles (4-state) | 3.0 | 9.0 | 1,000 |
| 100 cycles (6-state) | 30.0 | 90.0 | 100 |

## Gating Performance

Mahalanobis distance computation is Numba-optimized for low latency.

### mahalanobis_distance

| Dimension | Mean (ms) | P99 (ms) | Iterations |
|-----------|-----------|----------|------------|
| 2 | 0.015 | 0.060 | 50,000 |
| 3 | 0.024 | 0.090 | 50,000 |
| 6 | 0.045 | 0.150 | 50,000 |

### Batch Gating Scenarios

| Scenario | Mean (ms) | P99 (ms) | Iterations |
|----------|-----------|----------|------------|
| Batch 100 measurements | 3.0 | 10.0 | 1,000 |
| Batch 1000 measurements | 25.0 | 75.0 | 100 |
| 20 tracks × 50 measurements | 25.0 | 75.0 | 100 |

## Rotation Operations

Rotation utilities are Numba-optimized for high-frequency calls.

### Single Operations

| Function | Mean (ms) | P99 (ms) | Iterations |
|----------|-----------|----------|------------|
| euler2rotmat | 0.015 | 0.060 | 100,000 |
| quat2rotmat | 0.015 | 0.060 | 100,000 |
| quat_multiply | 0.009 | 0.045 | 100,000 |
| quat_rotate | 0.015 | 0.060 | 100,000 |

### Batch Operations (10,000 elements)

| Function | Mean (ms) | P99 (ms) | Iterations |
|----------|-----------|----------|------------|
| euler2rotmat batch | 50.0 | 150.0 | 100 |
| quat_multiply batch | 12.0 | 36.0 | 100 |
| quat_rotate batch | 30.0 | 90.0 | 100 |

## JPDA Performance

Joint Probabilistic Data Association scales with the product of tracks and measurements.

| Scenario | Mean (ms) | P99 (ms) | Iterations |
|----------|-----------|----------|------------|
| 5 tracks × 10 meas | 0.60 | 1.50 | 5,000 |
| 10 tracks × 20 meas | 3.00 | 9.00 | 1,000 |
| 20 tracks × 50 meas | 15.00 | 45.00 | 500 |

## CFAR Detection

CA-CFAR detection is Numba-optimized and scales linearly with signal length.

| Signal Length | Mean (ms) | P99 (ms) | Iterations |
|---------------|-----------|----------|------------|
| 1,000 | 0.90 | 3.00 | 2,000 |
| 5,000 | 4.50 | 15.00 | 500 |
| 10,000 | 9.00 | 30.00 | 200 |

## Clustering Performance

### K-Means

| Points | Mean (ms) | P99 (ms) | Iterations |
|--------|-----------|----------|------------|
| 100 | 15.0 | 45.0 | 500 |
| 500 | 25.0 | 75.0 | 200 |
| 1,000 | 45.0 | 135.0 | 100 |

### DBSCAN

| Points | Mean (ms) | P99 (ms) | Iterations |
|--------|-----------|----------|------------|
| 100 | 5.0 | 15.0 | 500 |
| 500 | 12.0 | 36.0 | 200 |
| 1,000 | 40.0 | 120.0 | 100 |

### Distance Matrix (Numba-optimized)

| Configuration | Mean (ms) | P99 (ms) | Iterations |
|---------------|-----------|----------|------------|
| 100 × 3D | 3.0 | 9.0 | 1,000 |
| 500 × 3D | 60.0 | 150.0 | 100 |
| 1,000 × 3D | 240.0 | 600.0 | 50 |

## Special Functions

Hypergeometric functions for statistical distributions.

| Function | Mean (ms) | P99 (ms) | Iterations |
|----------|-----------|----------|------------|
| ₃F₂(z=1) | 0.010 | 0.050 | 50,000 |
| ₃F₂(z=10) | 0.015 | 0.075 | 50,000 |
| ₃F₂(z=100) | 0.045 | 0.150 | 20,000 |
| ₃F₂(z=1000) | 0.300 | 0.900 | 5,000 |
| ₄F₃(z=100) | 0.045 | 0.150 | 20,000 |
| ₁F₁ | 0.015 | 0.075 | 50,000 |
| ₂F₁ | 0.060 | 0.180 | 20,000 |
| Pochhammer (single) | 0.006 | 0.030 | 100,000 |
| Pochhammer (array) | 0.009 | 0.045 | 100,000 |

## Regression Thresholds

CI pipelines enforce these thresholds:

| Threshold | Value | Action |
|-----------|-------|--------|
| Warning | +25% | Warning in PR comment |
| Failure | +50% | CI fails, blocks merge |
| Min Samples | 5 | Minimum runs for regression detection |

## Running Benchmarks

### Light Suite (for PRs)

```bash
python -m pytest benchmarks/test_kalman_bench.py benchmarks/test_gating_bench.py benchmarks/test_rotations_bench.py -v
```

### Full Suite (for main branch)

```bash
python -m pytest benchmarks/ -v
```

### Track Performance History

```bash
python scripts/track_performance.py
```

### Detect Regressions

```bash
python scripts/detect_regressions.py --baseline HEAD~5
```

## Caching Impact

Geophysical modules use LRU caching for significant speedups:

| Operation | Uncached | Cached | Speedup |
|-----------|----------|--------|---------|
| great_circle_distance | 1.5 µs | 0.3 µs | 5× |
| inverse_geodetic | 8.2 µs | 0.4 µs | 20× |
| precession_matrix | 12.5 µs | 0.5 µs | 25× |
| associated_legendre (n=360) | 850 µs | 1.2 µs | 700× |
| magnetic_field_spherical | ~500 µs | ~0.8 µs | 600× |

See [module-interdependencies.md](module-interdependencies.md) for cache configuration details.

## SLO Configuration

SLOs are defined in `.benchmarks/slos.json`. The format is:

```json
{
  "slos": {
    "module.function": {
      "description": "Function description",
      "scenario_name": {
        "mean_ms": 0.123,
        "p99_ms": 0.456,
        "iterations": 10000
      }
    }
  },
  "regression_thresholds": {
    "warning_percent": 25,
    "failure_percent": 50
  }
}
```

## Notes

- SLOs include 3× headroom for CI runner variability
- P99 thresholds are typically 3× the mean target
- Numba-accelerated functions have JIT compilation overhead on first call
- Cache-based functions show best speedups after warm-up
