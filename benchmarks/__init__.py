"""
Benchmark suite for pyTCL performance testing.

This package contains performance benchmarks using pytest-benchmark.
Benchmarks are organized into:

- Light benchmarks (run on PRs): kalman, gating, rotations
- Full benchmarks (run on main): jpda, cfar, clustering

Usage
-----
Run all benchmarks:
    pytest benchmarks/ --benchmark-only

Run light benchmarks only:
    pytest benchmarks/test_kalman_bench.py benchmarks/test_gating_bench.py \\
           benchmarks/test_rotations_bench.py --benchmark-only

Skip benchmarks in regular test runs:
    pytest tests/ --benchmark-skip
"""
