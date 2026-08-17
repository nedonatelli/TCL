"""Cubature point-generation benchmarks: the Smolyak value proposition."""

import pytest

from pytcl.mathematical_functions.numerical_integration import smolyak_points


@pytest.mark.benchmark
@pytest.mark.light
@pytest.mark.parametrize("n", [4, 8])
def test_smolyak_generation(benchmark, n):
    pts, w = benchmark(smolyak_points, n, 2)
    assert abs(w.sum() - 1.0) < 1e-10


@pytest.mark.benchmark
@pytest.mark.light
@pytest.mark.parametrize("n", [4, 8])
def test_smolyak_point_count_vs_tensor(benchmark, n):
    # documents the count ratio in the benchmark record: sparse grid at
    # degree-5 exactness vs the 3^n tensor Gauss-Hermite grid
    def counts():
        pts, _ = smolyak_points(n, 2)
        return len(pts), 3**n

    sparse, tensor = benchmark(counts)
    assert sparse < tensor
