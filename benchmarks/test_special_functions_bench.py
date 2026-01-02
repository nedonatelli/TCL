"""
Benchmarks for special functions (hypergeometric, etc).

Focuses on Numba-optimized generalized hypergeometric function.
"""

import numpy as np
import pytest


@pytest.fixture(scope="session")
def hypergeometric_test_data():
    """
    Pre-computed test data for hypergeometric benchmarks.

    Returns dict with parameter arrays and z values.
    """
    np.random.seed(42)
    return {
        # 3F2 parameters (general case using Numba)
        "a_3f2": np.array([1.0, 2.0, 3.0]),
        "b_3f2": np.array([4.0, 5.0]),
        # 4F3 parameters
        "a_4f3": np.array([1.0, 1.5, 2.0, 2.5]),
        "b_4f3": np.array([3.0, 3.5, 4.0]),
        # z values for batch processing
        "z_single": np.array([0.5]),
        "z_small": np.linspace(0.1, 0.9, 10),
        "z_medium": np.linspace(0.1, 0.9, 100),
        "z_large": np.linspace(0.1, 0.9, 1000),
    }


# =============================================================================
# Generalized Hypergeometric Benchmarks (Numba-optimized)
# =============================================================================


@pytest.mark.benchmark
@pytest.mark.light
@pytest.mark.parametrize("n_z", [1, 10, 100])
def test_generalized_hypergeometric_3f2(benchmark, hypergeometric_test_data, n_z):
    """Benchmark 3F2 generalized hypergeometric (Numba path)."""
    from pytcl.mathematical_functions.special_functions.hypergeometric import (
        generalized_hypergeometric,
    )

    a = hypergeometric_test_data["a_3f2"]
    b = hypergeometric_test_data["b_3f2"]

    if n_z == 1:
        z = hypergeometric_test_data["z_single"]
    elif n_z == 10:
        z = hypergeometric_test_data["z_small"]
    else:
        z = hypergeometric_test_data["z_medium"]

    # Warm up JIT compilation
    _ = generalized_hypergeometric(a, b, z)

    result = benchmark(generalized_hypergeometric, a, b, z)

    # Verify result shape and validity
    assert np.all(np.isfinite(result))


@pytest.mark.benchmark
@pytest.mark.full
@pytest.mark.parametrize("n_z", [1000])
def test_generalized_hypergeometric_3f2_large(benchmark, hypergeometric_test_data, n_z):
    """Benchmark 3F2 with large z array."""
    from pytcl.mathematical_functions.special_functions.hypergeometric import (
        generalized_hypergeometric,
    )

    a = hypergeometric_test_data["a_3f2"]
    b = hypergeometric_test_data["b_3f2"]
    z = hypergeometric_test_data["z_large"]

    # Warm up
    _ = generalized_hypergeometric(a, b, z[:10])

    result = benchmark(generalized_hypergeometric, a, b, z)
    assert np.all(np.isfinite(result))


@pytest.mark.benchmark
@pytest.mark.full
def test_generalized_hypergeometric_4f3(benchmark, hypergeometric_test_data):
    """Benchmark 4F3 generalized hypergeometric (higher order)."""
    from pytcl.mathematical_functions.special_functions.hypergeometric import (
        generalized_hypergeometric,
    )

    a = hypergeometric_test_data["a_4f3"]
    b = hypergeometric_test_data["b_4f3"]
    z = hypergeometric_test_data["z_medium"]

    # Warm up
    _ = generalized_hypergeometric(a, b, z[:10])

    result = benchmark(generalized_hypergeometric, a, b, z)
    assert np.all(np.isfinite(result))


# =============================================================================
# Scipy-routed benchmarks (for comparison)
# =============================================================================


@pytest.mark.benchmark
@pytest.mark.light
def test_hyp1f1_scipy_route(benchmark, hypergeometric_test_data):
    """Benchmark 1F1 via scipy route (baseline)."""
    from pytcl.mathematical_functions.special_functions.hypergeometric import (
        generalized_hypergeometric,
    )

    # These parameters route to scipy's hyp1f1
    a = np.array([1.0])
    b = np.array([2.0])
    z = hypergeometric_test_data["z_medium"]

    result = benchmark(generalized_hypergeometric, a, b, z)
    assert np.all(np.isfinite(result))


@pytest.mark.benchmark
@pytest.mark.light
def test_hyp2f1_scipy_route(benchmark, hypergeometric_test_data):
    """Benchmark 2F1 via scipy route (baseline)."""
    from pytcl.mathematical_functions.special_functions.hypergeometric import (
        generalized_hypergeometric,
    )

    # These parameters route to scipy's hyp2f1
    a = np.array([0.5, 0.5])
    b = np.array([1.5])
    z = hypergeometric_test_data["z_medium"] * 0.9  # Keep |z| < 1

    result = benchmark(generalized_hypergeometric, a, b, z)
    assert np.all(np.isfinite(result))


# =============================================================================
# Pochhammer symbol benchmarks
# =============================================================================


@pytest.mark.benchmark
@pytest.mark.light
def test_pochhammer_single(benchmark):
    """Benchmark Pochhammer symbol computation."""
    from pytcl.mathematical_functions.special_functions.hypergeometric import (
        pochhammer,
    )

    result = benchmark(pochhammer, 3.0, 5.0)
    assert np.isfinite(result)


@pytest.mark.benchmark
@pytest.mark.full
def test_pochhammer_array(benchmark):
    """Benchmark Pochhammer with array inputs."""
    from pytcl.mathematical_functions.special_functions.hypergeometric import (
        pochhammer,
    )

    a = np.linspace(1.0, 10.0, 100)
    n = np.ones(100) * 5.0

    result = benchmark(pochhammer, a, n)
    assert np.all(np.isfinite(result))
