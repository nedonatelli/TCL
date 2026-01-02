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


# =============================================================================
# Debye function benchmarks (Numba-optimized)
# =============================================================================


@pytest.fixture(scope="session")
def debye_test_data():
    """Pre-computed test data for Debye function benchmarks."""
    np.random.seed(42)
    return {
        "x_small": np.array([0.5, 1.0, 2.0]),
        "x_medium": np.linspace(0.1, 50, 100),
        "x_large": np.linspace(0.1, 50, 1000),
    }


@pytest.mark.benchmark
@pytest.mark.light
def test_debye_3_small(benchmark, debye_test_data):
    """Benchmark D_3(x) for small arrays (JIT-compiled)."""
    from pytcl.mathematical_functions.special_functions.debye import debye_3

    x = debye_test_data["x_small"]

    # Warm up JIT
    _ = debye_3(x)

    result = benchmark(debye_3, x)
    assert np.all(np.isfinite(result))
    assert np.all(result > 0)


@pytest.mark.benchmark
@pytest.mark.light
@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_debye_orders(benchmark, debye_test_data, order):
    """Benchmark Debye functions of different orders."""
    from pytcl.mathematical_functions.special_functions.debye import debye

    x = debye_test_data["x_medium"]

    # Warm up JIT
    _ = debye(order, x[:10])

    result = benchmark(debye, order, x)
    assert np.all(np.isfinite(result))


@pytest.mark.benchmark
@pytest.mark.full
def test_debye_3_large_batch(benchmark, debye_test_data):
    """Benchmark D_3(x) for large arrays (parallel execution)."""
    from pytcl.mathematical_functions.special_functions.debye import debye_3

    x = debye_test_data["x_large"]

    # Warm up JIT
    _ = debye_3(x[:10])

    result = benchmark(debye_3, x)
    assert np.all(np.isfinite(result))
    assert len(result) == 1000


@pytest.mark.benchmark
@pytest.mark.full
def test_debye_heat_capacity(benchmark, debye_test_data):
    """Benchmark Debye heat capacity calculation."""
    from pytcl.mathematical_functions.special_functions.debye import (
        debye_heat_capacity,
    )

    temperatures = np.linspace(10, 1000, 100)
    debye_temp = 428.0  # Aluminum

    # Warm up
    _ = debye_heat_capacity(temperatures[:10], debye_temp)

    result = benchmark(debye_heat_capacity, temperatures, debye_temp)
    assert np.all(np.isfinite(result))
    assert np.all(result > 0)
    assert np.all(result <= 1.0)


# =============================================================================
# Marcum Q function benchmarks
# =============================================================================


@pytest.fixture(scope="session")
def marcum_test_data():
    """Pre-computed test data for Marcum Q benchmarks."""
    np.random.seed(42)
    return {
        "a_small": np.array([1.0, 2.0, 3.0]),
        "b_small": np.array([2.0, 3.0, 4.0]),
        "a_medium": np.linspace(0.5, 10, 50),
        "b_medium": np.linspace(1.0, 15, 50),
        "snr_array": np.logspace(-1, 2, 100),  # 0.1 to 100 SNR
    }


@pytest.mark.benchmark
@pytest.mark.light
def test_marcum_q_single(benchmark):
    """Benchmark single Marcum Q computation."""
    from pytcl.mathematical_functions.special_functions.marcum_q import marcum_q

    result = benchmark(marcum_q, 3.0, 4.0)
    assert np.isfinite(result)
    assert 0 <= result <= 1


@pytest.mark.benchmark
@pytest.mark.light
def test_marcum_q_batch(benchmark, marcum_test_data):
    """Benchmark batch Marcum Q computation."""
    from pytcl.mathematical_functions.special_functions.marcum_q import marcum_q

    a = marcum_test_data["a_medium"]
    b = marcum_test_data["b_medium"]

    result = benchmark(marcum_q, a, b)
    assert np.all(np.isfinite(result))
    assert np.all((result >= 0) & (result <= 1))


@pytest.mark.benchmark
@pytest.mark.light
@pytest.mark.parametrize("m", [1, 2, 4])
def test_marcum_q_orders(benchmark, marcum_test_data, m):
    """Benchmark Marcum Q for different orders."""
    from pytcl.mathematical_functions.special_functions.marcum_q import marcum_q

    a = marcum_test_data["a_small"]
    b = marcum_test_data["b_small"]

    result = benchmark(marcum_q, a, b, m)
    assert np.all(np.isfinite(result))


@pytest.mark.benchmark
@pytest.mark.full
def test_swerling_detection(benchmark, marcum_test_data):
    """Benchmark Swerling detection probability calculation."""
    from pytcl.mathematical_functions.special_functions.marcum_q import (
        swerling_detection_probability,
    )

    snr = marcum_test_data["snr_array"]
    pfa = 1e-6

    result = benchmark(swerling_detection_probability, snr, pfa)
    assert np.all(np.isfinite(result))
    assert np.all((result >= 0) & (result <= 1))


@pytest.mark.benchmark
@pytest.mark.full
def test_marcum_q_inverse(benchmark, marcum_test_data):
    """Benchmark inverse Marcum Q computation."""
    from pytcl.mathematical_functions.special_functions.marcum_q import marcum_q_inv

    a = marcum_test_data["a_small"]
    q = np.array([0.9, 0.5, 0.1])

    result = benchmark(marcum_q_inv, a, q)
    assert np.all(np.isfinite(result))
    assert np.all(result >= 0)
