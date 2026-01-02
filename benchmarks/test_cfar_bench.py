"""
Benchmarks for CFAR detection algorithms.

These are full benchmarks that run on main branch merges and nightly builds.
"""

import pytest

from pytcl.mathematical_functions.signal_processing.detection import cfar_ca


class TestCFARBenchmarks:
    """Benchmark CFAR detection algorithms."""

    @pytest.mark.full
    def test_cfar_ca_1000(self, benchmark, cfar_test_signals):
        """Benchmark CA-CFAR on 1000-point signal."""
        signal = cfar_test_signals[1000]
        guard_cells = 2
        training_cells = 8
        pfa = 1e-4

        # Warm up Numba JIT
        _ = cfar_ca(signal, guard_cells, training_cells, pfa)

        result = benchmark(cfar_ca, signal, guard_cells, training_cells, pfa)

        assert result is not None

    @pytest.mark.full
    def test_cfar_ca_5000(self, benchmark, cfar_test_signals):
        """Benchmark CA-CFAR on 5000-point signal."""
        signal = cfar_test_signals[5000]
        guard_cells = 4
        training_cells = 16
        pfa = 1e-4

        # Warm up Numba JIT
        _ = cfar_ca(signal, guard_cells, training_cells, pfa)

        result = benchmark(cfar_ca, signal, guard_cells, training_cells, pfa)

        assert result is not None

    @pytest.mark.full
    def test_cfar_ca_10000(self, benchmark, cfar_test_signals):
        """Benchmark CA-CFAR on 10000-point signal."""
        signal = cfar_test_signals[10000]
        guard_cells = 4
        training_cells = 16
        pfa = 1e-4

        # Warm up Numba JIT
        _ = cfar_ca(signal, guard_cells, training_cells, pfa)

        result = benchmark(cfar_ca, signal, guard_cells, training_cells, pfa)

        assert result is not None


class TestCFARParameterBenchmarks:
    """Benchmark CFAR with different parameters."""

    @pytest.mark.full
    @pytest.mark.parametrize("training_cells", [8, 16, 32])
    def test_cfar_training_cells(self, benchmark, cfar_test_signals, training_cells):
        """Benchmark CA-CFAR with varying training cell counts."""
        signal = cfar_test_signals[5000]
        guard_cells = 4
        pfa = 1e-4

        # Warm up
        _ = cfar_ca(signal, guard_cells, training_cells, pfa)

        result = benchmark(cfar_ca, signal, guard_cells, training_cells, pfa)

        assert result is not None
