"""Tests for GPU utility functions that don't require a GPU.

These tests verify the platform detection and basic utility functions
work correctly, regardless of GPU availability.
"""

import platform

import numpy as np
import pytest
from numpy.testing import assert_allclose


class TestPlatformDetection:
    """Tests for platform detection functions."""

    def test_is_apple_silicon(self):
        """Test Apple Silicon detection."""
        from pytcl.gpu.utils import is_apple_silicon

        result = is_apple_silicon()
        assert isinstance(result, bool)

        # Verify the result matches expected platform
        expected = platform.system() == "Darwin" and platform.machine() == "arm64"
        assert result == expected

    def test_get_backend_returns_valid_value(self):
        """Test that get_backend returns a valid backend type."""
        from pytcl.gpu.utils import get_backend

        backend = get_backend()
        assert backend in ("mlx", "cupy", "numpy")

    def test_is_gpu_available_returns_bool(self):
        """Test GPU availability check returns boolean."""
        from pytcl.gpu.utils import is_gpu_available

        result = is_gpu_available()
        assert isinstance(result, bool)

    def test_backend_consistency(self):
        """Test that backend detection is consistent with availability checks."""
        from pytcl.gpu.utils import (
            get_backend,
            is_cupy_available,
            is_gpu_available,
            is_mlx_available,
        )

        backend = get_backend()

        if backend == "mlx":
            assert is_mlx_available()
            assert is_gpu_available()
        elif backend == "cupy":
            assert is_cupy_available()
            assert is_gpu_available()
        else:  # numpy
            assert not is_gpu_available()


class TestArrayModuleCPU:
    """Tests for array module detection with CPU arrays."""

    def test_get_array_module_numpy(self):
        """Test array module detection for numpy arrays."""
        from pytcl.gpu.utils import get_array_module

        x = np.array([1, 2, 3])
        xp = get_array_module(x)
        assert xp is np

    def test_to_cpu_numpy_passthrough(self):
        """Test that to_cpu returns numpy arrays unchanged."""
        from pytcl.gpu.utils import to_cpu

        x = np.array([1.0, 2.0, 3.0])
        x_cpu = to_cpu(x)
        assert x_cpu is x  # Should be the same object

    def test_get_gpu_memory_info_no_gpu(self):
        """Test memory info when no GPU is available."""
        from pytcl.gpu.utils import get_backend, get_gpu_memory_info

        info = get_gpu_memory_info()
        assert "backend" in info

        if get_backend() == "numpy":
            assert info["backend"] == "numpy"
            assert info["free"] == 0
            assert info["total"] == 0


class TestMLXBackend:
    """Tests for MLX backend (if available)."""

    @pytest.fixture(autouse=True)
    def check_mlx(self):
        """Skip tests if MLX is not available."""
        from pytcl.gpu.utils import is_mlx_available

        if not is_mlx_available():
            pytest.skip("MLX not available")

    def test_to_gpu_mlx(self):
        """Test array transfer to MLX."""
        from pytcl.gpu.utils import to_gpu

        x = np.array([1.0, 2.0, 3.0])
        x_gpu = to_gpu(x, backend="mlx")

        import mlx.core as mx

        assert isinstance(x_gpu, mx.array)

    def test_to_cpu_mlx(self):
        """Test array transfer from MLX to CPU."""
        from pytcl.gpu.utils import to_cpu, to_gpu

        x = np.array([1.0, 2.0, 3.0])
        x_gpu = to_gpu(x, backend="mlx")
        x_back = to_cpu(x_gpu)

        assert isinstance(x_back, np.ndarray)
        assert_allclose(x_back, x)

    def test_get_array_module_mlx(self):
        """Test array module detection for MLX arrays."""
        import mlx.core as mx

        from pytcl.gpu.utils import get_array_module

        x = mx.array([1, 2, 3])
        xp = get_array_module(x)
        assert xp is mx

    def test_roundtrip_mlx(self):
        """Test CPU -> GPU -> CPU roundtrip with MLX."""
        from pytcl.gpu.utils import to_cpu, to_gpu

        # Test various array shapes
        arrays = [
            np.array([1.0, 2.0, 3.0]),
            np.random.randn(10, 4),
            np.random.randn(5, 4, 4),
        ]

        for x in arrays:
            x_gpu = to_gpu(x, backend="mlx")
            x_back = to_cpu(x_gpu)
            assert_allclose(x_back, x.astype(np.float32), rtol=1e-5)

    def test_sync_gpu_mlx(self):
        """Test GPU synchronization with MLX."""
        from pytcl.gpu.utils import sync_gpu, to_gpu

        x = np.random.randn(1000)
        _ = to_gpu(x, backend="mlx")

        # Should not raise
        sync_gpu()

    def test_memory_info_mlx(self):
        """Test memory info with MLX backend."""
        from pytcl.gpu.utils import get_backend, get_gpu_memory_info

        if get_backend() != "mlx":
            pytest.skip("MLX not the active backend")

        info = get_gpu_memory_info()
        assert info["backend"] == "mlx"
        assert "device" in info
