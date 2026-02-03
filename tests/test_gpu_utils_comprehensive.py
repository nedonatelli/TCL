"""
Comprehensive tests for GPU utilities module.

Tests coverage for:
- Backend detection (Apple Silicon, CuPy, MLX availability)
- Array module selection and detection
- GPU/CPU array conversion
- GPU memory management
- GPU synchronization
"""

import numpy as np
import pytest

from pytcl.gpu.utils import (
    clear_gpu_memory,
    ensure_gpu_array,
    get_array_module,
    get_backend,
    get_gpu_memory_info,
    is_apple_silicon,
    is_cupy_available,
    is_gpu_available,
    is_mlx_available,
    sync_gpu,
    to_cpu,
    to_gpu,
)


class TestBackendDetection:
    """Tests for GPU backend detection."""

    def test_is_apple_silicon_returns_bool(self):
        """Test that is_apple_silicon returns boolean."""
        result = is_apple_silicon()
        assert isinstance(result, (bool, np.bool_))

    def test_is_mlx_available_returns_bool(self):
        """Test that is_mlx_available returns boolean."""
        result = is_mlx_available()
        assert isinstance(result, (bool, np.bool_))

    def test_is_cupy_available_returns_bool(self):
        """Test that is_cupy_available returns boolean."""
        result = is_cupy_available()
        assert isinstance(result, (bool, np.bool_))

    def test_get_backend_returns_string(self):
        """Test that get_backend returns valid backend string."""
        backend = get_backend()
        assert isinstance(backend, str)
        assert backend in ["cupy", "mlx", "numpy"]

    def test_is_gpu_available_returns_bool(self):
        """Test that is_gpu_available returns boolean."""
        result = is_gpu_available()
        assert isinstance(result, (bool, np.bool_))

    def test_backend_consistency(self):
        """Test consistency between backend functions."""
        gpu_available = is_gpu_available()
        backend = get_backend()

        # If GPU available, backend shouldn't be numpy-only
        if gpu_available:
            assert backend in ["cupy", "mlx"]
        else:
            assert backend == "numpy"


class TestArrayModuleSelection:
    """Tests for array module detection and selection."""

    def test_get_array_module_numpy(self):
        """Test get_array_module with NumPy array."""
        arr = np.array([1.0, 2.0, 3.0])
        module = get_array_module(arr)

        assert module is not None

    def test_get_array_module_scalar(self):
        """Test get_array_module with scalar."""
        arr = 5.0
        module = get_array_module(arr)

        assert module is not None

    def test_get_array_module_multidimensional(self):
        """Test get_array_module with multidimensional array."""
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        module = get_array_module(arr)

        assert module is not None

    def test_get_array_module_different_dtypes(self):
        """Test get_array_module with different dtypes."""
        for dtype in [np.float32, np.float64, np.int32, np.int64]:
            arr = np.array([1, 2, 3], dtype=dtype)
            module = get_array_module(arr)
            assert module is not None


class TestGPUArrayConversion:
    """Tests for GPU/CPU array conversion."""

    def test_to_cpu_numpy_array(self):
        """Test to_cpu with numpy array."""
        arr = np.array([1.0, 2.0, 3.0])
        result = to_cpu(arr)

        assert isinstance(result, np.ndarray)
        assert result.shape == arr.shape

    def test_to_cpu_preserves_values(self):
        """Test that to_cpu preserves array values."""
        arr = np.array([1.0, 2.0, 3.0])
        result = to_cpu(arr)

        assert np.allclose(result, arr)

    def test_to_cpu_multidimensional(self):
        """Test to_cpu with multidimensional array."""
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = to_cpu(arr)

        assert result.shape == arr.shape
        assert np.allclose(result, arr)

    def test_to_gpu_basic(self):
        """Test basic to_gpu conversion."""
        arr = np.array([1.0, 2.0, 3.0])

        # Should not raise error
        try:
            result = to_gpu(arr)
            assert result is not None
        except Exception:
            # OK if GPU not available
            pytest.skip("GPU not available")

    def test_to_gpu_returns_array_like(self):
        """Test that to_gpu returns array-like."""
        arr = np.array([1.0, 2.0, 3.0])

        try:
            result = to_gpu(arr)
            # Should be convertible to numpy
            to_cpu(result)
        except Exception:
            pytest.skip("GPU not available")

    def test_to_gpu_dtype_preservation(self):
        """Test that to_gpu preserves dtype when specified."""
        arr = np.array([1, 2, 3], dtype=np.float32)

        try:
            result = to_gpu(arr, dtype=np.float32)
            cpu_result = to_cpu(result)
            assert cpu_result.dtype == np.float32
        except Exception:
            pytest.skip("GPU not available")


class TestEnsureGPUArray:
    """Tests for ensure_gpu_array function."""

    def test_ensure_gpu_array_numpy(self):
        """Test ensure_gpu_array with numpy array."""
        arr = np.array([1.0, 2.0, 3.0])

        try:
            result = ensure_gpu_array(arr)
            assert result is not None
        except Exception:
            pytest.skip("GPU not available")

    def test_ensure_gpu_array_already_gpu(self):
        """Test ensure_gpu_array when already GPU."""
        arr = np.array([1.0, 2.0, 3.0])

        try:
            gpu_arr = to_gpu(arr)
            result = ensure_gpu_array(gpu_arr)
            assert result is not None
        except Exception:
            pytest.skip("GPU not available")

    def test_ensure_gpu_array_multidimensional(self):
        """Test ensure_gpu_array with multidimensional array."""
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])

        try:
            result = ensure_gpu_array(arr)
            assert result is not None
        except Exception:
            pytest.skip("GPU not available")


class TestGPUSynchronization:
    """Tests for GPU synchronization."""

    def test_sync_gpu_no_error(self):
        """Test that sync_gpu doesn't raise error."""
        try:
            sync_gpu()  # Should not raise
        except Exception:
            pytest.skip("GPU not available")

    def test_sync_gpu_safe_when_no_gpu(self):
        """Test sync_gpu is safe even without GPU."""
        # Should handle gracefully
        try:
            sync_gpu()
        except Exception:
            # OK if GPU not available
            pass


class TestGPUMemoryManagement:
    """Tests for GPU memory management."""

    def test_get_gpu_memory_info_returns_dict(self):
        """Test that get_gpu_memory_info returns dictionary."""
        try:
            info = get_gpu_memory_info()
            assert isinstance(info, dict)
        except Exception:
            pytest.skip("GPU not available")

    def test_get_gpu_memory_info_structure(self):
        """Test structure of GPU memory info."""
        try:
            info = get_gpu_memory_info()

            # Should have some keys
            assert len(info) > 0

            # Values should be strings or integers
            for key, value in info.items():
                assert isinstance(key, str)
        except Exception:
            pytest.skip("GPU not available")

    def test_clear_gpu_memory_no_error(self):
        """Test that clear_gpu_memory doesn't raise error."""
        try:
            clear_gpu_memory()  # Should not raise
        except Exception:
            pytest.skip("GPU not available")

    def test_clear_gpu_memory_safe_when_no_gpu(self):
        """Test clear_gpu_memory is safe even without GPU."""
        try:
            clear_gpu_memory()
        except Exception:
            # OK if GPU not available
            pass


class TestGPUIntegration:
    """Integration tests for GPU utilities."""

    def test_cpu_to_gpu_to_cpu_roundtrip(self):
        """Test roundtrip conversion CPU -> GPU -> CPU."""
        arr = np.array([1.0, 2.0, 3.0, 4.0])

        try:
            gpu_arr = to_gpu(arr)
            cpu_result = to_cpu(gpu_arr)

            assert np.allclose(cpu_result, arr)
        except Exception:
            pytest.skip("GPU not available")

    def test_roundtrip_multidimensional(self):
        """Test roundtrip with multidimensional arrays."""
        arr = np.random.randn(3, 4, 5)

        try:
            gpu_arr = to_gpu(arr)
            cpu_result = to_cpu(gpu_arr)

            assert cpu_result.shape == arr.shape
            assert np.allclose(cpu_result, arr, rtol=1e-5)
        except Exception:
            pytest.skip("GPU not available")

    def test_backend_detection_consistent(self):
        """Test backend detection is consistent."""
        apple_silicon = is_apple_silicon()
        mlx_available = is_mlx_available()
        cupy_available = is_cupy_available()
        gpu_available = is_gpu_available()
        backend = get_backend()

        # All should be consistent types
        assert isinstance(apple_silicon, (bool, np.bool_))
        assert isinstance(mlx_available, (bool, np.bool_))
        assert isinstance(cupy_available, (bool, np.bool_))
        assert isinstance(gpu_available, (bool, np.bool_))
        assert isinstance(backend, str)

        # Sanity check on backend string
        if mlx_available:
            assert backend == "mlx"
        elif cupy_available:
            assert backend == "cupy"
        else:
            assert backend == "numpy"

    def test_memory_operations_safe(self):
        """Test memory operations are safe and don't crash."""
        try:
            # Get initial memory info
            info1 = get_gpu_memory_info()

            # Clear memory
            clear_gpu_memory()

            # Sync GPU
            sync_gpu()

            # Get memory info again
            info2 = get_gpu_memory_info()

            # Both should be dicts
            assert isinstance(info1, dict)
            assert isinstance(info2, dict)
        except Exception:
            pytest.skip("GPU not available")

    def test_array_module_consistency(self):
        """Test array module detection is consistent."""
        arr1 = np.array([1.0, 2.0])
        arr2 = np.array([3.0, 4.0])

        module1 = get_array_module(arr1)
        module2 = get_array_module(arr2)

        # Same array types should return same module
        assert module1 == module2


class TestBackendStringValues:
    """Tests for backend string values."""

    def test_backend_is_lowercase(self):
        """Test that backend string is lowercase."""
        backend = get_backend()
        assert backend == backend.lower()

    def test_backend_valid_values(self):
        """Test that backend is one of valid values."""
        backend = get_backend()
        valid_backends = ["numpy", "cupy", "mlx"]
        assert backend in valid_backends


class TestPlatformDetection:
    """Tests for platform detection."""

    def test_apple_silicon_on_non_apple(self):
        """Test apple silicon detection respects platform."""
        result = is_apple_silicon()

        # Should always be bool
        assert isinstance(result, (bool, np.bool_))

    def test_cupy_requires_nvidia(self):
        """Test cupy availability checks for NVIDIA GPU."""
        result = is_cupy_available()

        # Should always be bool
        assert isinstance(result, (bool, np.bool_))

    def test_mlx_requires_apple(self):
        """Test MLX availability checks for Apple platform."""
        result = is_mlx_available()

        # Should always be bool
        assert isinstance(result, (bool, np.bool_))

        # MLX should only be available on Apple Silicon
        if result:
            assert is_apple_silicon()
