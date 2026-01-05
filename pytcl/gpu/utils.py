"""
GPU utility functions for array management and device detection.

This module provides utilities for:
- Checking GPU availability (CUDA via CuPy or Apple Silicon via MLX)
- Transferring arrays between CPU and GPU
- Getting the appropriate array module (numpy, cupy, or mlx)
- Memory management
- Automatic backend selection based on platform

The module automatically selects the appropriate backend:
- On Apple Silicon (M1/M2/M3): Uses MLX if available
- On systems with NVIDIA GPUs: Uses CuPy if available
- Falls back to CPU (numpy) if no GPU backend is available

Examples
--------
>>> from pytcl.gpu.utils import is_gpu_available, to_gpu, to_cpu
>>> if is_gpu_available():
...     x_gpu = to_gpu(x_numpy)
...     # ... perform GPU operations ...
...     x_cpu = to_cpu(x_gpu)
"""

import logging
import platform
from functools import lru_cache
from typing import Any, Literal, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pytcl.core.optional_deps import is_available

# Module logger
_logger = logging.getLogger("pytcl.gpu.utils")

# Type alias for arrays that could be numpy, cupy, or mlx
GPUArray = Any  # Would be cp.ndarray or mx.array if backend is available

# Backend type
BackendType = Literal["cupy", "mlx", "numpy"]


@lru_cache(maxsize=1)
def is_apple_silicon() -> bool:
    """
    Check if running on Apple Silicon (ARM64 Mac).

    Returns
    -------
    bool
        True if running on Apple Silicon (M1, M2, M3, etc.).

    Examples
    --------
    >>> from pytcl.gpu.utils import is_apple_silicon
    >>> if is_apple_silicon():
    ...     print("Running on Apple Silicon")
    """
    return platform.system() == "Darwin" and platform.machine() == "arm64"


@lru_cache(maxsize=1)
def is_mlx_available() -> bool:
    """
    Check if MLX acceleration is available (Apple Silicon).

    Returns True if:
    - Running on Apple Silicon (ARM64 Mac)
    - MLX is installed

    Returns
    -------
    bool
        True if MLX acceleration is available.

    Examples
    --------
    >>> from pytcl.gpu.utils import is_mlx_available
    >>> if is_mlx_available():
    ...     print("MLX acceleration enabled")
    """
    if not is_apple_silicon():
        _logger.debug("Not on Apple Silicon, MLX not applicable")
        return False

    if not is_available("mlx"):
        _logger.debug("MLX not installed")
        return False

    try:
        import mlx.core as mx

        # Verify MLX works by creating a simple array
        _ = mx.array([1.0, 2.0, 3.0])
        _logger.info("MLX available on Apple Silicon")
        return True
    except Exception as e:
        _logger.debug("MLX not functional: %s", e)
        return False


@lru_cache(maxsize=1)
def is_cupy_available() -> bool:
    """
    Check if CuPy (CUDA) acceleration is available.

    Returns True if:
    - CuPy is installed
    - A CUDA-capable GPU is detected
    - CUDA runtime is functional

    Returns
    -------
    bool
        True if CuPy acceleration is available.
    """
    if not is_available("cupy"):
        _logger.debug("CuPy not installed")
        return False

    try:
        import cupy as cp

        # Try to access a GPU device
        device = cp.cuda.Device(0)
        _ = device.compute_capability
        _logger.info("CuPy available: %s", device.pci_bus_id)
        return True
    except Exception as e:
        _logger.debug("CuPy/CUDA not available: %s", e)
        return False


@lru_cache(maxsize=1)
def get_backend() -> BackendType:
    """
    Get the best available GPU backend for the current platform.

    Priority:
    1. MLX on Apple Silicon
    2. CuPy on systems with NVIDIA GPUs
    3. numpy (CPU fallback)

    Returns
    -------
    str
        One of "mlx", "cupy", or "numpy".

    Examples
    --------
    >>> from pytcl.gpu.utils import get_backend
    >>> backend = get_backend()
    >>> print(f"Using {backend} backend")
    """
    if is_apple_silicon() and is_mlx_available():
        return "mlx"
    elif is_cupy_available():
        return "cupy"
    else:
        return "numpy"


@lru_cache(maxsize=1)
def is_gpu_available() -> bool:
    """
    Check if GPU acceleration is available.

    Returns True if either:
    - MLX is available (Apple Silicon)
    - CuPy is available with a CUDA GPU

    Returns
    -------
    bool
        True if GPU acceleration is available.

    Examples
    --------
    >>> from pytcl.gpu.utils import is_gpu_available
    >>> if is_gpu_available():
    ...     print("GPU acceleration enabled")
    ... else:
    ...     print("Falling back to CPU")

    Notes
    -----
    The result is cached after the first call for performance.
    Use `get_backend()` to determine which backend is being used.
    """
    return is_mlx_available() or is_cupy_available()


def get_array_module(arr: ArrayLike) -> Any:
    """
    Get the array module (numpy, cupy, or mlx.core) for the given array.

    This function enables writing code that works with numpy, cupy, and mlx
    arrays by returning the appropriate module.

    Parameters
    ----------
    arr : array_like
        Input array (numpy, cupy, or mlx).

    Returns
    -------
    module
        numpy, cupy, or mlx.core module, depending on the input array type.

    Examples
    --------
    >>> import numpy as np
    >>> from pytcl.gpu.utils import get_array_module
    >>> x = np.array([1, 2, 3])
    >>> xp = get_array_module(x)
    >>> xp is np
    True

    >>> # With CuPy array
    >>> import cupy as cp
    >>> x_gpu = cp.array([1, 2, 3])
    >>> xp = get_array_module(x_gpu)
    >>> xp is cp
    True

    >>> # With MLX array
    >>> import mlx.core as mx
    >>> x_mlx = mx.array([1, 2, 3])
    >>> xp = get_array_module(x_mlx)
    >>> xp.__name__
    'mlx.core'
    """
    # Check for MLX array first
    if is_available("mlx"):
        import mlx.core as mx

        if isinstance(arr, mx.array):
            return mx

    # Check for CuPy array
    if is_available("cupy"):
        import cupy as cp

        if isinstance(arr, cp.ndarray):
            return cp

    return np


def to_gpu(arr: ArrayLike, dtype: Any = None, backend: BackendType = None) -> GPUArray:
    """
    Transfer an array to GPU memory.

    Automatically selects the best available backend (MLX on Apple Silicon,
    CuPy on NVIDIA GPUs) unless a specific backend is requested.

    Parameters
    ----------
    arr : array_like
        Input array (typically numpy).
    dtype : dtype, optional
        Data type for the GPU array. If None, uses the input dtype.
    backend : str, optional
        Specific backend to use ("mlx", "cupy"). If None, auto-selects.

    Returns
    -------
    GPUArray
        Array in GPU memory (cupy.ndarray or mlx.array).

    Raises
    ------
    DependencyError
        If required backend is not installed.
    RuntimeError
        If no GPU is available.

    Examples
    --------
    >>> import numpy as np
    >>> from pytcl.gpu.utils import to_gpu, is_gpu_available
    >>> x = np.array([1.0, 2.0, 3.0])
    >>> if is_gpu_available():
    ...     x_gpu = to_gpu(x)
    ...     print(type(x_gpu).__name__)
    'ndarray'  # cupy.ndarray or 'array' for mlx

    Notes
    -----
    If the input is already a GPU array, it is returned as-is (or converted
    to the requested dtype).
    """
    from pytcl.core.optional_deps import import_optional

    if not is_gpu_available():
        raise RuntimeError(
            "No GPU available. Check CUDA installation or MLX availability."
        )

    # Determine backend
    if backend is None:
        backend = get_backend()

    # Use MLX backend
    if backend == "mlx":
        mx = import_optional(
            "mlx.core",
            package="mlx",
            extra="gpu-apple",
            feature="Apple Silicon GPU acceleration",
        )

        # If already an MLX array
        if isinstance(arr, mx.array):
            if dtype is not None:
                # MLX uses different dtype handling
                return arr.astype(_numpy_dtype_to_mlx(mx, dtype))
            return arr

        # Convert to numpy first if needed
        arr_np = np.asarray(arr)
        if dtype is not None:
            arr_np = arr_np.astype(dtype)

        return mx.array(arr_np)

    # Use CuPy backend
    else:
        cp = import_optional("cupy", extra="gpu", feature="GPU acceleration")

        # If already a CuPy array
        if isinstance(arr, cp.ndarray):
            if dtype is not None and arr.dtype != dtype:
                return arr.astype(dtype)
            return arr

        # Convert to numpy first if needed
        arr_np = np.asarray(arr)
        if dtype is not None:
            arr_np = arr_np.astype(dtype)

        return cp.asarray(arr_np)


def _numpy_dtype_to_mlx(mx, dtype) -> Any:
    """Convert numpy dtype to MLX dtype."""
    dtype_map = {
        np.float32: mx.float32,
        np.float64: mx.float32,  # MLX prefers float32
        np.int32: mx.int32,
        np.int64: mx.int64,
        np.bool_: mx.bool_,
    }
    if hasattr(dtype, "type"):
        dtype = dtype.type
    return dtype_map.get(dtype, mx.float32)


def to_cpu(arr: Union[ArrayLike, GPUArray]) -> NDArray[np.floating]:
    """
    Transfer an array from GPU to CPU memory.

    Parameters
    ----------
    arr : array_like, cupy.ndarray, or mlx.array
        Input array (numpy, cupy, or mlx).

    Returns
    -------
    numpy.ndarray
        Array in CPU memory.

    Examples
    --------
    >>> import numpy as np
    >>> from pytcl.gpu.utils import to_gpu, to_cpu, is_gpu_available
    >>> x = np.array([1.0, 2.0, 3.0])
    >>> if is_gpu_available():
    ...     x_gpu = to_gpu(x)
    ...     x_cpu = to_cpu(x_gpu)
    ...     np.allclose(x, x_cpu)
    True

    Notes
    -----
    If the input is already a numpy array, it is returned as-is.
    """
    # Already numpy
    if isinstance(arr, np.ndarray):
        return arr

    # Check if it's an MLX array
    if is_available("mlx"):
        import mlx.core as mx

        if isinstance(arr, mx.array):
            return np.array(arr)

    # Check if it's a CuPy array
    if is_available("cupy"):
        import cupy as cp

        if isinstance(arr, cp.ndarray):
            return cp.asnumpy(arr)

    # Fallback: convert via numpy
    return np.asarray(arr)


def ensure_gpu_array(
    arr: ArrayLike,
    dtype: Any = np.float64,
    backend: BackendType = None,
) -> GPUArray:
    """
    Ensure an array is on the GPU with the specified dtype.

    Parameters
    ----------
    arr : array_like
        Input array.
    dtype : dtype
        Desired data type.
    backend : str, optional
        Specific backend to use ("mlx", "cupy"). If None, auto-selects.

    Returns
    -------
    GPUArray
        Array on GPU with specified dtype (cupy.ndarray or mlx.array).
    """
    gpu_arr = to_gpu(arr, backend=backend)

    # MLX doesn't support float64 well, use float32
    if backend == "mlx" or (backend is None and get_backend() == "mlx"):
        if dtype == np.float64:
            dtype = np.float32

    if hasattr(gpu_arr, "dtype") and gpu_arr.dtype != dtype:
        if get_backend() == "mlx":
            import mlx.core as mx

            gpu_arr = gpu_arr.astype(_numpy_dtype_to_mlx(mx, dtype))
        else:
            gpu_arr = gpu_arr.astype(dtype)
    return gpu_arr


def sync_gpu() -> None:
    """
    Synchronize GPU operations.

    This blocks until all pending GPU operations are complete.
    Useful for accurate timing measurements.

    Examples
    --------
    >>> import time
    >>> from pytcl.gpu.utils import sync_gpu, is_gpu_available
    >>> if is_gpu_available():
    ...     # ... perform GPU operations ...
    ...     sync_gpu()  # Wait for completion
    ...     elapsed = time.time() - start
    """
    backend = get_backend()

    if backend == "mlx":
        import mlx.core as mx

        mx.eval()  # MLX uses lazy evaluation, eval() forces execution
    elif backend == "cupy":
        import cupy as cp

        cp.cuda.Stream.null.synchronize()


def get_gpu_memory_info() -> dict[str, int]:
    """
    Get GPU memory usage information.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'backend': Backend in use ("mlx", "cupy", or "numpy")
        - 'free': Free memory in bytes (if available)
        - 'total': Total memory in bytes (if available)
        - 'used': Used memory in bytes (if available)

    Examples
    --------
    >>> from pytcl.gpu.utils import get_gpu_memory_info, is_gpu_available
    >>> if is_gpu_available():
    ...     info = get_gpu_memory_info()
    ...     print(f"Backend: {info['backend']}")
    """
    backend = get_backend()

    if backend == "numpy":
        return {"backend": "numpy", "free": 0, "total": 0, "used": 0}

    if backend == "mlx":
        # MLX doesn't expose memory info directly, but we can get device info
        import mlx.core as mx

        device = mx.default_device()
        return {
            "backend": "mlx",
            "device": str(device),
            "free": -1,  # Not available
            "total": -1,  # Not available
            "used": -1,  # Not available
        }

    # CuPy backend
    import cupy as cp

    mempool = cp.get_default_memory_pool()
    free, total = cp.cuda.Device().mem_info

    return {
        "backend": "cupy",
        "free": free,
        "total": total,
        "used": total - free,
        "pool_used": mempool.used_bytes(),
        "pool_total": mempool.total_bytes(),
    }


def clear_gpu_memory() -> None:
    """
    Clear GPU memory pools.

    This frees cached memory blocks held by the GPU backend.
    Call this when you need to free GPU memory for other operations.

    Examples
    --------
    >>> from pytcl.gpu.utils import clear_gpu_memory, is_gpu_available
    >>> if is_gpu_available():
    ...     # ... perform GPU operations ...
    ...     clear_gpu_memory()  # Free cached memory
    """
    backend = get_backend()

    if backend == "mlx":
        import mlx.core as mx

        # MLX has automatic memory management, but we can force a sync
        mx.eval()
        # Note: MLX doesn't have explicit memory pool clearing like CuPy
    elif backend == "cupy":
        import cupy as cp

        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()


__all__ = [
    # Platform detection
    "is_apple_silicon",
    "is_mlx_available",
    "is_cupy_available",
    "get_backend",
    # Availability check
    "is_gpu_available",
    # Array operations
    "get_array_module",
    "to_gpu",
    "to_cpu",
    "ensure_gpu_array",
    # Synchronization and memory
    "sync_gpu",
    "get_gpu_memory_info",
    "clear_gpu_memory",
    # Type hints
    "GPUArray",
    "BackendType",
]
