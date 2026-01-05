"""
GPU-accelerated algorithms for the Tracker Component Library.

This module provides GPU-accelerated implementations of key tracking algorithms
using CuPy (NVIDIA GPUs) or MLX (Apple Silicon). These implementations offer
significant speedups (5-15x) for batch processing of multiple tracks or large
particle sets.

The module automatically selects the best available backend:
- On Apple Silicon (M1/M2/M3): Uses MLX if installed
- On systems with NVIDIA GPUs: Uses CuPy if installed
- Falls back to CPU (numpy) if no GPU backend is available

The GPU implementations mirror the CPU API but accept GPU arrays and return
GPU arrays. Use the utility functions to seamlessly transfer data between
CPU and GPU.

Requirements
------------
For NVIDIA GPUs:
- CUDA-capable GPU
- CuPy >= 12.0

For Apple Silicon:
- macOS with Apple Silicon (M1, M2, M3, etc.)
- MLX >= 0.5.0

Installation
------------
For NVIDIA CUDA:
    pip install pytcl[gpu]
    # or directly:
    pip install cupy-cuda12x  # For CUDA 12.x

For Apple Silicon:
    pip install pytcl[gpu-apple]
    # or directly:
    pip install mlx

Examples
--------
Basic usage with automatic backend selection:

>>> from pytcl.gpu import is_gpu_available, get_backend
>>> if is_gpu_available():
...     print(f"GPU available, using {get_backend()} backend")

Check platform:

>>> from pytcl.gpu import is_apple_silicon, is_mlx_available
>>> if is_apple_silicon():
...     print("Running on Apple Silicon")
>>> if is_mlx_available():
...     print("MLX acceleration available")

Batch processing example:

>>> from pytcl.gpu import batch_kf_predict, to_gpu, to_cpu
>>> # Move data to GPU (automatically uses best backend)
>>> x_gpu = to_gpu(x_batch)  # (n_tracks, state_dim)
>>> P_gpu = to_gpu(P_batch)  # (n_tracks, state_dim, state_dim)
>>> # Batch prediction
>>> x_pred, P_pred = batch_kf_predict(x_gpu, P_gpu, F, Q)
>>> # Move results back to CPU
>>> x_pred_cpu = to_cpu(x_pred)

See Also
--------
pytcl.dynamic_estimation.kalman : CPU Kalman filter implementations
pytcl.dynamic_estimation.particle_filters : CPU particle filter implementations
"""

from pytcl.gpu.utils import (
    get_array_module,
    get_backend,
    is_apple_silicon,
    is_cupy_available,
    is_gpu_available,
    is_mlx_available,
    to_cpu,
    to_gpu,
)

__all__ = [
    # Platform detection
    "is_apple_silicon",
    "is_mlx_available",
    "is_cupy_available",
    "get_backend",
    # Availability check
    "is_gpu_available",
    # Utility functions
    "get_array_module",
    "to_gpu",
    "to_cpu",
]


# Lazy imports for GPU implementations (only loaded if CuPy is available)
def __getattr__(name: str):
    """Lazy import GPU implementations."""
    if name in ("CuPyKalmanFilter", "batch_kf_predict", "batch_kf_update"):
        from pytcl.gpu.kalman import CuPyKalmanFilter, batch_kf_predict, batch_kf_update

        globals()[name] = locals()[name]
        return locals()[name]

    if name in ("CuPyExtendedKalmanFilter", "batch_ekf_predict", "batch_ekf_update"):
        from pytcl.gpu.ekf import (
            CuPyExtendedKalmanFilter,
            batch_ekf_predict,
            batch_ekf_update,
        )

        globals()[name] = locals()[name]
        return locals()[name]

    if name in ("CuPyUnscentedKalmanFilter", "batch_ukf_predict", "batch_ukf_update"):
        from pytcl.gpu.ukf import (
            CuPyUnscentedKalmanFilter,
            batch_ukf_predict,
            batch_ukf_update,
        )

        globals()[name] = locals()[name]
        return locals()[name]

    if name in (
        "CuPyParticleFilter",
        "gpu_resample_systematic",
        "gpu_resample_multinomial",
    ):
        from pytcl.gpu.particle_filter import (
            CuPyParticleFilter,
            gpu_resample_multinomial,
            gpu_resample_systematic,
        )

        globals()[name] = locals()[name]
        return locals()[name]

    if name in ("gpu_cholesky", "gpu_qr", "gpu_solve", "MemoryPool"):
        from pytcl.gpu.matrix_utils import (
            MemoryPool,
            gpu_cholesky,
            gpu_qr,
            gpu_solve,
        )

        globals()[name] = locals()[name]
        return locals()[name]

    raise AttributeError(f"module 'pytcl.gpu' has no attribute '{name}'")
