"""
GPU-accelerated algorithms for the Tracker Component Library.

This module provides GPU-accelerated implementations of key tracking algorithms
using CuPy (NVIDIA GPUs) or MLX (Apple Silicon). The batch implementations are
written once against a backend-dispatch layer (:mod:`pytcl.gpu._backend`) and
run unmodified on either backend.

Measured on Apple Silicon (MLX), batch linear Kalman predict+update against
a per-track CPU loop, end-to-end including host-device transfers and result
materialization, after warm-up: 1.6x at 100 tracks, 13x at 1,000, 40x at
20,000 (August 2026).

The module automatically selects the best available backend:
- On systems with NVIDIA GPUs: Uses CuPy if installed (float64)
- On Apple Silicon (M1/M2/M3): Uses MLX if installed (float32)
- Raises ``DependencyError`` if neither backend is installed: there is no
  NumPy compute fallback (``pytcl.gpu.utils.get_backend`` reports the string
  ``"numpy"`` for detection purposes only -- nothing computes on it)

Precision
---------
The MLX backend computes in **float32**: MLX raises on float64 GPU operations.
Batch results therefore agree with the reference CPU implementations to roughly
float32 precision (measured ~1e-7 relative for the linear and extended Kalman
filters) rather than to machine epsilon. Two consequences worth knowing:

- The **unscented** filter is sensitive to this. Merwe weights scale as
  1/alpha^2, so the common default ``alpha=1e-3`` produces weights of order
  1e6 and, in float32, a result with no significant digits (measured relative
  error ~1e2). ``batch_ukf_*`` emits a ``RuntimeWarning`` below
  ``alpha=1e-2``; use ``alpha >= 0.1`` on MLX, or a float64 backend.
- MLX linear-algebra kernels (``inv``, ``cholesky``, ``solve``, ``eigh``) run
  on the CPU stream, which the dispatch layer handles transparently. Unified
  memory makes this a scheduling change, not a copy.

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
- MLX >= 0.16.0

Installation
------------
For NVIDIA CUDA:
    pip install nrl-tracker[gpu]
    # or directly:
    pip install cupy-cuda12x  # For CUDA 12.x

For Apple Silicon:
    pip install nrl-tracker[gpu-apple]
    # or directly:
    pip install mlx

Examples
--------
Basic usage with automatic backend selection:

>>> from pytcl.gpu import is_gpu_available, get_backend
>>> isinstance(is_gpu_available(), bool)
True
>>> get_backend() in ("mlx", "cupy", "numpy")
True

Check platform:

>>> from pytcl.gpu import is_apple_silicon, is_mlx_available
>>> isinstance(is_apple_silicon(), bool)
True
>>> isinstance(is_mlx_available(), bool)
True

Batch processing example:

>>> import numpy as np
>>> from pytcl.gpu import batch_kf_predict, to_gpu, to_cpu
>>> x_batch = np.zeros((4, 2))                  # (n_tracks, state_dim)
>>> P_batch = np.stack([np.eye(2)] * 4)         # (n_tracks, state_dim, state_dim)
>>> F = np.array([[1.0, 1.0], [0.0, 1.0]])
>>> Q = np.eye(2) * 0.01
>>> # Move data to the GPU (automatically uses the best backend)
>>> x_gpu, P_gpu = to_gpu(x_batch), to_gpu(P_batch)
>>> x_pred, P_pred = batch_kf_predict(x_gpu, P_gpu, F, Q)
>>> # Move results back to the CPU
>>> to_cpu(x_pred).shape
(4, 2)

See Also
--------
pytcl.dynamic_estimation.kalman : CPU Kalman filter implementations
pytcl.dynamic_estimation.particle_filters : CPU particle filter implementations
"""

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

# __all__ now lists everything importable from this package, including the
# names resolved lazily below. It previously listed only the ten eager ones,
# so `batch_ekf_predict` imported fine while being absent from __all__ and
# from dir() -- which is exactly what made an earlier survey of this package
# report reachable names as missing.
__all__ = [
    # Platform detection
    "is_apple_silicon",
    "is_mlx_available",
    "is_cupy_available",
    "get_backend",
    # Availability check
    "is_gpu_available",
    # Array transfer and inspection
    "get_array_module",
    "to_gpu",
    "to_cpu",
    "ensure_gpu_array",
    # Device memory
    "clear_gpu_memory",
    "get_gpu_memory_info",
    "get_memory_pool",
    "sync_gpu",
    "MemoryPool",
    # Batch filters
    "CuPyKalmanFilter",
    "batch_kf_predict",
    "batch_kf_update",
    "batch_kf_predict_update",
    "CuPyExtendedKalmanFilter",
    "batch_ekf_predict",
    "batch_ekf_update",
    "CuPyUnscentedKalmanFilter",
    "batch_ukf_predict",
    "batch_ukf_update",
    # Particle filter
    "CuPyParticleFilter",
    "batch_particle_filter_update",
    "gpu_effective_sample_size",
    "gpu_normalize_weights",
    "gpu_resample_multinomial",
    "gpu_resample_stratified",
    "gpu_resample_systematic",
    # Linear algebra on the device
    "gpu_cholesky",
    "gpu_cholesky_safe",
    "gpu_eigh",
    "gpu_inv",
    "gpu_matrix_sqrt",
    "gpu_qr",
    "gpu_solve",
]


# Lazy imports for GPU implementations (only loaded if CuPy is available)
def __getattr__(name: str) -> object:
    """Lazy import GPU implementations."""
    if name in (
        "CuPyKalmanFilter",
        "batch_kf_predict",
        "batch_kf_update",
        "batch_kf_predict_update",
    ):
        from pytcl.gpu.kalman import (
            CuPyKalmanFilter,
            batch_kf_predict,
            batch_kf_predict_update,
            batch_kf_update,
        )

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
        "batch_particle_filter_update",
        "gpu_effective_sample_size",
        "gpu_normalize_weights",
        "gpu_resample_multinomial",
        "gpu_resample_stratified",
        "gpu_resample_systematic",
    ):
        from pytcl.gpu.particle_filter import (
            CuPyParticleFilter,
            batch_particle_filter_update,
            gpu_effective_sample_size,
            gpu_normalize_weights,
            gpu_resample_multinomial,
            gpu_resample_stratified,
            gpu_resample_systematic,
        )

        globals()[name] = locals()[name]
        return locals()[name]

    if name in (
        "MemoryPool",
        "get_memory_pool",
        "gpu_cholesky",
        "gpu_cholesky_safe",
        "gpu_eigh",
        "gpu_inv",
        "gpu_matrix_sqrt",
        "gpu_qr",
        "gpu_solve",
    ):
        from pytcl.gpu.matrix_utils import (
            MemoryPool,
            get_memory_pool,
            gpu_cholesky,
            gpu_cholesky_safe,
            gpu_eigh,
            gpu_inv,
            gpu_matrix_sqrt,
            gpu_qr,
            gpu_solve,
        )

        globals()[name] = locals()[name]
        return locals()[name]

    raise AttributeError(f"module 'pytcl.gpu' has no attribute '{name}'")
