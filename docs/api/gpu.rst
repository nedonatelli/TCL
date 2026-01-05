GPU Acceleration
================

.. module:: pytcl.gpu

The GPU module provides hardware-accelerated implementations of key tracking
algorithms using CuPy (NVIDIA CUDA) or MLX (Apple Silicon). These implementations
offer significant speedups (5-15x) for batch processing of multiple tracks.

The module automatically selects the best available backend:

- On Apple Silicon (M1/M2/M3): Uses MLX if installed
- On systems with NVIDIA GPUs: Uses CuPy if installed
- Falls back to CPU (numpy) if no GPU backend is available

Installation
------------

For NVIDIA CUDA GPUs::

    pip install nrl-tracker[gpu]
    # or directly:
    pip install cupy-cuda12x

For Apple Silicon (M1/M2/M3)::

    pip install nrl-tracker[gpu-apple]
    # or directly:
    pip install mlx

Quick Start
-----------

Check GPU availability and backend::

    from pytcl.gpu import is_gpu_available, get_backend, is_apple_silicon

    if is_gpu_available():
        print(f"GPU available, using {get_backend()} backend")

    if is_apple_silicon():
        print("Running on Apple Silicon")

Transfer arrays between CPU and GPU::

    from pytcl.gpu import to_gpu, to_cpu
    import numpy as np

    # CPU array
    x = np.random.randn(100, 4)

    # Transfer to GPU (uses best available backend)
    x_gpu = to_gpu(x)

    # Transfer back to CPU
    x_cpu = to_cpu(x_gpu)

Platform Detection
------------------

.. autofunction:: pytcl.gpu.utils.is_apple_silicon

.. autofunction:: pytcl.gpu.utils.is_mlx_available

.. autofunction:: pytcl.gpu.utils.is_cupy_available

.. autofunction:: pytcl.gpu.utils.get_backend

.. autofunction:: pytcl.gpu.utils.is_gpu_available

Array Operations
----------------

.. autofunction:: pytcl.gpu.utils.to_gpu

.. autofunction:: pytcl.gpu.utils.to_cpu

.. autofunction:: pytcl.gpu.utils.get_array_module

.. autofunction:: pytcl.gpu.utils.ensure_gpu_array

Memory Management
-----------------

.. autofunction:: pytcl.gpu.utils.sync_gpu

.. autofunction:: pytcl.gpu.utils.get_gpu_memory_info

.. autofunction:: pytcl.gpu.utils.clear_gpu_memory

Batch Kalman Filter
-------------------

GPU-accelerated batch Kalman filter operations for processing multiple tracks
in parallel. These functions provide 5-10x speedup compared to sequential CPU
processing.

.. autofunction:: pytcl.gpu.kalman.batch_kf_predict

.. autofunction:: pytcl.gpu.kalman.batch_kf_update

.. autoclass:: pytcl.gpu.kalman.CuPyKalmanFilter
   :members:
   :undoc-members:

Batch Extended Kalman Filter
----------------------------

GPU-accelerated Extended Kalman Filter for nonlinear dynamics.

.. autofunction:: pytcl.gpu.ekf.batch_ekf_predict

.. autofunction:: pytcl.gpu.ekf.batch_ekf_update

.. autoclass:: pytcl.gpu.ekf.CuPyExtendedKalmanFilter
   :members:
   :undoc-members:

Batch Unscented Kalman Filter
-----------------------------

GPU-accelerated Unscented Kalman Filter for highly nonlinear systems.

.. autofunction:: pytcl.gpu.ukf.batch_ukf_predict

.. autofunction:: pytcl.gpu.ukf.batch_ukf_update

.. autoclass:: pytcl.gpu.ukf.CuPyUnscentedKalmanFilter
   :members:
   :undoc-members:

GPU Particle Filter
-------------------

GPU-accelerated particle filtering with efficient resampling algorithms.

.. autofunction:: pytcl.gpu.particle_filter.gpu_resample_systematic

.. autofunction:: pytcl.gpu.particle_filter.gpu_resample_multinomial

.. autofunction:: pytcl.gpu.particle_filter.gpu_resample_stratified

.. autofunction:: pytcl.gpu.particle_filter.gpu_effective_sample_size

.. autofunction:: pytcl.gpu.particle_filter.gpu_normalize_weights

.. autoclass:: pytcl.gpu.particle_filter.CuPyParticleFilter
   :members:
   :undoc-members:

GPU Matrix Utilities
--------------------

GPU-accelerated matrix operations commonly used in tracking algorithms.

.. autofunction:: pytcl.gpu.matrix_utils.gpu_cholesky

.. autofunction:: pytcl.gpu.matrix_utils.gpu_cholesky_safe

.. autofunction:: pytcl.gpu.matrix_utils.gpu_qr

.. autofunction:: pytcl.gpu.matrix_utils.gpu_solve

.. autofunction:: pytcl.gpu.matrix_utils.gpu_inv

.. autofunction:: pytcl.gpu.matrix_utils.gpu_eigh

.. autofunction:: pytcl.gpu.matrix_utils.gpu_matrix_sqrt

.. autoclass:: pytcl.gpu.matrix_utils.MemoryPool
   :members:
   :undoc-members:

Example: Batch Track Processing
-------------------------------

Process multiple tracks in parallel using GPU acceleration::

    import numpy as np
    from pytcl.gpu import (
        is_gpu_available,
        to_gpu,
        to_cpu,
        batch_kf_predict,
        batch_kf_update,
    )

    if not is_gpu_available():
        raise RuntimeError("GPU not available")

    # Simulate 1000 tracks with 4D state (x, vx, y, vy)
    n_tracks = 1000
    state_dim = 4
    meas_dim = 2

    # Initial states and covariances
    x = np.random.randn(n_tracks, state_dim)
    P = np.tile(np.eye(state_dim), (n_tracks, 1, 1))

    # System matrices
    dt = 0.1
    F = np.array([
        [1, dt, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, dt],
        [0, 0, 0, 1]
    ])
    Q = np.eye(state_dim) * 0.1
    H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
    R = np.eye(meas_dim) * 0.5

    # Transfer to GPU
    x_gpu = to_gpu(x)
    P_gpu = to_gpu(P)

    # Batch predict (all 1000 tracks at once!)
    pred_result = batch_kf_predict(x_gpu, P_gpu, F, Q)

    # Generate measurements
    z = np.random.randn(n_tracks, meas_dim)

    # Batch update
    upd_result = batch_kf_update(
        pred_result.x, pred_result.P, z, H, R
    )

    # Transfer results back to CPU
    x_updated = to_cpu(upd_result.x)
    P_updated = to_cpu(upd_result.P)

    print(f"Processed {n_tracks} tracks in batch")

Performance Notes
-----------------

The GPU implementations achieve significant speedups for:

- **Large batch sizes**: Processing 100+ tracks simultaneously
- **Large particle counts**: Particle filters with 1000+ particles
- **Matrix operations**: Cholesky, QR, and eigendecompositions

For small batch sizes (< 10 tracks), CPU implementations may be faster due to
GPU transfer overhead.

Backend Differences
~~~~~~~~~~~~~~~~~~~

**CuPy (NVIDIA CUDA)**:
- Full float64 (double precision) support
- Explicit memory pool management
- CUDA stream synchronization

**MLX (Apple Silicon)**:
- Optimized for float32 (single precision)
- Automatic memory management
- Lazy evaluation with explicit sync via ``mx.eval()``
