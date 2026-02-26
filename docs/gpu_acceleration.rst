GPU Acceleration Guide
======================

Overview
--------

The Tracker Component Library provides GPU acceleration for compute-intensive operations through two backends:

- **CuPy** - NVIDIA GPU support (CUDA)
- **MLX** - Apple Silicon GPU support (Metal Performance Shaders)

This guide explains how to use GPU acceleration to speed up Kalman filters and particle filters.

.. note::

   GPU acceleration is optional. The library falls back to CPU automatically if GPU libraries aren't installed.

Installation
------------

**NVIDIA GPU (CUDA):**

.. code-block:: bash

   pip install cupy-cuda11x  # Replace 11x with your CUDA version (11.2, 12.0, etc.)
   # Or for a specific CUDA version:
   pip install cupy-cuda12 

**Apple Silicon (MLX):**

.. code-block:: bash

   pip install mlx

**Verify Installation:**

.. code-block:: python

   from pytcl.gpu import is_gpu_available
   print(is_gpu_available())  # Returns GPU backend name or None

Quick Start
-----------

Extended Kalman Filter on GPU:

.. code-block:: python

   import numpy as np
   from pytcl.dynamic_estimation.kalman import extended_kalman_filter
   
   # Your state transition and measurement functions
   def state_transition(x, dt):
       return x  # Example: constant velocity model
   
   def measurement_fn(x):
       return x[:2]  # Measure position only
   
   # Initialize on CPU (standard NumPy)
   x0 = np.array([0.0, 0.0, 1.0, 0.0])  # [x, y, vx, vy]
   P0 = np.eye(4)
   
   # If GPU available, convert to GPU arrays
   try:
       import cupy as cp
       x0_gpu = cp.asarray(x0)
       P0_gpu = cp.asarray(P0)
   except ImportError:
       x0_gpu = x0
       P0_gpu = P0
   
   # Run filter - automatically uses GPU if inputs are GPU arrays
   z = np.array([[1.0, 2.0]])  # Measurement: [x_meas, y_meas]
   x_new, P_new = extended_kalman_filter(x0_gpu, P0_gpu, z, state_transition, measurement_fn)

Performance Considerations
--------------------------

**When GPU Acceleration Helps:**

- Particle filters with 1,000+ particles
- Batch processing of many trajectories
- Large-scale data association problems
- Real-time systems processing multiple targets

**When GPU May Not Help:**

- Small filters (< 100 state dimensions)
- One-shot filtering operations (transfer overhead dominates)
- CPU-bound operations (coordinate conversions, etc.)

**Performance Best Practices:**

1. **Batch Operations**: Process multiple time steps before CPU transfer

   .. code-block:: python

      import cupy as cp
      
      # ✅ Good: Batch 100 measurements
      measurements = cp.asarray(measurements_array)  # [100, 2] measurements
      for z in measurements:
          x, P = extended_kalman_filter(x, P, z, ...)
          # Results stay on GPU
      
      # ❌ Avoid: Constant GPU<->CPU transfers
      for z in measurements:
          x_cpu = cp.asnumpy(x)  # Avoid repeated transfers
          x = cp.asarray(extended_kalman_filter(x_cpu, ...))

2. **Memory Management**: Monitor GPU memory usage

   .. code-block:: python

      import cupy as cp
      
      # Check available GPU memory
      mempool = cp.get_default_memory_pool()
      print(f"Used: {mempool.used_bytes() / 1e9:.2f} GB")
      print(f"Total: {mempool.total_bytes() / 1e9:.2f} GB")
      
      # Clear cache if needed
      mempool.free_all_blocks()

3. **Data Type Selection**: Use float32 for better GPU performance

   .. code-block:: python

      import cupy as cp
      
      # float32 is faster on most GPUs
      x = cp.asarray([0.0, 0.0, 1.0, 0.0], dtype=cp.float32)
      P = cp.eye(4, dtype=cp.float32)
      
      # float64 only if numerical precision is critical

Module-Specific GPU Support
----------------------------

**Kalman Filters** (Full Support)

- ``extended_kalman_filter()`` - EKF
- ``unscented_kalman_filter()`` - UKF
- ``cubature_kalman_filter()`` - CKF

All matrix operations (Cholesky, QR, etc.) automatically use GPU.

**Particle Filters** (Full Support)

- ``particle_filter()`` - Standard resampling
- ``sequential_importance_resampling()`` - SIR

GPU accelerates particle propagation and weight computation.

**Data Association** (Partial Support)

- ``assignment_nd()`` - Greedy and Hungarian algorithms
- Sparse assignment with large cost matrices benefits most

**Coordinate Conversions** (Limited Benefit)

- GPU acceleration not recommended for single conversions
- Batch conversions of 10,000+ points show ~2-5x speedup

Troubleshooting
---------------

**Issue: "ModuleNotFoundError: No module named 'cupy'"**

CuPy not installed. Install with your CUDA version:

.. code-block:: bash

   pip install cupy-cuda12

**Issue: CUDA Error "Device insufficient for this operation"**

GPU doesn't support required operations. Fall back to CPU:

.. code-block:: python

   import numpy as np
   x = np.asarray(x)  # Convert GPU arrays back to CPU
   # Continue processing on CPU

**Issue: Out of Memory Error**

Reduce batch size or switch to CPU:

.. code-block:: python

   # Process smaller batches
   batch_size = 100
   for i in range(0, len(measurements), batch_size):
       batch = measurements[i:i+batch_size]
       # Process batch...

**Issue: Slower than CPU**

GPU overhead > speedup. Common causes:

1. Small problem size (< 100 state dimension)
2. Frequent GPU<->CPU transfers
3. I/O bound (disk reading slower than computation)

Solution: Profile with `time` module to identify bottleneck:

.. code-block:: python

   import time
   
   # Time GPU operation
   x_gpu = cp.asarray(x)
   start = time.perf_counter()
   x_result = extended_kalman_filter(x_gpu, P_gpu, z_gpu, ...)
   cp.cuda.Stream.null.synchronize()  # Wait for GPU
   elapsed = time.perf_counter() - start
   print(f"GPU time: {elapsed:.4f}s")

Performance Benchmarks
----------------------

Typical speedups (relative to CPU NumPy):

=================================  ==============  ==============
Operation                          NVIDIA (CuPy)   Apple (MLX)
=================================  ==============  ==============
EKF with 20-dim state              5-8x            3-5x
Particle filter (1000 particles)   8-12x           4-7x
Sparse assignment (10k x 10k)      10-15x          5-10x
Coordinate conversion (1M points)  2-3x            1-2x
=================================  ==============  ==============

.. note::

   Benchmark results depend on GPU model, data size, and problem type.
   Always profile your specific use case.

Advanced Topics
---------------

**Custom GPU Kernels**

For extreme performance, write custom CUDA kernels with CuPy:

.. code-block:: python

   import cupy as cp
   
   # Define custom CUDA kernel
   kernel_code = '''
   __global__ void my_kernel(float *x, int n) {
       int idx = blockIdx.x * blockDim.x + threadIdx.x;
       if (idx < n) x[idx] *= 2.0f;
   }
   '''
   
   kernel = cp.RawKernel(kernel_code, 'my_kernel')

**Multi-GPU Processing**

For systems with multiple GPUs:

.. code-block:: python

   import cupy as cp
   
   # Distribute work across GPUs
   for gpu_id in range(cp.cuda.runtime.getDeviceCount()):
       with cp.cuda.Device(gpu_id):
           # Process on this GPU
           pass

See Also
--------

- :doc:`performance_optimization` - CPU optimization techniques
- :doc:`kalman_filter_tuning` - Filter tuning and diagnostics
- `CuPy Documentation <https://docs.cupy.dev>`_
- `MLX Documentation <https://ml-explore.github.io/mlx/>`_
