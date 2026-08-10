GPU Acceleration Guide
======================

Overview
--------

:mod:`pytcl.gpu` accelerates tracking workloads by processing **many tracks at
once**. Instead of running one filter per call, the batch functions take
arrays with a leading track dimension -- states of shape ``(n_tracks,
state_dim)``, covariances of shape ``(n_tracks, state_dim, state_dim)`` -- and
advance every track in a single device operation.

Two backends are supported behind one API:

- **CuPy** -- NVIDIA GPUs (CUDA), computes in float64
- **MLX** -- Apple Silicon (unified memory), computes in float32

The backend is selected automatically. If neither is installed, everything
falls back to NumPy on the CPU, so code written against :mod:`pytcl.gpu` runs
anywhere.

.. note::

   The batch functions accept plain NumPy arrays and move them to the device
   themselves. Use :func:`~pytcl.gpu.to_gpu` / :func:`~pytcl.gpu.to_cpu` when
   you want to control transfers explicitly, e.g. to keep intermediate results
   on the device across many steps.

Installation
------------

**NVIDIA GPU (CUDA):**

.. code-block:: bash

   pip install nrl-tracker[gpu]
   # or directly, matching your CUDA version:
   pip install cupy-cuda12x

**Apple Silicon (MLX):**

.. code-block:: bash

   pip install nrl-tracker[gpu-apple]
   # or directly:
   pip install mlx

**Check what you have:**

.. code-block:: python

   from pytcl.gpu import get_backend, is_cupy_available, is_gpu_available, is_mlx_available

   print("GPU available:", is_gpu_available())
   print("Backend:      ", get_backend())
   print("MLX:          ", is_mlx_available())
   print("CuPy:         ", is_cupy_available())

Output on an Apple Silicon machine with MLX installed:

.. code-block:: text

   GPU available: True
   Backend:       mlx
   MLX:           True
   CuPy:          False

Quick Start: Batch Linear Kalman Filter
---------------------------------------

Advance 1,000 constant-velocity tracks through one predict-update cycle.
``F``, ``Q``, ``H``, and ``R`` may be shared across the batch (2-D, as here)
or given per track as ``(n_tracks, dim, dim)`` stacks:

.. code-block:: python

   import numpy as np

   from pytcl.gpu import batch_kf_predict, batch_kf_update, to_cpu, to_gpu

   rng = np.random.default_rng(0)
   n_tracks = 1000

   # Constant-velocity model in 2D: state [x, vx, y, vy]
   dt = 1.0
   F = np.array(
       [[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]], dtype=float
   )
   Q = np.eye(4) * 0.01
   H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]], dtype=float)  # measure position
   R = np.eye(2) * 0.1

   x = rng.normal(size=(n_tracks, 4))
   P = np.tile(np.eye(4), (n_tracks, 1, 1))
   z = rng.normal(size=(n_tracks, 2))

   # One transfer in ...
   x_gpu, P_gpu = to_gpu(x), to_gpu(P)

   # ... all tracks advance in parallel on the device ...
   pred = batch_kf_predict(x_gpu, P_gpu, F, Q)
   upd = batch_kf_update(pred.x, pred.P, z, H, R)

   # ... one transfer out.
   x_new = to_cpu(upd.x)
   P_new = to_cpu(upd.P)

   print("Updated states:     ", x_new.shape)
   print("Updated covariances:", P_new.shape)
   print("Innovations:        ", to_cpu(upd.y).shape)
   print("Likelihoods:        ", to_cpu(upd.likelihood).shape)

.. code-block:: text

   Updated states:      (1000, 4)
   Updated covariances: (1000, 4, 4)
   Innovations:         (1000, 2)
   Likelihoods:         (1000,)

The update result is a named tuple with ``x``, ``P``, ``y`` (innovations),
``S`` (innovation covariances), ``K`` (gains), and ``likelihood`` -- one entry
per track, ready for gating and association. There is also
:func:`~pytcl.gpu.batch_kf_predict_update` for a fused step, and a stateful
:class:`~pytcl.gpu.CuPyKalmanFilter` class wrapping the same operations.

Nonlinear Filters: Batched Callbacks
------------------------------------

The batch EKF and UKF take user callables, and those callables receive the
**whole batch** as a single device array -- they are called once per step, not
once per track:

- ``f(x)`` and ``h(x)`` take ``(n_tracks, state_dim)`` and return
  ``(n_tracks, out_dim)``;
- ``F_jacobian(x)`` and ``H_jacobian(x)`` take ``(n_tracks, state_dim)`` and
  return ``(n_tracks, out_dim, state_dim)``.

Write callables against :func:`~pytcl.gpu.get_array_module` instead of NumPy
directly, so the same code runs on MLX, CuPy, or the NumPy fallback. (Mixing
a host NumPy array into a CuPy expression raises ``TypeError``.)

.. code-block:: python

   import numpy as np

   from pytcl.gpu import batch_ekf_predict, batch_ekf_update, get_array_module, to_cpu


   def f(x):
       # Nearly constant velocity with mild drag on the velocity component.
       xp = get_array_module(x)
       return xp.stack([x[:, 0] + x[:, 1], 0.99 * x[:, 1]], axis=1)


   def F_jac(x):
       # Constant Jacobian, broadcast over the batch: (n_tracks, 2, 2).
       xp = get_array_module(x)
       F = xp.array([[1.0, 1.0], [0.0, 0.99]])
       return xp.broadcast_to(F, (x.shape[0], 2, 2))


   def h(x):
       # Range measurement: (n_tracks, 2) -> (n_tracks, 1).
       xp = get_array_module(x)
       return xp.sqrt(x[:, 0:1] ** 2 + 1.0)


   def H_jac(x):
       # (n_tracks, 1, 2)
       xp = get_array_module(x)
       r = xp.sqrt(x[:, 0:1] ** 2 + 1.0)
       zeros = xp.zeros_like(r)
       return xp.stack([x[:, 0:1] / r, zeros], axis=2)


   rng = np.random.default_rng(1)
   n_tracks = 500
   x = rng.normal(size=(n_tracks, 2))
   P = np.tile(np.eye(2), (n_tracks, 1, 1))
   Q = np.eye(2) * 0.01
   R = np.array([[0.1]])
   z = rng.normal(loc=1.5, size=(n_tracks, 1))

   pred = batch_ekf_predict(x, P, f, F_jac, Q)
   upd = batch_ekf_update(pred.x, pred.P, z, h, H_jac, R)

   print("Predicted states:", to_cpu(pred.x).shape)
   print("Updated states:  ", to_cpu(upd.x).shape)
   print("Kalman gains:    ", to_cpu(upd.K).shape)

.. code-block:: text

   Predicted states: (500, 2)
   Updated states:   (500, 2)
   Kalman gains:     (500, 2, 1)

Pass ``None`` for a Jacobian argument to have it computed by finite
differences on the device.

**Unscented filter:** :func:`~pytcl.gpu.batch_ukf_predict` and
:func:`~pytcl.gpu.batch_ukf_update` use the same batched ``f``/``h`` contract
(no Jacobians needed). One MLX-specific caveat: the Merwe sigma-point weights
scale as ``1 / alpha**2``, and the conventional default ``alpha=1e-3`` gives
weights of order 1e6 -- unresolvable in float32. The library emits a
``RuntimeWarning`` below ``alpha=1e-2``; on MLX pass ``alpha`` of 0.1 or
larger:

.. code-block:: python

   from pytcl.gpu import batch_ukf_predict, batch_ukf_update

   pred = batch_ukf_predict(x, P, f, Q, alpha=0.5)
   upd = batch_ukf_update(pred.x, pred.P, z, h, R, alpha=0.5)

Particle Filters
----------------

:class:`~pytcl.gpu.CuPyParticleFilter` (the name is historical; it runs on
either backend) keeps its particle set on the device across predict, update,
and resample. The dynamics callable receives all particles as one
``(n_particles, state_dim)`` backend array; the likelihood callable receives
the particles and one measurement and returns per-particle likelihoods.

.. code-block:: python

   import numpy as np

   from pytcl.gpu import (
       CuPyParticleFilter,
       get_array_module,
       gpu_effective_sample_size,
       gpu_normalize_weights,
       gpu_resample_systematic,
   )


   def dynamics(particles):
       # Receives the whole particle set (n_particles, state_dim) as a
       # backend array; returns the propagated set with the same shape.
       return particles * 0.99


   def likelihood(particles, measurement):
       # Backend-agnostic: get_array_module returns mlx.core, cupy, or numpy.
       xp = get_array_module(particles)
       diff = particles[:, 0] - measurement
       return xp.exp(-0.5 * diff**2)


   np.random.seed(7)  # initialize() samples the prior with NumPy
   pf = CuPyParticleFilter(n_particles=10000, state_dim=2)
   pf.initialize(np.zeros(2), np.eye(2))
   pf.predict(dynamics)
   log_lik = pf.update(0.5, likelihood)

   print("Estimate shape:", pf.get_estimate().shape)
   print("ESS:           ", round(pf.get_ess(), 1))

   # The helpers also work standalone on plain NumPy or device arrays:
   weights = np.full(10000, 1.0 / 10000)
   print("Standalone ESS:", round(gpu_effective_sample_size(weights), 1))
   idx = gpu_resample_systematic(weights, seed=0)
   print("Resample index:", idx.shape, idx.dtype)
   w_norm, log_sum = gpu_normalize_weights(np.log(weights))
   print("Normalized sum:", round(float(w_norm.sum()), 6))

Output on MLX:

.. code-block:: text

   Estimate shape: (2,)
   ESS:            8402.5
   Standalone ESS: 10000.0
   Resample index: (10000,) mlx.core.int32
   Normalized sum: 1.0

Resampling is automatic when the effective sample size drops below
``resample_threshold * n_particles``; choose the scheme with
``resample_method`` (``"systematic"``, ``"stratified"``, or
``"multinomial"``). :func:`~pytcl.gpu.gpu_resample_stratified` and
:func:`~pytcl.gpu.gpu_resample_multinomial` mirror the systematic helper.
:func:`~pytcl.gpu.batch_particle_filter_update` updates many independent
particle filters (shape ``(n_filters, n_particles, state_dim)``) in one call.

Device Utilities
----------------

Transfers and introspection:

- :func:`~pytcl.gpu.to_gpu` / :func:`~pytcl.gpu.to_cpu` -- move arrays to and
  from the active backend; both accept arrays that are already where they
  belong
- :func:`~pytcl.gpu.ensure_gpu_array` -- like ``to_gpu`` but with a dtype
  guarantee
- :func:`~pytcl.gpu.get_array_module` -- returns ``mlx.core``, ``cupy``, or
  ``numpy`` for a given array, for backend-agnostic callables
- :func:`~pytcl.gpu.sync_gpu` -- block until queued device work completes;
  required for honest timing, since both backends are lazy or asynchronous

Memory:

.. code-block:: python

   from pytcl.gpu import clear_gpu_memory, get_gpu_memory_info, sync_gpu

   info = get_gpu_memory_info()
   print("Backend:", info["backend"])
   print("Used bytes:", info["used"])

   sync_gpu()  # block until queued device work completes (for timing)
   clear_gpu_memory()  # release cached device memory

.. code-block:: text

   Backend: mlx
   Used bytes: 0

On MLX the dictionary reports allocator state (``used``, ``peak``,
``cache``); ``free`` and ``total`` are ``-1`` because unified memory has no
separate device pool. On CuPy it reports the device's ``free`` and ``total``
plus the CuPy memory pool's usage. :func:`~pytcl.gpu.get_memory_pool` returns
a :class:`~pytcl.gpu.MemoryPool` manager wrapping the backend allocator, with
``get_stats()``, ``set_limit()``, and ``free_all()``.

Linear algebra helpers that run on the device and fall back transparently:
:func:`~pytcl.gpu.gpu_cholesky`, :func:`~pytcl.gpu.gpu_cholesky_safe`
(regularizing, returns a success flag), :func:`~pytcl.gpu.gpu_inv`,
:func:`~pytcl.gpu.gpu_solve`, :func:`~pytcl.gpu.gpu_qr`,
:func:`~pytcl.gpu.gpu_eigh`, and :func:`~pytcl.gpu.gpu_matrix_sqrt`.

Measured Performance
--------------------

The only benchmark we publish is one we have actually run. Conditions: Apple
Silicon, MLX backend, batch linear Kalman predict+update versus a per-track
CPU loop over the reference implementation, timed end-to-end **including**
host-device transfers and result materialization, after warm-up (August
2026):

==============  =============================
Batch size      Speedup vs per-track CPU loop
==============  =============================
100 tracks      1.6x
1,000 tracks    13x
20,000 tracks   40x
==============  =============================

The shape of that curve is the real lesson: the device does not make one
filter step faster, it makes *many* filter steps simultaneous. At 100 tracks
the fixed cost of dispatch and transfer eats most of the win; by 20,000
tracks it is negligible.

**CuPy:** correctness of the CuPy backend is validated against the CPU
reference on real NVIDIA hardware (RTX 5080), but we have not measured CuPy
speedups, so this guide quotes none. Expect the same qualitative behavior --
batch size pays for transfer overhead -- and profile your own workload.

**When the GPU helps:**

- Hundreds to tens of thousands of tracks stepped together
- Particle filters with large particle counts
- Pipelines that keep data on the device across many steps

**When it does not:**

- A single track, or a handful -- the CPU filters in
  :mod:`pytcl.dynamic_estimation` will be faster
- Per-step round-trips: converting to NumPy after every update discards the
  batching advantage
- Anything outside this module: :mod:`pytcl.gpu` accelerates batch Kalman,
  EKF, UKF, and particle filtering only. Assignment algorithms and coordinate
  conversions are CPU code paths and gain nothing from installing a GPU
  backend.

Precision on MLX
----------------

MLX computes in float32 and **raises on float64 GPU operations**, so the MLX
backend converts inputs to float32 throughout. Consequences:

- Batch results match the CPU reference implementations to roughly float32
  precision -- measured about 1e-7 relative error for the linear and extended
  Kalman filters -- rather than to machine epsilon.
- The UKF is the sensitive case: keep ``alpha`` at 0.1 or larger on MLX (see
  above). CuPy computes in float64 and has no such restriction.
- MLX linear-algebra kernels (``inv``, ``cholesky``, ``solve``, ``eigh``) run
  on the CPU stream; the dispatch layer handles this transparently, and
  unified memory makes it a scheduling change rather than a copy.

Troubleshooting
---------------

**"No GPU available" from to_gpu**

No backend is installed (or you are on hardware without one). The batch
functions themselves still work -- they fall back to NumPy -- but explicit
``to_gpu`` calls require a backend.

**ImportError: libcublas.so.12 (or another lib...so.12)**

Your system CUDA is 13.x (or missing entirely) and only ``cupy-cuda12x``
itself is installed. The ``[gpu]`` extra ships the CUDA 12 runtime
libraries as pip wheels on Linux, so reinstalling with
``pip install nrl-tracker[gpu]`` resolves this; on Windows, install a
system CUDA 12.8+ toolkit. NVRTC older than 12.8 also cannot compile
kernels for Blackwell (RTX 50-series) GPUs -- the pinned wheels cover
that case too.

**Slower than the CPU**

Almost always one of: the batch is too small to amortize transfer overhead,
or the loop transfers to NumPy every step. Keep results as backend arrays
between steps and convert once at the end. When timing, call
:func:`~pytcl.gpu.sync_gpu` before reading the clock -- both backends queue
work asynchronously, so un-synchronized timings measure dispatch, not
compute.

**Out of device memory**

Process the track set in chunks along the batch dimension, and call
:func:`~pytcl.gpu.clear_gpu_memory` between chunks if the allocator cache
grows.

See Also
--------

- :doc:`performance_optimization` - CPU optimization techniques
- :doc:`kalman_filter_tuning` - Filter tuning and diagnostics
- `CuPy Documentation <https://docs.cupy.dev>`_
- `MLX Documentation <https://ml-explore.github.io/mlx/>`_
