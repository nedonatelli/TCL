Troubleshooting Guide
=====================

Overview
--------

This guide covers common issues, errors, and their solutions. See :doc:`architecture` for module structure and :doc:`api_navigation` for function discovery.

Installation Issues
-------------------

**ImportError: No module named 'pytcl'**

**Cause**: pytcl not installed or wrong Python environment

**Solution**:

.. code-block:: bash

   # Check if installed
   python -c "import pytcl; print(pytcl.__version__)"
   
   # If not installed, install from PyPI
   pip install nrl-tracker
   
   # Or install from source
   git clone https://github.com/nedonatelli/TCL.git
   cd TCL
   pip install -e .

**Using Wrong Python Version**

**Cause**: Project requires Python 3.10+, but you're using an older version

**Solution**:

.. code-block:: bash

   # Check your Python version
   python --version
   
   # If too old, create a virtual environment with Python 3.10+
   python3.11 -m venv venv
   source venv/bin/activate
   pip install nrl-tracker

**GPU Backend Issues**

**ImportError: No module named 'cupy'**

**Cause**: CuPy not installed (GPU acceleration is optional)

**Solution**:

.. code-block:: bash

   # CuPy is optional - you can use CPU version
   # Or install CuPy for GPU acceleration
   pip install cupy-cuda11x  # Replace 11x with your CUDA version
   
   # Check if GPU backend works
   python -c "from pytcl.gpu.cupy_backend import to_gpu; print('CuPy OK')"

See :doc:`gpu_acceleration` for detailed GPU setup.

Import and Usage Issues
-----------------------

**ImportError: cannot import name 'function_name'**

**Cause**: Incorrect import path or function doesn't exist

**Solution**:

.. code-block:: python

   # ❌ Wrong: Function not in this location
   from pytcl.dynamic_estimation.kalman import kf_predict
   
   # ✅ Correct: Check the right module
   from pytcl.dynamic_estimation.kalman import kf_predict
   
   # Or use interactive discovery
   from pytcl.dynamic_estimation import kalman
   print([x for x in dir(kalman) if not x.startswith('_')])

**TypeError: Expected array-like input**

**Cause**: Function expects NumPy array, got Python list

**Solution**:

.. code-block:: python

   import numpy as np
   from pytcl.coordinate_systems.conversions import cart2sphere
   
   # ❌ Wrong: Python list
   result = cart2sphere([1.0, 0.0, 0.0])
   
   # ✅ Correct: NumPy array
   result = cart2sphere(np.array([1.0, 0.0, 0.0]))

**AttributeError: module 'pytcl' has no attribute 'xyz'**

**Cause**: Function not re-exported at top level

**Solution**:

.. code-block:: python

   # ✅ Use full import path
   from pytcl.dynamic_estimation.kalman import kf_predict
   
   # Instead of trying
   from pytcl import kf_predict  # Won't work

**Check available functions**:

.. code-block:: python

   from pytcl import dynamic_estimation
   print(dir(dynamic_estimation.kalman))

Array Shape and Dimension Issues
---------------------------------

**ValueError: Shapes (3,) and (4,4) not aligned**

**Cause**: Dimension mismatch in matrix operations

**Solution**: Ensure shapes are compatible

.. code-block:: python

   import numpy as np
   from pytcl.dynamic_estimation.kalman import kf_predict
   
   # State vector must match F rows
   F = np.eye(4)      # 4×4 motion model
   Q = np.eye(4) * 0.1
   x = np.array([0, 1, 0, 1])  # Must be length 4
   P = np.eye(4)      # Must be 4×4
   
   # ✅ Correct dimensions
   x_pred, P_pred = kf_predict(x, P, F, Q)

**Check dimensions**:

.. code-block:: python

   def check_shapes(x, P, F, Q, H, R):
       print(f"x shape: {x.shape}")
       print(f"P shape: {P.shape}")
       print(f"F shape: {F.shape}")
       print(f"Q shape: {Q.shape}")
       print(f"H shape: {H.shape}")
       print(f"R shape: {R.shape}")
   
   # Rules:
   # x.shape = (n,)
   # P.shape = (n, n)
   # F.shape = (n, n)
   # Q.shape = (n, n)
   # H.shape = (m, n)
   # R.shape = (m, m)

**RuntimeError: Expected 2-D array, got 1-D array instead**

**Cause**: Function expects 2D array but got 1D (or vice versa)

**Solution**:

.. code-block:: python

   import numpy as np
   
   # ❌ Wrong: 1D array
   x = np.array([1, 2, 3])
   x_2d = x.reshape(-1, 1)  # Convert to column vector
   
   # ✅ Correct: Use as-is if function expects 1D
   # Check function documentation
   from pytcl.mathematical_functions.special_functions import marcum_q
   a = 2.0
   b = 3.0
   q = marcum_q(a, b)  # Scalar inputs OK

Filter and Estimation Issues
-----------------------------

**Filter Divergence: Covariance Grows Over Time**

**Symptoms**: P increases instead of decreases, filter loses track

**Causes and Solutions**:

1. **Q too small** (process noise underestimated)

   .. code-block:: python

      # ❌ Q too small
      Q = np.eye(4) * 0.001  # Too small!
      
      # ✅ Reasonable Q
      Q = np.eye(4) * 0.1  # Depends on system

   See :doc:`kalman_filter_tuning` for proper Q estimation.

2. **R too large** (measurement noise overestimated)

   .. code-block:: python

      # ❌ R too large
      R = np.eye(2) * 1000  # Measurements are ignored
      
      # ✅ Match sensor accuracy
      R = np.eye(2) * 0.1

3. **Motion model wrong** (F matrix incorrect)

   .. code-block:: python

      # Verify motion model matches system
      T = 0.1  # time step
      F = f_constant_velocity(T, num_dims=2)
      # If system does accelerate, use:
      # F = f_constant_acceleration(T, num_dims=2)

4. **Measurement outliers** (sensor gives bad data)

   .. code-block:: python

      # Add outlier rejection
      innovation = z - H @ x_pred
      
      if np.linalg.norm(innovation) > gate_threshold:
          # Skip this measurement or downweight it
          pass

**Diagnostic Code**:

.. code-block:: python

   import numpy as np
   
   def diagnose_divergence(x_sequence, P_sequence):
       """Check if filter is diverging"""
       trace_P = [np.trace(P) for P in P_sequence]
       det_P = [np.linalg.det(P) for P in P_sequence]
       
       print("Trace of P (should decrease): ", trace_P)
       print("Det of P (should decrease): ", det_P)
       
       # Check if increasing
       if trace_P[-1] > trace_P[0]:
           print("⚠ Filter diverging: P trace increased")

See :doc:`kalman_filter_tuning` for detailed diagnostics.

**Filter Lags Behind Measurements**

**Symptom**: Filter predictions are always 1-2 steps behind actual motion

**Causes**:

1. **Q too large** (filter doesn't trust model)

   .. code-block:: python

      # ❌ Q too large
      Q = np.eye(4) * 10  # Filter doubts the model
      
      # ✅ Reduce Q
      Q = np.eye(4) * 0.1

2. **Motion model too simple**

   .. code-block:: python

      # If system has acceleration, need better model
      # Constant velocity model:
      F = f_constant_velocity(T, num_dims=2)
      
      # Better: constant acceleration
      F = f_constant_acceleration(T, num_dims=2)

3. **Measurement delay**

   Ensure your time stamps are accurate:

   .. code-block:: python

      # Check if measurements lag
      for i, (t, z) in enumerate(zip(times, measurements)):
          delay = current_time - t
          if delay > expected_delay:
              print(f"Measurement {i} delayed by {delay}s")

**Filter Too Jerky: Follows Noise**

**Symptom**: Filter output wiggles even when target not moving

**Cause**: Q too small (filter over-trusts measurements)

**Solution**:

.. code-block:: python

   # ❌ Q too small: filter tries to match every measurement
   Q = np.eye(4) * 0.001
   
   # ✅ Increase Q to smooth
   Q = np.eye(4) * 0.5

**Data Association Issues**

**"No assignment found for measurement"**

**Cause**: All measurements beyond gating distance

**Solution**:

.. code-block:: python

   from scipy.spatial.distance import cdist
   
   # Check gating distance
   positions = np.array([[0, 0], [1, 1]])
   measurements = np.array([[10, 10], [20, 20]])  # Far away!
   
   cost = cdist(positions, measurements)
   gate = 5.0  # Maximum distance
   
   # Mark out-of-range
   cost[cost > gate] = np.inf
   
   from pytcl.assignment_algorithms import relaxation_assignment_nd
   assignments = relaxation_assignment_nd(cost)
   # Result: mostly unassigned measurements

**Fix**:

.. code-block:: python

   # 1. Increase gate threshold
   gate = 10.0  # More permissive
   
   # 2. Or check if measurements are in right coordinate frame
   # Ensure measurements converted to same frame as state

**Multiple incorrect assignments**

**Cause**: Cost matrix has poor gradation between good/bad matches

**Solution**:

.. code-block:: python

   from pytcl.assignment_algorithms import relaxation_assignment_nd
   
   # Compute cost matrix more carefully
   # Option 1: Mahalanobis distance (includes covariance)
   from scipy.spatial.distance import cdist
   
   # Standard Euclidean
   cost = cdist(positions, measurements)
   
   # Better: Mahalanobis (weighted by covariance)
   from scipy.spatial.distance import cdist
   cost = cdist(positions, measurements, metric='mahalanobis')
   
   assignments = relaxation_assignment_nd(cost)

Coordinate Conversion Issues
-----------------------------

**"Result is NaN or Inf"**

**Cause**: Invalid input to conversion function (e.g., negative range, ±π issues)

**Solution**:

.. code-block:: python

   from pytcl.coordinate_systems.conversions import cart2sphere
   import numpy as np
   
   # ❌ Invalid: origin
   result = cart2sphere(np.array([0, 0, 0]))  # Undefined azimuth!
   
   # ✅ Valid: non-zero
   result = cart2sphere(np.array([1, 0, 0.1]))
   
   # ❌ Invalid: negative range
   result = cart2sphere(np.array([-1, 0, 0]))  # Range must be positive!

**Check input validity**:

.. code-block:: python

   def is_valid_spherical(spherical_coords):
       r, az, el = spherical_coords
       
       if r < 0:
           print("Range must be positive")
           return False
       
       if not (-np.pi <= az <= np.pi):
           print("Azimuth must be in [-π, π]")
           return False
       
       if not (-np.pi/2 <= el <= np.pi/2):
           print("Elevation must be in [-π/2, π/2]")
           return False
       
       return True

**Angle Wrapping Issues**

**Problem**: Angles greater than 2π or negative angles

**Solution**:

.. code-block:: python

   import numpy as np
   
   # Wrap angle to [-π, π]
   def wrap_angle(angle):
       return np.arctan2(np.sin(angle), np.cos(angle))
   
   # Test
   large_angle = 3 * np.pi
   wrapped = wrap_angle(large_angle)
   print(f"{large_angle} → {wrapped}")  # 9.42 → -2.57

Performance Issues
------------------

**"Program is too slow" or "Using too much memory"**

**Solution**: Profile first to find bottleneck

.. code-block:: python

   import cProfile
   import pstats
   
   def my_tracking_loop():
       for z in measurements:
           x, P = kf_predict(x, P, F, Q)
           x, P = kf_update(x, P, z, H, R)
   
   profiler = cProfile.Profile()
   profiler.enable()
   my_tracking_loop()
   profiler.disable()
   
   stats = pstats.Stats(profiler)
   stats.sort_stats('cumulative')
   stats.print_stats(10)  # Top 10 functions

See :doc:`performance_optimization` for detailed optimization techniques.

**"Processing measurements too slowly"**

**Causes and Solutions**:

1. **Too many array allocations**

   .. code-block:: python

      # ❌ Slow: allocate in loop
      for z in measurements:
          cost = np.zeros((n_targets, n_measurements))  # Allocate each time
          # ...
      
      # ✅ Fast: allocate once
      cost = np.zeros((n_targets, n_measurements))
      for z in measurements:
          cost[:] = compute_distances(...)  # Reuse

2. **Not vectorizing operations**

   .. code-block:: python

      # ❌ Slow: Python loop
      for measurement in measurements:
          predict_filter(measurement)
      
      # ✅ Fast: Batch
      predict_filter_batch(measurements)

3. **Using GPU backend without GPU**

   .. code-block:: python

      # If no GPU, CPU version is faster (no transfer overhead)
      import pytcl.gpu.cupy_backend as gpu
      
      # Check if GPU actually available
      try:
          gpu.check_gpu_available()
      except RuntimeError:
          print("GPU not available, using CPU")
          # Use CPU version instead

See :doc:`performance_optimization` and :doc:`gpu_acceleration`.

**"Memory usage keeps growing"**

**Cause**: Memory leaks or unbounded data accumulation

**Solution**:

.. code-block:: python

   import tracemalloc
   
   tracemalloc.start()
   
   # Run your code
   for i in range(1000):
       do_tracking_step()
   
   current, peak = tracemalloc.get_traced_memory()
   print(f"Current: {current / 1024 / 1024:.1f} MB")
   print(f"Peak: {peak / 1024 / 1024:.1f} MB")
   tracemalloc.stop()

**Fix**:

.. code-block:: python

   # Don't accumulate history indefinitely
   class Tracker:
       def __init__(self, max_history=100):
           self.history = []
       
       def update(self, measurement):
           self.history.append(measurement)
           
           # Keep only recent history
           if len(self.history) > self.max_history:
               self.history = self.history[-self.max_history:]

Numerical Issues
----------------

**Covariance Matrix Not Positive Definite**

**Symptom**: Eigenvalues have negative or very small values

**Cause**: Numerical drift, poor initialization, or invalid update

.. code-block:: python

   import numpy as np
   
   def check_positive_definite(P):
       eigs = np.linalg.eigvalsh(P)
       min_eig = np.min(eigs)
       
       if min_eig < 0:
           print(f"⚠ P not positive definite: min eigenvalue = {min_eig}")
           return False
       
       if min_eig < 1e-10:
           print(f"⚠ P nearly singular: min eigenvalue = {min_eig}")
           return False
       
       return True
   
   # Check P after initialization
   check_positive_definite(P)

**Solution**:

.. code-block:: python

   import numpy as np
   from scipy.linalg import cholesky
   
   # Fix: Enforce positive-definiteness
   def fix_covariance(P):
       try:
           # Try Cholesky decomposition
           L = cholesky(P, lower=True)
           return L @ L.T  # Rebuild
       except np.linalg.LinAlgError:
           # If that fails, add small regularization
           epsilon = 1e-8
           P_fixed = P + epsilon * np.eye(len(P))
           return P_fixed
   
   P = fix_covariance(P)

**"Numerical underflow/overflow"**

**Cause**: Very small or very large numbers in calculations

**Solution**:

.. code-block:: python

   import numpy as np
   
   # Use higher precision
   x = np.array([1e-300, 2e-300], dtype=np.float64)  # Not np.float32
   
   # Or use log-space for very small numbers
   log_prob = np.log(probability)  # -1000 is OK, but 1e-300 underflows

Debugging Tools
---------------

**Print Debug Information**:

.. code-block:: python

   import numpy as np
   from pytcl.dynamic_estimation.kalman import kf_predict, kf_update
   
   def debug_filter(x, P, z, F, Q, H, R):
       print(f"State before: {x}")
       print(f"Cov trace before: {np.trace(P):.4f}")
       
       x_p, P_p = kf_predict(x, P, F, Q)
       print(f"State after predict: {x_p}")
       print(f"Cov trace after predict: {np.trace(P_p):.4f}")
       
       innovation = z - H @ x_p
       print(f"Innovation: {innovation}")
       
       x_u, P_u = kf_update(x_p, P_p, z, H, R)
       print(f"State after update: {x_u}")
       print(f"Cov trace after update: {np.trace(P_u):.4f}")
       
       return x_u, P_u

**Use Assertions**:

.. code-block:: python

   import numpy as np
   
   def track_measurement(x, P, z):
       # Assertions catch bugs early
       assert isinstance(x, np.ndarray), "x must be array"
       assert x.ndim == 1, "x must be 1D"
       assert len(x) == 4, "x must be 4D state"
       assert np.isfinite(x).all(), "x contains NaN or Inf"
       assert np.isfinite(P).all(), "P contains NaN or Inf"
       
       # ... rest of function

**Use Logging**:

.. code-block:: python

   import logging
   
   logging.basicConfig(level=logging.DEBUG)
   logger = logging.getLogger(__name__)
   
   def update_filter(x, P, z):
       logger.debug(f"State: {x}")
       logger.debug(f"Measurement: {z}")
       
       x_pred, P_pred = kf_predict(x, P, F, Q)
       logger.debug(f"After predict: {x_pred}")
       
       x_upd, P_upd = kf_update(x_pred, P_pred, z, H, R)
       logger.debug(f"After update: {x_upd}")
       
       return x_upd, P_upd

Getting Help
------------

**If you still can't find the answer:**

1. Check the documentation at https://readthedocs.org/
2. Review examples in ``examples/`` folder
3. Check similar issues on GitHub: https://github.com/nedonatelli/TCL/issues
4. Create an issue describing:
   - What you're trying to do
   - Error message (full traceback)
   - Minimal reproducible example
   - Your environment (Python version, pytcl version)

**Minimal Reproducible Example**:

.. code-block:: python

   # Minimal code that shows the problem
   import numpy as np
   from pytcl.dynamic_estimation.kalman import kf_predict
   
   # Minimum setup to reproduce
   x = np.array([0.0, 1.0, 0.0, 1.0])
   P = np.eye(4) * 0.1
   F = np.eye(4)
   Q = np.eye(4) * 0.01
   
   # This fails with: [error message]
   x_pred, P_pred = kf_predict(x, P, F, Q)

See Also
~~~~~~~~

- :doc:`kalman_filter_tuning` - Filter parameter tuning
- :doc:`performance_optimization` - Performance profiling
- :doc:`gpu_acceleration` - GPU acceleration setup
- :doc:`api_navigation` - Function discovery
- Examples: ``examples/`` folder
