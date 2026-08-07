Custom Filter Implementation
=============================

Extend the Tracker Component Library with custom filtering algorithms. This guide covers designing, implementing, and integrating new filters while leveraging existing infrastructure for testing, logging, and integration.

.. contents:: Contents
   :local:
   :depth: 3

Why Implement Custom Filters
=============================

**Scenarios:**

1. **Proprietary Algorithms**: Company-specific filter designs not publicly available
2. **Research Extensions**: Novel variants for academic work or specialized applications
3. **Legacy Integration**: Wrapping existing MATLAB/C++ filter implementations
4. **Performance Optimization**: Heavily optimized versions for specific hardware
5. **Domain-Specific Logic**: Filters incorporating application-specific constraints

**Benefits of integrating with TCL:**

- ✅ Leverage existing coordinate systems, dynamic models, data association
- ✅ Use standard testing framework and diagnostics tools
- ✅ Automatic documentation generation
- ✅ Community review and potential contribution back to library

Design Patterns: Class-Based Wrappers
=====================================

pytcl itself does not define a filter class hierarchy: the shipped filter API is functional. ``kf_predict`` and ``kf_update`` are plain functions that take arrays and return NamedTuples (``KalmanPrediction(x, P)`` and ``KalmanUpdate(x, P, y, S, K, likelihood)``), and the same style holds for the EKF/UKF variants. There is no abstract base class to inherit from.

A class-based wrapper around those functions is nonetheless a reasonable design choice for *your own* code: it keeps ``x`` and ``P`` together, gives every filter the same ``predict``/``update``/``get_state`` surface, and makes filters interchangeable in a tracking loop. The rest of this guide builds such wrappers -- treat the base class below as a pattern you may adopt, not something pytcl requires or provides.

Filter Base Class Structure
----------------------------

A workable pattern:

.. code-block:: python

    from abc import ABC, abstractmethod
    import numpy as np
    from typing import Tuple, Optional, Dict, Any
    from numpy.typing import NDArray

    class BaseFilter(ABC):
        """
        Abstract base class for filtering algorithms.

        All filters should implement:
        - predict(): Propagate state forward
        - update(): Incorporate measurement
        - get_state(): Return current estimate
        """

        def __init__(self, state_dim: int, meas_dim: int):
            """
            Parameters
            ----------
            state_dim : int
                Dimension of state vector
            meas_dim : int
                Dimension of measurement vector
            """
            self.n = state_dim
            self.m = meas_dim

            # Subclasses should initialize:
            self.x: Optional[NDArray[np.floating[Any]]] = None  # State estimate
            self.P: Optional[NDArray[np.floating[Any]]] = None  # Covariance

        @abstractmethod
        def predict(self, **kwargs: Any) -> None:
            """
            Prediction step.

            Subclasses must implement. Typical signature:
                predict(F, Q, dt, ...)
            where F is state transition, Q is process noise covariance.
            """
            pass

        @abstractmethod
        def update(self, z: NDArray[np.floating[Any]],
                  **kwargs: Any) -> None:
            """
            Measurement update step.

            Subclasses must implement. Typical signature:
                update(z, H, R, ...)
            where z is measurement, H is measurement matrix, R is noise covariance.
            """
            pass

        @abstractmethod
        def get_state(self) -> Tuple[NDArray[np.floating[Any]],
                                    NDArray[np.floating[Any]]]:
            """
            Return current state estimate and covariance.

            Returns
            -------
            (x, P) tuple
                x: State vector (n,)
                P: Covariance matrix (n, n)
            """
            pass

        @property
        def state_dim(self) -> int:
            """State dimension."""
            return self.n

        @property
        def meas_dim(self) -> int:
            """Measurement dimension."""
            return self.m

Example 1: Custom Adaptive Constant Velocity Filter
===================================================

Create a specialized filter for tracking objects moving at constant velocity with adaptive noise estimation.

.. code-block:: python

    import numpy as np
    from scipy.linalg import cholesky
    from typing import Dict, Any, Tuple, Optional
    from numpy.typing import NDArray

    class AdaptiveConstantVelocityFilter:
        """
        Specialized filter for constant velocity targets.

        Features:
        - Assumes constant velocity motion model
        - Adaptively estimates measurement noise
        - Detects target acceleration (maneuvers)
        - Simplified interface for CV-only scenarios
        """

        def __init__(self, pos_init: NDArray[np.floating[Any]],
                     vel_init: Optional[NDArray[np.floating[Any]]] = None,
                     P_pos: float = 1.0, P_vel: float = 0.1):
            """
            Parameters
            ----------
            pos_init : (d,) array
                Initial position (d = 2 or 3 for 2D/3D)
            vel_init : (d,) array, optional
                Initial velocity (assumed zero if not provided)
            P_pos : float
                Initial position uncertainty
            P_vel : float
                Initial velocity uncertainty
            """
            self.d = len(pos_init)  # Dimension (2D, 3D, etc.)
            self.n = 2 * self.d     # State: [pos, vel]

            # Initialize state [x, y, vx, vy] (in 2D)
            self.x = np.zeros(self.n)
            self.x[:self.d] = pos_init
            if vel_init is not None:
                self.x[self.d:] = vel_init

            # Initialize covariance
            self.P = np.diag(np.concatenate([
                np.full(self.d, P_pos),
                np.full(self.d, P_vel)
            ]))

            # Adaptive noise estimation
            self.R_est = 1.0  # Will be updated
            self.innovation_history = []
            self.forgetting_factor = 0.95

            # Maneuver detection
            self.maneuver_prob = 0.0
            self.Q_base = np.eye(self.n) * 0.01

            # Diagnostics
            self.update_count = 0

        def predict(self, dt: float) -> None:
            """
            Predict using constant velocity model.

            Parameters
            ----------
            dt : float
                Time step
            """
            # State transition matrix F = [[I, I*dt], [0, I]]
            F = np.eye(self.n)
            F[:self.d, self.d:] = np.eye(self.d) * dt

            # Predictor-corrector: adjust Q based on maneuver detection
            Q = self.Q_base.copy()
            if self.maneuver_prob > 0.5:
                # Increase process noise if maneuvering likely
                Q *= (1.0 + 2.0 * self.maneuver_prob)

            # Propagate
            self.x = F @ self.x
            self.P = F @ self.P @ F.T + Q

        def update(self, z: NDArray[np.floating[Any]]) -> None:
            """
            Update with position measurement.

            Parameters
            ----------
            z : (d,) array
                Measured position
            """
            # Measurement matrix (observe position only)
            H = np.hstack([np.eye(self.d), np.zeros((self.d, self.d))])

            # Innovation
            z_pred = H @ self.x
            innovation = z - z_pred

            # Innovation covariance
            S = H @ self.P @ H.T + self.R_est * np.eye(self.d)

            # Kalman gain
            K = self.P @ H.T @ np.linalg.inv(S)

            # Update
            self.x = self.x + K @ innovation
            self.P = (np.eye(self.n) - K @ H) @ self.P

            # Adaptive R estimation
            innov_squared = (innovation ** 2).sum()
            self.innovation_history.append(innov_squared)

            if len(self.innovation_history) > 1:
                # Exponential moving average
                self.R_est = (self.forgetting_factor * self.R_est +
                             (1 - self.forgetting_factor) * innov_squared)

                # Limit R to prevent numerical issues
                self.R_est = np.clip(self.R_est, 0.01, 100.0)

            # Detect maneuvers: innovation should be small for CV model
            expected_innov = self.d * self.R_est
            innov_ratio = innov_squared / (expected_innov + 1e-6)

            # Update maneuver probability
            self.maneuver_prob = 0.9 * self.maneuver_prob + 0.1 * (innov_ratio > 2.0)

            self.update_count += 1

        def get_state(self) -> Tuple[NDArray[np.floating[Any]],
                                     NDArray[np.floating[Any]]]:
            """Return state and covariance."""
            return self.x.copy(), self.P.copy()

        def get_position(self) -> NDArray[np.floating[Any]]:
            """Return position estimate only."""
            return self.x[:self.d].copy()

        def get_velocity(self) -> NDArray[np.floating[Any]]:
            """Return velocity estimate only."""
            return self.x[self.d:].copy()

        def get_diagnostics(self) -> Dict[str, Any]:
            """Return diagnostic information."""
            return {
                'updates': self.update_count,
                'R_estimate': self.R_est,
                'maneuver_probability': self.maneuver_prob,
                'innovations_recent_mean': (np.mean(self.innovation_history[-10:])
                                           if self.innovation_history else 0.0)
            }

**Usage:**

.. code-block:: python

    # 2D tracking scenario
    pos_init = np.array([0.0, 0.0])
    cvf = AdaptiveConstantVelocityFilter(pos_init)

    for k in range(100):
        # Predict
        cvf.predict(dt=0.1)

        # Measurement
        true_pos = np.array([k * 0.1, 0.05 * k])
        z = true_pos + 0.5 * np.random.randn(2)

        # Update
        cvf.update(z)

        if k % 20 == 0:
            pos, vel = cvf.get_position(), cvf.get_velocity()
            diag = cvf.get_diagnostics()
            print(f"Step {k}: pos=[{pos[0]:.3f}, {pos[1]:.3f}], "
                  f"vel=[{vel[0]:.3f}, {vel[1]:.3f}], "
                  f"R_est={diag['R_estimate']:.2f}, "
                  f"maneuver_prob={diag['maneuver_probability']:.2f}")

Example 2: Wrapping External C++ Filter
========================================

Integrate a pre-existing C++ filter implementation via ctypes or Cython.

.. code-block:: python

    import ctypes
    import numpy as np
    from pathlib import Path
    from typing import Tuple, Any
    from numpy.typing import NDArray

    class ExternalCppFilterWrapper:
        """
        Wrapper for external C++ filter implementation.

        Assumes compiled library with C interface:
        - kf_init(n, m)
        - kf_predict(F, Q, dt)
        - kf_update(z, H, R)
        - kf_get_state(x_out, P_out)
        """

        def __init__(self, state_dim: int, meas_dim: int,
                     lib_path: str = None):
            """
            Parameters
            ----------
            state_dim : int
                State space dimension
            meas_dim : int
                Measurement space dimension
            lib_path : str, optional
                Path to compiled .so/.dll file
                If None, looks for libcppfilter.so in current directory
            """
            self.n = state_dim
            self.m = meas_dim

            # Load external library
            if lib_path is None:
                lib_path = "./libcppfilter.so"

            try:
                self.lib = ctypes.CDLL(lib_path)
            except OSError as e:
                raise RuntimeError(f"Failed to load C++ library: {lib_path}") from e

            # Define C function signatures
            self._define_c_functions()

            # Initialize filter handle
            init_func = self.lib.kf_init
            init_func.argtypes = [ctypes.c_int, ctypes.c_int]
            init_func.restype = ctypes.c_void_p

            self.handle = init_func(ctypes.c_int(state_dim),
                                   ctypes.c_int(meas_dim))

            if not self.handle:
                raise RuntimeError("Failed to initialize C++ filter")

            # Cache current state (C++ ownership)
            self.x = np.zeros(state_dim)
            self.P = np.eye(state_dim)

        def _define_c_functions(self) -> None:
            """Define ctypes signatures for C functions."""
            # kf_predict(handle, F, dt, Q)
            predict_func = self.lib.kf_predict
            predict_func.argtypes = [
                ctypes.c_void_p,
                np.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),
                ctypes.c_double,
                np.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS')
            ]
            predict_func.restype = ctypes.c_int
            self._predict_c = predict_func

            # kf_update(handle, z, H, R)
            update_func = self.lib.kf_update
            update_func.argtypes = [
                ctypes.c_void_p,
                np.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),
                np.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),
                np.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS')
            ]
            update_func.restype = ctypes.c_int
            self._update_c = update_func

            # kf_get_state(handle, x_out, P_out)
            get_state_func = self.lib.kf_get_state
            get_state_func.argtypes = [
                ctypes.c_void_p,
                np.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),
                np.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS')
            ]
            get_state_func.restype = ctypes.c_int
            self._get_state_c = get_state_func

        def predict(self, F: NDArray[np.floating[Any]],
                   Q: NDArray[np.floating[Any]],
                   dt: float = 1.0) -> None:
            """
            Call C++ predict function.

            Parameters
            ----------
            F : (n, n) array
                State transition matrix
            Q : (n, n) array
                Process noise covariance
            dt : float
                Time step
            """
            # Ensure arrays are contiguous and double precision
            F_c = np.ascontiguousarray(F, dtype=np.float64)
            Q_c = np.ascontiguousarray(Q, dtype=np.float64)

            ret = self._predict_c(self.handle, F_c, dt, Q_c)
            if ret != 0:
                raise RuntimeError(f"C++ predict failed with code {ret}")

        def update(self, z: NDArray[np.floating[Any]],
                  H: NDArray[np.floating[Any]],
                  R: NDArray[np.floating[Any]]) -> None:
            """Call C++ update function."""
            z_c = np.ascontiguousarray(z, dtype=np.float64)
            H_c = np.ascontiguousarray(H, dtype=np.float64)
            R_c = np.ascontiguousarray(R, dtype=np.float64)

            ret = self._update_c(self.handle, z_c, H_c, R_c)
            if ret != 0:
                raise RuntimeError(f"C++ update failed with code {ret}")

            # Sync state from C++ to Python
            self._sync_state()

        def _sync_state(self) -> None:
            """Fetch current state from C++ implementation."""
            x_out = np.zeros(self.n, dtype=np.float64)
            P_out = np.zeros((self.n, self.n), dtype=np.float64)

            ret = self._get_state_c(self.handle, x_out, P_out)
            if ret != 0:
                raise RuntimeError(f"C++ get_state failed with code {ret}")

            self.x = x_out.copy()
            self.P = P_out.copy()

        def get_state(self) -> Tuple[NDArray[np.floating[Any]],
                                    NDArray[np.floating[Any]]]:
            """Return state estimate and covariance."""
            return self.x.copy(), self.P.copy()

        def __del__(self) -> None:
            """Clean up C++ resources."""
            if hasattr(self, 'lib') and hasattr(self, 'handle'):
                try:
                    cleanup = self.lib.kf_cleanup
                    cleanup.argtypes = [ctypes.c_void_p]
                    cleanup(self.handle)
                except:
                    pass

Integration with TCL Components
================================

Integrate custom filters with existing TCL infrastructure.

Using Dynamic Models
--------------------

.. code-block:: python

    from pytcl.dynamic_models import f_constant_velocity, q_constant_velocity

    class FilterWithTCLModels:
        """
        Custom filter using TCL dynamic models.
        """

        def __init__(self, x0: NDArray[np.floating[Any]],
                     P0: NDArray[np.floating[Any]],
                     num_dims: int = 2):
            self.x = x0.copy()
            self.P = P0.copy()
            self.num_dims = num_dims
            self.n = len(x0)

        def predict(self, dt: float, Q_sigma_a: float = 0.1) -> None:
            """
            Predict using TCL constant velocity model.
            """
            # Get TCL motion models
            F = f_constant_velocity(T=dt, num_dims=self.num_dims)
            Q = q_constant_velocity(T=dt, sigma_a=Q_sigma_a,
                                   num_dims=self.num_dims)

            # Propagate
            self.x = F @ self.x
            self.P = F @ self.P @ F.T + Q

        def update(self, z: NDArray[np.floating[Any]],
                  R: float) -> None:
            """Update with position measurement."""
            # Measurement matrix (position only).
            # f_constant_velocity builds one [position, velocity] block per
            # dimension, so the state is ordered [x, vx, y, vy] -- NOT
            # [x, y, vx, vy]. The position components sit at even indices.
            H = np.zeros((self.num_dims, self.n))
            for i in range(self.num_dims):
                H[i, 2 * i] = 1.0

            # Standard Kalman update
            S = H @ self.P @ H.T + R * np.eye(self.num_dims)
            K = self.P @ H.T @ np.linalg.inv(S)

            self.x = self.x + K @ (z - H @ self.x)
            self.P = (np.eye(self.n) - K @ H) @ self.P

        def get_state(self) -> Tuple[NDArray[np.floating[Any]],
                                    NDArray[np.floating[Any]]]:
            return self.x.copy(), self.P.copy()

.. note::

   State layout: ``f_constant_velocity(T, num_dims=d)`` is block-diagonal with
   one ``[[1, T], [0, 1]]`` block per spatial dimension, so the state vector is
   interleaved as ``[x, vx, y, vy, ...]``. A measurement matrix written for the
   ``[x, y, vx, vy]`` ordering would silently "measure" a velocity component.

Using Coordinate Transformations
---------------------------------

.. code-block:: python

    import numpy as np
    from pytcl.coordinate_systems import cart2sphere, sphere2cart

    # For 3-D sensors, use the shipped conversions. cart2sphere takes 3-D
    # points and returns a (range, azimuth, elevation) triple:
    r, az, el = cart2sphere(np.array([3.0, 4.0, 12.0]), system_type='az-el')

    class PolarCoordinateFilter:
        """
        Filter that works in polar coordinates natively.
        Useful for radar systems.
        """

        def __init__(self, r_init: float, theta_init: float,
                     P_r: float = 1.0, P_theta: float = 0.1):
            """
            Parameters
            ----------
            r_init, theta_init : float
                Initial range and bearing
            P_r, P_theta : float
                Uncertainties
            """
            # State: [r, theta, dr, dtheta]
            self.x = np.array([r_init, theta_init, 0.0, 0.0])
            self.P = np.diag([P_r, P_theta, 0.1, 0.01])

        def predict(self, dt: float) -> None:
            """Predict in polar coordinates."""
            F = np.eye(4)
            F[0, 2] = dt  # r += dr*dt
            F[1, 3] = dt  # theta += dtheta*dt

            Q = np.eye(4) * 0.01

            self.x = F @ self.x
            self.P = F @ self.P @ F.T + Q

        def update_from_cartesian(self, z_cart: NDArray[np.floating[Any]]) -> None:
            """
            Update with 2-D Cartesian measurement.
            Convert to polar, then update.

            Note: cart2sphere handles 3-D points (returning range, azimuth,
            elevation). This filter is planar, so the range/bearing
            conversion is just two lines of numpy.
            """
            # Convert measurement to polar
            r_meas = np.hypot(z_cart[0], z_cart[1])
            theta_meas = np.arctan2(z_cart[1], z_cart[0])

            # Use simplified update (nonlinear)
            H = np.array([[1, 0, 0, 0],  # Measure range
                         [0, 1, 0, 0]])  # Measure bearing

            z_pred = H @ self.x
            z_meas = np.array([r_meas, theta_meas])

            innovation = z_meas - z_pred

            # Handle angle wrapping
            innovation[1] = np.arctan2(np.sin(innovation[1]),
                                      np.cos(innovation[1]))

            R = np.diag([0.5, 0.1])

            S = H @ self.P @ H.T + R
            K = self.P @ H.T @ np.linalg.inv(S)

            self.x = self.x + K @ innovation
            self.P = (np.eye(4) - K @ H) @ self.P

        def get_state_cartesian(self) -> Tuple[NDArray[np.floating[Any]],
                                               NDArray[np.floating[Any]]]:
            """Return state in Cartesian coordinates."""
            # Convert position estimate to Cartesian
            r, theta = self.x[:2]
            x_cart = np.array([r * np.cos(theta),
                              r * np.sin(theta)])
            return x_cart, self.P[:2, :2]

Testing Custom Filters
======================

Implement comprehensive tests for your filter.

For statistical consistency checks -- is the covariance honest about the actual estimation errors? -- do not hand-roll the metrics: :mod:`pytcl.performance_evaluation` ships ``nees`` / ``nees_sequence`` (state-space, needs truth data), ``nis`` / ``nis_sequence`` (measurement-space, works on innovations), and ``consistency_test`` (chi-square bounds on either sequence). The bounded-error and shrinking-trace checks below are coarse smoke tests; ``consistency_test`` is the principled version.

.. code-block:: python

    import unittest
    import numpy as np
    from numpy.testing import assert_allclose, assert_array_less

    class TestCustomFilter(unittest.TestCase):
        """Unit tests for custom filter."""

        def setUp(self) -> None:
            """Initialize test fixtures."""
            self.x0 = np.array([0.0, 0.0])
            self.P0 = np.eye(2) * 0.1
            self.filter = AdaptiveConstantVelocityFilter(self.x0)

        def test_initialization(self) -> None:
            """Test filter initialization."""
            x, P = self.filter.get_state()
            assert_allclose(x[:2], self.x0)
            assert_allclose(np.diag(P)[:2], 1.0, rtol=0.1)

        def test_predict_updates_covariance(self) -> None:
            """Covariance should increase during prediction."""
            self.filter.predict(dt=0.1)
            x, P = self.filter.get_state()

            # Covariance norm should increase
            P_norm = np.linalg.norm(P)
            P0_norm = np.linalg.norm(self.P0)
            self.assertGreater(P_norm, P0_norm)

        def test_update_reduces_uncertainty(self) -> None:
            """Update should reduce uncertainty."""
            self.filter.predict(dt=0.1)
            P_pred = self.filter.get_state()[1]

            # Measurement
            z = np.array([1.0, 1.0])
            self.filter.update(z)
            P_post = self.filter.get_state()[1]

            # Trace of covariance should decrease
            self.assertLess(np.trace(P_post), np.trace(P_pred))

        def test_constant_velocity_tracking(self) -> None:
            """Test steady-state tracking of constant velocity."""
            # Truth: constant velocity
            v_true = np.array([1.0, 0.5])

            errors = []
            for k in range(100):
                pos_true = v_true * 0.1 * k
                z = pos_true + 0.1 * np.random.randn(2)

                self.filter.predict(dt=0.1)
                self.filter.update(z)

                x, _ = self.filter.get_state()
                error = np.linalg.norm(x[:2] - pos_true)
                errors.append(error)

            # Steady-state error should be bounded
            steady_state_error = np.mean(errors[-10:])
            self.assertLess(steady_state_error, 0.5)

        def test_filter_interface_compatibility(self) -> None:
            """Verify filter implements expected interface."""
            # Should have these methods
            self.assertTrue(hasattr(self.filter, 'predict'))
            self.assertTrue(hasattr(self.filter, 'update'))
            self.assertTrue(hasattr(self.filter, 'get_state'))

        def test_numerical_stability(self) -> None:
            """Test for NaN/Inf propagation."""
            for k in range(50):
                self.filter.predict(dt=0.1)
                z = np.array([1e6 * np.random.randn(),
                             1e6 * np.random.randn()])
                self.filter.update(z)

                x, P = self.filter.get_state()

                # Check no NaNs or Infs
                self.assertTrue(np.all(np.isfinite(x)))
                self.assertTrue(np.all(np.isfinite(P)))

**Run tests:**

.. code-block:: bash

    # Run all tests
    python -m pytest test_custom_filter.py -v

    # Run specific test
    python -m pytest test_custom_filter.py::TestCustomFilter::test_constant_velocity_tracking -v

Performance Optimization
========================

Optimize custom filter for production use.

Profiling
---------

.. code-block:: python

    import cProfile
    import pstats
    from io import StringIO

    def profile_filter() -> None:
        """Profile filter performance."""
        pr = cProfile.Profile()
        pr.enable()

        # Run filter for 1000 steps
        filt = AdaptiveConstantVelocityFilter(np.array([0.0, 0.0]))
        for k in range(1000):
            filt.predict(dt=0.1)
            z = np.array([k * 0.1, 0.05 * k]) + 0.1 * np.random.randn(2)
            filt.update(z)

        pr.disable()

        # Print stats
        s = StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats(10)  # Top 10 functions
        print(s.getvalue())

Vectorization with NumPy
------------------------

.. code-block:: python

    def batch_filter_update(filter_obj, measurements: NDArray[np.floating[Any]]) -> None:
        """
        Update filter with batch of measurements.

        More efficient than loop for large batches.
        """
        for z in measurements:
            filter_obj.predict(dt=0.1)
            filter_obj.update(z)

    # Benchmark
    import timeit

    measurements = np.random.randn(10000, 2)

    t = timeit.timeit(
        lambda: batch_filter_update(AdaptiveConstantVelocityFilter(np.array([0.0, 0.0])),
                                   measurements[:100]),
        number=100
    )
    print(f"Average time per update: {t/100/100*1000:.2f} ms")

Numba JIT Compilation
---------------------

For compute-intensive filters, use Numba JIT:

.. code-block:: python

    from numba import njit

    @njit
    def kalman_gain_update(P: NDArray[np.floating[Any]],
                          H: NDArray[np.floating[Any]],
                          R: float) -> NDArray[np.floating[Any]]:
        """
        Compute Kalman gain (JIT compiled).

        Much faster for repeated calls.
        """
        S = H @ P @ H.T + R
        K = P @ H.T / S  # Scalar R
        return K

Practical Workflow: Algorithm to Integration
=============================================

Step-by-step example from research paper to production.

**Scenario:** Implement a novel "Adaptive Fading Memory Filter" from a paper.

1. **Understand Algorithm**

   - Read paper carefully
   - Note all equations and parameters
   - Identify state/measurement dimensions
   - Check numerical considerations

2. **Prototype (NumPy)**

.. code-block:: python

    class AdaptiveFadingMemoryFilter:
        """
        From: Smith et al. (2024) "Adaptive Fading Memory for Tracking"

        Key equations:
        - Fading factor λ_k controls memory length
        - λ_k = 1 - α * innovation_normalized
        - Covariance scaling: P_k *= λ_k
        """

        def __init__(self, x0, P0, alpha: float = 0.01):
            self.x = x0.copy()
            self.P = P0.copy()
            self.alpha = alpha
            self.lambda_k = 1.0

        def predict(self, F, Q):
            P_prev = self.P.copy()

            self.x = F @ self.x
            self.P = self.lambda_k * (F @ self.P @ F.T + Q)

        def update(self, z, H, R):
            innovation = z - H @ self.x
            S = H @ self.P @ H.T + R
            K = self.P @ H.T @ np.linalg.inv(S)

            # Compute innovation normalized squared
            innov_norm_sq = innovation.T @ np.linalg.inv(S) @ innovation

            # Update fading factor
            # λ_k = 1 - α * (γ_k - m) where m = measurement_dim
            m = len(z)
            self.lambda_k = max(0.5, 1.0 - self.alpha * (innov_norm_sq - m))

            self.x = self.x + K @ innovation
            self.P = (np.eye(len(self.x)) - K @ H) @ self.P

3. **Test (Unit Tests)**

.. code-block:: python

    def test_fading_factor_adaptation():
        """Fading factor should respond to anomalies."""
        filt = AdaptiveFadingMemoryFilter(np.array([0.0, 0.0]), np.eye(2))

        # Normal measurement
        filt.predict(np.eye(2), np.eye(2) * 0.01)
        filt.update(np.array([0.0, 0.0]), np.eye(2), np.eye(2) * 0.1)
        lambda1 = filt.lambda_k

        # Large residual -> fading factor should decrease
        filt.predict(np.eye(2), np.eye(2) * 0.01)
        filt.update(np.array([10.0, 10.0]), np.eye(2), np.eye(2) * 0.1)
        lambda2 = filt.lambda_k

        assert lambda2 < lambda1  # Fading factor decreases with anomaly

4. **Integrate with TCL**

.. code-block:: python

    from pytcl.dynamic_models import f_constant_velocity

    class TCLAdaptiveFadingMemoryFilter(AdaptiveFadingMemoryFilter):
        """Integration with TCL models."""

        def predict_cv(self, dt: float, sigma_a: float = 0.1):
            """Predict using TCL constant velocity model."""
            from pytcl.dynamic_models import q_constant_velocity

            F = f_constant_velocity(T=dt, num_dims=2)
            Q = q_constant_velocity(T=dt, sigma_a=sigma_a, num_dims=2)

            self.predict(F, Q)

5. **Validate (Benchmarks)**

.. code-block:: python

    def benchmark_against_standard_kf():
        """Compare performance against baseline."""
        from pytcl.dynamic_estimation.kalman import kf_predict, kf_update

        # Scenario: tracking with variable noise
        errors_adaptive = []
        errors_standard = []

        for trial in range(100):
            adaptive = TCLAdaptiveFadingMemoryFilter(
                np.array([0.0, 0.0, 1.0, 0.0]),
                np.eye(4) * 0.1
            )

            # ... simulate and compare ...

        print(f"Adaptive RMSE: {np.mean(errors_adaptive):.4f}")
        print(f"Standard RMSE: {np.mean(errors_standard):.4f}")

6. **Document & Release**

.. code-block:: python

    """
    Adaptive Fading Memory Filter
    =============================

    Implementation of the algorithm from Smith et al. (2024).

    Features:
    - Automatic memory length adaptation
    - Response to target maneuvers
    - No parameter tuning required

    Example:
        >>> filt = TCLAdaptiveFadingMemoryFilter(x0, P0)
        >>> for t in range(N):
        ...     filt.predict_cv(dt=0.1)
        ...     filt.update(z, H, R)
        ...     x, P = filt.get_state()

    References:
        Smith et al. (2024). Adaptive Fading Memory for Tracking.
    """

Documentation and Type Hints
=============================

Ensure proper type hints and documentation.

.. code-block:: python

    from typing import Tuple, Optional, Union, Dict, Any
    from numpy.typing import NDArray
    import numpy as np

    class WellDocumentedFilter:
        """
        Example of properly documented custom filter.

        This filter demonstrates best practices for documentation,
        type hints, and code organization.

        Attributes
        ----------
        x : NDArray[np.floating[Any]]
            Current state estimate (n,)
        P : NDArray[np.floating[Any]]
            Current covariance matrix (n, n)
        n : int
            State dimension
        m : int
            Measurement dimension
        """

        def __init__(self,
                    x0: NDArray[np.floating[Any]],
                    P0: NDArray[np.floating[Any]]) -> None:
            """
            Initialize filter.

            Parameters
            ----------
            x0 : (n,) array
                Initial state estimate
            P0 : (n, n) array
                Initial error covariance matrix
                Must be positive definite.

            Raises
            ------
            ValueError
                If P0 is not positive definite

            Examples
            --------
            >>> x0 = np.array([0.0, 1.0])
            >>> P0 = np.eye(2) * 0.1
            >>> filt = WellDocumentedFilter(x0, P0)
            """
            self.n = len(x0)
            self.m: Optional[int] = None

            # Validate inputs
            if P0.shape != (self.n, self.n):
                raise ValueError("P0 must be square with shape (n, n)")

            self.x = x0.copy()
            self.P = P0.copy()

        def predict(self,
                   F: NDArray[np.floating[Any]],
                   Q: Optional[NDArray[np.floating[Any]]] = None) -> None:
            """
            Predict state forward using linear transition.

            Updates state estimate and covariance according to:
                x_pred = F @ x
                P_pred = F @ P @ F.T + Q

            Parameters
            ----------
            F : (n, n) array
                State transition matrix
            Q : (n, n) array, optional
                Process noise covariance
                Default: zero process noise

            Raises
            ------
            ValueError
                If F has wrong shape

            Notes
            -----
            The update is performed in-place. For numerical stability,
            ensure Q is positive semi-definite.
            """
            if F.shape != (self.n, self.n):
                raise ValueError(f"F must have shape ({self.n}, {self.n})")

            if Q is None:
                Q = np.zeros((self.n, self.n))

            self.x = F @ self.x
            self.P = F @ self.P @ F.T + Q

        def update(self,
                  z: NDArray[np.floating[Any]],
                  H: NDArray[np.floating[Any]],
                  R: NDArray[np.floating[Any]]) -> Tuple[float, float]:
            """
            Update state using measurement.

            Standard Kalman measurement update:
                K = P @ H.T @ inv(H @ P @ H.T + R)
                x = x + K @ (z - H @ x)
                P = (I - K @ H) @ P

            Parameters
            ----------
            z : (m,) array
                Measurement vector
            H : (m, n) array
                Measurement matrix
            R : (m, m) array
                Measurement noise covariance

            Returns
            -------
            innovation : float
                Innovation (measurement residual) norm
            gain : float
                Kalman gain magnitude (Frobenius norm)

            Raises
            ------
            ValueError
                If dimensions don't match

            Examples
            --------
            >>> innov, gain = filt.update(z, H, R)
            >>> print(f"Innovation: {innov:.4f}, Gain: {gain:.4f}")
            """
            m = len(z)

            if H.shape != (m, self.n):
                raise ValueError(f"H must have shape ({m}, {self.n})")
            if R.shape != (m, m):
                raise ValueError(f"R must have shape ({m}, {m})")

            self.m = m

            # Innovation covariance
            S = H @ self.P @ H.T + R

            # Kalman gain
            try:
                K = self.P @ H.T @ np.linalg.inv(S)
            except np.linalg.LinAlgError:
                K = self.P @ H.T @ np.linalg.pinv(S)

            # Innovation
            innov = z - H @ self.x

            # Update
            self.x = self.x + K @ innov
            self.P = (np.eye(self.n) - K @ H) @ self.P

            return float(np.linalg.norm(innov)), float(np.linalg.norm(K))

        def get_state(self) -> Tuple[NDArray[np.floating[Any]],
                                    NDArray[np.floating[Any]]]:
            """
            Return current state estimate and covariance.

            Returns
            -------
            x : (n,) array
                State estimate
            P : (n, n) array
                Error covariance matrix
            """
            return self.x.copy(), self.P.copy()

Common Pitfalls and Solutions
=============================

**Pitfall 1: Modifying Input Arrays**

.. code-block:: python

    # ❌ Bad: modifies input
    def predict_bad(self, F):
        self.x = F @ self.x  # If F is mutable reference...

    # ✅ Good: makes copy
    def predict_good(self, F):
        F = np.asarray(F, dtype=np.float64)
        self.x = F @ self.x

**Pitfall 2: Not Checking Matrix Dimensions**

.. code-block:: python

    # ❌ Bad: silent failure on dimension mismatch
    def update_bad(self, z, H, R):
        S = H @ self.P @ H.T + R  # No dimension check

    # ✅ Good: validate dimensions
    def update_good(self, z, H, R):
        if H.shape[1] != self.n:
            raise ValueError(f"H must have {self.n} columns")

**Pitfall 3: Numerical Instability**

.. code-block:: python

    # Bad: unstable covariance update
    def update_covariance_bad(self, K, H):
        self.P = (np.eye(self.n) - K @ H) @ self.P

    # Good: Joseph form (more stable)
    def update_covariance_good(self, K, H, R):
        I_KH = np.eye(self.n) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

See Also
========

- :doc:`kalman_filter_tuning` -- Tuning custom filters
- :doc:`advanced_kf_variants` -- Other filter types to extend
- :doc:`performance_optimization` -- Profiling and optimization
- :doc:`troubleshooting` -- Debugging filter issues

**Resources:**

- TCL GitHub: https://github.com/nedonatelli/TCL
- NumPy Documentation: https://numpy.org/doc/
- Type Hints Reference: https://typing.readthedocs.io/
- Numba JIT: https://numba.readthedocs.io/
