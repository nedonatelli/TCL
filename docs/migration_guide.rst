MATLAB to Python Migration Guide
================================

This guide helps users transition from the original MATLAB Tracker Component Library
to the Python port (pyTCL/nrl-tracker).

Installation
------------

.. code-block:: bash

   pip install nrl-tracker

For optional features:

.. code-block:: bash

   pip install nrl-tracker[astronomy]    # Orbital mechanics with astropy
   pip install nrl-tracker[geodesy]      # Advanced geodetic functions
   pip install nrl-tracker[signal]       # Wavelet transforms
   pip install nrl-tracker[all]          # Everything

Naming Conventions
------------------

Function names follow Python conventions (snake_case) instead of MATLAB's mixed camelCase:

.. list-table:: Function Name Mapping
   :header-rows: 1
   :widths: 40 40 20

   * - MATLAB
     - Python
     - Module
   * - ``Cart2Sphere``
     - ``cart2sphere``
     - ``coordinate_systems``
   * - ``spher2Cart``
     - ``sphere2cart``
     - ``coordinate_systems``
   * - ``Cart2Ellipse``
     - ``ecef2geodetic``
     - ``coordinate_systems``
   * - ``KalmanUpdate``
     - ``kf_update``
     - ``dynamic_estimation``
   * - ``discKalPred``
     - ``kf_predict``
     - ``dynamic_estimation``
   * - ``EKFUpdate``
     - ``ekf_update``
     - ``dynamic_estimation``
   * - ``UKFUpdate``
     - ``ukf_update``
     - ``dynamic_estimation``
   * - ``cubKalUpdate``
     - ``ckf_update``
     - ``dynamic_estimation``
   * - ``FPolyKal``
     - ``f_constant_velocity``
     - ``dynamic_models``
   * - ``QPolyKal``
     - ``q_constant_velocity``
     - ``dynamic_models``
   * - ``assign2D``
     - ``assign2d``
     - ``assignment_algorithms``
   * - ``assign2DHungarian``
     - ``hungarian``
     - ``assignment_algorithms``

Import Structure
----------------

MATLAB (flat namespace):

.. code-block:: matlab

   % MATLAB - all functions in path
   F = FPolyKal(T, 4, 1);
   [xPred, PPred] = discKalPred(x, P, F, Q);

Python (hierarchical modules):

.. code-block:: python

   # Python - import from modules
   import numpy as np
   from pytcl.dynamic_models import f_constant_velocity, q_constant_velocity
   from pytcl.dynamic_estimation import kf_predict

   x = np.zeros(4)  # [x, vx, y, vy]
   P = np.eye(4) * 100
   F = f_constant_velocity(T=1.0, num_dims=2)
   Q = q_constant_velocity(T=1.0, sigma_a=0.1, num_dims=2)
   pred = kf_predict(x, P, F, Q)

   # Or import the whole module
   import pytcl.dynamic_estimation as de
   pred = de.kf_predict(x, P, F, Q)

Return Values
-------------

MATLAB uses multiple output arguments; Python uses named tuples. Note the
argument order: MATLAB ``KalmanUpdate`` takes ``R`` before ``H``, while
``kf_update`` takes ``H`` before ``R`` (and requires both):

MATLAB:

.. code-block:: matlab

   [xUpdate, PUpdate, innov, Pzz, W] = KalmanUpdate(xPred, PPred, z, R, H);

Python:

.. code-block:: python

   from pytcl.dynamic_estimation import kf_update

   H = np.array([[1.0, 0, 0, 0], [0, 0, 1.0, 0]])
   R = np.eye(2) * 10
   z = np.array([1.0, 2.0])

   result = kf_update(pred.x, pred.P, z, H, R)
   # kf_update returns a 6-field NamedTuple; MATLAB names are renamed:
   # innov -> y, Pzz -> S, W (gain) -> K
   x_update = result.x
   P_update = result.P
   innovation = result.y
   S = result.S
   gain = result.K
   likelihood = result.likelihood

Array Indexing
--------------

MATLAB uses 1-based indexing; Python/NumPy uses 0-based:

.. code-block:: matlab

   % MATLAB
   x = [1, 2, 3, 4, 5];
   first = x(1);      % 1
   last = x(end);     % 5
   subset = x(2:4);   % [2, 3, 4]

.. code-block:: python

   # Python
   import numpy as np
   x = np.array([1, 2, 3, 4, 5])
   first = x[0]       # 1
   last = x[-1]       # 5
   subset = x[1:4]    # [2, 3, 4]

Matrix Operations
-----------------

Most operations are similar, but some differ:

.. list-table:: Matrix Operations
   :header-rows: 1
   :widths: 30 35 35

   * - Operation
     - MATLAB
     - Python (NumPy)
   * - Matrix multiply
     - ``A * B``
     - ``A @ B``
   * - Element-wise multiply
     - ``A .* B``
     - ``A * B``
   * - Transpose
     - ``A'`` or ``A.'``
     - ``A.T``
   * - Inverse
     - ``inv(A)``
     - ``np.linalg.inv(A)``
   * - Solve Ax=b
     - ``A \ b``
     - ``np.linalg.solve(A, b)``
   * - Concatenate horizontal
     - ``[A, B]``
     - ``np.hstack([A, B])``
   * - Concatenate vertical
     - ``[A; B]``
     - ``np.vstack([A, B])``
   * - Identity matrix
     - ``eye(n)``
     - ``np.eye(n)``
   * - Zeros matrix
     - ``zeros(m, n)``
     - ``np.zeros((m, n))``
   * - Diagonal matrix
     - ``diag(v)``
     - ``np.diag(v)``

Example Migration: Kalman Filter
--------------------------------

MATLAB:

.. code-block:: matlab

   % Initialize
   x = [0; 0; 0; 0];  % [x, vx, y, vy]
   P = eye(4) * 100;

   % Motion model
   T = 1.0;  % time step
   F = FPolyKal(T, 4, 1);  % 2D constant velocity (xDim=4, order=1)
   q = 0.1;  % process noise
   Q = QPolyKal(T, 4, 1, q);

   % Measurement model
   H = [1, 0, 0, 0; 0, 0, 1, 0];  % position only
   R = eye(2) * 10;

   % Measurements: one column per scan
   measurements = [1.0, 2.1, 3.2; 2.0, 4.2, 6.1];

   % Filter loop
   for k = 1:size(measurements, 2)
       % Predict
       [xPred, PPred] = discKalPred(x, P, F, Q);

       % Update (note: R before H in MATLAB)
       z = measurements(:, k);
       [x, P] = KalmanUpdate(xPred, PPred, z, R, H);
   end

Python:

.. code-block:: python

   import numpy as np
   from pytcl.dynamic_estimation import kf_predict, kf_update
   from pytcl.dynamic_models import f_constant_velocity, q_constant_velocity

   # Initialize
   x = np.array([0.0, 0.0, 0.0, 0.0])  # [x, vx, y, vy]
   P = np.eye(4) * 100

   # Motion model
   T = 1.0  # time step
   F = f_constant_velocity(T=T, num_dims=2)  # 2D constant velocity
   Q = q_constant_velocity(T=T, sigma_a=0.1, num_dims=2)

   # Measurement model
   H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])  # position only
   R = np.eye(2) * 10

   # Measurements: one row per scan
   measurements = np.array([[1.0, 2.0], [2.1, 4.2], [3.2, 6.1]])

   # Filter loop
   for z in measurements:
       # Predict
       pred = kf_predict(x, P, F, Q)

       # Update
       upd = kf_update(pred.x, pred.P, z, H, R)
       x, P = upd.x, upd.P

Example Migration: Coordinate Conversion
----------------------------------------

MATLAB:

.. code-block:: matlab

   % Cartesian to spherical (returns a stacked [r; az; el] point)
   cartPoint = [1000; 2000; 3000];
   sphPoint = Cart2Sphere(cartPoint);

   % Back to Cartesian
   cartBack = spher2Cart(sphPoint);

   % Geodetic to ECEF
   lat = 40.7128 * pi/180;  % NYC latitude
   lon = -74.0060 * pi/180;
   alt = 10;  % meters
   ecef = ellips2Cart([lat; lon; alt]);

Python:

.. code-block:: python

   import numpy as np
   from pytcl.coordinate_systems import (
       cart2sphere, sphere2cart,
       geodetic2ecef, ecef2geodetic
   )

   # Cartesian to spherical: returns a (r, az, el) tuple of arrays
   # instead of a stacked point; a system_type keyword selects the
   # angle convention ('standard', 'az-el', or 'range-az-el')
   cart_point = np.array([1000, 2000, 3000])
   r, az, el = cart2sphere(cart_point)

   # Back to Cartesian
   cart_back = sphere2cart(r, az, el)

   # Geodetic to ECEF
   lat = np.radians(40.7128)  # NYC latitude
   lon = np.radians(-74.0060)
   alt = 10  # meters
   ecef = geodetic2ecef(lat, lon, alt)

Example Migration: Data Association
-----------------------------------

MATLAB:

.. code-block:: matlab

   % Cost matrix (tracks x measurements)
   C = [10, 5, 13; 3, 15, 8; 12, 7, 9];

   % 2D assignment (Jonker-Volgenant)
   [col4row, row4col, gain] = assign2D(C);
   % col4row is 1-based; 0 marks an unassigned row

Python:

.. code-block:: python

   import numpy as np
   from pytcl.assignment_algorithms import (
       assign2d, hungarian, gated_gnn_association
   )

   # Cost matrix (tracks x measurements)
   C = np.array([[10, 5, 13], [3, 15, 8], [12, 7, 9]], dtype=float)

   # 2D assignment: 0-based index pairs plus explicit absence
   result = assign2d(C)
   rows = result.row_indices
   cols = result.col_indices
   cost = result.cost
   # result.unassigned_rows / result.unassigned_cols instead of 0 sentinels

   # hungarian returns a plain 3-tuple
   row_ind, col_ind, cost = hungarian(C)

   # Gated GNN in one call (gating + assignment; no single
   # MATLAB TCL equivalent)
   track_preds = np.array([[10.0, 20.0], [30.0, 40.0]])
   track_covs = np.array([np.eye(2) * 4 for _ in range(2)])
   measurements = np.array([[10.5, 19.8], [30.2, 40.5], [100.0, 100.0]])

   assoc = gated_gnn_association(track_preds, track_covs, measurements)
   # assoc.track_to_measurement[i] is the measurement index for
   # track i, or -1 if unassigned

Module Mapping Reference
------------------------

.. list-table:: Module Mapping
   :header-rows: 1
   :widths: 40 60

   * - MATLAB Folder
     - Python Module
   * - ``Coordinate_Systems/``
     - ``pytcl.coordinate_systems``
   * - ``Dynamic_Estimation/``
     - ``pytcl.dynamic_estimation``
   * - ``Dynamic_Models/``
     - ``pytcl.dynamic_models``
   * - ``Assignment_Algorithms/``
     - ``pytcl.assignment_algorithms``
   * - ``Mathematical_Functions/``
     - ``pytcl.mathematical_functions``
   * - ``Navigation/``
     - ``pytcl.navigation``
   * - ``Astronomical_Code/``
     - ``pytcl.astronomical``
   * - ``Gravity/``
     - ``pytcl.gravity``
   * - ``Magnetism/``
     - ``pytcl.magnetism``
   * - ``Terrain/``
     - ``pytcl.terrain``
   * - ``Atmosphere_and_Refraction/``
     - ``pytcl.atmosphere`` (atmosphere models only; refraction is unported)
   * - ``Mathematical_Functions/Signal_Processing/``
     - ``pytcl.mathematical_functions.signal_processing``

Common Gotchas
--------------

1. **Row vs Column Vectors**

   MATLAB distinguishes between row and column vectors. NumPy 1D arrays are neither:

   .. code-block:: python

      x = np.array([1, 2, 3])      # Shape: (3,) - neither row nor column
      x_row = x.reshape(1, -1)     # Shape: (1, 3) - row vector
      x_col = x.reshape(-1, 1)     # Shape: (3, 1) - column vector

2. **In-place Operations**

   NumPy arrays can be modified in-place, which may cause unexpected behavior:

   .. code-block:: python

      # This modifies the original!
      x = np.array([1, 2, 3])
      y = x
      y[0] = 999  # x is now [999, 2, 3]

      # Use .copy() to avoid this
      x = np.array([1, 2, 3])
      y = x.copy()
      y[0] = 999  # x is still [1, 2, 3]

3. **Angle Units**

   pyTCL uses **radians** consistently (like MATLAB TCL), but be careful with NumPy:

   .. code-block:: python

      # Convert degrees to radians
      lat_rad = np.radians(40.7128)

      # Convert radians to degrees
      lat_deg = np.degrees(lat_rad)

4. **Matrix vs Array**

   Use ``@`` for matrix multiplication, ``*`` for element-wise:

   .. code-block:: python

      A = np.array([[1, 2], [3, 4]])
      B = np.array([[5, 6], [7, 8]])

      A @ B   # Matrix multiply: [[19, 22], [43, 50]]
      A * B   # Element-wise: [[5, 12], [21, 32]]

5. **Complex Conjugate Transpose**

   MATLAB's ``'`` is conjugate transpose. Use ``.conj().T`` in NumPy:

   .. code-block:: python

      A = np.array([[1+2j, 3+4j]])
      A.T           # Transpose only: [[1+2j], [3+4j]]
      A.conj().T    # Conjugate transpose: [[1-2j], [3-4j]]

Getting Help
------------

- **API Documentation**: https://pytcl.readthedocs.io
- **GitHub Issues**: https://github.com/nedonatelli/TCL/issues
- **Original MATLAB Library**: https://github.com/USNavalResearchLaboratory/TrackerComponentLibrary

Type Hints
----------

pyTCL includes type hints for better IDE support:

.. code-block:: python

   from pytcl.dynamic_estimation import kf_predict, KalmanPrediction
   from numpy.typing import NDArray
   import numpy as np

   def my_filter(
       x: NDArray[np.floating],
       P: NDArray[np.floating],
       F: NDArray[np.floating],
       Q: NDArray[np.floating],
   ) -> KalmanPrediction:
       return kf_predict(x, P, F, Q)

Performance Tips
----------------

1. **Use NumPy vectorized operations** instead of Python loops
2. **Pre-allocate arrays** for large simulations
3. **Use Numba** (included as dependency) for custom numerical functions
4. **Consider scipy.linalg** for specialized linear algebra

.. code-block:: python

   data = np.arange(1000.0)
   some_function = np.sqrt

   # Slow: Python loop
   result = []
   for i in range(1000):
       result.append(some_function(data[i]))

   # Fast: Vectorized
   result = some_function(data)  # If function supports arrays

   # Fast: Pre-allocated
   result = np.zeros(1000)
   for i in range(1000):
       result[i] = some_function(data[i])
