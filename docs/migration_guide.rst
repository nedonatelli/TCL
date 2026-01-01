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

Function names follow Python conventions (snake_case) instead of MATLAB's PascalCase:

.. list-table:: Function Name Mapping
   :header-rows: 1
   :widths: 40 40 20

   * - MATLAB
     - Python
     - Module
   * - ``Cart2Sphere``
     - ``cart2sphere``
     - ``coordinate_systems``
   * - ``Sphere2Cart``
     - ``sphere2cart``
     - ``coordinate_systems``
   * - ``Cart2Ellipse``
     - ``cart2ellipse``
     - ``coordinate_systems``
   * - ``KalmanUpdate``
     - ``kf_update``
     - ``dynamic_estimation``
   * - ``KalmanPredict``
     - ``kf_predict``
     - ``dynamic_estimation``
   * - ``EKFUpdate``
     - ``ekf_update``
     - ``dynamic_estimation``
   * - ``UKFUpdate``
     - ``ukf_update``
     - ``dynamic_estimation``
   * - ``CKFUpdate``
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
   * - ``GNNAssociation``
     - ``gnn_association``
     - ``assignment_algorithms``

Import Structure
----------------

MATLAB (flat namespace):

.. code-block:: matlab

   % MATLAB - all functions in path
   F = FPolyKal(T, 2);
   [xPred, PPred] = KalmanPredict(x, P, F, Q);

Python (hierarchical modules):

.. code-block:: python

   # Python - import from modules
   from pytcl.dynamic_models import f_constant_velocity
   from pytcl.dynamic_estimation import kf_predict

   F = f_constant_velocity(T=1.0, num_dims=2)
   pred = kf_predict(x, P, F, Q)

   # Or import the whole module
   import pytcl.dynamic_estimation as de
   pred = de.kf_predict(x, P, F, Q)

Return Values
-------------

MATLAB uses multiple output arguments; Python uses named tuples:

MATLAB:

.. code-block:: matlab

   [xUpdate, PUpdate, innov, S] = KalmanUpdate(xPred, PPred, z, H, R);

Python:

.. code-block:: python

   from pytcl.dynamic_estimation import kf_update

   result = kf_update(x_pred, P_pred, z, H, R)
   # Access fields by name
   x_update = result.x
   P_update = result.P
   innovation = result.innovation
   S = result.S

   # Or unpack directly
   x, P, innov, S = result.x, result.P, result.innovation, result.S

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
   F = FPolyKal(T, 2);  % 2D constant velocity
   q = 0.1;  % process noise
   Q = QPolyKal(T, [q; q], 2);

   % Measurement model
   H = [1, 0, 0, 0; 0, 0, 1, 0];  % position only
   R = eye(2) * 10;

   % Filter loop
   for k = 1:length(measurements)
       % Predict
       [xPred, PPred] = KalmanPredict(x, P, F, Q);

       % Update
       z = measurements(:, k);
       [x, P] = KalmanUpdate(xPred, PPred, z, H, R);
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

   % Cartesian to spherical
   cartPoint = [1000; 2000; 3000];
   [r, az, el] = Cart2Sphere(cartPoint);

   % Back to Cartesian
   cartBack = Sphere2Cart([r; az; el]);

   % Geodetic to ECEF
   lat = 40.7128 * pi/180;  % NYC latitude
   lon = -74.0060 * pi/180;
   alt = 10;  % meters
   ecef = Ellipse2Cart([lat; lon; alt]);

Python:

.. code-block:: python

   import numpy as np
   from pytcl.coordinate_systems import (
       cart2sphere, sphere2cart,
       geodetic2ecef, ecef2geodetic
   )

   # Cartesian to spherical
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

   % Hungarian algorithm
   [assignment, cost] = assign2D(C);

   % GNN with gating
   trackStates = {[10; 20], [30; 40]};
   trackCovs = {eye(2)*4, eye(2)*4};
   measurements = [10.5, 30.2, 100; 19.8, 40.5, 100];

   result = GNNAssociation(trackStates, trackCovs, measurements, H, R);

Python:

.. code-block:: python

   import numpy as np
   from pytcl.assignment_algorithms import (
       hungarian, gnn_association, gated_gnn_association
   )

   # Cost matrix (tracks x measurements)
   C = np.array([[10, 5, 13], [3, 15, 8], [12, 7, 9]], dtype=float)

   # Hungarian algorithm
   result = hungarian(C)
   assignment = result.assignment
   cost = result.cost

   # GNN with gating
   track_preds = np.array([[10.0, 20.0], [30.0, 40.0]])
   track_covs = np.array([np.eye(2) * 4 for _ in range(2)])
   measurements = np.array([[10.5, 19.8], [30.2, 40.5], [100.0, 100.0]])

   result = gated_gnn_association(track_preds, track_covs, measurements)

Module Mapping Reference
------------------------

.. list-table:: Module Mapping
   :header-rows: 1
   :widths: 40 60

   * - MATLAB Folder
     - Python Module
   * - ``Coordinate Systems/``
     - ``pytcl.coordinate_systems``
   * - ``Dynamic Estimation/``
     - ``pytcl.dynamic_estimation``
   * - ``Dynamic Models/``
     - ``pytcl.dynamic_models``
   * - ``Assignment Algorithms/``
     - ``pytcl.assignment_algorithms``
   * - ``Mathematical Functions/``
     - ``pytcl.mathematical_functions``
   * - ``Navigation/``
     - ``pytcl.navigation``
   * - ``Astronomical Code/``
     - ``pytcl.astronomical``
   * - ``Gravity/``
     - ``pytcl.gravity``
   * - ``Magnetism/``
     - ``pytcl.magnetism``
   * - ``Terrain/``
     - ``pytcl.terrain``
   * - ``Atmospheric Models/``
     - ``pytcl.atmosphere``
   * - ``Signal Processing/``
     - ``pytcl.signal_processing``

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

   from pytcl.dynamic_estimation import kf_predict, KFPredictResult
   from numpy.typing import NDArray
   import numpy as np

   def my_filter(
       x: NDArray[np.floating],
       P: NDArray[np.floating],
       F: NDArray[np.floating],
       Q: NDArray[np.floating],
   ) -> KFPredictResult:
       return kf_predict(x, P, F, Q)

Performance Tips
----------------

1. **Use NumPy vectorized operations** instead of Python loops
2. **Pre-allocate arrays** for large simulations
3. **Use Numba** (included as dependency) for custom numerical functions
4. **Consider scipy.linalg** for specialized linear algebra

.. code-block:: python

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
