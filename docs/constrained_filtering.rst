Constrained State Estimation
=============================

Overview
--------

The Constrained Extended Kalman Filter (CEKF) enforces state constraints
during filtering, keeping estimates physically valid. Common applications
include:

- **Position bounds**: Aircraft within geofence, satellite orbits in valid regions
- **Velocity limits**: Maximum speed constraints for vehicles
- **Proportional constraints**: Mixture fractions that sum to unity
- **Momentum conservation**: Constrained collision dynamics

The prediction step is a standard EKF prediction. Constraints are enforced
in the update step: after the usual EKF update, any violated constraints are
handled by projecting the estimate onto the constraint surface with a
covariance-weighted Lagrange multiplier method (Simon, 2010).

Constraint Types
----------------

**Equality Constraints** (g(x) = 0)
  Must be satisfied exactly:

  .. code-block:: python

     from pytcl.dynamic_estimation.kalman import ConstraintFunction

     # Example: Mixture fractions sum to 1
     def mixture_constraint(x):
         # g(x) = 0 means x[0] + x[1] + x[2] = 1
         return x[0] + x[1] + x[2] - 1.0

     equality = ConstraintFunction(mixture_constraint, constraint_type="equality")

**Inequality Constraints** (g(x) <= 0)
  Define feasible regions:

  .. code-block:: python

     import numpy as np

     # Example: Position within 10 m of origin
     def position_bound(x):
         # g(x) <= 0 means sqrt(x[0]^2 + x[1]^2) <= 10
         return np.sqrt(x[0]**2 + x[1]**2) - 10.0

     inequality = ConstraintFunction(position_bound)  # "inequality" is the default

Basic Usage
-----------

.. code-block:: python

   import numpy as np
   from pytcl.dynamic_estimation.kalman import (
       ConstraintFunction,
       constrained_ekf_predict,
       constrained_ekf_update,
   )

   # 1. Define a constraint: position must stay at or below 100 m
   def constraint_fn(x):
       return x[0] - 100.0  # g(x) <= 0 means x[0] <= 100

   constraint = ConstraintFunction(constraint_fn)

   # 2. Initialize filter
   x0 = np.array([99.5, 1.0])  # Initial state [position, velocity]
   P0 = np.diag([1.0, 0.01])   # Initial covariance

   # 3. Define dynamics and measurement models
   dt = 0.1

   def f(x):
       """Constant-velocity dynamics."""
       return np.array([x[0] + x[1] * dt, x[1]])

   F = np.array([[1.0, dt], [0.0, 1.0]])  # Jacobian of f (constant here)

   def h(x):
       """Measure position only."""
       return np.array([x[0]])

   H = np.array([[1.0, 0.0]])  # Jacobian of h

   # 4. Prediction step (standard EKF prediction; constraints are not
   #    enforced here -- enforcement happens in the update step)
   Q = np.diag([0.001, 0.0001])
   pred = constrained_ekf_predict(x0, P0, f, F, Q)

   # 5. Update step with constraint enforcement
   z = np.array([100.4])  # Measurement pulls the estimate past the bound
   R = np.array([[0.1]])
   upd = constrained_ekf_update(pred.x, pred.P, z, h, H, R, constraints=[constraint])

   print(f"Unconstrained prediction: {pred.x[0]:.3f}")
   print(f"Constrained update:       {upd.x[0]:.3f}")

Output (the update is clipped back to the constraint surface)::

   Unconstrained prediction: 99.600
   Constrained update:       100.000

Both functions return the same named tuples as the unconstrained filters:
``constrained_ekf_predict`` returns a ``KalmanPrediction`` with fields
``x`` and ``P``; ``constrained_ekf_update`` returns a ``KalmanUpdate`` with
fields ``x``, ``P``, ``y`` (innovation), ``S`` (innovation covariance),
``K`` (gain), and ``likelihood``.

Class-Based API
---------------

For repeated use, ``ConstrainedEKF`` keeps a persistent constraint list:

.. code-block:: python

   from pytcl.dynamic_estimation.kalman import ConstrainedEKF

   cekf = ConstrainedEKF()
   cekf.add_constraint(constraint)

   pred = cekf.predict(x0, P0, f, F, Q)
   upd = cekf.update(pred.x, pred.P, z, h, H, R)

Advanced Constraint Handling
-----------------------------

**Multiple Constraints**

Combine equality and inequality constraints:

.. code-block:: python

   # Constraint 1: Position >= 0
   def pos_lower(x):
       return -x[0]

   # Constraint 2: Position <= 100
   def pos_upper(x):
       return x[0] - 100.0

   # Constraint 3: Velocity must be positive
   def vel_positive(x):
       return -x[1]

   constraints = [
       ConstraintFunction(pos_lower),
       ConstraintFunction(pos_upper),
       ConstraintFunction(vel_positive),
   ]

**Analytical Jacobians**

By default the constraint Jacobian is computed by numerical differentiation.
For better performance and accuracy, provide it via the ``G`` argument:

.. code-block:: python

   def constraint_jacobian(x):
       """Jacobian of the constraint function, shape (1, n)."""
       return np.array([[1.0, 0.0]])  # dg/dx for a linear constraint

   constraint = ConstraintFunction(constraint_fn, G=constraint_jacobian)

Real-World Example: Geofenced Vehicle
--------------------------------------

Estimate vehicle position and velocity while respecting a rectangular boundary:

.. code-block:: python

   import numpy as np
   from pytcl.dynamic_estimation.kalman import (
       ConstraintFunction,
       constrained_ekf_predict,
       constrained_ekf_update,
   )

   # State: [x, y, vx, vy]
   x = np.array([50.0, 50.0, 1.0, 0.5])
   P = np.eye(4)

   # Define geofence: 0 <= x <= 100, 0 <= y <= 100
   geofence_constraints = [
       ConstraintFunction(lambda x: -x[0]),        # x >= 0
       ConstraintFunction(lambda x: x[0] - 100),   # x <= 100
       ConstraintFunction(lambda x: -x[1]),        # y >= 0
       ConstraintFunction(lambda x: x[1] - 100),   # y <= 100
   ]

   # Constant-velocity dynamics
   dt = 0.1

   def dynamics(x):
       x_new = x.copy()
       x_new[0] += x[2] * dt  # x += vx * dt
       x_new[1] += x[3] * dt  # y += vy * dt
       return x_new

   F = np.eye(4)
   F[0, 2] = dt
   F[1, 3] = dt

   # Process and measurement noise
   Q = np.diag([0.001, 0.001, 0.0001, 0.0001])
   R = np.eye(2) * 0.01

   # Measurement: [x, y] positions
   z = np.array([50.5, 49.8])

   def measurement(x):
       return x[:2]

   H = np.zeros((2, 4))
   H[0, 0] = 1.0
   H[1, 1] = 1.0

   # Prediction
   pred = constrained_ekf_predict(x, P, dynamics, F, Q)

   # Update (constraints enforced here)
   upd = constrained_ekf_update(
       pred.x, pred.P, z, measurement, H, R,
       constraints=geofence_constraints,
   )
   x, P = upd.x, upd.P

Constraint Satisfaction Properties
-----------------------------------

The CEKF provides:

1. **Feasibility**: After each update, violated constraints are projected
   back onto the constraint surface (to a small numerical tolerance)
2. **Optimality**: The projection minimizes the covariance-weighted distance
   to the unconstrained estimate, subject to the linearized constraints
3. **Stability**: The projected covariance is re-symmetrized and its
   eigenvalues floored, keeping it positive definite

Trade-offs:

- Computational cost grows with state dimension, O(n^3) per step
- Nonlinear constraints are handled by iterating the linearized projection
  (up to 10 internal iterations)
- The prediction step is unconstrained; a prediction may leave the feasible
  region until the next update
- Constraint infeasibility indicates modeling errors

Troubleshooting
---------------

**Constraint Infeasibility**
  If constraints cannot be satisfied, check:

  - Constraint logic (bounds are achievable)
  - Initial state satisfies all constraints
  - Measurement noise is reasonable

**Covariance Growth**
  If uncertainty grows despite measurements:

  - Verify measurement function h(x) is correct
  - Check measurement noise R scaling
  - Ensure constraints don't over-tighten estimates

**Divergence**
  Filter diverges despite valid setup:

  - Add process noise Q
  - Check your analytical constraint Jacobian: leave ``G=None`` so
    ``ConstraintFunction`` differentiates ``g`` numerically, and compare
    the two results

See Also
~~~~~~~~

- :doc:`getting_started` - Basic filter usage
- :doc:`adaptive_filtering` - Adaptive constraint handling
- :ref:`Constrained Extended Kalman Filter <constrained-ekf>` - API Reference
