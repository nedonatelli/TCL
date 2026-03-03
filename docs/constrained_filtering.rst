Constrained State Estimation
=============================

Overview
--------

The Constrained Extended Kalman Filter (CEKF) enforces state constraints
during filtering, ensuring estimates remain physically valid. Common applications
include:

- **Position bounds**: Aircraft within geofence, satellite orbits in valid regions
- **Velocity limits**: Maximum speed constraints for vehicles
- **Proportional constraints**: Mixture fractions that sum to unity
- **Momentum conservation**: Constrained collision dynamics

Constraint Types
----------------

**Equality Constraints** (g(x) = 0)
  Must be strictly satisfied at all times:
  
  .. code-block:: python
  
     # Example: Mixture fractions sum to 1
     def mixture_constraint(x):
         # g(x) = 0 means x[0] + x[1] + x[2] = 1
         return (x[0] + x[1] + x[2] - 1.0)

**Inequality Constraints** (g(x) ≤ 0)
  Define feasible regions:
  
  .. code-block:: python
  
     # Example: Position within 10m of origin
     def position_bound(x):
         # g(x) <= 0 means sqrt(x[0]^2 + x[1]^2) <= 10
         return np.sqrt(x[0]**2 + x[1]**2) - 10.0

Basic Usage
-----------

.. code-block:: python

   import numpy as np
   from pytcl.dynamic_estimation.kalman import (
       ConstraintFunction,
       ConstrainedEKF,
       constrained_ekf_predict,
       constrained_ekf_update,
   )

   # 1. Define constraint functions
   def constraint_fn(x):
       """Constraint: position between 0 and 100m"""
       return x[0] - 100.0  # x[0] <= 100
   
   constraint = ConstraintFunction(constraint_fn)

   # 2. Initialize filter
   x0 = np.array([50.0, 1.0])  # Initial state [position, velocity]
   P0 = np.diag([1.0, 0.01])  # Initial covariance

   # 3. Define dynamics and measurement Jacobians
   def F_jacobian(x):
       """State transition Jacobian"""
       dt = 0.1
       return np.array([[1.0, dt], [0.0, 1.0]])
   
   def H_jacobian(x):
       """Measurement Jacobian"""
       return np.array([[1.0, 0.0]])  # Measure position only

   # 4. Prediction step with constraint
   Q = np.diag([0.001, 0.0001])
   x_pred, P_pred = constrained_ekf_predict(
       x0, P0,
       f=lambda x: x + np.array([x[1] * 0.1, 0.0]),  # Dynamics
       F_jac=F_jacobian,
       Q=Q,
       constraints=[constraint]
   )

   # 5. Update step with constraint
   z = np.array([51.0])  # Measurement
   R = np.array([[0.1]])
   x_upd, P_upd = constrained_ekf_update(
       x_pred, P_pred,
       z=z,
       h=lambda x: np.array([x[0]]),  # Measurement function
       H_jac=H_jacobian,
       R=R,
       constraints=[constraint]
   )

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

For better performance, provide constraint Jacobians manually:

.. code-block:: python

   def constraint_jacobian(x):
       """Jacobian of constraint function"""
       return np.array([[1.0, 0.0]])  # dc/dx for linear constraint
   
   constraint = ConstraintFunction(
       constraint_fn,
       jacobian=constraint_jacobian
   )

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

   # Define dynamics with constant velocity model
   def dynamics(x):
       dt = 0.1
       x_new = x.copy()
       x_new[0] += x[2] * dt  # x += vx * dt
       x_new[1] += x[3] * dt  # y += vy * dt
       return x_new

   def dynamics_jacobian(x):
       dt = 0.1
       F = np.eye(4)
       F[0, 2] = dt
       F[1, 3] = dt
       return F

   # Process and measurement noise
   Q = np.diag([0.001, 0.001, 0.0001, 0.0001])
   R = np.eye(2) * 0.01

   # Measurement: [x, y] positions
   z = np.array([50.5, 49.8])
   
   def measurement(x):
       return x[:2]
   
   def measurement_jacobian(x):
       H = np.zeros((2, 4))
       H[0, 0] = 1.0
       H[1, 1] = 1.0
       return H

   # Prediction
   x_pred, P_pred = constrained_ekf_predict(
       x, P,
       f=dynamics,
       F_jac=dynamics_jacobian,
       Q=Q,
       constraints=geofence_constraints
   )

   # Update
   x_upd, P_upd = constrained_ekf_update(
       x_pred, P_pred,
       z=z,
       h=measurement,
       H_jac=measurement_jacobian,
       R=R,
       constraints=geofence_constraints
   )

Constraint Satisfaction Properties
-----------------------------------

The CEKF provides:

1. **Feasibility**: All estimates satisfy constraints exactly
2. **Optimality**: Under Gaussian assumptions and linear constraints, solutions approximate
   constrained MLE
3. **Stability**: Maintains positive-definite covariance matrices through projection

Trade-offs:

- Computational cost increases with constraint complexity O(n³) per step
- Nonlinear constraints may require iteration to converge
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
  - Use lower constraint violation tolerance
  - Add process noise Q
  - Verify Jacobians numerically: `pytcl.automatic_jacobian()`

Performance Considerations
---------------------------

For real-time applications:

.. code-block:: python

   # Use Square-Root CEKF for numerical stability
   from pytcl.dynamic_estimation.kalman import constrained_srckf_predict
   
   # Operates on Cholesky factor S where P = S @ S.T
   x_pred, S_pred = constrained_srckf_predict(
       x, S,
       f=dynamics,
       F_jac=dynamics_jacobian,
       Q=Q,
       constraints=geofence_constraints
   )

See Also
~~~~~~~~

- :doc:`getting_started` - Basic filter usage
- :doc:`adaptive_filtering` - Adaptive constraint handling
- :ref:`api/dynamic_estimation:Constrained Extended Kalman Filter` - API Reference
