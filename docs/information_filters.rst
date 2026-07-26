Information Filters and SRIF
=============================

Information filters represent uncertainty by the **inverse of covariance** rather than covariance itself. This approach offers superior numerical stability, natural handling of weak measurements, and efficient batch processing. This guide covers the information filter, Square Root Information Filter (SRIF), and practical implementations.

.. contents:: Contents
   :local:
   :depth: 3

---

When to Use Information Filters
================================

**Advantages over Kalman Filters:**

1. **Numerical Stability**: Works well with poorly-conditioned systems
   
   - No matrix inversion of covariance (direct accumulation of information)
   - Square Root Information Filter is more robust than standard Kalman

2. **Weak Measurement Handling**: Graceful degradation when measurements have high uncertainty
   
   - Kalman gain can become ill-conditioned with high $R$
   - Information filter naturally handles rank-deficient $R$

3. **Batch Processing**: Natural for off-line data processing and smoothing

4. **Decentralized Systems**: Multiple information sources combine trivially (additive information)

5. **Initial Uncertainty**: Easy to handle unknown initial state (low information)

**When NOT to use:**

- You need real-time performance and computational resources are limited (Kalman is simpler)
- Your system is well-conditioned and measurements are always present (standard Kalman is fine)
- You need fast forward-only filtering of large data streams (Kalman is more standard)

---

Information Filter Fundamentals
================================

**Standard Kalman Filter** represents uncertainty as covariance $\mathbf{P}$:

$$\mathbf{P}_k = E[(\mathbf{x}_k - \hat{\mathbf{x}}_k)(\mathbf{x}_k - \hat{\mathbf{x}}_k)^T]$$

**Information Filter** represents uncertainty by the information matrix:

$$\mathbf{Y}_k = \mathbf{P}_k^{-1}$$

and the information vector:

$$\mathbf{y}_k = \mathbf{P}_k^{-1} \hat{\mathbf{x}}_k = \mathbf{Y}_k \hat{\mathbf{x}}_k$$

Recover the state estimate via:

$$\hat{\mathbf{x}}_k = \mathbf{Y}_k^{-1} \mathbf{y}_k$$

**Information Gain:**

When a measurement $z_k$ arrives with covariance $R_k$, the information gain is:

$$\Delta \mathbf{Y}_k = \mathbf{H}_k^T \mathbf{R}_k^{-1} \mathbf{H}_k$$

$$\Delta \mathbf{y}_k = \mathbf{H}_k^T \mathbf{R}_k^{-1} z_k$$

The information is updated additively:

$$\mathbf{Y}_{k|k} = \mathbf{Y}_{k|k-1} + \Delta \mathbf{Y}_k$$

$$\mathbf{y}_{k|k} = \mathbf{y}_{k|k-1} + \Delta \mathbf{y}_k$$

**Key Insight:** Information matrices add directly (unlike covariances which combine via matrix algebra). This makes decentralized fusion trivial.

---

Standard Information Filter Algorithm
=====================================

Prediction Step
---------------

The prediction is more complex than Kalman because we must invert covariances. Assume discrete linear system:

$$\mathbf{x}_{k+1} = \mathbf{F}_k \mathbf{x}_k + \mathbf{w}_k, \quad \mathbf{w}_k \sim \mathcal{N}(0, \mathbf{Q}_k)$$

Predicted information and state:

$$\mathbf{Y}_{k+1|k} = [\mathbf{F}_k \mathbf{P}_{k|k} \mathbf{F}_k^T + \mathbf{Q}_k]^{-1}$$

$$\mathbf{y}_{k+1|k} = \mathbf{Y}_{k+1|k} \mathbf{F}_k \mathbf{P}_{k|k} \mathbf{y}_{k|k}$$

where $\mathbf{P}_{k|k} = \mathbf{Y}_{k|k}^{-1}$.

**Implementation** (requires inversion—more costly than Kalman):

.. code-block:: python

    import numpy as np
    from scipy.linalg import inv, solve
    
    class InformationFilter:
        """
        Standard Information Filter.
        
        State represented by (information matrix, information vector):
            Y = P^{-1}    (information matrix)
            y = P^{-1} * x_hat  (information vector)
        
        Recover state: x_hat = Y^{-1} * y
        """
        
        def __init__(self, x0: np.ndarray, P0: np.ndarray):
            """
            Initialize information filter.
            
            Parameters
            ----------
            x0 : (n,) array
                Initial state estimate
            P0 : (n, n) array
                Initial covariance
            """
            self.n = len(x0)
            
            # Store covariance for predict step (information filter needs it)
            self.P = P0.copy()
            
            # Information representation
            self.Y = inv(P0)  # Information matrix
            self.y = self.Y @ x0  # Information vector
            
            self.x = x0.copy()
        
        def predict(self, F: np.ndarray, Q: np.ndarray) -> None:
            """
            Prediction step.
            
            Parameters
            ----------
            F : (n, n) array
                State transition matrix
            Q : (n, n) array
                Process noise covariance
            """
            # Covariance prediction (need P for next step)
            # P_pred = F @ P @ F.T + Q
            self.P = F @ self.P @ F.T + Q
            
            # Information prediction
            # Y_pred = (F @ P @ F.T + Q)^{-1}
            self.Y = inv(self.P)
            
            # Recover state and convert to information vector
            self.x = self.Y @ self.y  # Recover x from information rep
        
        def update(self, z: np.ndarray, H: np.ndarray, 
                  R: np.ndarray) -> None:
            """
            Measurement update.
            
            Parameters
            ----------
            z : (m,) array
                Measurement
            H : (m, n) array
                Measurement matrix
            R : (m, m) array
                Measurement noise covariance
            """
            # Information gain from measurement
            R_inv = inv(R)
            
            # Update information matrix and vector (additive!)
            self.Y = self.Y + H.T @ R_inv @ H
            self.y = self.y + H.T @ R_inv @ z
            
            # Recover covariance and state
            self.P = inv(self.Y)
            self.x = self.P @ self.y
        
        def get_state(self) -> tuple:
            """Return current state and covariance."""
            return self.x.copy(), self.P.copy()

**Example: Simple 1D Tracking**

.. code-block:: python

    # 1D position tracking with constant velocity
    x0 = np.array([0.0, 1.0])  # pos, vel
    P0 = np.eye(2) * 0.1
    
    ifilt = InformationFilter(x0, P0)
    
    # Process model
    dt = 0.1
    F = np.array([[1, dt], [0, 1]])
    Q = np.eye(2) * 0.01
    
    # Measurement model (position only)
    H = np.array([[1, 0]])
    R = np.array([[0.5]])
    
    for k in range(20):
        # Predict
        ifilt.predict(F, Q)
        
        # Measurement
        z = np.array([0.1 * k + 0.5 * np.random.randn()])
        
        # Update
        ifilt.update(z, H, R)
        
        x, P = ifilt.get_state()
        if k % 5 == 0:
            print(f"Step {k}: x = {x}, σ = {np.sqrt(np.diag(P))}")

---

Square Root Information Filter (SRIF)
====================================

Standard information filters require matrix inversion at each predict step, which can be numerically unstable. SRIF avoids this by working with the **square root** of the information matrix.

Let $\mathbf{S} = \mathbf{Y}^{1/2}$ (Cholesky factor of information matrix).

**Advantages:**

- No explicit matrix inversion (use QR decomposition)
- Better numerical conditioning (square root of matrix is better-conditioned)
- Works reliably with rank-deficient problems
- Decorrelated measurements are natural

Algorithm: Measurement Update
------------------------------

Given current square root: $\mathbf{S}_k \mathbf{S}_k^T = \mathbf{Y}_k$

New measurement with $\mathbf{H}_{k+1}$, $z_{k+1}$, $R_{k+1}$:

Stack and QR decompose:

$$\begin{bmatrix} \mathbf{S}_k \\ \sqrt{\mathbf{R}_{k+1}^{-1}} \mathbf{H}_{k+1} \end{bmatrix} \leftarrow \text{QR Decomposition}$$

Result: Upper triangular $\mathbf{S}_{k+1}$ (new square root information matrix).

**Implementation:**

.. code-block:: python

    from scipy.linalg import qr, cholesky, solve_triangular, inv
    
    class SquareRootInformationFilter:
        """
        Square Root Information Filter (SRIF).
        
        Maintains square root of information matrix: S where Y = S @ S.T
        More numerically stable than standard information filter.
        """
        
        def __init__(self, x0: np.ndarray, P0: np.ndarray):
            """
            Initialize SRIF.
            
            Parameters
            ----------
            x0 : (n,) array
                Initial state estimate
            P0 : (n, n) array
                Initial covariance (should be positive definite)
            """
            self.n = len(x0)
            self.x = x0.copy()
            
            # Square root of information matrix
            # S @ S.T = Y = P^{-1}
            Y = inv(P0)
            self.S = cholesky(Y, lower=False)  # Upper triangular
            
            # Current covariance
            self.P = P0.copy()
        
        def update_measurement(self, z: np.ndarray, H: np.ndarray,
                              R: np.ndarray) -> None:
            """
            Measurement update via QR decomposition.
            
            Parameters
            ----------
            z : (m,) array
                Measurement
            H : (m, n) array
                Measurement matrix
            R : (m, m) array
                Measurement noise covariance
            """
            m = len(z)
            
            # Compute square root of R^{-1}
            # For stability, use Cholesky if possible
            try:
                R_inv_sq = cholesky(inv(R), lower=False)
            except:
                # Fallback: eigenvalue decomposition
                eigvals, eigvecs = np.linalg.eigh(inv(R))
                eigvals = np.maximum(eigvals, 1e-10)  # Regularize
                R_inv_sq = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            
            # Stacked matrix: [S_k^T; sqrt(R^{-1}) @ H]
            matrix_stacked = np.vstack([
                self.S.T,
                R_inv_sq @ H
            ])
            
            # QR decomposition
            Q, R_upper = qr(matrix_stacked.T)  # QR of transpose
            
            # New square root (transpose back to upper triangular form)
            self.S = R_upper[:self.n, :self.n].T
            if not np.allclose(self.S, np.triu(self.S)):
                # Ensure upper triangular
                self.S = np.triu(self.S)
            
            # Updated information vector
            # y_new = y_old + H.T @ R^{-1} @ z
            Y = self.S.T @ self.S
            y = Y @ self.x + H.T @ inv(R) @ z
            
            # Update state estimate
            self.P = inv(Y)
            self.x = self.P @ y
        
        def predict(self, F: np.ndarray, Q: np.ndarray) -> None:
            """
            Prediction step (requires covariance update).
            
            Parameters
            ----------
            F : (n, n) array
                State transition matrix
            Q : (n, n) array
                Process noise covariance
            """
            # Standard covariance prediction
            self.P = F @ self.P @ F.T + Q
            
            # Update square root of information matrix
            Y_new = inv(self.P)
            self.S = cholesky(Y_new, lower=False)
        
        def get_state(self) -> tuple:
            """Return state estimate and covariance."""
            return self.x.copy(), self.P.copy()

**Example: SRIF vs Standard IF**

.. code-block:: python

    import numpy as np
    from scipy.linalg import inv
    
    # Poorly conditioned problem
    x0 = np.array([1.0, 1.0, 1.0])
    P0 = np.array([
        [1e6, 1e5, 1e4],
        [1e5, 1e6, 1e5],
        [1e4, 1e5, 1e6]
    ])
    
    # Standard IF
    if_filt = InformationFilter(x0, P0)
    
    # SRIF
    srif_filt = SquareRootInformationFilter(x0, P0)
    
    # Measurement matrix
    H = np.eye(3)
    R = np.eye(3) * 0.1
    z = np.array([1.1, 0.9, 1.05])
    
    # Update both
    if_filt.update(z, H, R)
    srif_filt.update_measurement(z, H, R)
    
    x_if, P_if = if_filt.get_state()
    x_srif, P_srif = srif_filt.get_state()
    
    print(f"IF error:   {np.linalg.norm(x_if - x_srif):.2e}")
    print(f"IF cond:    {np.linalg.cond(P_if):.2e}")
    print(f"SRIF cond:  {np.linalg.cond(P_srif):.2e}")

---

Extended Information Filter (EKIF)
==================================

For nonlinear systems, extend information filtering with linearization (similar to EKF from Kalman filters).

.. code-block:: python

    class ExtendedInformationFilter:
        """
        Extended Information Filter for nonlinear systems.
        
        Linearizes around current state estimate:
            x_{k+1} = f(x_k) + w_k
            z_k = h(x_k) + v_k
        """
        
        def __init__(self, x0: np.ndarray, P0: np.ndarray):
            self.n = len(x0)
            self.x = x0.copy()
            self.P = P0.copy()
            
            # Information representation
            self.Y = inv(P0)
            self.y = self.Y @ x0
        
        def predict(self, f_func, F_jacobian, Q: np.ndarray) -> None:
            """
            Nonlinear prediction.
            
            Parameters
            ----------
            f_func : callable
                Nonlinear state function f(x)
            F_jacobian : callable
                Jacobian of f at current x
            Q : (n, n) array
                Process noise covariance
            """
            # Nonlinear prediction
            self.x = f_func(self.x)
            
            # Linearize
            F = F_jacobian(self.x)
            
            # Covariance prediction
            self.P = F @ self.P @ F.T + Q
            
            # Information update
            self.Y = inv(self.P)
            self.y = self.Y @ self.x
        
        def update(self, z: np.ndarray, h_func, H_jacobian, 
                  R: np.ndarray) -> None:
            """
            Nonlinear measurement update.
            
            Parameters
            ----------
            z : (m,) array
                Measurement
            h_func : callable
                Measurement function h(x)
            H_jacobian : callable
                Jacobian of h at current x
            R : (m, m) array
                Measurement noise covariance
            """
            # Linearize measurement
            H = H_jacobian(self.x)
            
            # Information update (additive)
            R_inv = inv(R)
            
            self.Y = self.Y + H.T @ R_inv @ H
            self.y = self.y + H.T @ R_inv @ z
            
            # Recover state
            self.P = inv(self.Y)
            self.x = self.P @ self.y

---

Decorrelated Measurement Processing
===================================

Information filters naturally handle measurements processed one at a time without correlation issues.

**Benefit:** Process high-rate measurements sequentially without loss of numerical precision.

.. code-block:: python

    class DecorrelatedInformationFilter:
        """
        Process measurements one-at-a-time (scalar).
        
        Avoids manipulation of measurement covariance matrix.
        More numerically stable for high-dimensional systems.
        """
        
        def __init__(self, x0: np.ndarray, P0: np.ndarray):
            self.n = len(x0)
            self.x = x0.copy()
            self.P = P0.copy()
            self.Y = inv(P0)
            self.y = self.Y @ x0
        
        def update_scalar(self, z_scalar: float, h_row: np.ndarray,
                         r_scalar: float) -> None:
            """
            Update with scalar (1D) measurement.
            
            Parameters
            ----------
            z_scalar : float
                Scalar measurement value
            h_row : (n,) array
                Single row of measurement matrix (Jacobian for one sensor)
            r_scalar : float
                Scalar measurement variance
            """
            # Information update (simple additive form)
            self.Y = self.Y + (1/r_scalar) * np.outer(h_row, h_row)
            self.y = self.y + (1/r_scalar) * h_row * z_scalar
            
            # Recover state
            self.P = inv(self.Y)
            self.x = self.P @ self.y
        
        def update_batch_scalar(self, measurements: np.ndarray,
                               H_matrix: np.ndarray,
                               R_diag: np.ndarray) -> None:
            """
            Update with multiple scalar measurements (processed one-by-one).
            
            Parameters
            ----------
            measurements : (m,) array
                Measurement values
            H_matrix : (m, n) array
                Measurement matrix
            R_diag : (m,) array
                Diagonal measurement variances
            """
            for i in range(len(measurements)):
                self.update_scalar(measurements[i], H_matrix[i, :], 
                                  R_diag[i])

**Example: Multi-Sensor Fusion**

.. code-block:: python

    # 2D localization with 3 independent distance sensors
    x0 = np.array([0.0, 0.0])  # [x, y]
    P0 = np.eye(2) * 10.0
    
    dif = DecorrelatedInformationFilter(x0, P0)
    
    # Sensor 1: distance to origin
    h1 = np.array([0, 0])  # Simplified: just x position
    r1 = 0.5
    z1 = 1.1
    dif.update_scalar(z1, h1, r1)
    
    # Sensor 2: y measurement
    h2 = np.array([0, 1])
    r2 = 0.5
    z2 = 0.9
    dif.update_scalar(z2, h2, r2)
    
    # Sensor 3: noisy z measurement
    h3 = np.array([0.5, 0.5]) / np.sqrt(0.5)
    r3 = 0.8
    z3 = 1.4
    dif.update_scalar(z3, h3, r3)
    
    x_est, P_est = dif.x, dif.P
    print(f"Estimated position: {x_est}")
    print(f"Uncertainty: {np.sqrt(np.diag(P_est))}")

---

Comparison: Information Filter vs Kalman Filter
================================================

**Numerical Behavior:**

+----------------------------------+-----------------------+---------------------+
| Aspect                           | Kalman Filter         | Information Filter  |
+==================================+=======================+=====================+
| State representation             | x-hat, P (covariance) | y, Y (information)  |
+----------------------------------+-----------------------+---------------------+
| Measurement update               | Multiplicative (K)    | Additive (Y, y)     |
+----------------------------------+-----------------------+---------------------+
| Prediction step                  | Direct (P_pred)       | Requires inversion  |
+----------------------------------+-----------------------+---------------------+
| Weak measurement (R → ∞)         | Kalman gain → 0       | Information → 0     |
+----------------------------------+-----------------------+---------------------+
| Rank-deficient R                 | Ill-conditioned       | Natural             |
+----------------------------------+-----------------------+---------------------+
| Numerical stability              | Good                  | Better (SRIF)       |
+----------------------------------+-----------------------+---------------------+
| Batch processing                 | Awkward               | Natural             |
+----------------------------------+-----------------------+---------------------+
| Decentralized fusion             | Complex               | Trivial (add info)  |
+----------------------------------+-----------------------+---------------------+

**When Information Filter Excels:**

1. **Unknown initial state**: Set $\mathbf{Y}_0 = \epsilon \mathbf{I}$ (weak information)
2. **Intermittent measurements**: Missing measurements = no information update (skip it)
3. **Batch smoothing**: Natural representation for all-at-once processing
4. **Multi-source fusion**: Information from all sensors add directly

.. code-block:: python

    class HybridFilter:
        """
        Use Kalman for forward pass, Information for batch smoothing.
        
        Best of both worlds:
        - Kalman: efficient forward prediction
        - Information: stable batch refinement
        """
        
        def __init__(self, x0: np.ndarray, P0: np.ndarray):
            self.x_list = [x0.copy()]
            self.P_list = [P0.copy()]
            
            self.x = x0.copy()
            self.P = P0.copy()
        
        def run_forward_kalman(self, F_list, Q_list, 
                              z_list, H_list, R_list) -> None:
            """Run Kalman filter forward and store trajectory."""
            for k, (F, Q) in enumerate(zip(F_list, Q_list)):
                # Predict
                self.P = F @ self.P @ F.T + Q
                self.x = F @ self.x
                
                # Measurement available?
                if z_list[k] is not None:
                    H, R, z = H_list[k], R_list[k], z_list[k]
                    
                    # Standard Kalman update
                    S = H @ self.P @ H.T + R
                    K = self.P @ H.T @ inv(S)
                    
                    self.x = self.x + K @ (z - H @ self.x)
                    self.P = (np.eye(len(self.x)) - K @ H) @ self.P
                
                self.x_list.append(self.x.copy())
                self.P_list.append(self.P.copy())
        
        def batch_information_smooth(self) -> None:
            """
            Refine entire trajectory using information filter smoother.
            Similar to RTS but using information representation.
            """
            # Convert to information representation
            Y_list = [inv(P) for P in self.P_list]
            y_list = [Y @ x for Y, x in zip(Y_list, self.x_list)]
            
            # Information smoothing (simplified)
            # In practice, use dedicated batch smoother
            # Here just demonstrating the concept
            print(f"Smoothing {len(self.x_list)} states in batch mode")
            
            return self.x_list, self.P_list

---

Weak Initial Conditions (High Uncertainty)
===========================================

Information filters excel when initial state is highly uncertain.

.. code-block:: python

    class WeakInitialStateIF:
        """
        Handle unknown initial state via weak information.
        
        If x_0 is completely unknown:
        Y_0 = ε * I  (very small information)
        y_0 = 0      (no information about x)
        
        As measurements arrive, information accumulates.
        """
        
        def __init__(self, state_dim: int, epsilon: float = 1e-6):
            """
            Parameters
            ----------
            state_dim : int
                State dimension
            epsilon : float
                Weak information intensity (smaller = more uncertain)
            """
            self.n = state_dim
            
            # Weak initial information
            self.Y = epsilon * np.eye(state_dim)
            self.y = np.zeros(state_dim)
        
        def update_with_measurements(self, z_list, H_list, R_list):
            """Process batch of measurements to initialize state."""
            for z, H, R in zip(z_list, H_list, R_list):
                # Accumulate information
                self.Y = self.Y + H.T @ inv(R) @ H
                self.y = self.y + H.T @ inv(R) @ z
            
            # Recover state estimate
            try:
                P = inv(self.Y)
                x = P @ self.y
                return x, P
            except np.linalg.LinAlgError:
                print("Warning: Information matrix still singular")
                print(f"  Information rank: {np.linalg.matrix_rank(self.Y)}")
                return None, None

**Example: Initialize from measurements**

.. code-block:: python

    # Unknown initial state, but have 5 measurements
    weak_if = WeakInitialStateIF(state_dim=2)
    
    # Process measurements
    z_batch = [np.array([1.0]), np.array([1.1]), 
               np.array([0.9]), np.array([1.05]), np.array([1.02])]
    H_batch = [np.array([[1, 0]]) for _ in z_batch]
    R_batch = [np.array([[0.1]]) for _ in z_batch]
    
    x_init, P_init = weak_if.update_with_measurements(z_batch, 
                                                       H_batch, R_batch)
    print(f"Initialized state: {x_init}")
    print(f"Initialization uncertainty: {np.sqrt(np.diag(P_init))}")

---

Practical Diagnostic Tools
==========================

Monitor information filter health.

.. code-block:: python

    class InformationFilterDiagnostics:
        """Diagnostics for information filters."""
        
        def __init__(self, if_object):
            self.filter = if_object
            self.history = {
                'information_rank': [],
                'condition_numbers': [],
                'information_norms': [],
                'state_norms': []
            }
        
        def log_step(self) -> None:
            """Record diagnostic data."""
            Y, y = self.filter.Y, self.filter.y
            
            # Information matrix diagnostics
            self.history['information_rank'].append(np.linalg.matrix_rank(Y))
            self.history['condition_numbers'].append(np.linalg.cond(Y))
            self.history['information_norms'].append(np.linalg.norm(Y))
            self.history['state_norms'].append(np.linalg.norm(self.filter.x))
        
        def check_degeneracy(self) -> dict:
            """Check if information is sufficient."""
            Y = self.filter.Y
            rank = np.linalg.matrix_rank(Y)
            expected_rank = Y.shape[0]
            
            return {
                'is_degenerate': rank < expected_rank,
                'rank': rank,
                'expected_rank': expected_rank,
                'rank_deficit': expected_rank - rank,
                'condition_number': np.linalg.cond(Y)
            }
        
        def check_information_bounds(self) -> dict:
            """Verify information matrix is reasonable."""
            Y = self.filter.Y
            eigvals = np.linalg.eigvals(Y)
            
            return {
                'min_eigenvalue': np.min(np.abs(eigvals)),
                'max_eigenvalue': np.max(np.abs(eigvals)),
                'all_positive': np.all(eigvals > 0),
                'is_spd': np.all(eigvals > 1e-10)  # Check positive definite
            }
        
        def plot_diagnostics(self):
            """Generate diagnostic plots."""
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            
            # Information rank
            axes[0, 0].plot(self.history['information_rank'], alpha=0.7)
            axes[0, 0].set_title('Information Matrix Rank')
            axes[0, 0].set_ylabel('rank(Y)')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Condition number
            axes[0, 1].semilogy(self.history['condition_numbers'], alpha=0.7)
            axes[0, 1].set_title('Information Matrix Conditioning')
            axes[0, 1].set_ylabel('cond(Y)')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Information norm (total information content)
            axes[1, 0].plot(self.history['information_norms'], alpha=0.7)
            axes[1, 0].set_title('Information Accumulation')
            axes[1, 0].set_ylabel('||Y||_F')
            axes[1, 0].grid(True, alpha=0.3)
            
            # State norm
            axes[1, 1].plot(self.history['state_norms'], alpha=0.7)
            axes[1, 1].set_title('State Magnitude')
            axes[1, 1].set_ylabel('||x||')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            return fig

---

Batch Information Filter (BIF)
==============================

Process entire measurement batch at once for optimal smoothing.

.. code-block:: python

    class BatchInformationFilter:
        """
        Batch information filter for offline processing.
        
        Process all measurements simultaneously for global optimality.
        """
        
        def __init__(self, x0: np.ndarray, P0: np.ndarray):
            self.x_init = x0.copy()
            self.P_init = P0.copy()
        
        def process_batch(self, measurements, H_list, R_list, 
                         dynamics=None, Q_list=None):
            """
            Process entire batch of measurements.
            
            Parameters
            ----------
            measurements : list of arrays
                Measurement sequence
            H_list : list of arrays
                Measurement matrices
            R_list : list of arrays
                Measurement noise covariances
            dynamics : callable, optional
                State transition function f(x)
            Q_list : list of arrays, optional
                Process noise covariances
                
            Returns
            -------
            x_batch : (N, n) array
                Batch state estimates
            P_batch : list of (n, n) arrays
                Batch covariances
            """
            N = len(measurements)  # Number of timesteps
            n = len(self.x_init)   # State dimension
            
            # Initialize information representation
            Y_accum = inv(self.P_init)
            y_accum = Y_accum @ self.x_init
            
            # Accumulate all measurements
            for z, H, R in zip(measurements, H_list, R_list):
                R_inv = inv(R)
                Y_accum = Y_accum + H.T @ R_inv @ H
                y_accum = y_accum + H.T @ R_inv @ z
            
            # Recover global estimate
            P_batch = inv(Y_accum)
            x_batch = P_batch @ y_accum
            
            # If dynamics provided, do batch trajectory
            if dynamics is not None:
                x_trajectory = self._batch_trajectory_estimate(
                    x_batch, P_batch, dynamics, Q_list
                )
            else:
                x_trajectory = x_batch
            
            return x_trajectory, P_batch
        
        def _batch_trajectory_estimate(self, x_batch, P_batch, 
                                      dynamics, Q_list):
            """
            Refine trajectory using dynamics constraints.
            (Simplified version)
            """
            return x_batch

---

Tuning Guidelines
=================

**Choose Information Filter When:**

1. ✅ System is poorly conditioned (high aspect ratios in state/measurement spaces)
2. ✅ Multiple sensors with different measurement rates/uncertainties
3. ✅ Batch processing or offline smoothing
4. ✅ Unknown or weak initial state
5. ✅ Need numerical robustness

**Parameter Selection:**

- **Weak initial information**: $\epsilon = 10^{-6}$ to $10^{-8} \times \text{trace}(\mathbf{Y}_{\text{nominal}})$
- **SRIF vs Standard IF**: Use SRIF always if numerics are critical
- **Regularization**: Add $\delta \mathbf{I}$ to prevent singularity:
  
  $$\mathbf{Y}_k \leftarrow \mathbf{Y}_k + \delta \mathbf{I}, \quad \delta = 10^{-10}$$

---

Common Pitfalls
===============

1. **Singular Information Matrix**: Measurements don't constrain all states
   
   - *Fix*: Monitor rank, add weak priors, ensure observable states

2. **Numerical Inversion**: Computing $\mathbf{Y}^{-1}$ becomes ill-conditioned
   
   - *Fix*: Use SRIF or regularize (add regularization term)

3. **Weak Measurements Cause Overflow**: Information accumulates without bound
   
   - *Fix*: Use information bounds, monitor condition numbers

4. **Loss of Process Dynamics**: SRIF predict step is complex
   
   - *Fix*: Hybrid approach (Kalman forward, Information backward)

---

See Also
========

- [Kalman Filter Tuning](kalman_filter_tuning.rst) — Standard filtering approach
- [Smoothing Algorithms](smoothing.rst) — RTS smoother for trajectory refinement
- [Adaptive Filtering](adaptive_filtering.rst) — Tuning parameters online
- [Troubleshooting Guide](troubleshooting.rst) — Numerical issues diagnosis

**References:**

- Bierman (1977) — *Factorization Methods for Discrete Sequential Estimation* — Foundational SRIF work
- Bar-Shalom, Li, Kirubarajan (2001) — *Estimation with Applications to Tracking and Navigation*
- Kailath, Sayed, Hassibi (2000) — *Linear Estimation* — Comprehensive treatment with information filters
- Dyer & McReynolds (1969) — *Extension of Square Root Filtering to Include Process Noise* — Original SRIF
