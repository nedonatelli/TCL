Advanced Kalman Filter Variants
===============================

Beyond the Extended Kalman Filter (EKF) and Unscented Kalman Filter (UKF), advanced variants use sophisticated numerical integration schemes, sigma-point strategies, and ensemble methods to achieve superior accuracy for highly nonlinear systems. This guide covers Cubature Kalman Filter, Centered Difference Kalman Filter, Ensemble Kalman Filter, and their practical applications.

.. contents:: Contents
   :local:
   :depth: 3

---

When to Use Advanced KF Variants
=================================

**Problem Scenarios:**

1. **Highly Nonlinear Systems**: EKF linearization is too coarse
   
   - Launch vehicle ascent (extreme acceleration changes)
   - Radar tracking in near-field (range-dependent nonlinearity)
   - Atmospheric re-entry (drag coefficient varies drastically)

2. **Non-Gaussian Error Distributions**: Measurements are heavy-tailed
   
   - *Solution*: Ensemble or particle filter approaches

3. **Ill-Conditioned Jacobians**: Linearization is numerically unstable
   
   - *Solution*: Cubature (uses numerical integration instead)

4. **Large Computational Budget**: Can afford extra complexity
   
   - Cubature/Sigma-point: Slightly more expensive than EKF
   - Ensemble: Computationally intensive but parallelizable

5. **High-Dimensional Systems**: Need scalable uncertainty propagation
   
   - *Solution*: Ensemble Kalman Filter (scales to 1000s of states)

---

Cubature Kalman Filter (CKF)
=============================

The Cubature Kalman Filter uses **cubature integration rules** to compute transformed mean and covariance through nonlinear functions with high accuracy.

**Key Idea:**

Numerical integration via cubature points (symmetric sampling):

$$\int_{\mathbb{R}^n} g(\mathbf{x}) \mathcal{N}(\mathbf{x}; \mathbf{0}, \mathbf{I}) d\mathbf{x} \approx \frac{1}{2n} \sum_{i=1}^{2n} g(\sqrt{n} \boldsymbol{\xi}_i)$$

where $\boldsymbol{\xi}_i$ are the $2n$ cubature points.

**Advantages:**

- No Jacobian computation required (derivative-free)
- Third-order numerical accuracy for Gaussian inputs
- Better accuracy than UKF for many nonlinear problems
- Symmetric sampling provides numerical stability

Theory: Spherical Cubature Rule
--------------------------------

For an $n$-dimensional system, use $2n$ cubature points on a sphere:

$$\boldsymbol{\xi}_i = \sqrt{n} \mathbf{e}_i, \quad i = 1, \ldots, n$$
$$\boldsymbol{\xi}_{i+n} = -\sqrt{n} \mathbf{e}_i, \quad i = 1, \ldots, n$$

where $\mathbf{e}_i$ are standard basis vectors.

**Transformed Mean:**

$$\hat{\mathbf{y}} = \frac{1}{2n} \sum_{i=1}^{2n} g(\mathbf{m} + \mathbf{S} \boldsymbol{\xi}_i)$$

where $\mathbf{P} = \mathbf{S} \mathbf{S}^T$ (Cholesky decomposition).

**Transformed Covariance:**

$$\mathbf{Q}_{yy} = \frac{1}{2n} \sum_{i=1}^{2n} (g(\mathbf{m} + \mathbf{S} \boldsymbol{\xi}_i) - \hat{\mathbf{y}})(g(\ldots) - \hat{\mathbf{y}})^T$$

Implementation
--------------

.. code-block:: python

    import numpy as np
    from scipy.linalg import cholesky, qr
    
    class CubatureKalmanFilter:
        """
        Cubature Kalman Filter.
        
        Uses spherical cubature rules for derivative-free nonlinear filtering.
        Numerically accurate for smooth nonlinearities without Jacobians.
        """
        
        def __init__(self, x0: np.ndarray, P0: np.ndarray):
            """
            Parameters
            ----------
            x0 : (n,) array
                Initial state estimate
            P0 : (n, n) array
                Initial covariance (should be positive definite)
            """
            self.n = len(x0)
            self.x = x0.copy()
            self.P = P0.copy()
            
            # Generate cubature points (derived from spherical rule)
            self._generate_cubature_points()
        
        def _generate_cubature_points(self) -> None:
            """Generate 2n cubature points on unit sphere."""
            self.num_points = 2 * self.n
            
            # Cubature points: ±sqrt(n) * e_i
            self.points = np.zeros((self.num_points, self.n))
            for i in range(self.n):
                self.points[i, i] = np.sqrt(self.n)
                self.points[self.n + i, i] = -np.sqrt(self.n)
            
            # Weight for each point
            self.weight = 1.0 / self.num_points
        
        def _transform_cubature(self, func, m: np.ndarray, 
                               P: np.ndarray) -> tuple:
            """
            Apply nonlinear transformation via cubature.
            
            Returns
            -------
            (transformed_mean, transformed_cov, cubature_samples)
            """
            # Cholesky decomposition for stable sampling
            try:
                S = cholesky(P, lower=True)
            except np.linalg.LinAlgError:
                # Fallback to regularization
                S = cholesky(P + 1e-6 * np.eye(self.n), lower=True)
            
            # Generate cubature samples
            # x_i = m + S * points_i
            samples = m[:, np.newaxis] + S @ self.points.T
            
            # Propagate through nonlinearity
            transformed = np.array([func(samples[:, i]) for i in range(self.num_points)])
            
            # Compute transformed mean
            mean = self.weight * np.sum(transformed, axis=0)
            
            # Compute transformed covariance
            # Q = E[(y - mean)(y - mean)^T]
            residuals = transformed - mean
            cov = self.weight * (residuals.T @ residuals)
            
            return mean, cov, transformed.T
        
        def predict(self, f_func, Q: np.ndarray) -> None:
            """
            Predict via nonlinear state function.
            
            Parameters
            ----------
            f_func : callable
                Nonlinear state function: x_{k+1} = f(x_k)
            Q : (n, n) array
                Process noise covariance
            """
            # Cubature integration of nonlinear function
            self.x, P_no_noise, _ = self._transform_cubature(f_func, self.x, self.P)
            
            # Add process noise
            self.P = P_no_noise + Q
        
        def update(self, z: np.ndarray, h_func, R: np.ndarray) -> None:
            """
            Update via nonlinear measurement function.
            
            Parameters
            ----------
            z : (m,) array
                Measurement
            h_func : callable
                Measurement function: z = h(x)
            R : (m, m) array
                Measurement noise covariance
            """
            # Propagate state through h
            z_mean, Pzz, z_samples = self._transform_cubature(h_func, 
                                                               self.x, self.P)
            
            # Cross-covariance P_xz
            # Recompute state samples
            S = cholesky(self.P, lower=True)
            x_samples = self.x[:, np.newaxis] + S @ self.points.T
            
            state_residuals = x_samples - self.x[:, np.newaxis]
            z_residuals = z_samples - z_mean[:, np.newaxis]
            
            Pxz = self.weight * (state_residuals @ z_residuals.T)
            
            # Add measurement noise to z covariance
            Pzz = Pzz + R
            
            # Kalman gain and update
            K = Pxz @ np.linalg.inv(Pzz)
            
            self.x = self.x + K @ (z - z_mean)
            self.P = self.P - K @ Pzz @ K.T
        
        def get_state(self) -> tuple:
            """Return state estimate and covariance."""
            return self.x.copy(), self.P.copy()

**Example: Nonlinear Pendulum Tracking**

.. code-block:: python

    # Nonlinear pendulum state: [angle, angular_velocity]
    def f_pendulum(x, dt=0.1, g=9.81, L=1.0):
        """Nonlinear pendulum dynamics."""
        theta, theta_dot = x
        
        # Simple Euler integration (use RK4 in practice)
        theta_new = theta + theta_dot * dt
        theta_dot_new = theta_dot - (g/L) * np.sin(theta) * dt
        
        return np.array([theta_new, theta_dot_new])
    
    def h_pendulum(x):
        """Measure angle (nonlinear: position sensor on arc)."""
        theta = x[0]
        return np.array([np.sin(theta)])  # Nonlinear measurement
    
    # CKF
    x0 = np.array([0.1, 0.0])
    P0 = np.diag([0.01, 0.01])
    ckf = CubatureKalmanFilter(x0, P0)
    
    Q = np.diag([0.001, 0.001])
    R = np.array([[0.01]])
    
    for k in range(100):
        # Predict
        ckf.predict(lambda x: f_pendulum(x, dt=0.1), Q)
        
        # Measurement (true angle + noise)
        true_angle = 0.1 * np.sin(0.1 * k)
        z = np.array([np.sin(true_angle) + 0.1 * np.random.randn()])
        
        # Update
        ckf.update(z, h_pendulum, R)
        
        if k % 25 == 0:
            x_est, P_est = ckf.get_state()
            print(f"Step {k}: θ = {x_est[0]:.4f}, σ_θ = {np.sqrt(P_est[0,0]):.4f}")

---

Sigma-Point Kalman Filters
===========================

**Unscented Kalman Filter (UKF)** and variants use sigma points (deterministic samples) to represent the probability distribution.

Unscented Transform
-------------------

Given mean $\mathbf{m}$ and covariance $\mathbf{P}$, generate $2n+1$ sigma points:

$$\boldsymbol{\sigma}_0 = \mathbf{m}$$

$$\boldsymbol{\sigma}_i = \mathbf{m} + \sqrt{(n + \kappa)} \mathbf{S}_i, \quad i = 1, \ldots, n$$

$$\boldsymbol{\sigma}_{i+n} = \mathbf{m} - \sqrt{(n + \kappa)} \mathbf{S}_i, \quad i = 1, \ldots, n$$

where $\mathbf{S}$ is Cholesky decomposition of $\mathbf{P}$, and $\kappa$ is a tuning parameter.

**Weights:**

$$W_0^m = \frac{\kappa}{n + \kappa}, \quad W_0^c = \frac{\kappa}{n + \kappa} + (1 - \alpha^2 + \beta)$$

$$W_i^m = W_i^c = \frac{1}{2(n + \kappa)}, \quad i = 1, \ldots, 2n$$

.. code-block:: python

    class UnscentedKalmanFilter:
        """
        Unscented Kalman Filter (UKF).
        
        Sigma-point based approach for nonlinear filtering.
        Parameters α, β, κ control sigma point placement.
        """
        
        def __init__(self, x0: np.ndarray, P0: np.ndarray, 
                     alpha: float = 1e-3, beta: float = 2.0, 
                     kappa: float = 0.0):
            """
            Parameters
            ----------
            x0, P0 : Initial state and covariance
            alpha : float
                Spread of sigma points (typically 1e-3 to 1)
                - Smaller α: More concentrated around mean
                - Larger α: More spread out
            beta : float
                Incorporates prior knowledge of distribution
                - 2.0: Optimal for Gaussian
            kappa : float
                Secondary scaling parameter (typically 0 or 3-n)
            """
            self.n = len(x0)
            self.x = x0.copy()
            self.P = P0.copy()
            
            self.alpha = alpha
            self.beta = beta
            self.kappa = kappa
            
            self._compute_weights_and_scaling()
        
        def _compute_weights_and_scaling(self) -> None:
            """Pre-compute weights and scaling factors."""
            self.lambda_ = self.alpha**2 * (self.n + self.kappa) - self.n
            self.gamma = np.sqrt(self.n + self.lambda_)
            
            # Weights for mean
            self.W_m = np.zeros(2 * self.n + 1)
            self.W_m[0] = self.lambda_ / (self.n + self.lambda_)
            self.W_m[1:] = 1.0 / (2 * (self.n + self.lambda_))
            
            # Weights for covariance
            self.W_c = self.W_m.copy()
            self.W_c[0] = self.lambda_ / (self.n + self.lambda_) + \
                         (1 - self.alpha**2 + self.beta)
        
        def _generate_sigma_points(self, m: np.ndarray, 
                                   P: np.ndarray) -> np.ndarray:
            """Generate 2n+1 sigma points."""
            try:
                S = cholesky(P, lower=True)
            except np.linalg.LinAlgError:
                # Regularization
                S = cholesky(P + 1e-6 * np.eye(self.n), lower=True)
            
            sigma = np.zeros((2 * self.n + 1, self.n))
            sigma[0] = m
            
            for i in range(self.n):
                sigma[i + 1] = m + self.gamma * S[:, i]
                sigma[i + self.n + 1] = m - self.gamma * S[:, i]
            
            return sigma
        
        def predict(self, f_func, Q: np.ndarray) -> None:
            """Predict using unscented transform."""
            # Generate sigma points from current estimate
            sigma = self._generate_sigma_points(self.x, self.P)
            
            # Propagate through nonlinearity
            sigma_pred = np.array([f_func(s) for s in sigma])
            
            # Compute predicted mean
            self.x = np.sum(self.W_m[:, np.newaxis] * sigma_pred, axis=0)
            
            # Compute predicted covariance
            residuals = sigma_pred - self.x
            self.P = np.sum([self.W_c[i] * np.outer(residuals[i], residuals[i]) 
                            for i in range(len(sigma_pred))], axis=0) + Q
        
        def update(self, z: np.ndarray, h_func, R: np.ndarray) -> None:
            """Update using unscented transform."""
            # Generate sigma points
            sigma = self._generate_sigma_points(self.x, self.P)
            
            # Propagate through measurement function
            z_sigma = np.array([h_func(s) for s in sigma])
            
            # Predicted measurement mean
            z_pred = np.sum(self.W_m[:, np.newaxis] * z_sigma, axis=0)
            
            # Measurement covariance
            z_residuals = z_sigma - z_pred
            Pzz = np.sum([self.W_c[i] * np.outer(z_residuals[i], z_residuals[i]) 
                         for i in range(len(z_sigma))], axis=0) + R
            
            # Cross-covariance
            x_residuals = sigma - self.x
            Pxz = np.sum([self.W_c[i] * np.outer(x_residuals[i], z_residuals[i]) 
                         for i in range(len(sigma))], axis=0)
            
            # Kalman gain and update
            K = Pxz @ np.linalg.inv(Pzz)
            
            self.x = self.x + K @ (z - z_pred)
            self.P = self.P - K @ Pzz @ K.T
        
        def get_state(self) -> tuple:
            """Return state and covariance."""
            return self.x.copy(), self.P.copy()

---

Centered Difference Kalman Filter
==================================

The Centered Difference Kalman Filter (CDKF) uses **finite differences** instead of derivatives to linearize the system locally.

**Key Idea:**

Approximate Jacobian via central differences:

$$\mathbf{F}_{approx}[i,j] \approx \frac{f_i(\mathbf{x} + \delta \mathbf{e}_j) - f_i(\mathbf{x} - \delta \mathbf{e}_j)}{2\delta}$$

where $\delta$ is the difference step size.

**Advantages:**

- No Jacobian code needed (numerical differentiation)
- Better approximation than forward differences (O(δ²) vs O(δ))
- Works for complex or implicit dynamics
- Slightly more expensive than EKF (2n extra function calls vs 1)

Implementation
--------------

.. code-block:: python

    class CenteredDifferenceKalmanFilter:
        """
        Centered Difference Kalman Filter (CDKF).
        
        Computes Jacobians numerically via centered differences.
        More accurate linearization than forward differences.
        """
        
        def __init__(self, x0: np.ndarray, P0: np.ndarray, 
                     delta: float = 1e-4):
            """
            Parameters
            ----------
            x0, P0 : Initial state and covariance
            delta : float
                Finite difference step size
                - Smaller: More accurate but numerically sensitive
                - Larger: More robust but less accurate
                - Typically 1e-4 to 1e-5
            """
            self.n = len(x0)
            self.x = x0.copy()
            self.P = P0.copy()
            self.delta = delta
        
        def _jacobian_centered(self, func, x: np.ndarray) -> np.ndarray:
            """
            Compute Jacobian via centered differences.
            
            J[i,j] ≈ (f_i(x + δ*e_j) - f_i(x - δ*e_j)) / (2δ)
            """
            n = len(x)
            m = len(func(x))
            
            J = np.zeros((m, n))
            
            for j in range(n):
                x_plus = x.copy()
                x_plus[j] += self.delta
                
                x_minus = x.copy()
                x_minus[j] -= self.delta
                
                f_plus = func(x_plus)
                f_minus = func(x_minus)
                
                J[:, j] = (f_plus - f_minus) / (2 * self.delta)
            
            return J
        
        def predict(self, f_func, Q: np.ndarray) -> None:
            """
            Predict with numerical Jacobian.
            
            Similar to EKF but uses finite differences for F.
            """
            # Estimate state
            self.x = f_func(self.x)
            
            # Compute Jacobian via centered differences
            F = self._jacobian_centered(f_func, self.x)
            
            # Covariance propagation (like EKF)
            self.P = F @ self.P @ F.T + Q
        
        def update(self, z: np.ndarray, h_func, R: np.ndarray) -> None:
            """
            Update with numerical Jacobian for h.
            """
            # Predicted measurement
            z_pred = h_func(self.x)
            
            # Jacobian
            H = self._jacobian_centered(h_func, self.x)
            
            # Standard Kalman update
            S = H @ self.P @ H.T + R
            K = self.P @ H.T @ np.linalg.inv(S)
            
            self.x = self.x + K @ (z - z_pred)
            self.P = (np.eye(self.n) - K @ H) @ self.P
        
        def get_state(self) -> tuple:
            return self.x.copy(), self.P.copy()

**Example: Radar Range-Rate Nonlinearity**

.. code-block:: python

    # Radar measurement is highly nonlinear in range-rate
    def h_radar(x):
        """
        Radar measurement: [range, range_rate].
        x = [x_pos, y_pos, x_vel, y_vel]
        """
        pos = x[:2]
        vel = x[2:4]
        
        r = np.linalg.norm(pos)
        r_rate = np.dot(pos, vel) / (r + 1e-6)
        
        return np.array([r, r_rate])
    
    # CDKF
    x0 = np.array([1000.0, 0.0, 0.0, 100.0])  # pos_x, pos_y, vel_x, vel_y
    P0 = np.diag([100.0, 100.0, 10.0, 10.0])
    
    cdkf = CenteredDifferenceKalmanFilter(x0, P0, delta=1e-4)
    
    Q = np.diag([1.0, 1.0, 0.1, 0.1])
    R = np.diag([10.0, 1.0])
    
    for k in range(50):
        # Simple linear motion
        def f_linear(x):
            dt = 0.1
            F = np.array([[1, 0, dt, 0],
                         [0, 1, 0, dt],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
            return F @ x
        
        cdkf.predict(f_linear, Q)
        
        # Measurement
        z = h_radar(x0) + 0.1 * np.random.randn(2)
        cdkf.update(z, h_radar, R)

---

Ensemble Kalman Filter (EnKF)
=============================

The Ensemble Kalman Filter represents uncertainty via an ensemble (collection) of state realizations rather than explicit covariance matrices.

**Key Advantages:**

1. **Scalability**: Works efficiently in very high dimensions (1000s-millions of states)
2. **Non-Gaussian Errors**: Naturally handles non-Gaussian distributions
3. **Nonlinearity Handling**: Implicit handling via ensemble propagation
4. **Parallelization**: Each ensemble member can run independently

**Algorithm:**

Given ensemble $\{\mathbf{x}^{(i)}\}_{i=1}^{N}$ with $N$ members:

1. **Predict**: Propagate each member independently
2. **Update**: Add random perturbations to measurements, update ensemble members

.. code-block:: python

    class EnsembleKalmanFilter:
        """
        Ensemble Kalman Filter (EnKF).
        
        Represents uncertainty via ensemble of state realizations.
        Naturally handles high-dimensional systems and nonlinearity.
        """
        
        def __init__(self, x0: np.ndarray, P0: np.ndarray, 
                     num_members: int = 100):
            """
            Parameters
            ----------
            x0 : (n,) array
                Mean state
            P0 : (n, n) array
                Initial covariance
            num_members : int
                Number of ensemble members (typically 50-1000)
            """
            self.n = len(x0)
            self.num_members = num_members
            
            # Initialize ensemble
            L = cholesky(P0, lower=True)
            self.ensemble = x0[:, np.newaxis] + L @ np.random.randn(self.n, num_members)
        
        def get_state(self) -> tuple:
            """Return mean and covariance from ensemble."""
            x_mean = np.mean(self.ensemble, axis=1)
            
            # Compute covariance from ensemble
            anomalies = self.ensemble - x_mean[:, np.newaxis]
            P = (1.0 / (self.num_members - 1)) * (anomalies @ anomalies.T)
            
            return x_mean, P
        
        def predict(self, f_func, Q: np.ndarray) -> None:
            """
            Propagate ensemble members.
            
            Parameters
            ----------
            f_func : callable
                Nonlinear state function
            Q : (n, n) array
                Process noise covariance
            """
            # Propagate each ensemble member
            for i in range(self.num_members):
                self.ensemble[:, i] = f_func(self.ensemble[:, i])
            
            # Add process noise to all members
            L_Q = cholesky(Q, lower=True)
            noise = L_Q @ np.random.randn(self.n, self.num_members)
            self.ensemble = self.ensemble + noise
        
        def update(self, z: np.ndarray, h_func, R: np.ndarray) -> None:
            """
            Update ensemble via perturbed measurement.
            
            Parameters
            ----------
            z : (m,) array
                Measurement
            h_func : callable
                Measurement function
            R : (m, m) array
                Measurement noise covariance
            """
            m = len(z)
            
            # Propagate ensemble through measurement function
            z_ensemble = np.array([h_func(self.ensemble[:, i]) 
                                   for i in range(self.num_members)]).T
            
            # Ensemble mean measurement
            z_mean = np.mean(z_ensemble, axis=1)
            
            # Ensemble measurement anomalies
            Z_anom = z_ensemble - z_mean[:, np.newaxis]
            
            # Measurement covariance
            Pzz = (1.0 / (self.num_members - 1)) * (Z_anom @ Z_anom.T) + R
            
            # State anomalies
            x_mean = np.mean(self.ensemble, axis=1)
            X_anom = self.ensemble - x_mean[:, np.newaxis]
            
            # Cross-covariance
            Pxz = (1.0 / (self.num_members - 1)) * (X_anom @ Z_anom.T)
            
            # Kalman gain
            K = Pxz @ np.linalg.inv(Pzz)
            
            # Add perturbations to measurement
            L_R = cholesky(R, lower=True)
            z_pert = z[:, np.newaxis] + L_R @ np.random.randn(m, 
                                                               self.num_members)
            
            # Update each ensemble member
            for i in range(self.num_members):
                self.ensemble[:, i] = self.ensemble[:, i] + K @ (z_pert[:, i] - 
                                                                  z_ensemble[:, i])

**Example: Atmospheric Data Assimilation (Simplified)**

.. code-block:: python

    # Simplified 1D temperature profile
    def f_temp_diffusion(x, dt=0.01, diffusion=0.1):
        """Temperature diffusion: dT/dt = α d²T/dx²"""
        # Very simplified: just smooth the profile
        n = len(x)
        x_new = x.copy()
        for i in range(1, n-1):
            x_new[i] += diffusion * dt * (x[i-1] - 2*x[i] + x[i+1])
        return x_new
    
    def h_temp_obs(x):
        """Observe temperature at specific points."""
        # Observe every 5th grid point
        return x[::5]
    
    # EnKF
    n_grid = 50
    x0 = 20.0 + 5.0 * np.sin(np.linspace(0, 2*np.pi, n_grid))
    P0 = np.eye(n_grid) * 1.0
    
    enkf = EnsembleKalmanFilter(x0, P0, num_members=100)
    
    Q = np.eye(n_grid) * 0.1
    R = np.eye(10) * 0.5
    
    for k in range(100):
        # Predict
        enkf.predict(f_temp_diffusion, Q)
        
        # Measurement (at sparse locations)
        x_true = f_temp_diffusion(x0)
        z = h_temp_obs(x_true) + np.sqrt(0.5) * np.random.randn(10)
        
        # Update
        enkf.update(z, h_temp_obs, R)
        
        if k % 25 == 0:
            x_est, P_est = enkf.get_state()
            print(f"Step {k}: Mean T = {np.mean(x_est):.2f}, "
                  f"Uncertainty = {np.mean(np.diag(P_est)):.3f}")

---

Comparison: Advanced KF Variants
=================================

**Accuracy and Computational Cost:**

+--------------------------------+----------+-----------+-----------+-----------+
| Filter Type                    | CKF      | UKF       | CDKF      | EnKF      |
+================================+==========+===========+===========+===========+
| **Accuracy**                   |          |           |           |           |
|  Nonlinearity (mild)           | EKF+     | EKF+      | EKF+      | EKF+      |
|  Nonlinearity (strong)         | UKF      | Good      | EKF       | UKF       |
|  Non-Gaussian errors           | Fair     | Fair      | Fair      | **Good**  |
+--------------------------------+----------+-----------+-----------+-----------+
| **Speed** (relative to EKF)    |          |           |           |           |
|  Single step                   | 1.5×     | 1.3×      | 2.0×      | N×        |
|  Dimension (n)                 | 4n       | 2n+1      | 2n        | N members |
+--------------------------------+----------+-----------+-----------+-----------+
| **Jacobian Required**          | No       | No        | Yes*      | No        |
|  (estimates internally)        |          |           |           |           |
+--------------------------------+----------+-----------+-----------+-----------+
| **Memory (relative)**          | 1×       | 1×        | 1×        | N×        |
+--------------------------------+----------+-----------+-----------+-----------+
| **Parallelizable**             | No       | No        | No        | **Yes**   |
+--------------------------------+----------+-----------+-----------+-----------+
| **High Dimensions** (n>1000)   | ✗        | ✗         | ✗         | ✓         |
+--------------------------------+----------+-----------+-----------+-----------+

**When to Use Each:**

1. **Cubature Kalman Filter (CKF)**
   
   - ✅ Moderate-dimensional systems (n < 100)
   - ✅ Smooth nonlinearities
   - ✅ Need derivative-free approach
   - ❌ High dimensions or real-time constraints

2. **Unscented Kalman Filter (UKF)**
   
   - ✅ Balance accuracy and speed
   - ✅ Most nonlinearities
   - ✅ Standard choice for modern tracking
   - ✅ Well-understood theory and tuning

3. **Centered Difference KF (CDKF)**
   
   - ✅ Complex dynamics only available as code
   - ✅ Numerical precision issues make analytical Jacobians unreliable
   - ⚠️ Slightly more expensive than EKF
   - ❌ Not significantly more accurate than EKF for most problems

4. **Ensemble Kalman Filter (EnKF)**
   
   - ✅ Very high dimensions (1000s-millions)
   - ✅ Non-Gaussian errors
   - ✅ Parallelizable across ensemble members
   - ✅ Data assimilation (geophysics, oceanography)
   - ❌ More complex, requires careful tuning
   - ❌ Smaller ensemble → sampling errors

---

Hybrid Approaches
=================

Combine multiple variants for specific problem structures.

.. code-block:: python

    class HybridMultiFilterSystem:
        """
        Use different filters for different states or modes.
        
        For example:
        - UKF for highly nonlinear position estimation
        - Linear Kalman for well-modeled velocity
        - EnKF for distributed inference
        """
        
        def __init__(self):
            self.filters = {}
        
        def register_filter(self, name: str, filter_obj):
            """Register a filter for a subsystem."""
            self.filters[name] = filter_obj
        
        def predict(self, predictions: dict) -> None:
            """Predict each subsystem with its filter."""
            for name, pred_func in predictions.items():
                if name in self.filters:
                    self.filters[name].predict(lambda x: pred_func(x))
        
        def update(self, measurements: dict) -> None:
            """Update each subsystem."""
            for name, (z, h_func, R) in measurements.items():
                if name in self.filters:
                    self.filters[name].update(z, h_func, R)
        
        def get_state_dict(self) -> dict:
            """Retrieve state from all filters."""
            return {name: filt.get_state() 
                   for name, filt in self.filters.items()}

---

Practical Diagnostics
======================

Monitor filter performance across variants.

.. code-block:: python

    class AdvancedFilterDiagnostics:
        """Compare and diagnose advanced filter performance."""
        
        def __init__(self, filter_dict: dict):
            """
            Parameters
            ----------
            filter_dict : dict
                Named filters: {'ckf': ckf_obj, 'ukf': ukf_obj, ...}
            """
            self.filters = filter_dict
            self.metrics = {name: {'innovations': [], 'uncertainties': []} 
                          for name in filter_dict}
        
        def innovation_consistency_test(self, z: np.ndarray, 
                                       h_func, R: np.ndarray) -> dict:
            """
            Evaluate innovation consistency for all filters.
            
            Innovations should have zero mean and covariance S.
            """
            results = {}
            
            for name, filt in self.filters.items():
                x, P = filt.get_state()
                
                z_pred = h_func(x)
                innov = z - z_pred
                
                H_approx = self._compute_jacobian(h_func, x)
                S = H_approx @ P @ H_approx.T + R
                
                # Normalized innovation
                try:
                    innov_norm_sq = innov.T @ np.linalg.inv(S) @ innov
                except:
                    innov_norm_sq = np.inf
                
                results[name] = {
                    'innovation': innov,
                    'innovations_norm_sq': innov_norm_sq,
                    'consistency': 'good' if 0.5 < innov_norm_sq < 2.0 else 'poor'
                }
            
            return results
        
        def _compute_jacobian(self, func, x: np.ndarray, 
                            delta: float = 1e-4) -> np.ndarray:
            """Numerical Jacobian."""
            n = len(x)
            m = len(func(x))
            J = np.zeros((m, n))
            
            for j in range(n):
                x_plus = x.copy()
                x_plus[j] += delta
                J[:, j] = (func(x_plus) - func(x)) / delta
            
            return J

---

Tuning Guidelines
=================

**CKF Tuning:**

- Usually minimal tuning needed (derivative-free, symmetric)
- Primary parameter: process noise $Q$ (same as standard Kalman)

**UKF Tuning:**

- $\alpha$ (spread): Typically $10^{-3}$ (start conservative)
- $\beta$ (prior knowledge): 2.0 for Gaussian
- $\kappa$ (secondary): Often 0, or $3-n$ for some applications

**CDKF Tuning:**

- $\delta$ (step size): $10^{-4}$ to $10^{-5}$ (problem-dependent)
- Smaller $\delta$: More accurate but numerically sensitive
- Larger $\delta$: More robust but less accurate

**EnKF Tuning:**

- Ensemble size $N$: $50-1000$ typical
  - Larger $N$: Better approximation, more expensive
  - Smaller $N$: Faster, but sampling errors
- Localization: For spatial systems, limit update region

**Rule of Thumb:**

1. Start with standard Kalman → UKF → advanced variant
2. Use **CKF** if derivatives cause numerical issues
3. Use **EnKF** if dimension > 500
4. Switch to **CDKF** only if UKF underperforms

---

Common Pitfalls
===============

1. **Tuning Proliferation**: Advanced filters have more parameters
   
   - *Fix*: Use defaults initially, tune conservatively

2. **High Ensemble Size Overhead**: EnKF with 1000 members is expensive
   
   - *Fix*: Use localization, data assimilation techniques

3. **Numerical Issues in Derivatives**: CDKF can amplify roundoff errors
   
   - *Fix*: Use appropriate $\delta$, consider analytical Jacobians

4. **Overconfidence in Ensemble Mean**: EnKF ensemble can collapse
   
   - *Fix*: Monitor ensemble spread, use covariance inflation

5. **Mode Switches**: IMM + advanced filter combinations complex
   
   - *Fix*: Test thoroughly, start simple

---

See Also
========

- [Kalman Filter Tuning](kalman_filter_tuning.rst) — Basics and standard Kalman
- [Adaptive Filtering](adaptive_filtering.rst) — Parameter tuning online
- [Information Filters](information_filters.rst) — Numerically stable alternatives
- [Particle Filters](particle_filters.rst) — For multi-modal distributions
- [Troubleshooting](troubleshooting.rst) — Debugging filter issues

**References:**

- Arasaratnam & Haykin (2009) — *Cubature Kalman Filters* — Foundational CKF paper
- Sarkka (2013) — *Bayesian Filtering and Smoothing* — Comprehensive sigma-point theory
- Evensen (2003) — *Ensemble Kalman Filter* — Ensemble methods origins
- Bar-Shalom, Li, Kirubarajan (2001) — *Estimation with Applications* — Comprehensive reference
