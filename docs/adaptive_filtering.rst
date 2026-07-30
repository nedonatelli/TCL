Adaptive Filtering
==================

Adaptive filtering techniques allow Kalman and particle filters to automatically adjust their parameters in response to changing system dynamics and measurement conditions. This guide covers divergence detection, noise covariance estimation, and adaptive Kalman gain strategies for real-time filter tuning.

.. contents:: Contents
   :local:
   :depth: 3

---

When to Use Adaptive Filtering
===============================

Standard Kalman filters assume **known and constant** process and measurement noise covariances ($Q$ and $R$). However, real systems often violate this assumption:

**Problem Scenarios:**

- **Unknown noise statistics**: Initial $Q$ and $R$ estimates are inaccurate
- **Time-varying systems**: Process noise changes due to environmental conditions
- **Sensor degradation**: Measurement noise increases over time
- **Model mismatch**: System dynamics drift from the assumed model
- **Multi-sensor fusion**: Different sensors have varying noise characteristics

**Filter Divergence Symptoms:**

1. **Residual saturation**: Innovation sequence shows bias or unusual statistics
2. **Filter overfitting**: Confidence bounds contract despite poor estimate quality
3. **Estimate instability**: State estimates diverge from measurements unexpectedly
4. **Consistency tests fail**: Normalized innovations exceed $3\sigma$ bounds repeatedly

**When NOT to adapt:** If your system parameters are well-characterized and constant, standard tuning (static $Q$ and $R$) is more robust.

---

Divergence Detection Techniques
================================

Before adapting filter parameters, detect when divergence is occurring.

Normalized Innovation Squared Test
----------------------------------

The normalized innovation (residual) should follow $\chi^2(m)$ distribution if filter is consistent:

$$\gamma_k = \mathbf{y}_k^T \mathbf{S}_k^{-1} \mathbf{y}_k \sim \chi^2(m)$$

where $\mathbf{y}_k$ is the innovation and $\mathbf{S}_k$ is the innovation covariance.

**Algorithm:**

.. code-block:: python

    import numpy as np
    from scipy.stats import chi2
    from pytcl.dynamic_estimation.kalman import ekf_predict, ekf_update
    
    class DivergenceDetector:
        """Detects filter divergence using statistical tests."""
        
        def __init__(self, measurement_dim: int, window_size: int = 100):
            self.m = measurement_dim
            self.window_size = window_size
            self.innovations = []
            self.normalized_sq = []
            
        def update(self, innovation: np.ndarray, 
                   innovation_cov: np.ndarray) -> dict:
            """
            Check divergence at current timestep.
            
            Parameters
            ----------
            innovation : (m,) array
                Measurement residual y_k
            innovation_cov : (m, m) array
                Innovation covariance S_k
                
            Returns
            -------
            dict with keys:
                - normalized_sq: Normalized innovation squared γ_k
                - mahal_distance: Mahalanobis distance (sqrt of γ_k)
                - p_value: χ²(m) tail probability
                - is_outlier: True if γ_k > chi2.ppf(0.99, m)
                - is_diverging: True if recent mean γ_k exceeds threshold
            """
            # Normalized innovation squared
            try:
                S_inv = np.linalg.inv(innovation_cov)
                gamma = float(innovation.T @ S_inv @ innovation)
            except np.linalg.LinAlgError:
                return {'error': 'Singular innovation covariance'}
            
            self.normalized_sq.append(gamma)
            self.innovations.append(innovation.copy())
            
            # Keep window
            if len(self.normalized_sq) > self.window_size:
                self.normalized_sq.pop(0)
                self.innovations.pop(0)
            
            # Statistical test
            p_value = 1.0 - chi2.cdf(gamma, self.m)
            critical_value = chi2.ppf(0.99, self.m)  # 99% quantile
            is_outlier = gamma > critical_value
            
            # Divergence test: mean γ_k should be ≈ m
            recent_mean = np.mean(self.normalized_sq)
            divergence_threshold = self.m * 1.5  # Allow 50% excess
            is_diverging = recent_mean > divergence_threshold
            
            return {
                'normalized_sq': gamma,
                'mahal_distance': np.sqrt(gamma),
                'p_value': p_value,
                'is_outlier': is_outlier,
                'recent_mean': recent_mean,
                'is_diverging': is_diverging,
            }
    
    # Example usage
    detector = DivergenceDetector(measurement_dim=3)
    
    # Simulate measurements from a simple filter
    for k in range(200):
        # Your filter update produces innovation and innovation_cov
        innovation = np.random.randn(3)  # Simulated
        innovation_cov = np.eye(3)
        
        status = detector.update(innovation, innovation_cov)
        
        if status.get('is_diverging'):
            print(f"Divergence detected at step {k}")
            print(f"  Mean normalized_sq: {status['recent_mean']:.2f} (expected ~{detector.m})")

Autocorrelation Test
--------------------

White residuals should have zero autocorrelation. Significant autocorrelation indicates model mismatch or parameter errors.

.. code-block:: python

    def autocorrelation_test(innovations: np.ndarray, lag: int = 10) -> dict:
        """
        Test residual autocorrelation.
        
        Parameters
        ----------
        innovations : (N, m) array
            Time series of innovations
        lag : int
            Maximum lag to examine
            
        Returns
        -------
        dict with autocorrelation values and significance tests
        """
        N, m = innovations.shape
        acf_values = {}
        
        for component in range(m):
            signal = innovations[:, component]
            mean = np.mean(signal)
            var = np.var(signal)
            
            acf = np.zeros(lag + 1)
            for k in range(lag + 1):
                if k == 0:
                    acf[k] = 1.0
                else:
                    acf[k] = np.mean((signal[:-k] - mean) * 
                                     (signal[k:] - mean)) / var
            
            acf_values[f'component_{component}'] = acf
        
        # Ljung-Box test
        Q_stat = N * (N + 2) * sum(acf[k]**2 / (N - k) 
                                   for k in range(1, lag + 1))
        
        from scipy.stats import chi2
        p_value = 1.0 - chi2.cdf(Q_stat, lag - 1)
        
        return {
            'acf': acf_values,
            'ljung_box_statistic': Q_stat,
            'p_value': p_value,
            'is_white': p_value > 0.05
        }

---

Noise Covariance Estimation
=============================

Estimate $Q$ and $R$ from batch or online data.

Batch Estimation: Maximum Likelihood
-------------------------------------

Given a batch of filter residuals, estimate noise covariances via maximum likelihood.

$$\hat{\mathbf{Q}} = \frac{1}{N} \sum_{k=1}^{N} \mathbf{x}_k \mathbf{x}_k^T$$

$$\hat{\mathbf{R}} = \frac{1}{N} \sum_{k=1}^{N} \mathbf{y}_k \mathbf{y}_k^T$$

.. code-block:: python

    class NoiseEstimator:
        """Batch and online noise covariance estimation."""
        
        @staticmethod
        def estimate_from_residuals(innovations: np.ndarray,
                                   state_predictions: np.ndarray) -> dict:
            """
            Estimate Q and R from filter residuals and predictions.
            
            Parameters
            ----------
            innovations : (N, m) array
                Measurement residuals
            state_predictions : (N, n) array
                State prediction errors (x_true - x_pred)
                
            Returns
            -------
            dict with estimated R and Q matrices
            """
            N = innovations.shape[0]
            
            # Measurement noise covariance
            R_est = innovations.T @ innovations / N
            
            # Process noise covariance (approximation)
            Q_est = state_predictions.T @ state_predictions / N
            
            return {
                'R': R_est,
                'Q': Q_est,
                'R_condition_number': np.linalg.cond(R_est),
                'Q_condition_number': np.linalg.cond(Q_est),
            }
        
        @staticmethod
        def estimate_diagonal(innovations: np.ndarray) -> np.ndarray:
            """
            Quick estimate: diagonal covariance (assumes independent noise).
            More numerically stable for ill-conditioned problems.
            """
            return np.diag(np.var(innovations, axis=0))

Adaptive Q Estimation (Online)
-------------------------------

Track time-varying process noise online. Use exponential weighting to emphasize recent data:

$$\hat{\mathbf{Q}}_k = \alpha \hat{\mathbf{Q}}_{k-1} + (1-\alpha) (\mathbf{y}_{k} - \mathbf{H} \hat{\mathbf{x}}_{k|k-1})(\cdot)^T$$

.. code-block:: python

    class AdaptiveNoiseEstimator:
        """Online adaptive noise covariance estimation."""
        
        def __init__(self, state_dim: int, meas_dim: int, 
                     forgetting_factor: float = 0.95):
            """
            Parameters
            ----------
            state_dim : int
                State dimension
            meas_dim : int
                Measurement dimension
            forgetting_factor : float
                Exponential weighting (0.9-0.99 typical)
                - 1.0: infinite memory (pure averaging)
                - 0.9: recent ~10 steps emphasized
            """
            self.n = state_dim
            self.m = meas_dim
            self.alpha = forgetting_factor
            
            self.R_hat = np.eye(meas_dim)
            self.Q_hat = np.eye(state_dim)
            self.update_count = 0
        
        def update_R(self, innovation: np.ndarray) -> None:
            """Update measurement noise estimate."""
            self.update_count += 1
            
            # Exponentially weighted update
            self.R_hat = (self.alpha * self.R_hat + 
                         (1 - self.alpha) * np.outer(innovation, innovation))
        
        def update_Q(self, state_residual: np.ndarray) -> None:
            """Update process noise estimate."""
            self.Q_hat = (self.alpha * self.Q_hat + 
                         (1 - self.alpha) * np.outer(state_residual, 
                                                     state_residual))
        
        def get_estimates(self) -> dict:
            """Return current noise estimates."""
            return {
                'R': self.R_hat.copy(),
                'Q': self.Q_hat.copy(),
                'updates': self.update_count
            }

**Best Practices:**

- Start with `forgetting_factor = 0.95` (effective window ≈ 20 steps)
- Increase forgetting factor if system is slowly time-varying
- Decrease (e.g., 0.90) for rapidly changing conditions
- Regularize: Add small $\epsilon I$ to prevent singularity
- Monitor condition numbers to detect ill-posedness

---

Adaptive Kalman Filtering
==========================

Dynamically adjust Kalman filter gains and covariances.

Gain Adaptation Strategy
------------------------

Instead of adapting noise covariances, directly scale the Kalman gain:

$$\mathbf{K}_k^{adaptive} = \beta_k \mathbf{K}_k$$

where $\beta_k$ controls filter responsiveness based on residual statistics.

.. code-block:: python

    class AdaptiveKalmanFilter:
        """
        Adaptive Kalman filter with automatic gain tuning.
        
        Adjusts gain based on:
        1. Innovation magnitude (residual statistics)
        2. Consistency tests (are we under/overconfident?)
        3. Divergence indicators
        """
        
        def __init__(self, x, P, H, Q, R, initial_beta: float = 1.0):
            """
            Parameters
            ----------
            x, P : ndarray
                Initial state and covariance. The library exposes predict and
                update as functions rather than a filter object, so the
                wrapper owns this state itself.
            H : ndarray
                Measurement matrix.
            Q, R : ndarray
                Process and measurement noise covariances.
            initial_beta : float
                Initial gain scaling factor (typically 1.0)
            """
            self.x = np.asarray(x, dtype=float)
            self.P = np.asarray(P, dtype=float)
            self.H = np.asarray(H, dtype=float)
            self.Q = np.asarray(Q, dtype=float)
            self.R = np.asarray(R, dtype=float)
            self.beta = initial_beta
            self.R_est = self.R.copy()
            self.innovation_history = []
            self.beta_history = [initial_beta]
            self.divergence_state = 'normal'
        
        def predict(self, F) -> np.ndarray:
            """Prediction step (standard KF)."""
            pred = kf_predict(self.x, self.P, F, self.Q)
            self.x, self.P = pred.x, pred.P
            return self.x
        
        def update(self, z: np.ndarray, dt: float) -> tuple:
            """
            Update step with adaptive gain.
            
            Returns
            -------
            (state, covariance, diagnostics)
            """
            # Compute innovation with predicted state
            H = self.H
            x_pred = self.x
            y = z - H @ x_pred  # Innovation
            
            # Innovation covariance
            S = H @ self.P @ H.T + self.R
            
            # Normalize innovation
            try:
                S_inv = np.linalg.inv(S)
                gamma = float(y.T @ S_inv @ y)
            except np.linalg.LinAlgError:
                gamma = 1.0
            
            # Compute scaling factor β_k based on innovation magnitude
            m = len(z)
            expected_gamma = m  # Expected value of χ²(m)
            beta = self._compute_adaptive_gain(gamma, expected_gamma)
            
            # Scale Kalman gain
            K = self.P @ H.T @ np.linalg.inv(S)
            K_adapted = beta * K
            
            # Update with adapted gain
            x_new = x_pred + K_adapted @ y
            P_new = (np.eye(len(x_pred)) - K_adapted @ H) @ self.P
            
            # Update filter state
            self.x = x_new
            self.P = P_new
            
            # Record history
            self.innovation_history.append(gamma)
            self.beta_history.append(beta)
            
            return x_new, P_new, {
                'innovation_norm_sq': gamma,
                'adaptive_beta': beta,
                'divergence_state': self.divergence_state
            }
        
        def _compute_adaptive_gain(self, gamma: float, 
                                   expected_gamma: float) -> float:
            """
            Compute adaptive gain scaling β based on normalized innovation.
            
            Strategy:
            - If γ ≈ expected: No scaling (β = 1)
            - If γ >> expected: Innovation is large → reduce gain (β < 1)
              (System is noisier than assumed, filter is overconfident)
            - If γ << expected: Innovation is small → increase gain (β > 1)
              (Filter is underconfident, not tracking changes)
            """
            # Clip to prevent extreme swings
            ratio = gamma / expected_gamma
            
            # Use smooth function: β = 1 / sqrt(ratio)
            # This is the "optimal" scaling for Gaussian errors
            beta = 1.0 / np.sqrt(max(ratio, 0.5))  # Limit 0.5 ≤ β ≤ 1.4
            beta = np.clip(beta, 0.5, 1.4)
            
            # Divergence detection: if repeated large innovations
            if len(self.innovation_history) >= 10:
                recent_mean = np.mean(self.innovation_history[-10:])
                if recent_mean > expected_gamma * 2.0:
                    self.divergence_state = 'diverging'
                    beta *= 0.8  # Reduce gain further
                else:
                    self.divergence_state = 'normal'
            
            return beta

**Example: GPS/INS Navigation**

.. code-block:: python

    # Simulate adaptive filtering for GPS/INS fusion
    from pytcl.dynamic_estimation.kalman import kf_predict

    # The library exposes predict/update functions rather than a filter class,
    # so the state the wrapper adapts is just these arrays.
    x = np.array([0.0, 0.0, 0.0])          # pos_x, pos_y, vel_x
    P = np.eye(3)
    Q = np.diag([0.01, 0.01, 0.001])
    R = np.diag([1.0, 1.0])                # GPS measurement noise

    H = np.array([[1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0]])   # GPS observes position only

    # Wrap with adaptive filtering
    adaptive_ekf = AdaptiveKalmanFilter(x, P, H, Q, R, initial_beta=1.0)
    
    # Simulate GPS signal degradation
    for k in range(100):
        # Predict
        adaptive_ekf.predict(dt=0.1)
        
        # GPS measurement with time-varying noise
        if k < 50:
            gps_noise_std = 1.0
        else:
            gps_noise_std = 5.0  # GPS signal degrades (e.g., urban canyon)
        
        z = np.array([0.1*k, 0.05*k**2]) + gps_noise_std * np.random.randn(2)
        
        x_est, P_est, diagnostics = adaptive_ekf.update(z, dt=0.1)
        
        if k % 20 == 0:
            print(f"Step {k}: β = {diagnostics['adaptive_beta']:.2f}, "
                  f"γ = {diagnostics['innovation_norm_sq']:.1f}")

---

Least Mean Squares (LMS) Adaptation
====================================

For problems where you want to minimize prediction error directly, use adaptive LMS filtering.

Algorithm
---------

At each step, update filter weights to minimize squared error:

$$\mathbf{w}_{k+1} = \mathbf{w}_k + \mu \mathbf{x}_k (z_k - \mathbf{w}_k^T \mathbf{x}_k)$$

where $\mu$ is the step size (learning rate).

.. code-block:: python

    class LMSFilter:
        """
        Least Mean Squares adaptive filter.
        
        Minimizes MSE directly using gradient descent on filter weights.
        """
        
        def __init__(self, filter_order: int, step_size: float = 0.01):
            """
            Parameters
            ----------
            filter_order : int
                Number of filter taps (order)
            step_size : float
                Learning rate μ (typically 0 < μ < 1/(3*power))
            """
            self.order = filter_order
            self.mu = step_size
            self.weights = np.zeros(filter_order)
            self.buffer = np.zeros(filter_order)
            self.error_history = []
        
        def update(self, measurement: float) -> tuple:
            """
            Update filter with new measurement.
            
            Parameters
            ----------
            measurement : float
                New input sample
                
            Returns
            -------
            (estimate, error, normalized_error_sq)
            """
            # Shift buffer and insert new measurement
            self.buffer = np.roll(self.buffer, 1)
            self.buffer[0] = measurement
            
            # Filtering (estimate via convolution with weights)
            estimate = np.dot(self.weights, self.buffer)
            
            # Error
            error = measurement - estimate
            
            # Gradient descent update
            # ∂MSE/∂w = -2 * error * x (x is buffer)
            self.weights += self.mu * error * self.buffer
            
            error_sq = error ** 2
            self.error_history.append(error_sq)
            
            return estimate, error, error_sq
        
        def set_step_size(self, mu: float) -> None:
            """
            Dynamically adjust learning rate.
            
            Larger μ: Faster adaptation, but less stable
            Smaller μ: Slower adaptation, but more stability
            """
            self.mu = mu
        
        def get_mse(self, window: int = 100) -> float:
            """Get recent Mean Squared Error."""
            if len(self.error_history) < window:
                return np.mean(self.error_history)
            return np.mean(self.error_history[-window:])

**Example: Adaptive System Identification**

.. code-block:: python

    # Adaptive filter to identify unknown system response
    # True system has impulse response [1.0, 0.5, -0.2]
    true_system = np.array([1.0, 0.5, -0.2])
    
    # Adaptive LMS filter
    lms = LMSFilter(filter_order=3, step_size=0.05)
    
    for k in range(500):
        # Input signal (white noise)
        x_input = np.random.randn()
        
        # True system output
        y_true = np.dot(true_system, 
                       np.array([x_input, 
                                 0.5 * np.random.randn(),
                                 0.1 * np.random.randn()]))
        
        # Measurement with noise
        z = y_true + 0.1 * np.random.randn()
        
        # LMS update
        y_est, error, mse = lms.update(z)
        
        if k % 100 == 0:
            print(f"Step {k}: Estimated weights = {lms.weights}, "
                  f"MSE = {lms.get_mse():.4f}")

---

Recursive Least Squares (RLS) Adaptation
========================================

RLS is more sophisticated than LMS: it maintains a covariance matrix of the problem and achieves faster convergence.

Algorithm
---------

$$\mathbf{w}_{k+1} = \mathbf{w}_k + \mathbf{K}_k (z_k - \mathbf{x}_k^T \mathbf{w}_k)$$

$$\mathbf{K}_k = \frac{\mathbf{P}_{k-1} \mathbf{x}_k}{\lambda + \mathbf{x}_k^T \mathbf{P}_{k-1} \mathbf{x}_k}$$

$$\mathbf{P}_k = \frac{1}{\lambda}\left(\mathbf{P}_{k-1} - \frac{\mathbf{P}_{k-1} \mathbf{x}_k \mathbf{x}_k^T \mathbf{P}_{k-1}}{\lambda + \mathbf{x}_k^T \mathbf{P}_{k-1} \mathbf{x}_k}\right)$$

where $\lambda$ is the forgetting factor (0.95-0.99 typical).

.. code-block:: python

    class RLSFilter:
        """
        Recursive Least Squares adaptive filter.
        
        Faster convergence than LMS at cost of higher computational load.
        """
        
        def __init__(self, filter_order: int, 
                     forgetting_factor: float = 0.95,
                     regularization: float = 1e-6):
            """
            Parameters
            ----------
            filter_order : int
                Number of filter coefficients
            forgetting_factor : float
                λ ∈ (0, 1]. Smaller = faster fade of old data
                - 1.0: full recursive LS (no forgetting)
                - 0.95: ~20 step effective window
                - 0.90: ~10 step effective window
            regularization : float
                Small regularization to prevent ill-conditioning
            """
            self.order = filter_order
            self.lamda = forgetting_factor
            self.reg = regularization
            
            self.weights = np.zeros(filter_order)
            self.P = (1.0 / regularization) * np.eye(filter_order)
            self.buffer = np.zeros(filter_order)
            self.error_history = []
        
        def update(self, measurement: float) -> tuple:
            """
            RLS update step.
            
            Returns
            -------
            (estimate, error, condition_number)
            """
            # Shift buffer
            self.buffer = np.roll(self.buffer, 1)
            self.buffer[0] = measurement
            
            # Prediction error
            y_est = np.dot(self.weights, self.buffer)
            error = measurement - y_est
            
            # Compute gain matrix K_k
            denom = self.lamda + self.buffer @ self.P @ self.buffer
            K = (self.P @ self.buffer) / denom
            
            # Update weights
            self.weights += K * error
            
            # Update covariance matrix (Joseph form for stability)
            P_new = (1.0 / self.lamda) * (self.P - np.outer(K, self.buffer @ self.P))
            
            # Regularization to maintain numerical stability
            eigenvalues = np.linalg.eigvals(P_new)
            if np.min(eigenvalues) < 1e-10:
                P_new += self.reg * np.eye(self.order)
            
            self.P = P_new
            
            self.error_history.append(error ** 2)
            
            cond_number = np.linalg.cond(self.P)
            
            return y_est, error, cond_number
        
        def get_mse(self, window: int = 50) -> float:
            """Get recent MSE."""
            if len(self.error_history) < window:
                return np.mean(self.error_history)
            return np.mean(self.error_history[-window:])

**Comparison: LMS vs RLS**

.. code-block:: python

    # Compare convergence speed
    true_weights = np.array([0.8, -0.3, 0.2])
    
    lms = LMSFilter(filter_order=3, step_size=0.1)
    rls = RLSFilter(filter_order=3, forgetting_factor=0.95)
    
    lms_errors = []
    rls_errors = []
    
    for k in range(200):
        # Input
        x = np.random.randn()
        y = np.random.randn()
        z = 0.8*x - 0.3*y + 0.2*np.random.randn() + 0.05*np.random.randn()
        
        # LMS
        y_lms, err_lms, _ = lms.update(z)
        lms_errors.append(np.mean(lms.error_history[-10:]) if lms.error_history else 0)
        
        # RLS
        y_rls, err_rls, _ = rls.update(z)
        rls_errors.append(rls.get_mse(window=10))
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 4))
    plt.semilogy(lms_errors, label='LMS', alpha=0.7)
    plt.semilogy(rls_errors, label='RLS', alpha=0.7)
    plt.xlabel('Iteration')
    plt.ylabel('MSE (log scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title('Convergence Speed: LMS vs RLS')
    plt.tight_layout()
    plt.show()
    
    print(f"LMS final MSE: {lms_errors[-1]:.4f}")
    print(f"RLS final MSE: {rls_errors[-1]:.4f}")

---

Practical Adaptive Filter Systems
==================================

Multi-Sensor Adaptive Fusion
----------------------------

Combine multiple sensors with unknown noise characteristics.

.. code-block:: python

    class AdaptiveMultiSensorFusion:
        """
        Fuse multiple sensors with adaptive noise estimation.
        """
        
        def __init__(self, num_sensors: int, state_dim: int):
            self.num_sensors = num_sensors
            self.state_dim = state_dim
            
            # Adaptive noise estimates for each sensor
            self.R_estimates = [np.eye(1) for _ in range(num_sensors)]
            self.measurement_history = [[] for _ in range(num_sensors)]
            
            # State estimate
            self.x = np.zeros(state_dim)
            self.P = np.eye(state_dim)
        
        def update_sensor(self, sensor_id: int, measurement: float,
                         H: np.ndarray, alpha: float = 0.95) -> None:
            """
            Update with single sensor measurement.
            
            Parameters
            ----------
            sensor_id : int
                Which sensor (0 to num_sensors-1)
            measurement : float
                Scalar measurement
            H : (1, state_dim) array
                Measurement matrix
            alpha : float
                Exponential weighting for R estimation
            """
            z = np.array([measurement])
            
            # Predicted measurement
            z_pred = H @ self.x
            innovation = z - z_pred
            
            # Update R estimate for this sensor
            R_old = self.R_estimates[sensor_id]
            self.R_estimates[sensor_id] = (alpha * R_old + 
                                          (1 - alpha) * innovation @ innovation.T)
            
            # Standard Kalman update with estimated R
            S = H @ self.P @ H.T + self.R_estimates[sensor_id]
            K = self.P @ H.T / S
            
            self.x = self.x + K @ innovation
            self.P = (np.eye(self.state_dim) - K @ H) @ self.P
            
            self.measurement_history[sensor_id].append(measurement)
        
        def get_sensor_reliability(self, sensor_id: int) -> dict:
            """Rate each sensor based on noise estimates."""
            R = self.R_estimates[sensor_id]
            noise_std = np.sqrt(R[0, 0])
            
            # Lower noise = more reliable
            reliability = 1.0 / (1.0 + noise_std)
            
            return {
                'sensor_id': sensor_id,
                'noise_estimate': noise_std,
                'reliability_score': reliability,
                'measurements': len(self.measurement_history[sensor_id])
            }

GPS/INS Adaptive Integration
-----------------------------

Real-world example: Adaptive GPS/INS navigation with sensor degradation detection.

.. code-block:: python

    class AdaptiveGPSINS:
        """
        Adaptive GPS/INS integration.
        
        - INS provides high-rate state propagation
        - GPS provides low-rate absolute measurements
        - Adapts GPS measurement noise when signal degrades
        """
        
        def __init__(self):
            # INS state: [pos_x, pos_y, vel_x, vel_y]
            self.x = np.array([0.0, 0.0, 1.0, 0.0])
            self.P = np.eye(4) * 0.1
            
            # Process noise (INS integration)
            self.Q = np.diag([0.01, 0.01, 0.001, 0.001]) ** 2
            
            # Measurement noise estimates (GPS)
            self.R_gps = np.eye(2) * 1.0  # Will adapt
            
            # Tuning parameters
            self.R_max = 100.0  # Max allowed GPS noise
            self.R_min = 0.1
            self.forgetting_factor = 0.95
            
            self.gps_measurement_count = 0
        
        def ins_predict(self, dt: float, 
                       accel: np.ndarray) -> None:
            """Propagate INS."""
            F = np.eye(4)
            F[0, 2] = dt
            F[1, 3] = dt
            
            # State transition
            self.x = F @ self.x  # Simplified (IRL use 9-state IMU model)
            
            # Covariance propagation
            self.P = F @ self.P @ F.T + self.Q
        
        def gps_update(self, gps_pos: np.ndarray) -> dict:
            """
            GPS measurement update with adaptive noise.
            Detects multipath/signal degradation.
            """
            H = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0]])
            
            z = gps_pos
            z_pred = H @ self.x
            innovation = z - z_pred
            
            # Innovation covariance
            S = H @ self.P @ H.T + self.R_gps
            
            # Check if innovation is unusually large (signal loss)
            innovation_norm_sq = innovation @ np.linalg.inv(S) @ innovation
            expected_norm = 2.0  # Expected χ²(2)
            
            is_degraded = innovation_norm_sq > 2.0 * expected_norm
            
            # Adaptive R: Increase if GPS is degraded, decrease if good
            if is_degraded:
                # GPS signal degraded → reduce gain
                self.R_gps = np.minimum(self.R_gps * 1.2, 
                                       self.R_max * np.eye(2))
            else:
                # GPS signal good → normal noise
                self.R_gps = (self.forgetting_factor * self.R_gps + 
                            (1 - self.forgetting_factor) * 
                            np.outer(innovation, innovation))
                self.R_gps = np.clip(self.R_gps[0, 0], self.R_min, 
                                    self.R_max) * np.eye(2)
            
            # Standard Kalman update
            K = self.P @ H.T @ np.linalg.inv(S)
            self.x = self.x + K @ innovation
            self.P = (np.eye(4) - K @ H) @ self.P
            
            self.gps_measurement_count += 1
            
            return {
                'is_degraded': is_degraded,
                'innovation_norm_sq': innovation_norm_sq,
                'gps_noise_std': np.sqrt(self.R_gps[0, 0]),
                'gps_count': self.gps_measurement_count
            }

---

Diagnostic Tools
================

Monitor and debug adaptive filters.

.. code-block:: python

    class AdaptiveFilterDiagnostics:
        """Comprehensive diagnostics for adaptive filters."""
        
        def __init__(self, filter_object):
            self.filter = filter_object
            self.history = {
                'innovations': [],
                'residual_stats': [],
                'covariance_norms': [],
                'condition_numbers': [],
                'divergence_indicators': []
            }
        
        def log_step(self, innovation: np.ndarray, 
                    covariance: np.ndarray) -> None:
            """Log diagnostic data after each filter step."""
            self.history['innovations'].append(innovation.copy())
            self.history['covariance_norms'].append(np.linalg.norm(covariance))
            self.history['condition_numbers'].append(np.linalg.cond(covariance))
            
            # Innovation stats
            innov_stats = {
                'mean': np.mean(innovation),
                'std': np.std(innovation),
                'norm': np.linalg.norm(innovation),
            }
            self.history['residual_stats'].append(innov_stats)
        
        def plot_diagnostics(self) -> None:
            """Generate diagnostic plots."""
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            
            # Innovation norms
            innovation_norms = [np.linalg.norm(y) for y in self.history['innovations']]
            axes[0, 0].plot(innovation_norms, alpha=0.7)
            axes[0, 0].set_title('Innovation Magnitude Over Time')
            axes[0, 0].set_ylabel('||innovation||')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Condition number
            axes[0, 1].semilogy(self.history['condition_numbers'], alpha=0.7)
            axes[0, 1].set_title('Covariance Matrix Condition Number')
            axes[0, 1].set_ylabel('cond(P)')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Covariance trace
            axes[1, 0].plot(self.history['covariance_norms'], alpha=0.7)
            axes[1, 0].set_title('State Uncertainty (||P||)')
            axes[1, 0].set_ylabel('||P||_F')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Innovation mean trend
            means = [s['mean'] for s in self.history['residual_stats']]
            axes[1, 1].plot(means, alpha=0.7, label='Mean')
            axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
            axes[1, 1].set_title('Innovation Mean (Should Be ≈ 0)')
            axes[1, 1].set_ylabel('Mean innovation')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].legend()
            
            plt.tight_layout()
            return fig

---

Tuning Guidelines
=================

**Choose Adaptation Strategy:**

+-----------------------------+-------------+------------+---------------+
| Method                      | Convergence | Robustness | Computational |
+=============================+=============+============+===============+
| Static tuning               | N/A (fixed) | High       | Low           |
+-----------------------------+-------------+------------+---------------+
| Gain adaptation (β scaling) | Medium      | High       | Very Low      |
+-----------------------------+-------------+------------+---------------+
| R estimation (adaptive)     | Medium      | Medium     | Low           |
+-----------------------------+-------------+------------+---------------+
| LMS                         | Slow        | Low        | Very Low      |
+-----------------------------+-------------+------------+---------------+
| RLS                         | Fast        | Medium     | Medium        |
+-----------------------------+-------------+------------+---------------+

**Parameter Selection:**

- **Forgetting factor** ($\alpha$ or $\lambda$): 
  
  - Fast-changing system: $\alpha = 0.90$ (short memory ~10 steps)
  - Medium dynamics: $\alpha = 0.95$ (medium memory ~20 steps)
  - Slow dynamics: $\alpha = 0.99$ (long memory ~100 steps)

- **LMS step size** ($\mu$):
  
  - Too large: Instability, oscillation
  - Too small: Slow convergence
  - Typical: $\mu = 0.01$ to $0.1$
  - Rule: $\mu < \frac{1}{3 \times \text{signal power}}$

- **Gain scaling limits**:
  
  - Prevent extreme swings: $0.5 \leq \beta \leq 1.5$
  - More conservative: $0.7 \leq \beta \leq 1.3$

---

Common Pitfalls
===============

1. **Over-adaptation**: Tracking noise instead of signal
   
   - *Fix*: Increase forgetting factor, add regularization

2. **Singularity/ill-conditioning**: Covariance matrix becomes rank-deficient
   
   - *Fix*: Add regularization, limit condition numbers, use Joseph form

3. **Divergence undetected**: Filter fails silently
   
   - *Fix*: Monitor normalized innovation statistics continuously

4. **Poor initialization**: Adaptive phase leads to bad estimates
   
   - *Fix*: Start with conservative (high) noise estimates

5. **Sensor bias**: Adaptive filter cannot correct for systematic bias
   
   - *Fix*: Detect and remove bias separately (e.g., least-squares batch estimation)

---

See Also
========

- [Kalman Filter Tuning Guide](kalman_filter_tuning.rst) — Static parameter selection
- [Particle Filters](particle_filters.rst) — Non-adaptive but handles non-Gaussian distributions
- [Smoothing Algorithms](smoothing.rst) — Batch refinement after filtering
- [Troubleshooting Guide](troubleshooting.rst) — Filter divergence diagnosis

**References:**

- Bar-Shalom, Li & Kirubarajan (2001) — *Estimation with Applications to Tracking and Navigation*
- Grewal & Andrews (2015) — *Kalman Filtering Theory and Practice*
- Haykin (2002) — *Adaptive Filter Theory* — Comprehensive LMS/RLS theory
