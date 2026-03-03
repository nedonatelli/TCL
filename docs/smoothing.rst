Smoothing Algorithms & Offline Estimation
========================================

*Comprehensive guide to batch and fixed-lag smoothing for improved state estimation using forward and backward passes.*

Smoothing differs from filtering: while filters use only past measurements, smoothers use all available measurements (past and future) for better accuracy. Ideal for offline analysis, trajectory refinement, and post-processing.

**Table of Contents:**

- Smoothing Fundamentals
- Rauch-Tsch iersky-Striebel (RTS) Smoother
- Fixed-Lag Smoothing
- Two-Filter Approach
- Fixed-Interval Smoothing
- Performance Comparison
- Practical Implementation
- Applications
- Common Issues & Solutions
- Best Practices

Smoothing Fundamentals
---------------------

**Filtering vs Smoothing:**

- **Filter**: $p(x_k | z_{1:k})$ - estimate at time $k$ using measurements up to $k$
- **Smoother**: $p(x_k | z_{1:N})$ - estimate at time $k$ using all $N$ measurements

**Why Smoothing?**

1. Better accuracy: ~30-50% lower RMS error vs filtering
2. No real-time constraint: can use all data
3. Ideal for trajectory analysis and post-processing
4. Symmetric uncertainty (less bias than filtering)

**Trade-off:**

- Filtering: Real-time, lower latency
- Smoothing: Offline only, requires complete dataset

**Mathematical Concept:**

Smoothing likelihood:
$$p(x_k|z_{1:N}) \propto p(z_{k+1:N}|x_k) p(x_k|z_{1:k})$$

Where:
- $p(x_k|z_{1:k})$: forward filter estimate
- $p(z_{k+1:N}|x_k)$: backward filter estimate


Rauch-Tschiersky-Striebel (RTS) Smoother
----------------------------------------

**Algorithm: Forward-Backward Smoothing**

The RTS smoother combines forward (filtering) and backward (smoothing) passes.

**Step 1: Forward Pass (Standard Kalman Filter)**

.. code-block:: python

    def forward_pass(measurements, F, H, Q, R, x0, P0):
        """
        Standard Kalman filter forward pass.
        
        Args:
            measurements: N × m array of measurements
            F: n × n state transition matrix
            H: m × n measurement matrix
            Q: n × n process noise covariance
            R: m × m measurement noise covariance
            x0: initial state estimate
            P0: initial covariance
        
        Returns:
            x_f: N × n forward estimates
            P_f: N × n × n forward covariances
            x_pred: N × n predicted states
            P_pred: N × n × n predicted covariances
        """
        N = len(measurements)
        n = len(x0)
        
        x_f = np.zeros((N, n))
        P_f = np.zeros((N, n, n))
        x_pred = np.zeros((N, n))
        P_pred = np.zeros((N, n, n))
        
        x = x0.copy()
        P = P0.copy()
        
        for k in range(N):
            # Predict
            x_pred[k] = F @ x
            P_pred[k] = F @ P @ F.T + Q
            
            # Update
            y = measurements[k] - H @ x_pred[k]
            S = H @ P_pred[k] @ H.T + R
            K = P_pred[k] @ H.T @ np.linalg.inv(S)
            
            x = x_pred[k] + K @ y
            P = (np.eye(n) - K @ H) @ P_pred[k]
            
            x_f[k] = x
            P_f[k] = P
        
        return x_f, P_f, x_pred, P_pred


**Step 2: Backward Pass (RTS Smoother)**

.. code-block:: python

    def rts_backward_pass(x_f, P_f, x_pred, P_pred, F):
        """
        RTS backward smoothing pass.
        
        Args:
            x_f: N × n forward filter estimates
            P_f: N × n × n forward covariances
            x_pred: N × n predicted states
            P_pred: N × n × n predicted covariances
            F: state transition matrix
        
        Returns:
            x_s: N × n smoothed estimates (RTS smoother)
            P_s: N × n × n smoothed covariances
        """
        N = len(x_f)
        n = x_f.shape[1]
        
        x_s = np.zeros_like(x_f)
        P_s = np.zeros_like(P_f)
        
        # Initialize: last estimate is same as filtering
        x_s[-1] = x_f[-1]
        P_s[-1] = P_f[-1]
        
        # Backward pass
        for k in range(N - 2, -1, -1):
            # Smoother gain
            D = P_f[k] @ F.T @ np.linalg.inv(P_pred[k+1])
            
            # Smoothed estimate
            x_s[k] = x_f[k] + D @ (x_s[k+1] - x_pred[k+1])
            
            # Smoothed covariance
            P_s[k] = P_f[k] + D @ (P_s[k+1] - P_pred[k+1]) @ D.T
        
        return x_s, P_s


**Complete RTS Smoother:**

.. code-block:: python

    class RTSSmoother:
        """Rauch-Tschiersky-Striebel backward smoother."""
        
        def __init__(self, F, H, Q, R):
            """
            Args:
                F: state transition matrix
                H: measurement matrix
                Q: process noise covariance
                R: measurement noise covariance
            """
            self.F = F
            self.H = H
            self.Q = Q
            self.R = R
        
        def smooth(self, measurements, x0, P0):
            """
            Perform complete RTS smoothing.
            
            Returns:
                x_smooth: smoothed trajectory
                P_smooth: smoothed covariances
                x_filter: for comparison (forward pass only)
                P_filter: for comparison
            """
            # Forward pass
            x_f, P_f, x_pred, P_pred = forward_pass(
                measurements, self.F, self.H, self.Q, self.R, x0, P0
            )
            
            # Backward pass
            x_s, P_s = rts_backward_pass(x_f, P_f, x_pred, P_pred, self.F)
            
            return x_s, P_s, x_f, P_f
        
        def improvement(self, P_filter, P_smooth):
            """Quantify smoothing improvement."""
            # Ratio of filter to smoother uncertainty
            ratio = np.trace(P_filter) / np.trace(P_smooth)
            percent_improvement = (1 - 1/ratio) * 100
            return percent_improvement


**Practical Example:**

.. code-block:: python

    def example_rts_smoothing():
        """Simple 1D constant velocity model."""
        # System
        dt = 0.1
        F = np.array([[1, dt], [0, 1]])  # state transition
        H = np.array([[1, 0]])  # measure position only
        Q = np.eye(2) * 0.01  # process noise
        R = np.array([[1.0]])  # measurement noise
        
        # Measurement data (noisy observations)
        true_pos = np.linspace(0, 10, 100)
        measurements = true_pos + np.random.normal(0, 1, 100)
        
        # Initial estimates
        x0 = np.array([measurements[0], 0])
        P0 = np.eye(2)
        
        # Smooth
        smoother = RTSSmoother(F, H, Q, R)
        x_smooth, P_smooth, x_filter, P_filter = smoother.smooth(
            measurements[:, np.newaxis], x0, P0
        )
        
        # Compare
        improvement = smoother.improvement(P_filter, P_smooth)
        print(f"Smoothing improvement: {improvement:.1f}%")
        
        return x_smooth, P_smooth, x_filter, P_filter


Fixed-Lag Smoothing
-------------------

**Real-Time Smoothing with Bounded Delay**

Fixed-lag smoother maintains a sliding window of past estimates, providing lower latency than batch smoothing.

.. code-block:: python

    class FixedLagSmoother:
        """
        Fixed-lag smoother: combines current filter update with lag-step 
        past estimates for near-real-time smoothing.
        """
        
        def __init__(self, F, H, Q, R, lag=5):
            """
            Args:
                lag: number of timesteps to remember (fixed lag)
            """
            self.F = F
            self.H = H
            self.Q = Q
            self.R = R
            self.lag = lag
            
            # Buffers
            self.history_x = []  # state estimates
            self.history_P = []  # covariances
            self.history_x_pred = []
            self.history_P_pred = []
        
        def predict_update(self, measurement):
            """One step: predict, update current, smooth past."""
            n = self.F.shape[0]
            
            if len(self.history_x) == 0:
                # Initialize
                x = np.zeros(n)
                P = np.eye(n)
            else:
                # Predict from last
                x = self.F @ self.history_x[-1]
                P = self.F @ self.history_P[-1] @ self.F.T + self.Q
            
            # Store predicted
            self.history_x_pred.append(x.copy())
            self.history_P_pred.append(P.copy())
            
            # Update with measurement
            y = measurement - self.H @ x
            S = self.H @ P @ self.H.T + self.R
            K = P @ self.H.T @ np.linalg.inv(S)
            
            x = x + K @ y
            P = (np.eye(n) - K @ self.H) @ P
            
            # Store updated
            self.history_x.append(x)
            self.history_P.append(P)
            
            # Smooth lag steps back
            if len(self.history_x) > self.lag:
                self._smooth_backward(lag_steps=self.lag)
            
            # Return current estimate (already smoothed if possible)
            return x, P
        
        def _smooth_backward(self, lag_steps):
            """Smooth past estimates using RTS-like update."""
            # Smooth back lag_steps
            idx = len(self.history_x) - lag_steps - 1
            
            if idx < 0:
                return
            
            # Multiple RTS iterations for the lag
            for _ in range(lag_steps):
                if idx < 0:
                    break
                
                # RTS smoother gain
                D = (self.history_P[idx] @ self.F.T @ 
                     np.linalg.inv(self.history_P_pred[idx+1]))
                
                # Update laggy estimate
                self.history_x[idx] = (self.history_x[idx] + 
                    D @ (self.history_x[idx+1] - self.history_x_pred[idx+1]))
                self.history_P[idx] = (self.history_P[idx] + 
                    D @ (self.history_P[idx+1] - self.history_P_pred[idx+1]) @ D.T)
                
                idx -= 1
        
        def get_delayed_estimate(self, steps_back):
            """Get smoothed estimate from steps_back ago."""
            if steps_back < len(self.history_x):
                idx = len(self.history_x) - steps_back - 1
                return self.history_x[idx], self.history_P[idx]
            return None, None


Two-Filter Approach
-------------------

**Forward and Backward Filtering**

Alternative to RTS: run forward and backward filters independently, then combine.

.. code-block:: python

    class TwoFilterSmoother:
        """Combine forward and backward filters for smoothing."""
        
        def __init__(self, F, H, Q, R):
            self.F = F
            self.H = H
            self.Q = Q
            self.R = R
        
        def backward_filter(self, measurements, x0_back, P0_back):
            """
            Run Kalman filter backward (in reverse time).
            
            Reverse dynamics: x_{k-1} = F^{-T} x_k
            """
            N = len(measurements)
            n = len(x0_back)
            
            x_b = np.zeros((N, n))
            P_b = np.zeros((N, n, n))
            
            F_inv = np.linalg.inv(self.F)
            
            x = x0_back.copy()
            P = P0_back.copy()
            
            # Process in reverse
            for k in range(N - 1, -1, -1):
                # Update at current time
                y = measurements[k] - self.H @ x
                S = self.H @ P @ self.H.T + self.R
                K = P @ self.H.T @ np.linalg.inv(S)
                
                x = x + K @ y
                P = (np.eye(n) - K @ self.H) @ P
                
                x_b[k] = x
                P_b[k] = P
                
                # Predict backward
                if k > 0:
                    x = F_inv.T @ x
                    P = F_inv.T @ P @ F_inv + self.Q
            
            return x_b, P_b
        
        def combine_estimates(self, x_f, P_f, x_b, P_b):
            """
            Combine forward and backward filter estimates.
            
            Information filter form: combine inverses of covariances.
            """
            N = len(x_f)
            x_combined = np.zeros_like(x_f)
            P_combined = np.zeros_like(P_f)
            
            for k in range(N):
                # Information form combination
                P_f_inv = np.linalg.inv(P_f[k])
                P_b_inv = np.linalg.inv(P_b[k])
                
                # Combined information and state
                P_combined[k] = np.linalg.inv(P_f_inv + P_b_inv)
                x_combined[k] = P_combined[k] @ (
                    P_f_inv @ x_f[k] + P_b_inv @ x_b[k]
                )
            
            return x_combined, P_combined
        
        def smooth(self, measurements, x0_f, P0_f, x0_b, P0_b):
            """Complete two-filter smoothing."""
            # Forward filter
            x_f, P_f, _, _ = forward_pass(
                measurements, self.F, self.H, self.Q, self.R, x0_f, P0_f
            )
            
            # Backward filter
            x_b, P_b = self.backward_filter(measurements, x0_b, P0_b)
            
            # Combine
            x_smooth, P_smooth = self.combine_estimates(x_f, P_f, x_b, P_b)
            
            return x_smooth, P_smooth


Fixed-Interval Smoothing
------------------------

**Batch Smoothing with Fixed Time Window**

Process complete measurement epochs, then refine.

.. code-block:: python

    def fixed_interval_smooth(measurements, batch_size, overlap=0):
        """
        Smooth in batches with optional overlap.
        
        Args:
            measurements: all measurements
            batch_size: number of measurements per batch
            overlap: number of overlapping measurements between batches
        """
        N = len(measurements)
        smoothed_trajectory = np.zeros_like(measurements)
        
        stride = batch_size - overlap
        
        for start_idx in range(0, N, stride):
            end_idx = min(start_idx + batch_size, N)
            batch = measurements[start_idx:end_idx]
            
            if len(batch) < 2:
                continue
            
            # Smooth this batch
            smoother = RTSSmoother(F, H, Q, R)
            x_smooth, _ = smoother.forward_backward(batch)
            
            # Store (with handling for overlap)
            smoothed_trajectory[start_idx:end_idx] = x_smooth
        
        return smoothed_trajectory


Performance Comparison
---------------------

**Accuracy Metrics:**

.. code-block:: python

    def compare_filter_vs_smoother(x_true, x_filter, x_smooth, P_filter, P_smooth):
        """Quantify smoothing improvement."""
        
        # RMS errors
        error_filter = np.sqrt(np.mean((x_filter - x_true)**2, axis=0))
        error_smooth = np.sqrt(np.mean((x_smooth - x_true)**2, axis=0))
        
        # Uncertainty reduction
        uncertainty_filter = np.sqrt(np.mean(np.diagonal(P_filter, axis1=1, axis2=2)[:, 0]))
        uncertainty_smooth = np.sqrt(np.mean(np.diagonal(P_smooth, axis1=1, axis2=2)[:, 0]))
        
        print("=== Filter vs Smoother Comparison ===")
        print(f"Filter RMS error:    {error_filter[0]:.4f}")
        print(f"Smoother RMS error:  {error_smooth[0]:.4f}")
        print(f"Improvement:         {(1 - error_smooth[0]/error_filter[0])*100:.1f}%")
        print()
        print(f"Filter uncertainty:   {uncertainty_filter:.4f}")
        print(f"Smoother uncertainty: {uncertainty_smooth:.4f}")
        print(f"Reduction:           {(1 - uncertainty_smooth/uncertainty_filter)*100:.1f}%")


Practical Implementation
------------------------

**Complete Smoothing System:**

.. code-block:: python

    class TrajectorySmoother:
        """Full trajectory smoothing pipeline."""
        
        def __init__(self, process_model, measurement_model, process_noise, 
                     measurement_noise):
            self.F = process_model
            self.H = measurement_model
            self.Q = process_noise
            self.R = measurement_noise
        
        def smooth_trajectory(self, measurements, x0, P0):
            """Smooth complete trajectory."""
            smoother = RTSSmoother(self.F, self.H, self.Q, self.R)
            x_smooth, P_smooth, x_filter, P_filter = smoother.smooth(
                measurements, x0, P0
            )
            
            return {
                'smoothed': x_smooth,
                'filtered': x_filter,
                'smoother_cov': P_smooth,
                'filter_cov': P_filter
            }
        
        def estimate_uncertainties(self, P_smooth):
            """Extract uncertainty bounds."""
            std_devs = np.sqrt(np.diagonal(P_smooth, axis1=1, axis2=2))
            return std_devs
        
        def compute_likelihoods(self, measurements, x_smooth, P_smooth):
            """Evaluate measurement likelihoods at smoothed trajectory."""
            N = len(measurements)
            likelihoods = np.zeros(N)
            
            for k in range(N):
                # Predicted measurement
                z_pred = self.H @ x_smooth[k]
                
                # Innovation
                y = measurements[k] - z_pred
                
                # Innovation covariance
                S = self.H @ P_smooth[k] @ self.H.T + self.R
                
                # Gaussian likelihood
                likelihoods[k] = np.exp(-0.5 * y @ np.linalg.inv(S) @ y)
            
            return likelihoods


Applications
-----------

**1. Post-Mission Analysis**

Analyze complete flight/mission after recording:

.. code-block:: python

    def analyze_mission_post_flight(telemetry_log):
        """Process complete mission using batch smoothing."""
        measurements = telemetry_log['measurements']
        timestamps = telemetry_log['timestamps']
        
        smoother = RTSSmoother(F, H, Q, R)
        x_smooth, P_smooth = smoother.smooth(measurements, x0, P0)
        
        # Report
        report = {
            'trajectory': x_smooth,
            'uncertainties': np.sqrt(np.diagonal(P_smooth, axis1=1, axis2=2)),
            'timestamps': timestamps,
            'filter_vs_smooth_improvement': compute_improvement(P_filter, P_smooth)
        }
        
        return report


**2. Multi-Sensor Fusion with Latency**

Resync delayed measurements:

.. code-block:: python

    def fuse_delayed_measurements(primary_measurements, delayed_measurements_with_time, 
                                  delay_samples):
        """
        Insert delayed measurement into smoothed trajectory.
        
        Smoother handles time-correlation naturally.
        """
        smoother = RTSSmoother(F, H, Q, R)
        x_smooth, P_smooth = smoother.smooth(primary_measurements, x0, P0)
        
        # Align delayed measurement
        insertion_idx = len(primary_measurements) - delay_samples
        proposed_state = x_smooth[insertion_idx]
        
        return proposed_state


**3. GPS/INS Trajectory Refinement**

Smooth GPS-corrected inertial trajectory:

.. code-block:: python

    def refine_inertial_trajectory(ins_trajectory, gps_fixes, gps_times):
        """
        INS provides high-rate, GPS provides occasional corrections.
        Smoother integrates both optimally.
        """
        # Blend measurements
        measurements = np.zeros_like(ins_trajectory)
        for i, (time, fix) in enumerate(zip(gps_times, gps_fixes)):
            idx = int(time / dt)
            measurements[idx] = fix
        
        smoother = RTSSmoother(F_ins, H_gps, Q_ins, R_gps)
        x_refined, P_refined = smoother.smooth(measurements, x0, P0)
        
        return x_refined


Common Issues & Solutions
-------------------------

**Problem: Backward Pass Diverges**

Solution: Check matrix invertibility in RTS gain:

.. code-block:: python

    def stable_rts_gain(P_f, P_pred, F):
        """Compute RTS gain with numerical stability."""
        try:
            # Standard RTS
            D = P_f @ F.T @ np.linalg.inv(P_pred)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if singular
            D = P_f @ F.T @ np.linalg.pinv(P_pred)
        
        return D


**Problem: Backward Covariance Grows**

Indicates filter was too optimistic (P too small). Solutions:

1. Increase process noise Q
2. Reduce measurement noise R (if overestimated)
3. Use Joseph form covariance update

.. code-block:: python

    def joseph_covariance_update(P, K, H):
        """Joseph stabilized covariance (numerically stable)."""
        n = P.shape[0]
        I_KH = np.eye(n) - K @ H
        return I_KH @ P @ I_KH.T + K @ R @ K.T


**Problem: Smoother Solution Violates Constraints**

Use constrained smoothing:

.. code-block:: python

    def constrained_smooth(x_filter, P_filter, constraints):
        """Project smoothed estimates into constraint set."""
        # Quadratic programming to find closest feasible point
        from scipy.optimize import minimize
        
        def constraint_cost(x):
            # Minimize distance from filter estimate
            return np.sum((x - x_filter)**2)
        
        # ... set up constraints, solve


Best Practices
--------------

1. **Batch Smoothing** (Offline Analysis)
   - Use RTS smoother for complete datasets
   - Provides optimal estimates given all data
   - ~30-50% accuracy improvement over filtering

2. **Fixed-Lag** (Near Real-Time)
   - When you need low latency but can wait for lag
   - Typical lag: 5-10 timesteps
   - Good balance: ~20% improvement, low delay

3. **Monitor Consistency**
   - Check that P_smooth < P_filter everywhere (if not, filter issue)
   - Verify backward pass using simulated data
   - Compare forward-only vs full smoother

4. **Memory Efficient**
   - Don't store full history; process in batches
   - Use fixed-interval smoothing for long missions
   - Overlap batches to avoid edge artifacts

5. **Numerical Stability**
   - Use Joseph covariance form
   - Check matrix condition numbers
   - Use pseudo-inverse for near-singular matrices

6. **Parameter Tuning**
   - Validate process/measurement noise with smoothed residuals
   - Residuals should be white noise (check autocorrelation)
   - Use smoother to detect parameter mismatches

Troubleshooting
---------------

**Diagnostic Checklist:**

.. code-block:: python

    def diagnose_smoother(measurements, x_filter, x_smooth, P_filter, P_smooth):
        """Check smoother quality."""
        
        issues = []
        
        # Check 1: P_smooth < P_filter
        ratio = np.trace(P_filter) / np.trace(P_smooth)
        if ratio < 1.1:
            issues.append("Minimal improvement - check data quality")
        
        # Check 2: Backward covariance growth
        if P_smooth[0, 0, 0] > P_smooth[-1, 0, 0]:
            issues.append("Backward covariance grows - check Q/R tuning")
        
        # Check 3: Smoothed state sanity
        max_vel = np.max(np.abs(x_smooth[:, 1]))
        if max_vel > 1000:
            issues.append("Unrealistic velocities in smoother")
        
        # Check 4: Residual whiteness
        residuals = measurements - (H @ x_smooth.T).T
        acf = np.correlate(residuals[:, 0], residuals[:, 0], mode='same')
        if acf[len(acf)//2 + 1] > 0.3 * acf[len(acf)//2]:
            issues.append("Residuals auto-correlated - check measurements")
        
        return issues if issues else ["✓ Smoother diagnostics OK"]


See Also
~~~~~~~~

- :doc:`recipes` - Ready-to-use smoothing implementations
- :doc:`kalman_filter_tuning` - Parameter selection for smoothers
- :doc:`troubleshooting` - Smoothing issues and debugging
- API: `tcl.smoothers` module for RTS, fixed-lag implementations
