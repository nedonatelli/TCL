Navigation & Inertial Measurement Systems
=========================================

*Comprehensive guide to inertial navigation systems (INS), mechanization, GNSS/INS integration, and practical implementation.*

This guide covers the complete workflow for building navigation systems using inertial sensors, including INS propagation, GNSS integration, and real-time filtering architectures.

**Table of Contents:**

- INS Fundamentals
- INS Mechanization
- Error Modeling & Propagation
- GNSS/INS Integration Architectures
- Practical Implementation
- Diagnostics & Troubleshooting
- Best Practices

INS Fundamentals
----------------

Inertial Navigation Systems compute position, velocity, and attitude by integrating acceleration and rotation rate measurements from accelerometers and gyroscopes.

**Key Sensors:**

- **Accelerometers**: Measure specific force (acceleration + gravity) in body frame
- **Gyroscopes**: Measure rotation rates about body axes
- **Integrated IMU**: Combined 6-DOF sensor package (3 accelerometers, 3 gyroscopes)

**INS Coordinate Frames:**

1. **Body Frame (b-frame)**: Fixed to vehicle, rotated with aircraft/spacecraft
2. **Navigation Frame (n-frame)**: Local tangent plane (NED or ENU)
3. **ECEF Frame**: Earth-centered, Earth-fixed (for global navigation)
4. **Inertial Frame (i-frame)**: Inertial reference (for precision work)

**Mechanization Equation:**

The fundamental INS equations relate specific force and rotation rates to position and velocity:

.. math::

    \dot{\mathbf{v}}^n = \mathbf{C}_b^n \mathbf{f}^b - (\mathbf{2}\boldsymbol{\omega}_{ie}^n + \boldsymbol{\omega}_{en}^n) \times \mathbf{v}^n + \mathbf{g}^n

    \dot{\mathbf{p}}^n = \mathbf{v}^n

    \dot{\mathbf{C}}_b^n = \mathbf{C}_b^n [\boldsymbol{\omega}_{ib}^b]_\times - [\boldsymbol{\omega}_{in}^n]_\times \mathbf{C}_b^n

Where:
- :math:`\mathbf{v}^n`: velocity in nav frame
- :math:`\mathbf{C}_b^n`: direction cosine matrix (body to nav)
- :math:`\mathbf{f}^b`: specific force measurements
- :math:`\boldsymbol{\omega}_{ie}^n`: Earth rotation rate
- :math:`\boldsymbol{\omega}_{en}^n`: transport rate (due to vehicle motion)
- :math:`\mathbf{g}^n`: gravity vector in nav frame

Basic INS Propagation
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import numpy as np
    from tcl.coordinate_systems import ecef2enu, enu2ecef
    from tcl.rotations import dcm_from_euler, euler_from_dcm
    
    class BasicINS:
        """Simple strapdown INS propagator (NED frame)."""
        
        def __init__(self, lat0, lon0, alt0, euler0=(0, 0, 0)):
            """Initialize INS state at reference position."""
            self.lat = lat0  # latitude (rad)
            self.lon = lon0  # longitude (rad)
            self.alt = alt0  # altitude (m)
            self.v_ned = np.array([0.0, 0.0, 0.0])  # velocity NED (m/s)
            
            # Attitude (Euler angles: roll, pitch, yaw)
            self.euler = np.array(euler0, dtype=float)
            self.C_b2n = dcm_from_euler(*euler0)  # body to NED DCM
            
            # Earth parameters
            self.we = 7.2921e-5  # Earth rotation rate (rad/s)
            self.Re = 6378137.0  # Earth radius (m)
            self.g = 9.81  # gravity (m/s²)
        
        def propagate(self, accel_b, gyro_b, dt):
            """Propagate INS state over time step dt."""
            
            # Transform specific force to NED frame
            f_ned = self.C_b2n @ accel_b
            
            # Coriolis and centrifugal accelerations (simplified)
            coriolis = -2.0 * np.array([0, self.we * np.cos(self.lat), 
                                        self.we * np.sin(self.lat)])
            
            # Transport rate (due to vehicle motion over Earth)
            dlat_dt = self.v_ned[0] / (self.Re + self.alt)
            dlon_dt = self.v_ned[1] / ((self.Re + self.alt) * np.cos(self.lat))
            
            omega_en = np.array([dlon_dt * np.sin(self.lat),
                                -dlat_dt,
                                -dlon_dt * np.cos(self.lat)])
            
            # Gravity (simplified - constant vector in local frame)
            g_ned = np.array([0, 0, self.g])
            
            # Velocity update: dv/dt = f + coriolis + gravity_anomaly
            gravity_anomaly = coriolis + np.cross(omega_en, self.v_ned)
            a_ned = f_ned + gravity_anomaly + g_ned
            
            self.v_ned += a_ned * dt
            
            # Position update
            self.lat += dlat_dt * dt
            self.lon += dlon_dt * dt
            self.alt += -self.v_ned[2] * dt  # negative because NED is down-positive
            
            # Attitude update using gyro measurements
            # Angular velocity in navigation frame
            omega_ib_b = gyro_b  # body rates from gyro
            
            # Skew-symmetric form
            omega_ib_b_skew = np.array([
                [0, -omega_ib_b[2], omega_ib_b[1]],
                [omega_ib_b[2], 0, -omega_ib_b[0]],
                [-omega_ib_b[1], omega_ib_b[0], 0]
            ])
            
            omega_en_skew = np.array([
                [0, -omega_en[2], omega_en[1]],
                [omega_en[2], 0, -omega_en[0]],
                [-omega_en[1], omega_en[0], 0]
            ])
            
            # DCM update: dC/dt = C @ [omega_ib]_x - [omega_in]_x @ C
            dC_dt = self.C_b2n @ omega_ib_b_skew - omega_en_skew @ self.C_b2n
            self.C_b2n += dC_dt * dt
            
            # Normalize DCM (optional but improves numerical stability)
            U, _, VT = np.linalg.svd(self.C_b2n)
            self.C_b2n = U @ VT
        
        def get_position(self):
            """Get current position (lat, lon, alt)."""
            return np.array([self.lat, self.lon, self.alt])
        
        def get_velocity(self):
            """Get current velocity in NED frame."""
            return self.v_ned.copy()


INS Error Sources & Modeling
-----------------------------

Real inertial sensors exhibit various error characteristics that cause INS drift:

**Accelerometer Errors:**

.. math::

    \mathbf{f}_{measured} = \mathbf{f}_{true} + \mathbf{b}_a + \mathbf{S}_a \mathbf{f}_{true} + \mathbf{n}_a + \text{temp effects}

Where:
- :math:`\mathbf{b}_a`: bias (constant offset, drifts over time)
- :math:`\mathbf{S}_a`: scale factor (gain error)
- :math:`\mathbf{n}_a`: white noise (high frequency)

**Gyroscope Errors:**

- **Bias**: Several types (constant, random walk, rate-dependent)
- **Scale factor**: Gain errors on rotation rates
- **Noise**: Angle random walk and rate random walk
- **Coupling**: Accelerometer-induced errors (g-sensitivity)

**Practical Error Magnitudes (Mid-Grade INS):**

========== ========================= ======================
Sensor     Typical Bias              Random Walk Rate
========== ========================= ======================
Accel      50-100 mg (0.5-1 m/s²)   ~0.01 m/s²/√hr
Gyro       50-200 °/hr (0.01-0.055 °/s) ~0.3 °/hr/√hr
========== ========================= ======================

**INS Divergence Over Time (No Updates):**

For unaided INS (no GNSS), position error grows approximately as:

.. math::

    \sigma_{\text{position}} \approx 0.5 \sigma_{\text{accel\_bias}} \cdot t^2

Example: With 100 mg accelerometer bias, position error ~ 1 km after 1 hour.

GNSS/INS Integration Architectures
----------------------------------

**1. Loosely Coupled Integration**

GNSS and INS process measurements independently. GNSS provides position/velocity updates.

Advantages:
- Simple implementation
- Works with standard GNSS receivers
- Easy to debug

Disadvantages:
- Slower convergence after GNSS outage
- Cannot use GNSS during high dynamics

.. code-block:: python

    class LooselyIntegratedNav:
        """Loosely coupled GNSS/INS navigation."""
        
        def __init__(self, ins, q_accel=0.01, q_gyro=1e-6):
            """
            ins: BasicINS or similar
            q_accel: process noise for accel bias (m²/s⁵)
            q_gyro: process noise for gyro bias (rad²/s³)
            """
            self.ins = ins
            
            # State: [lat, lon, alt, v_n, v_e, v_d, euler_angles, accel_bias, gyro_bias]
            self.state = np.zeros(15)
            self.state[0:3] = ins.get_position()
            self.state[3:6] = ins.get_velocity()
            self.state[6:9] = ins.euler
            
            # Covariance (diagonal for simplicity)
            self.P = np.eye(15)
            self.P[0:3, 0:3] *= 1e4  # position uncertainty (m²)
            self.P[3:6, 3:6] *= 1e2  # velocity uncertainty (m²/s²)
            self.P[6:9, 6:9] *= 0.01  # attitude uncertainty (rad²)
            self.P[9:12, 9:12] *= 0.1  # accel bias (m²/s⁴)
            self.P[12:15, 12:15] *= 1e-4  # gyro bias (rad²/s²)
            
            self.q_accel = q_accel
            self.q_gyro = q_gyro
        
        def predict_ins(self, accel_b, gyro_b, dt):
            """INS propagation step."""
            # Compensate measurements with estimated biases
            accel_corr = accel_b - self.state[9:12]
            gyro_corr = gyro_b - self.state[12:15]
            
            # Propagate INS
            self.ins.propagate(accel_corr, gyro_corr, dt)
            
            # Predict Kalman filter state (simplified - no cross-coupling)
            self.state[0:3] = self.ins.get_position()
            self.state[3:6] = self.ins.get_velocity()
            self.state[6:9] = self.ins.euler
            
            # Increase covariance (process noise)
            self.P[0:3, 0:3] += np.eye(3) * 10 * dt  # position drift
            self.P[3:6, 3:6] += np.eye(3) * 0.1 * dt  # velocity drift
            self.P[9:12, 9:12] += np.eye(3) * self.q_accel * dt
            self.P[12:15, 12:15] += np.eye(3) * self.q_gyro * dt
        
        def update_gnss(self, gnss_pos, gnss_vel, pos_cov=100, vel_cov=1):
            """GNSS measurement update."""
            # GNSS provides position and velocity updates
            z = np.hstack([gnss_pos, gnss_vel])
            
            # Measurement matrix (observe position and velocity directly)
            H = np.zeros((6, 15))
            H[0:3, 0:3] = np.eye(3)
            H[3:6, 3:6] = np.eye(3)
            
            # Measurement covariance
            R = np.diag([pos_cov]*3 + [vel_cov]*3)
            
            # Kalman update
            innovation = z - self.state[0:6]
            S = H @ self.P @ H.T + R
            K = self.P @ H.T @ np.linalg.inv(S)
            
            self.state += K @ innovation
            self.P = (np.eye(15) - K @ H) @ self.P


**2. Tightly Coupled Integration**

INS and GNSS share a single Kalman filter. GNSS measurements are raw pseudoranges, not derived position.

Advantages:
- Better performance during signal degradation
- Faster convergence
- More robust to GNSS outages

Disadvantages:
- Complex implementation
- Requires raw GNSS data
- Needs GNSS receiver control

**3. Ultra-Tight Coupling**

INS state is used to predict GNSS signal tracking parameters (carrier frequency, code delay). The tracking loops and INS filter are integrated.

Advantages:
- Works in severe signal degradation
- Maintains tracking in high-dynamic environments

Disadvantages:
- Requires custom GNSS receiver
- Complex real-time implementation


Practical Implementation Considerations
---------------------------------------

**Coning and Sculling Algorithms**

When integrating gyro measurements over finite time steps, simple integration accumulates rotation errors. Coning algorithms reduce this error:

.. code-block:: python

    def coning_correction(omega_prev, omega_curr, omega_next, dt):
        """
        Reduce coning error in gyro integration using three-point algorithm.
        
        Args:
            omega_prev, omega_curr, omega_next: angular rates at three time steps
        
        Returns:
            corrected_rotation: rotation vector for current interval
        """
        # Simple first-order coning correction
        coning_term = (1.0/12.0) * np.cross(omega_prev, omega_curr)
        
        rotation = omega_curr * dt + coning_term * dt**2
        return rotation


**Handling GNSS Outages**

During GNSS signal loss, only INS measurements are available. Position/velocity drift is proportional to sensor errors.

.. code-block:: python

    def estimate_gnss_outage_duration(accel_bias_std, desired_error=100.0):
        """
        Estimate maximum GNSS outage duration before position error exceeds limit.
        
        Args:
            accel_bias_std: standard deviation of accel bias (m/s²)
            desired_error: acceptable position error (m)
        
        Returns:
            max_duration: seconds of GNSS outage
        """
        # Position error ~ 0.5 * bias * t²
        # Solving for t: t = sqrt(2 * error / bias)
        max_duration = np.sqrt(2.0 * desired_error / accel_bias_std)
        return max_duration


**Alignment & Initialization**

Proper system alignment is critical for INS accuracy:

.. code-block:: python

    class INSAlignment:
        """Compute initial attitude from accelerometer and magnetometer data."""
        
        @staticmethod
        def level_alignment(accel_samples, num_samples=100):
            """
            Estimate pitch and roll from accelerometer samples (assumed stationary).
            """
            # Average accelerometer readings (removes noise)
            accel_avg = np.mean(accel_samples[-num_samples:], axis=0)
            
            # Roll: atan2(a_y, a_z)
            roll = np.arctan2(accel_avg[1], accel_avg[2])
            
            # Pitch: atan2(-a_x, sqrt(a_y² + a_z²))
            pitch = np.arctan2(-accel_avg[0], 
                              np.sqrt(accel_avg[1]**2 + accel_avg[2]**2))
            
            return roll, pitch
        
        @staticmethod
        def heading_initialization(mag_samples, roll, pitch, num_samples=100):
            """
            Estimate yaw/heading from magnetometer data (assuming level attitude).
            """
            mag_avg = np.mean(mag_samples[-num_samples:], axis=0)
            
            # Rotate magnetometer to level frame
            sin_roll = np.sin(roll)
            cos_roll = np.cos(roll)
            sin_pitch = np.sin(pitch)
            cos_pitch = np.cos(pitch)
            
            mx_level = (mag_avg[0] * cos_pitch + 
                       mag_avg[1] * sin_roll * sin_pitch + 
                       mag_avg[2] * cos_roll * sin_pitch)
            
            my_level = (mag_avg[1] * cos_roll - 
                       mag_avg[2] * sin_roll)
            
            # Heading: atan2(-my, mx)
            heading = np.arctan2(-my_level, mx_level)
            
            return heading


State Vector Estimation & Diagnostics
-------------------------------------

**Typical GNSS/INS Kalman Filter State:**

.. code-block:: python

    class NavigationFilter:
        """15-state GNSS/INS Kalman filter with error estimation."""
        
        STATE_LABELS = [
            'lat', 'lon', 'alt',  # Position (0-2)
            'v_n', 'v_e', 'v_d',  # Velocity NED (3-5)
            'roll', 'pitch', 'yaw',  # Attitude Euler (6-8)
            'accel_bias_x', 'accel_bias_y', 'accel_bias_z',  # Accel bias (9-11)
            'gyro_bias_x', 'gyro_bias_y', 'gyro_bias_z'  # Gyro bias (12-14)
        ]
        
        def diagnose_convergence(self, P_diagonal):
            """Check if filter is properly converged."""
            diagnostics = {}
            
            # Position standard deviations
            diagnostics['pos_std'] = np.sqrt(P_diagonal[0:3])
            
            # Velocity standard deviations
            diagnostics['vel_std'] = np.sqrt(P_diagonal[3:6])
            
            # Attitude standard deviations (convert to degrees)
            diagnostics['att_std_deg'] = np.sqrt(P_diagonal[6:9]) * 180 / np.pi
            
            # Bias estimates
            diagnostics['accel_bias'] = np.sqrt(P_diagonal[9:12])
            diagnostics['gyro_bias'] = np.sqrt(P_diagonal[12:15])
            
            # Check if converged (empirical thresholds)
            is_converged = (
                np.all(diagnostics['pos_std'] < 500) and  # Position < 500 m
                np.all(diagnostics['vel_std'] < 50) and   # Velocity < 50 m/s
                np.all(diagnostics['att_std_deg'] < 10)   # Attitude < 10°
            )
            
            return diagnostics, is_converged


Common Issues & Solutions
~~~~~~~~~~~~~~~~~~~~~~~~

**Problem: INS diverges without GNSS updates**

Solution: Use predictable process noise to limit divergence rate. After known maximum outage, perform GNSS-denied mode:

.. code-block:: python

    def gnss_denied_update(state, P, timeout_sec=3600):
        """Increase covariance during GNSS outage."""
        # Position uncertainty grows at least 1 m/s - typical drift rate
        for _ in range(int(timeout_sec)):
            P[0:3, 0:3] += np.diag([1.0, 1.0, 1.0])  # accumulate drift
        return state, P


**Problem: Large attitude errors at startup**

Solution: Perform stationary alignment for 30-60 seconds:

.. code-block:: python

    def stationary_alignment_filter(accel_buffer, mag_buffer):
        """Align attitude assuming vehicle is stationary."""
        roll, pitch = INSAlignment.level_alignment(accel_buffer)
        yaw = INSAlignment.heading_initialization(mag_buffer, roll, pitch)
        return np.array([roll, pitch, yaw])


**Problem: Altitude diverges when filter loses GNSS**

Solution: Use barometer as fallback altimeter with appropriate process/measurement noise.

Best Practices
--------------

1. **Initialization & Alignment**
   - Always perform stationary alignment on level ground
   - Verify heading with compass or known reference
   - Allow 1-2 minute convergence period

2. **GNSS/INS Fusion Strategy**
   - Use loosely coupled for robustness with standard receivers
   - Use tightly coupled when raw GNSS data available
   - Implement automatic switching based on GNSS signal quality

3. **Sensor Fusion Architecture**
   - Include barometer for altitude correction
   - Add magnetometer for heading reference
   - Use wheel speed/odometry if available

4. **Error Monitoring**
   - Check innovation sequences (should be white Gaussian noise)
   - Monitor covariance traces to detect filter divergence
   - Implement chi-square test for outlier detection

5. **Real-Time Performance**
   - Use fixed-lag smoothing to avoid filter lag
   - Implement output buffering for consistent message rates
   - Profile computation time for embedded systems

6. **Sensor Calibration**
   - Pre-flight: accelerometer bias estimation from static data
   - In-flight: calibrate gyro bias during coordinated turns
   - Temperature compensation for long mission duration

References & Further Reading
----------------------------

- **Titterton & Weston** (2004): "Strapdown Inertial Navigation Technology"
- **Rogers** (2007): "Applied Mathematics in Integrated Navigation Systems"
- **IEEE Aerospace & Electronic Systems Magazine**: Regular INS/GNSS papers
- **TCWG Standards** (RFP-004): INS/GNSS Integration

See Also
--------

- :doc:`coordinate_systems` - Coordinate transformations for navigation frames
- :doc:`astronomical` - Precision reference frames (ECEF/ECI)
- :doc:`recipes` - Ready-to-use Kalman filter implementations
- :doc:`troubleshooting` - Navigation system diagnostics
