"""Unit tests for Rao-Blackwellized Particle Filter (RBPF).

Tests cover:
- Particle initialization and management
- Prediction step (nonlinear + linear propagation)
- Update step (weight adaptation)
- Resampling and merging
- State estimation
- Comparison with standard particle filter
"""

import numpy as np
import pytest

from pytcl.dynamic_estimation.rbpf import (
    RBPFFilter,
    RBPFParticle,
    rbpf_predict,
    rbpf_update,
)


class TestRBPFParticle:
    """Test RBPFParticle namedtuple."""

    def test_create_particle(self):
        """Test creating an RBPF particle."""
        y = np.array([1.0, 2.0])
        x = np.array([0.5, 0.5])
        P = np.eye(2) * 0.1
        w = 0.25

        particle = RBPFParticle(y=y, x=x, P=P, w=w)

        assert np.allclose(particle.y, y)
        assert np.allclose(particle.x, x)
        assert np.allclose(particle.P, P)
        assert particle.w == 0.25

    def test_particle_immutable(self):
        """Test that particles are immutable."""
        particle = RBPFParticle(
            y=np.array([1.0]),
            x=np.array([1.0]),
            P=np.array([[1.0]]),
            w=1.0,
        )

        with pytest.raises(AttributeError):
            particle.w = 0.5


class TestRBPFInitialization:
    """Test RBPF initialization."""

    def test_create_filter(self):
        """Test RBPF instantiation."""
        rbpf = RBPFFilter(max_particles=50)

        assert rbpf.max_particles == 50
        assert len(rbpf.particles) == 0

    def test_initialize_particles(self):
        """Test particle initialization."""
        rbpf = RBPFFilter()

        y0 = np.array([0.0, 0.0])
        x0 = np.array([0.0, 0.0])
        P0 = np.eye(2) * 0.1

        num_particles = 20

        rbpf.initialize(y0, x0, P0, num_particles=num_particles)

        assert len(rbpf.particles) == num_particles
        assert np.isclose(sum(p.w for p in rbpf.particles), 1.0)

    def test_initialize_weights_normalized(self):
        """Test that initial weights are normalized."""
        rbpf = RBPFFilter()

        y0 = np.array([1.0])
        x0 = np.array([1.0])
        P0 = np.array([[1.0]])

        for num_p in [5, 10, 50]:
            rbpf.initialize(y0, x0, P0, num_particles=num_p)

            total_weight = sum(p.w for p in rbpf.particles)

            assert np.isclose(total_weight, 1.0)


class TestRBPFPrediction:
    """Test RBPF prediction step."""

    def setup_method(self):
        """Set up test system."""
        self.rbpf = RBPFFilter()

        y0 = np.array([0.0, 0.0])
        x0 = np.array([0.0, 0.0])
        P0 = np.eye(2) * 0.01

        self.rbpf.initialize(y0, x0, P0, num_particles=10)

        # Nonlinear system
        self.g = lambda y: 0.9 * y + 0.1 * np.sin(y)
        self.G = np.array([[0.9, 0.0], [0.0, 0.9]])
        self.Qy = np.eye(2) * 0.001

        # Linear system
        self.F = np.eye(2)
        self.Qx = np.eye(2) * 0.001

    def test_predict_propagates_particles(self):
        """Test that predict propagates both components."""

        def f(x, y):
            return x

        old_y = [p.y.copy() for p in self.rbpf.particles]

        self.rbpf.predict(self.g, self.G, self.Qy, f, self.F, self.Qx)

        new_y = [p.y for p in self.rbpf.particles]

        # At least some particles should have changed
        changed = sum(1 for o, n in zip(old_y, new_y) if not np.allclose(o, n))
        assert changed > 0

    def test_predict_preserves_weights(self):
        """Test that predict preserves weights."""

        def f(x, y):
            return x

        old_weights = [p.w for p in self.rbpf.particles]

        self.rbpf.predict(self.g, self.G, self.Qy, f, self.F, self.Qx)

        new_weights = [p.w for p in self.rbpf.particles]

        assert np.allclose(old_weights, new_weights)

    def test_predict_increases_covariance(self):
        """Test that predict increases covariance."""

        def f(x, y):
            return x

        old_traces = [np.trace(p.P) for p in self.rbpf.particles]

        self.rbpf.predict(self.g, self.G, self.Qy, f, self.F, self.Qx)

        new_traces = [np.trace(p.P) for p in self.rbpf.particles]

        for old_tr, new_tr in zip(old_traces, new_traces):
            assert new_tr >= old_tr


class TestRBPFUpdate:
    """Test RBPF update step."""

    def setup_method(self):
        """Set up test system."""
        self.rbpf = RBPFFilter()

        y0 = np.array([0.0, 0.0])
        x0 = np.array([0.0, 0.0])
        P0 = np.eye(2)

        self.rbpf.initialize(y0, x0, P0, num_particles=10)

        self.H = np.array([[1.0, 0.0]])
        self.R = np.array([[0.1]])

    def test_update_normalizes_weights(self):
        """Test that update normalizes weights."""

        def h(x, y):
            return self.H @ x

        z = np.array([0.5])

        self.rbpf.update(z, h, self.H, self.R)

        total_weight = sum(p.w for p in self.rbpf.particles)

        assert np.isclose(total_weight, 1.0)

    def test_update_reduces_covariance(self):
        """Test that update reduces covariance."""

        def h(x, y):
            return self.H @ x

        old_traces = [np.trace(p.P) for p in self.rbpf.particles]

        z = np.array([0.5])

        self.rbpf.update(z, h, self.H, self.R)

        new_traces = [np.trace(p.P) for p in self.rbpf.particles]

        # Average trace should decrease
        assert np.mean(new_traces) < np.mean(old_traces)

    def test_update_adapts_weights(self):
        """Test that weights adapt based on measurement."""
        # Initialize with different x values
        rbpf = RBPFFilter()

        x_vals = [
            np.array([0.0, 0.0]),
            np.array([5.0, 0.0]),
        ]

        rbpf.particles = [
            RBPFParticle(y=np.array([0.0, 0.0]), x=x_vals[i], P=np.eye(2), w=0.5)
            for i in range(2)
        ]

        def h(x, y):
            return self.H @ x

        # Measurement close to first particle
        z = np.array([0.1])

        rbpf.update(z, h, self.H, self.R)

        # First particle should have higher weight
        assert rbpf.particles[0].w > rbpf.particles[1].w


class TestRBPFResampling:
    """Test RBPF resampling."""

    def test_resample_if_needed(self):
        """Test resampling condition."""
        rbpf = RBPFFilter(resample_threshold=0.5)

        # Create particles with highly unequal weights
        particles = [
            RBPFParticle(
                y=np.array([float(i), 0.0]), x=np.array([0.0, 0.0]), P=np.eye(2), w=w
            )
            for i, w in enumerate([0.99, 0.005, 0.005])
        ]

        rbpf.particles = particles

        rbpf._resample_if_needed()

        # After resampling, weights should be more uniform
        weights = [p.w for p in rbpf.particles]
        assert np.isclose(sum(weights), 1.0)

    def test_resample_preserves_particle_count(self):
        """Test that resampling preserves particle count."""
        rbpf = RBPFFilter(resample_threshold=0.1)

        # Highly unequal weights to trigger resampling
        particles = [
            RBPFParticle(
                y=np.array([float(i), 0.0]), x=np.array([0.0, 0.0]), P=np.eye(2), w=w
            )
            for i, w in enumerate([0.99, 0.005, 0.005])
        ]

        rbpf.particles = particles

        initial_count = len(rbpf.particles)

        rbpf._resample_if_needed()

        assert len(rbpf.particles) == initial_count


class TestRBPFMerging:
    """Test RBPF merging."""

    def test_merge_reduces_particle_count(self):
        """Test that merging reduces particle count."""
        rbpf = RBPFFilter(max_particles=2, merge_threshold=10.0)

        # Create 4 particles
        particles = [
            RBPFParticle(
                y=np.array([float(i), 0.0]),
                x=np.array([0.0, 0.0]),
                P=np.eye(2) * 0.01,
                w=0.25,
            )
            for i in range(4)
        ]

        rbpf.particles = particles

        rbpf._merge_particles()

        assert len(rbpf.particles) <= rbpf.max_particles

    def test_merge_preserves_weights(self):
        """Test that merging preserves total weight."""
        rbpf = RBPFFilter(max_particles=2)

        particles = [
            RBPFParticle(
                y=np.array([float(i), 0.0]),
                x=np.array([0.0, 0.0]),
                P=np.eye(2) * 0.01,
                w=0.25,
            )
            for i in range(4)
        ]

        rbpf.particles = particles

        old_weight = sum(p.w for p in rbpf.particles)

        rbpf._merge_particles()

        new_weight = sum(p.w for p in rbpf.particles)

        assert np.isclose(old_weight, new_weight)


class TestRBPFEstimate:
    """Test RBPF state estimation."""

    def test_estimate_single_particle(self):
        """Test estimate with single particle."""
        rbpf = RBPFFilter()

        y0 = np.array([1.0, 2.0])
        x0 = np.array([3.0, 4.0])
        P0 = np.eye(2)

        rbpf.particles = [RBPFParticle(y=y0, x=x0, P=P0, w=1.0)]

        y_est, x_est, P_est = rbpf.estimate()

        assert np.allclose(y_est, y0)
        assert np.allclose(x_est, x0)
        assert np.allclose(P_est, P0)

    def test_estimate_multiple_particles(self):
        """Test estimate with multiple particles."""
        rbpf = RBPFFilter()

        y1 = np.array([0.0, 0.0])
        y2 = np.array([2.0, 0.0])
        x = np.array([0.0, 0.0])
        P = np.eye(2)

        rbpf.particles = [
            RBPFParticle(y=y1, x=x, P=P, w=0.5),
            RBPFParticle(y=y2, x=x, P=P, w=0.5),
        ]

        y_est, x_est, P_est = rbpf.estimate()

        # Weighted mean
        expected_y = 0.5 * y1 + 0.5 * y2
        assert np.allclose(y_est, expected_y)

    def test_estimate_covariance_positive_definite(self):
        """Test that estimated covariance is positive definite."""
        rbpf = RBPFFilter()

        particles = [
            RBPFParticle(
                y=np.array([float(i), 0.0]),
                x=np.array([float(i) * 0.1, 0.0]),
                P=np.eye(2) * 0.1,
                w=1.0 / 10,
            )
            for i in range(10)
        ]

        rbpf.particles = particles

        y_est, x_est, P_est = rbpf.estimate()

        # Check positive definiteness
        eigvals = np.linalg.eigvalsh(P_est)
        assert np.all(eigvals > -1e-10)


class TestRBPFConvenience:
    """Test convenience functions."""

    def test_rbpf_predict_function(self):
        """Test rbpf_predict convenience function."""
        particles = [
            RBPFParticle(
                y=np.array([float(i), 0.0]),
                x=np.array([0.0, 0.0]),
                P=np.eye(2),
                w=0.1,
            )
            for i in range(10)
        ]

        def g(y):
            return 0.9 * y

        G = np.eye(2) * 0.9
        Qy = np.eye(2) * 0.001

        def f(x, y):
            return x

        F = np.eye(2)
        Qx = np.eye(2) * 0.001

        new_particles = rbpf_predict(particles, g, G, Qy, f, F, Qx)

        assert len(new_particles) == len(particles)
        assert np.isclose(sum(p.w for p in new_particles), 1.0)

    def test_rbpf_update_function(self):
        """Test rbpf_update convenience function."""
        particles = [
            RBPFParticle(
                y=np.array([0.0, 0.0]),
                x=np.array([float(i), 0.0]),
                P=np.eye(2),
                w=0.1,
            )
            for i in range(10)
        ]

        H = np.array([[1.0, 0.0]])
        R = np.array([[0.1]])
        z = np.array([0.5])

        def h(x, y):
            return H @ x

        updated = rbpf_update(particles, z, h, H, R)

        assert len(updated) == len(particles)
        assert np.isclose(sum(p.w for p in updated), 1.0)


class TestRBPFIntegration:
    """Integration tests for RBPF."""

    def test_full_predict_update_cycle(self):
        """Test full predict-update cycle."""
        rbpf = RBPFFilter()

        y0 = np.array([0.0, 0.0])
        x0 = np.array([0.0, 0.0])
        P0 = np.eye(2)

        rbpf.initialize(y0, x0, P0, num_particles=10)

        # Nonlinear system
        def g(y):
            return 0.95 * y

        G = np.eye(2) * 0.95
        Qy = np.eye(2) * 0.001

        # Linear system
        def f(x, y):
            return 0.99 * x

        F = np.eye(2) * 0.99
        Qx = np.eye(2) * 0.001

        # Measurement
        H = np.array([[1.0, 0.0]])
        R = np.array([[0.1]])

        def h(x, y):
            return H @ x

        # Predict
        rbpf.predict(g, G, Qy, f, F, Qx)

        # Update
        z = np.array([0.5])
        rbpf.update(z, h, H, R)

        # Estimate
        y_est, x_est, P_est = rbpf.estimate()

        assert y_est.shape == y0.shape
        assert x_est.shape == x0.shape
        assert P_est.shape == P0.shape

    def test_multi_step_filtering(self):
        """Test RBPF over multiple time steps."""
        rbpf = RBPFFilter()

        y0 = np.array([0.0, 0.0])
        x0 = np.array([0.0, 0.0])
        P0 = np.eye(2) * 0.1

        rbpf.initialize(y0, x0, P0, num_particles=20)

        def g(y):
            return 0.95 * y

        G = np.eye(2) * 0.95
        Qy = np.eye(2) * 0.005

        def f(x, y):
            return x + 0.01 * y

        F = np.eye(2)
        Qx = np.eye(2) * 0.001

        H = np.array([[1.0, 0.0]])
        R = np.array([[0.05]])

        def h(x, y):
            return H @ (x + y)

        for k in range(10):
            rbpf.predict(g, G, Qy, f, F, Qx)

            z = np.array([np.random.randn() * 0.05])

            rbpf.update(z, h, H, R)

            y_est, x_est, P_est = rbpf.estimate()

            # Check validity
            assert P_est.shape == (2, 2)
            assert np.all(np.linalg.eigvalsh(P_est) > -1e-10)
            assert np.isclose(sum(p.w for p in rbpf.particles), 1.0)

    def test_nonlinear_system_tracking(self):
        """Test RBPF on nonlinear tracking scenario."""
        rbpf = RBPFFilter(max_particles=50)

        # Target angle (nonlinear) and range (linear)
        theta0 = np.array([0.0])  # angle
        r0 = np.array([100.0])  # range
        x0 = r0

        P0 = np.eye(1) * 10.0

        rbpf.initialize(theta0, x0, P0, num_particles=30)

        # Nonlinear dynamics: angle increases
        def g(y):
            return y + 0.05 + np.random.randn(1) * 0.01

        G = np.array([[1.0]])
        Qy = np.array([[0.001]])

        # Linear dynamics: range constant
        def f(x, y):
            return x

        F = np.array([[1.0]])
        Qx = np.array([[0.5]])

        # Measurement of range from noisy sensor
        H = np.array([[1.0]])
        R = np.array([[1.0]])

        def h(x, y):
            return x

        for k in range(10):
            rbpf.predict(g, G, Qy, f, F, Qx)

            # Noisy measurement
            z = np.array([100.0 + np.random.randn() * 1.0])

            rbpf.update(z, h, H, R)

            y_est, x_est, P_est = rbpf.estimate()

            # Range estimate should stay around 100
            assert 95 < x_est[0] < 105

    def test_divergence_handling(self):
        """Test that filter handles measurement divergence."""
        rbpf = RBPFFilter()

        y0 = np.array([0.0])
        x0 = np.array([0.0])
        P0 = np.array([[10.0]])

        rbpf.initialize(y0, x0, P0, num_particles=20)

        def g(y):
            return y

        G = np.array([[1.0]])
        Qy = np.array([[0.1]])

        def f(x, y):
            return x

        F = np.array([[1.0]])
        Qx = np.array([[0.1]])

        H = np.array([[1.0]])
        R = np.array([[1.0]])

        def h(x, y):
            return x

        # Very different measurements
        measurements = [10.0, -10.0, 20.0, -20.0, 5.0]

        for z_val in measurements:
            rbpf.predict(g, G, Qy, f, F, Qx)
            z = np.array([z_val])
            rbpf.update(z, h, H, R)

            y_est, x_est, P_est = rbpf.estimate()

            # Should still have valid estimates
            assert not np.any(np.isnan(x_est))
            assert not np.any(np.isinf(x_est))
            assert np.isclose(sum(p.w for p in rbpf.particles), 1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
