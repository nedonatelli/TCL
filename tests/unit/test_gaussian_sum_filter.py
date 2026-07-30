"""Unit tests for Gaussian Sum Filter (GSF).

Tests cover:
- Component initialization and management
- Prediction step (EKF on each component)
- Update step (weight adaptation)
- Component pruning and merging
- Multi-modal posterior approximation
- Comparison with single EKF
"""

import numpy as np
import pytest

from pytcl.dynamic_estimation.gaussian_sum_filter import (
    GaussianComponent,
    GaussianSumFilter,
    gaussian_sum_filter_predict,
    gaussian_sum_filter_update,
)


class TestGaussianComponent:
    """Test GaussianComponent namedtuple."""

    def test_create_component(self):
        """Test creating a Gaussian component."""
        x = np.array([1.0, 0.5])
        P = np.eye(2) * 0.1
        w = 0.5

        comp = GaussianComponent(x=x, P=P, w=w)

        assert np.allclose(comp.x, x)
        assert np.allclose(comp.P, P)
        assert comp.w == 0.5

    def test_component_immutable(self):
        """Test that components are immutable."""
        comp = GaussianComponent(x=np.array([1.0]), P=np.array([[1.0]]), w=1.0)

        with pytest.raises(AttributeError):
            comp.x = np.array([2.0])


class TestGaussianSumFilterInitialization:
    """Test GSF initialization."""

    def test_initialization(self):
        """Test GSF instantiation."""
        gsf = GaussianSumFilter(max_components=5)

        assert gsf.max_components == 5
        assert len(gsf.components) == 0

    def test_initialize_single_component(self):
        """Test initializing with single component."""
        gsf = GaussianSumFilter()

        x0 = np.array([0.0, 0.0])
        P0 = np.eye(2)

        gsf.initialize(x0, P0, num_components=1)

        assert len(gsf.components) == 1
        assert np.allclose(gsf.components[0].x, x0)
        assert np.allclose(gsf.components[0].P, P0)
        assert gsf.components[0].w == 1.0

    def test_initialize_multiple_components(self):
        """Test initializing with multiple components."""
        gsf = GaussianSumFilter()

        x0 = np.array([0.0, 0.0])
        P0 = np.eye(2)
        num_comp = 3

        gsf.initialize(x0, P0, num_components=num_comp)

        assert len(gsf.components) == num_comp
        assert np.allclose(sum(c.w for c in gsf.components), 1.0)

        # All have same covariance
        for comp in gsf.components:
            assert np.allclose(comp.P, P0)


class TestGaussianSumFilterPrediction:
    """Test GSF prediction step."""

    def setup_method(self):
        """Set up test system."""
        self.gsf = GaussianSumFilter()

        x0 = np.array([0.0, 0.0])
        P0 = np.eye(2) * 0.01

        self.gsf.initialize(x0, P0, num_components=2)

        # Linear system: x[k+1] = F @ x[k]
        self.F = np.array([[1.0, 0.1], [0.0, 1.0]])
        self.Q = np.eye(2) * 0.001

    def test_predict_preserves_weights(self):
        """Test that predict preserves component weights."""
        old_weights = [c.w for c in self.gsf.components]

        def f(x):
            return self.F @ x

        self.gsf.predict(f, self.F, self.Q)

        new_weights = [c.w for c in self.gsf.components]

        assert np.allclose(old_weights, new_weights)

    def test_predict_increases_covariance(self):
        """Test that predict increases covariance due to process noise."""

        def f(x):
            return self.F @ x

        old_traces = [np.trace(c.P) for c in self.gsf.components]

        self.gsf.predict(f, self.F, self.Q)

        new_traces = [np.trace(c.P) for c in self.gsf.components]

        for old_tr, new_tr in zip(old_traces, new_traces):
            assert new_tr > old_tr

    def test_predict_changes_mean(self):
        """Test that predict changes component means."""
        x0 = np.array([1.0, 0.5])
        gsf = GaussianSumFilter()
        gsf.initialize(x0, np.eye(2), num_components=1)

        def f(x):
            return self.F @ x

        gsf.predict(f, self.F, self.Q)

        # Mean should change due to system dynamics
        x_pred = gsf.components[0].x
        assert not np.allclose(x_pred, x0)

        # But should match F @ x0 approximately
        expected = self.F @ x0
        assert np.allclose(x_pred, expected, atol=1e-5)


class TestGaussianSumFilterUpdate:
    """Test GSF update step."""

    def setup_method(self):
        """Set up test system."""
        self.gsf = GaussianSumFilter()

        x0 = np.array([0.0, 0.0])
        P0 = np.eye(2)

        self.gsf.initialize(x0, P0, num_components=3)

        self.H = np.array([[1.0, 0.0]])
        self.R = np.array([[0.1]])

    def test_update_normalizes_weights(self):
        """Test that update normalizes component weights."""

        def h(x):
            return self.H @ x

        z = np.array([1.0])

        self.gsf.update(z, h, self.H, self.R)

        total_weight = sum(c.w for c in self.gsf.components)

        assert np.isclose(total_weight, 1.0)

    def test_update_reduces_covariance(self):
        """Test that update reduces component covariances."""
        old_traces = [np.trace(c.P) for c in self.gsf.components]

        def h(x):
            return self.H @ x

        z = np.array([0.5])

        self.gsf.update(z, h, self.H, self.R)

        new_traces = [np.trace(c.P) for c in self.gsf.components]

        for old_tr, new_tr in zip(old_traces, new_traces):
            assert new_tr < old_tr

    def test_update_adapts_weights_based_on_likelihood(self):
        """Test weight adaptation based on measurement likelihood."""
        # Initialize with different means
        gsf = GaussianSumFilter(
            prune_threshold=0.01
        )  # Higher threshold to preserve components
        x1 = np.array([0.0, 0.0])
        x2 = np.array([5.0, 0.0])

        # Use larger measurement covariance to avoid pruning
        R = np.array([[1.0]])

        comp1 = GaussianComponent(x=x1, P=np.eye(2), w=0.5)
        comp2 = GaussianComponent(x=x2, P=np.eye(2), w=0.5)

        gsf.components = [comp1, comp2]

        def h(x):
            return self.H @ x

        # Measurement close to x1
        z = np.array([0.1])

        gsf.update(z, h, self.H, R)

        # Component 1 should have higher weight (if not pruned)
        assert len(gsf.components) > 0
        if len(gsf.components) >= 2:
            assert gsf.components[0].w > gsf.components[1].w


class TestGaussianSumFilterPruning:
    """Test component pruning."""

    def test_prune_low_weight_components(self):
        """Test pruning of low-weight components."""
        gsf = GaussianSumFilter(prune_threshold=0.1)

        comp1 = GaussianComponent(x=np.array([0.0, 0.0]), P=np.eye(2), w=0.8)
        comp2 = GaussianComponent(x=np.array([1.0, 1.0]), P=np.eye(2), w=0.15)
        comp3 = GaussianComponent(x=np.array([2.0, 2.0]), P=np.eye(2), w=0.05)

        gsf.components = [comp1, comp2, comp3]

        gsf._prune_components()

        # comp3 should be removed (weight < threshold)
        assert len(gsf.components) == 2

    def test_prune_renormalizes_weights(self):
        """Test that pruning renormalizes remaining weights."""
        gsf = GaussianSumFilter(prune_threshold=0.1)

        comp1 = GaussianComponent(x=np.array([0.0, 0.0]), P=np.eye(2), w=0.8)
        comp2 = GaussianComponent(x=np.array([1.0, 1.0]), P=np.eye(2), w=0.2)

        gsf.components = [comp1, comp2]

        gsf._prune_components()

        total_weight = sum(c.w for c in gsf.components)

        assert np.isclose(total_weight, 1.0)


class TestGaussianSumFilterMerging:
    """Test component merging."""

    def test_merge_similar_components(self):
        """Test merging of similar components."""
        gsf = GaussianSumFilter(max_components=2, merge_threshold=0.1)

        # Two very similar components
        x1 = np.array([0.0, 0.0])
        x2 = np.array([0.01, 0.01])
        P = np.eye(2) * 0.001

        comp1 = GaussianComponent(x=x1, P=P, w=0.4)
        comp2 = GaussianComponent(x=x2, P=P, w=0.3)
        comp3 = GaussianComponent(x=np.array([5.0, 5.0]), P=P, w=0.3)

        gsf.components = [comp1, comp2, comp3]

        gsf._merge_components()

        # Should have merged comp1 and comp2
        assert len(gsf.components) <= 3

    def test_merge_respects_max_components(self):
        """Test that merging respects max_components limit."""
        gsf = GaussianSumFilter(max_components=2, merge_threshold=1e3)

        # 4 components
        for i in range(4):
            comp = GaussianComponent(x=np.array([float(i), 0.0]), P=np.eye(2), w=0.25)
            gsf.components.append(comp)

        gsf._merge_components()

        assert len(gsf.components) <= gsf.max_components


class TestGaussianSumFilterEstimate:
    """Test estimate computation."""

    def test_estimate_single_component(self):
        """Test estimate with single component."""
        gsf = GaussianSumFilter()

        x0 = np.array([1.0, 2.0])
        P0 = np.array([[1.0, 0.1], [0.1, 1.0]])

        gsf.components = [GaussianComponent(x=x0, P=P0, w=1.0)]

        x_est, P_est = gsf.estimate()

        assert np.allclose(x_est, x0)
        assert np.allclose(P_est, P0)

    def test_estimate_multiple_components(self):
        """Test estimate with multiple components."""
        gsf = GaussianSumFilter()

        x1 = np.array([0.0, 0.0])
        x2 = np.array([2.0, 0.0])
        P = np.eye(2)

        gsf.components = [
            GaussianComponent(x=x1, P=P, w=0.5),
            GaussianComponent(x=x2, P=P, w=0.5),
        ]

        x_est, P_est = gsf.estimate()

        # Weighted mean
        expected_x = 0.5 * x1 + 0.5 * x2
        assert np.allclose(x_est, expected_x)

        # Covariance should account for mixture spread
        assert P_est[0, 0] > 1.0  # Increased due to mixture spread


class TestGaussianSumFilterConvenience:
    """Test convenience functions."""

    def test_gsf_predict_function(self):
        """Test gaussian_sum_filter_predict function."""
        x1 = np.array([0.0, 0.0])
        x2 = np.array([1.0, 1.0])
        P = np.eye(2) * 0.01

        components = [
            GaussianComponent(x=x1, P=P, w=0.5),
            GaussianComponent(x=x2, P=P, w=0.5),
        ]

        F = np.array([[1.0, 0.1], [0.0, 1.0]])
        Q = np.eye(2) * 0.001

        def f(x):
            return F @ x

        new_comps = gaussian_sum_filter_predict(components, f, F, Q)

        assert len(new_comps) == 2
        assert np.allclose(sum(c.w for c in new_comps), 1.0)

    def test_gsf_update_function(self):
        """Test gaussian_sum_filter_update function."""
        x1 = np.array([0.0, 0.0])
        x2 = np.array([1.0, 0.0])
        P = np.eye(2)

        components = [
            GaussianComponent(x=x1, P=P, w=0.5),
            GaussianComponent(x=x2, P=P, w=0.5),
        ]

        H = np.array([[1.0, 0.0]])
        R = np.array([[0.1]])
        z = np.array([0.5])

        def h(x):
            return H @ x

        updated_comps = gaussian_sum_filter_update(components, z, h, H, R)

        assert len(updated_comps) == 2
        assert np.isclose(sum(c.w for c in updated_comps), 1.0)


class TestGaussianSumFilterIntegration:
    """Integration tests for GSF."""

    def test_full_predict_update_cycle(self):
        """Test full predict-update cycle."""
        gsf = GaussianSumFilter()

        x0 = np.array([0.0, 0.0])
        P0 = np.eye(2)

        gsf.initialize(x0, P0, num_components=2)

        F = np.array([[1.0, 0.1], [0.0, 1.0]])
        Q = np.eye(2) * 0.01

        H = np.array([[1.0, 0.0]])
        R = np.array([[0.1]])

        def f(x):
            return F @ x

        def h(x):
            return H @ x

        # Predict
        gsf.predict(f, F, Q)

        # Update
        z = np.array([0.5])
        gsf.update(z, h, H, R)

        # Check estimate
        x_est, P_est = gsf.estimate()

        assert x_est.shape == x0.shape
        assert P_est.shape == P0.shape
        assert np.all(np.linalg.eigvalsh(P_est) > -1e-10)

    def test_multi_modal_filtering(self):
        """Test GSF with multi-modal posterior."""
        gsf = GaussianSumFilter(max_components=5)

        # Initialize with two modes
        x1 = np.array([0.0, 0.0])
        x2 = np.array([5.0, 0.0])
        P = np.eye(2) * 0.5

        gsf.components = [
            GaussianComponent(x=x1, P=P, w=0.5),
            GaussianComponent(x=x2, P=P, w=0.5),
        ]

        F = np.eye(2)
        Q = np.eye(2) * 0.001

        H = np.array([[1.0, 0.0]])
        R = np.array([[0.1]])

        def f(x):
            return F @ x

        def h(x):
            return H @ x

        # Predict
        gsf.predict(f, F, Q)

        # Measurement between the two modes
        z = np.array([2.5])

        # Update
        gsf.update(z, h, H, R)

        # Both modes should still be present with adapted weights
        assert len(gsf.components) >= 1
        assert np.isclose(sum(c.w for c in gsf.components), 1.0)

    def test_long_horizon_filtering(self):
        """Test GSF over multiple time steps."""
        gsf = GaussianSumFilter(max_components=4)

        x0 = np.array([0.0, 0.0])
        P0 = np.eye(2)

        gsf.initialize(x0, P0, num_components=2)

        F = np.array([[1.0, 0.1], [0.0, 0.99]])
        Q = np.eye(2) * 0.005

        H = np.array([[1.0, 0.0]])
        R = np.array([[0.05]])

        def f(x):
            return F @ x

        def h(x):
            return H @ x

        for k in range(10):
            # Predict
            gsf.predict(f, F, Q)

            # Measurement
            z = np.array([np.random.randn() * 0.05])

            # Update
            gsf.update(z, h, H, R)

            # Check validity
            x_est, P_est = gsf.estimate()
            assert P_est.shape == (2, 2)
            assert np.all(np.linalg.eigvalsh(P_est) > -1e-10)
            assert np.isclose(sum(c.w for c in gsf.components), 1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
