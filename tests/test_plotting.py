"""
Tests for the plotting module.

These tests verify the core utility functions work correctly.
Plotting functions that return Plotly figures are tested for proper output types.
"""

import numpy as np
import pytest

from pytcl.plotting import (  # Ellipse utilities; Track utilities (non-plotting)
    confidence_region_radius,
    covariance_ellipse_points,
    covariance_ellipsoid_points,
    ellipse_parameters,
    plot_tracking_result,
    plot_trajectory_2d,
    plot_trajectory_3d,
)

# Check if plotly is available
try:
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


class TestCovarianceEllipsePoints:
    """Tests for covariance_ellipse_points function."""

    def test_identity_covariance(self):
        """Test with identity covariance (circle)."""
        mean = [0, 0]
        cov = [[1, 0], [0, 1]]
        x, y = covariance_ellipse_points(mean, cov, n_std=1.0, n_points=100)

        assert len(x) == 100
        assert len(y) == 100
        # Should be approximately unit circle
        radii = np.sqrt(x**2 + y**2)
        np.testing.assert_allclose(radii, 1.0, rtol=0.01)

    def test_diagonal_covariance(self):
        """Test with diagonal covariance (axis-aligned ellipse)."""
        mean = [0, 0]
        cov = [[4, 0], [0, 1]]  # 2:1 aspect ratio
        x, y = covariance_ellipse_points(mean, cov, n_std=1.0, n_points=100)

        # X should span [-2, 2], Y should span [-1, 1] (at 1-sigma)
        assert np.max(np.abs(x)) > 1.8
        assert np.max(np.abs(x)) < 2.2
        assert np.max(np.abs(y)) > 0.8
        assert np.max(np.abs(y)) < 1.2

    def test_offset_mean(self):
        """Test with non-zero mean."""
        mean = [5, 10]
        cov = [[1, 0], [0, 1]]
        x, y = covariance_ellipse_points(mean, cov, n_std=1.0, n_points=100)

        # Center should be at mean
        np.testing.assert_allclose(np.mean(x), 5, rtol=0.01)
        np.testing.assert_allclose(np.mean(y), 10, rtol=0.01)

    def test_correlated_covariance(self):
        """Test with correlated covariance."""
        mean = [0, 0]
        cov = [[1, 0.8], [0.8, 1]]
        x, y = covariance_ellipse_points(mean, cov, n_std=1.0, n_points=100)

        # Should produce tilted ellipse
        assert len(x) == 100
        assert len(y) == 100

    def test_n_std_scaling(self):
        """Test that n_std scales the ellipse properly."""
        mean = [0, 0]
        cov = [[1, 0], [0, 1]]

        x1, y1 = covariance_ellipse_points(mean, cov, n_std=1.0)
        x2, y2 = covariance_ellipse_points(mean, cov, n_std=2.0)

        # 2-sigma ellipse should have 2x radius
        r1 = np.max(np.sqrt(x1**2 + y1**2))
        r2 = np.max(np.sqrt(x2**2 + y2**2))
        np.testing.assert_allclose(r2 / r1, 2.0, rtol=0.01)

    def test_invalid_covariance_shape(self):
        """Test error on invalid covariance shape."""
        mean = [0, 0]
        cov = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # 3x3

        with pytest.raises(ValueError, match="2x2"):
            covariance_ellipse_points(mean, cov)


class TestCovarianceEllipsoidPoints:
    """Tests for covariance_ellipsoid_points function."""

    def test_identity_covariance(self):
        """Test with identity covariance (sphere)."""
        mean = [0, 0, 0]
        cov = np.eye(3)
        x, y, z = covariance_ellipsoid_points(mean, cov, n_std=1.0, n_points=10)

        assert x.shape == (10, 10)
        assert y.shape == (10, 10)
        assert z.shape == (10, 10)

        # All points should be approximately on unit sphere
        radii = np.sqrt(x**2 + y**2 + z**2)
        np.testing.assert_allclose(radii, 1.0, rtol=0.1)

    def test_diagonal_covariance(self):
        """Test with diagonal covariance."""
        mean = [0, 0, 0]
        cov = np.diag([4, 1, 9])  # Semi-axes: 2, 1, 3
        x, y, z = covariance_ellipsoid_points(mean, cov, n_std=1.0)

        # Check extents
        assert np.max(np.abs(x)) > 1.5  # Should extend to ~2
        assert np.max(np.abs(z)) > 2.5  # Should extend to ~3

    def test_offset_mean(self):
        """Test with non-zero mean."""
        mean = [5, 10, 15]
        cov = np.eye(3)
        x, y, z = covariance_ellipsoid_points(mean, cov, n_std=1.0)

        # Center should be at mean
        np.testing.assert_allclose(np.mean(x), 5, rtol=0.1)
        np.testing.assert_allclose(np.mean(y), 10, rtol=0.1)
        np.testing.assert_allclose(np.mean(z), 15, rtol=0.1)

    def test_invalid_covariance_shape(self):
        """Test error on invalid covariance shape."""
        mean = [0, 0, 0]
        cov = [[1, 0], [0, 1]]  # 2x2

        with pytest.raises(ValueError, match="3x3"):
            covariance_ellipsoid_points(mean, cov)


class TestEllipseParameters:
    """Tests for ellipse_parameters function."""

    def test_identity_covariance(self):
        """Test with identity covariance."""
        cov = [[1, 0], [0, 1]]
        a, b, theta = ellipse_parameters(cov)

        assert a == pytest.approx(1.0)
        assert b == pytest.approx(1.0)

    def test_diagonal_covariance(self):
        """Test with diagonal covariance."""
        cov = [[4, 0], [0, 1]]
        a, b, theta = ellipse_parameters(cov)

        assert a == pytest.approx(2.0)  # sqrt(4)
        assert b == pytest.approx(1.0)  # sqrt(1)
        # Angle should be 0 or pi (aligned with x-axis)
        assert abs(theta) < 0.1 or abs(abs(theta) - np.pi) < 0.1

    def test_rotated_ellipse(self):
        """Test with rotated ellipse."""
        # 45-degree rotated ellipse
        angle = np.pi / 4
        R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        D = np.diag([4, 1])
        cov = R @ D @ R.T

        a, b, theta = ellipse_parameters(cov)

        assert a == pytest.approx(2.0, rel=0.01)
        assert b == pytest.approx(1.0, rel=0.01)

    def test_invalid_shape(self):
        """Test error on invalid covariance shape."""
        cov = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

        with pytest.raises(ValueError):
            ellipse_parameters(cov)


class TestConfidenceRegionRadius:
    """Tests for confidence_region_radius function."""

    def test_2d_95_percent(self):
        """Test 2D 95% confidence region."""
        r = confidence_region_radius(2, 0.95)
        # Chi-squared(2) at 95% is approximately 5.991
        expected = np.sqrt(5.991)
        assert r == pytest.approx(expected, rel=0.01)

    def test_2d_99_percent(self):
        """Test 2D 99% confidence region."""
        r = confidence_region_radius(2, 0.99)
        # Chi-squared(2) at 99% is approximately 9.21
        expected = np.sqrt(9.21)
        assert r == pytest.approx(expected, rel=0.01)

    def test_3d_95_percent(self):
        """Test 3D 95% confidence region."""
        r = confidence_region_radius(3, 0.95)
        # Chi-squared(3) at 95% is approximately 7.815
        expected = np.sqrt(7.815)
        assert r == pytest.approx(expected, rel=0.01)

    def test_1d_confidence(self):
        """Test 1D confidence region (should match normal distribution)."""
        r = confidence_region_radius(1, 0.6827)  # ~1 sigma
        assert r == pytest.approx(1.0, rel=0.1)


@pytest.mark.skipif(not HAS_PLOTLY, reason="plotly not installed")
class TestPlotlyFunctions:
    """Tests for Plotly-dependent plotting functions."""

    def test_plot_trajectory_2d(self):
        """Test 2D trajectory plotting."""
        states = np.random.randn(50, 4)
        trace = plot_trajectory_2d(states, x_idx=0, y_idx=2)

        assert isinstance(trace, go.Scatter)
        assert len(trace.x) == 50
        assert len(trace.y) == 50

    def test_plot_trajectory_3d(self):
        """Test 3D trajectory plotting."""
        states = np.random.randn(50, 6)
        trace = plot_trajectory_3d(states, x_idx=0, y_idx=2, z_idx=4)

        assert isinstance(trace, go.Scatter3d)
        assert len(trace.x) == 50

    def test_plot_tracking_result(self):
        """Test tracking result plotting."""
        true_states = np.cumsum(np.random.randn(50, 4), axis=0)
        estimates = true_states + 0.1 * np.random.randn(50, 4)
        measurements = true_states[:, [0, 2]] + 0.5 * np.random.randn(50, 2)

        fig = plot_tracking_result(
            true_states=true_states,
            estimates=estimates,
            measurements=measurements,
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 3  # At least true, meas, estimate

    def test_plot_tracking_result_with_covariances(self):
        """Test tracking result with covariance ellipses."""
        n_steps = 20
        true_states = np.cumsum(np.random.randn(n_steps, 4), axis=0)
        estimates = true_states + 0.1 * np.random.randn(n_steps, 4)
        covariances = [np.diag([1, 0.1, 1, 0.1]) for _ in range(n_steps)]

        fig = plot_tracking_result(
            true_states=true_states,
            estimates=estimates,
            covariances=covariances,
            ellipse_interval=5,
        )

        assert isinstance(fig, go.Figure)


class TestPlotCovarianceEllipse:
    """Tests for plot_covariance_ellipse function."""

    @pytest.mark.skipif(not HAS_PLOTLY, reason="plotly not installed")
    def test_basic_ellipse(self):
        """Test basic ellipse trace creation."""
        from pytcl.plotting import plot_covariance_ellipse

        mean = [0, 0]
        cov = [[1, 0], [0, 1]]
        trace = plot_covariance_ellipse(mean, cov)

        assert isinstance(trace, go.Scatter)
        assert trace.fill == "toself"

    @pytest.mark.skipif(not HAS_PLOTLY, reason="plotly not installed")
    def test_unfilled_ellipse(self):
        """Test unfilled ellipse."""
        from pytcl.plotting import plot_covariance_ellipse

        mean = [0, 0]
        cov = [[1, 0], [0, 1]]
        trace = plot_covariance_ellipse(mean, cov, fill=False)

        assert isinstance(trace, go.Scatter)
        assert trace.fill is None


class TestPlotCovarianceEllipses:
    """Tests for plot_covariance_ellipses function."""

    @pytest.mark.skipif(not HAS_PLOTLY, reason="plotly not installed")
    def test_multiple_ellipses(self):
        """Test plotting multiple ellipses."""
        from pytcl.plotting import plot_covariance_ellipses

        means = [[0, 0], [5, 5], [10, 0]]
        covariances = [
            [[1, 0], [0, 1]],
            [[2, 0.5], [0.5, 1]],
            [[1, -0.3], [-0.3, 2]],
        ]

        fig = plot_covariance_ellipses(means, covariances)

        assert isinstance(fig, go.Figure)
        # 3 ellipses + 3 center points
        assert len(fig.data) == 6


class TestPlotCovarianceEllipsoid:
    """Tests for plot_covariance_ellipsoid function."""

    @pytest.mark.skipif(not HAS_PLOTLY, reason="plotly not installed")
    def test_basic_ellipsoid(self):
        """Test basic ellipsoid trace creation."""
        from pytcl.plotting import plot_covariance_ellipsoid

        mean = [0, 0, 0]
        cov = np.diag([1, 2, 3])
        trace = plot_covariance_ellipsoid(mean, cov)

        assert isinstance(trace, go.Surface)


class TestCoordinatePlotting:
    """Tests for coordinate system plotting functions."""

    @pytest.mark.skipif(not HAS_PLOTLY, reason="plotly not installed")
    def test_plot_coordinate_axes_3d(self):
        """Test 3D coordinate axes plotting."""
        from pytcl.plotting import plot_coordinate_axes_3d

        traces = plot_coordinate_axes_3d()

        assert len(traces) == 3  # X, Y, Z
        assert all(isinstance(t, go.Scatter3d) for t in traces)

    @pytest.mark.skipif(not HAS_PLOTLY, reason="plotly not installed")
    def test_plot_coordinate_axes_with_rotation(self):
        """Test coordinate axes with rotation."""
        from pytcl.plotting import plot_coordinate_axes_3d

        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])  # 90 degree rotation about z
        traces = plot_coordinate_axes_3d(rotation_matrix=R)

        assert len(traces) == 3

    @pytest.mark.skipif(not HAS_PLOTLY, reason="plotly not installed")
    def test_plot_rotation_comparison(self):
        """Test rotation comparison plotting."""
        from pytcl.plotting import plot_rotation_comparison

        R1 = np.eye(3)
        R2 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

        fig = plot_rotation_comparison(R1, R2)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 6  # 3 axes x 2 frames

    @pytest.mark.skipif(not HAS_PLOTLY, reason="plotly not installed")
    def test_plot_spherical_grid(self):
        """Test spherical grid plotting."""
        from pytcl.plotting import plot_spherical_grid

        fig = plot_spherical_grid(r=1.0)

        assert isinstance(fig, go.Figure)


class TestMetricsPlotting:
    """Tests for performance metrics plotting functions."""

    @pytest.mark.skipif(not HAS_PLOTLY, reason="plotly not installed")
    def test_plot_nees_sequence(self):
        """Test NEES sequence plotting."""
        from pytcl.plotting import plot_nees_sequence

        rng = np.random.default_rng(42)
        nees_values = rng.chisquare(df=4, size=50)
        fig = plot_nees_sequence(nees_values, n_dims=4)

        assert isinstance(fig, go.Figure)

    @pytest.mark.skipif(not HAS_PLOTLY, reason="plotly not installed")
    def test_plot_ospa_over_time(self):
        """Test OSPA plotting."""
        from pytcl.plotting import plot_ospa_over_time

        ospa = np.random.rand(50) * 10
        fig = plot_ospa_over_time(ospa)

        assert isinstance(fig, go.Figure)

    @pytest.mark.skipif(not HAS_PLOTLY, reason="plotly not installed")
    def test_plot_error_histogram(self):
        """Test error histogram plotting."""
        from pytcl.plotting import plot_error_histogram

        errors = np.random.randn(1000, 3)
        fig = plot_error_histogram(errors)

        assert isinstance(fig, go.Figure)

    @pytest.mark.skipif(not HAS_PLOTLY, reason="plotly not installed")
    def test_plot_cardinality_over_time(self):
        """Test cardinality plotting."""
        from pytcl.plotting import plot_cardinality_over_time

        true_card = np.array([3, 3, 4, 4, 4, 5, 5, 5, 4, 4])
        est_card = np.array([2, 3, 3, 4, 4, 4, 5, 5, 5, 4])

        fig = plot_cardinality_over_time(true_card, est_card)

        assert isinstance(fig, go.Figure)


class TestAnimatedTracking:
    """Tests for animated tracking visualization."""

    @pytest.mark.skipif(not HAS_PLOTLY, reason="plotly not installed")
    def test_create_animated_tracking(self):
        """Test animated tracking creation."""
        from pytcl.plotting import create_animated_tracking

        n_steps = 20
        true_states = np.cumsum(np.random.randn(n_steps + 1, 4), axis=0)
        estimates = true_states + 0.1 * np.random.randn(n_steps + 1, 4)
        measurements = true_states[1:, [0, 2]] + 0.5 * np.random.randn(n_steps, 2)
        covariances = [np.diag([1, 0.1, 1, 0.1]) for _ in range(n_steps + 1)]

        fig = create_animated_tracking(
            true_states=true_states,
            estimates=estimates,
            measurements=measurements,
            covariances=covariances,
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.frames) == n_steps
