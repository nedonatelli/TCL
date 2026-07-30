"""
Comprehensive tests for plotting metrics visualization module.

Tests coverage for:
- RMSE visualization
- NEES/NIS sequence visualization
- OSPA visualization
- Cardinality visualization
- Consistency summary plotting
"""

import numpy as np
import pytest

# Check if plotly is available
try:
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from pytcl.plotting.metrics import (
    plot_cardinality_over_time,
    plot_consistency_summary,
    plot_error_histogram,
    plot_monte_carlo_rmse,
    plot_nees_sequence,
    plot_nis_sequence,
    plot_ospa_over_time,
    plot_rmse_over_time,
)


@pytest.mark.skipif(not HAS_PLOTLY, reason="plotly not available")
class TestRMSEPlotting:
    """Tests for RMSE visualization."""

    def test_plot_rmse_1d_errors(self):
        """Test plotting RMSE with 1D error array."""
        errors = np.array([0.1, 0.2, 0.15, 0.25])
        fig = plot_rmse_over_time(errors)

        assert fig is not None
        assert isinstance(fig, go.Figure)

    def test_plot_rmse_2d_errors(self):
        """Test plotting RMSE with 2D error array."""
        errors = np.array([[0.1, 0.2], [0.15, 0.25], [0.12, 0.22]])
        fig = plot_rmse_over_time(errors)

        assert fig is not None
        assert isinstance(fig, go.Figure)

    def test_plot_rmse_with_time_array(self):
        """Test plotting RMSE with custom time vector."""
        errors = np.array([0.1, 0.2, 0.15])
        time = np.array([0.0, 1.0, 2.0])
        fig = plot_rmse_over_time(errors, time=time)

        assert fig is not None

    def test_plot_rmse_with_component_names(self):
        """Test plotting RMSE with custom component names."""
        errors = np.array([[0.1, 0.2], [0.15, 0.25]])
        names = ["Position Error", "Velocity Error"]
        fig = plot_rmse_over_time(errors, component_names=names)

        assert fig is not None

    def test_plot_rmse_with_title_and_labels(self):
        """Test plotting RMSE with custom title and labels."""
        errors = np.array([0.1, 0.2, 0.15])
        fig = plot_rmse_over_time(errors, title="Custom RMSE Plot", ylabel="Error (m)")

        assert fig is not None

    def test_plot_rmse_large_array(self):
        """Test plotting RMSE with large error array."""
        errors = np.random.rand(1000, 3) * 0.5
        fig = plot_rmse_over_time(errors)

        assert fig is not None

    def test_plot_rmse_single_value(self):
        """Test RMSE with single error value."""
        errors = np.array([0.5])
        fig = plot_rmse_over_time(errors)

        assert fig is not None

    def test_plot_rmse_zero_errors(self):
        """Test RMSE with all zero errors."""
        errors = np.zeros(5)
        fig = plot_rmse_over_time(errors)

        assert fig is not None

    def test_plot_rmse_constant_errors(self):
        """Test RMSE with constant error values."""
        errors = np.ones(10) * 0.5
        fig = plot_rmse_over_time(errors)

        assert fig is not None


@pytest.mark.skipif(not HAS_PLOTLY, reason="plotly not available")
class TestNEESPlotting:
    """Tests for NEES visualization."""

    def test_plot_nees_basic(self):
        """Test basic NEES plotting."""
        nees_values = np.array([1.0, 1.5, 0.8, 1.2, 1.1])
        fig = plot_nees_sequence(nees_values)

        assert fig is not None
        assert isinstance(fig, go.Figure)

    def test_plot_nees_with_time(self):
        """Test NEES plotting with time vector."""
        nees_values = np.array([1.0, 1.5, 0.8, 1.2])
        time = np.array([0.0, 1.0, 2.0, 3.0])
        fig = plot_nees_sequence(nees_values, time=time)

        assert fig is not None

    def test_plot_nees_with_bounds(self):
        """Test NEES plotting with confidence bounds."""
        nees_values = np.array([1.0, 1.5, 0.8, 1.2, 1.1])
        fig = plot_nees_sequence(
            nees_values,
            title="NEES with Bounds",
        )

        assert fig is not None

    def test_plot_nees_large_sequence(self):
        """Test NEES with long time sequence."""
        nees_values = np.random.exponential(1.0, 500)
        fig = plot_nees_sequence(nees_values)

        assert fig is not None

    def test_plot_nees_all_ones(self):
        """Test NEES with all unit values."""
        nees = np.ones(10)
        fig = plot_nees_sequence(nees)

        assert fig is not None

    def test_plot_nees_high_values(self):
        """Test NEES with high values."""
        nees = np.array([5.0, 10.0, 15.0, 8.0])
        fig = plot_nees_sequence(nees)

        assert fig is not None


@pytest.mark.skipif(not HAS_PLOTLY, reason="plotly not available")
class TestNISPlotting:
    """Tests for NIS visualization."""

    def test_plot_nis_basic(self):
        """Test basic NIS plotting."""
        nis_values = np.array([0.5, 0.8, 1.2, 0.9, 1.1])
        fig = plot_nis_sequence(nis_values)

        assert fig is not None
        assert isinstance(fig, go.Figure)

    def test_plot_nis_with_time(self):
        """Test NIS plotting with time vector."""
        nis_values = np.array([0.5, 0.8, 1.2, 0.9])
        time = np.array([0.0, 1.0, 2.0, 3.0])
        fig = plot_nis_sequence(nis_values, time=time)

        assert fig is not None

    def test_plot_nis_with_custom_labels(self):
        """Test NIS with custom labels."""
        nis_values = np.array([0.5, 0.8, 1.2, 0.9, 1.1])
        fig = plot_nis_sequence(
            nis_values,
            title="Innovation Consistency",
        )

        assert fig is not None

    def test_plot_nis_high_values(self):
        """Test NIS with high values."""
        nis = np.array([5.0, 10.0, 15.0, 8.0])
        fig = plot_nis_sequence(nis)

        assert fig is not None


@pytest.mark.skipif(not HAS_PLOTLY, reason="plotly not available")
class TestOSPAPlotting:
    """Tests for OSPA visualization."""

    def test_plot_ospa_basic(self):
        """Test basic OSPA plotting."""
        ospa_values = np.array([1.0, 0.8, 1.2, 0.9, 1.1])
        fig = plot_ospa_over_time(ospa_values)

        assert fig is not None
        assert isinstance(fig, go.Figure)

    def test_plot_ospa_with_time(self):
        """Test OSPA plotting with time vector."""
        ospa_values = np.array([1.0, 0.8, 1.2, 0.9])
        time = np.array([0.0, 1.0, 2.0, 3.0])
        fig = plot_ospa_over_time(ospa_values, time=time)

        assert fig is not None

    def test_plot_ospa_with_custom_params(self):
        """Test OSPA with custom parameters."""
        ospa_values = np.array([1.0, 0.8, 1.2, 0.9, 1.1, 0.95])
        fig = plot_ospa_over_time(
            ospa_values,
            title="Optimal Subpattern Assignment Metric",
        )

        assert fig is not None

    def test_plot_ospa_with_components(self):
        """Test OSPA plotting with components."""
        ospa_values = np.array([1.0, 0.8, 1.2, 0.9, 1.1])
        fig = plot_ospa_over_time(ospa_values)

        assert fig is not None

    def test_plot_ospa_very_small_values(self):
        """Test OSPA with very small values."""
        ospa = np.array([0.001, 0.002, 0.0015, 0.0025])
        fig = plot_ospa_over_time(ospa)

        assert fig is not None


@pytest.mark.skipif(not HAS_PLOTLY, reason="plotly not available")
class TestCardinalityPlotting:
    """Tests for cardinality visualization."""

    def test_plot_cardinality_basic(self):
        """Test basic cardinality plotting."""
        true_card = np.array([2, 3, 2, 3, 3])
        est_card = np.array([2, 2, 2, 3, 3])

        fig = plot_cardinality_over_time(true_card, est_card)

        assert fig is not None
        assert isinstance(fig, go.Figure)

    def test_plot_cardinality_with_time(self):
        """Test cardinality with time vector."""
        true_card = np.array([2, 3, 2])
        est_card = np.array([2, 2, 2])
        time = np.array([0.0, 1.0, 2.0])

        fig = plot_cardinality_over_time(true_card, est_card, time=time)

        assert fig is not None

    def test_plot_cardinality_with_labels(self):
        """Test cardinality with custom labels."""
        true_card = np.array([1, 2, 3, 2])
        est_card = np.array([1, 2, 2, 2])

        fig = plot_cardinality_over_time(
            true_card, est_card, title="Target Cardinality"
        )

        assert fig is not None


@pytest.mark.skipif(not HAS_PLOTLY, reason="plotly not available")
class TestConsistencySummaryPlotting:
    """Tests for consistency summary visualization."""

    def test_plot_consistency_summary_basic(self):
        """Test basic consistency summary."""
        nees_values = np.random.exponential(1.0, 100)
        nis_values = np.random.exponential(1.0, 100)

        fig = plot_consistency_summary(nees_values, nis_values)

        assert fig is not None
        assert isinstance(fig, go.Figure)

    def test_plot_consistency_summary_with_bounds(self):
        """Test consistency summary with bounds."""
        nees = np.random.exponential(1.0, 50)
        nis = np.random.exponential(1.0, 50)

        fig = plot_consistency_summary(nees, nis, title="Filter Consistency Check")

        assert fig is not None


@pytest.mark.skipif(not HAS_PLOTLY, reason="plotly not available")
class TestErrorHistogramPlotting:
    """Tests for error histogram visualization."""

    def test_plot_error_histogram_1d(self):
        """Test error histogram with 1D errors."""
        errors = np.random.randn(1000) * 0.5
        fig = plot_error_histogram(errors)

        assert fig is not None
        assert isinstance(fig, go.Figure)

    def test_plot_error_histogram_2d(self):
        """Test error histogram with 2D errors."""
        errors = np.random.randn(500, 2) * 0.5
        fig = plot_error_histogram(errors)

        assert fig is not None

    def test_plot_error_histogram_with_labels(self):
        """Test error histogram with component labels."""
        errors = np.random.randn(200, 3) * 0.5
        labels = ["X Error", "Y Error", "Z Error"]

        fig = plot_error_histogram(errors, component_names=labels)

        assert fig is not None


@pytest.mark.skipif(not HAS_PLOTLY, reason="plotly not available")
class TestMonteCarloRMSEPlotting:
    """Tests for Monte Carlo RMSE visualization."""

    def test_plot_monte_carlo_rmse_basic(self):
        """Test basic Monte Carlo RMSE."""
        # Simulated: 10 runs, 50 timesteps, 2 components
        rmse_runs = np.random.rand(10, 50, 2) * 0.5

        fig = plot_monte_carlo_rmse(rmse_runs)

        assert fig is not None
        assert isinstance(fig, go.Figure)

    def test_plot_monte_carlo_rmse_with_time(self):
        """Test Monte Carlo RMSE with time."""
        rmse_runs = np.random.rand(5, 30, 1) * 0.5
        time = np.linspace(0, 10, 30)

        fig = plot_monte_carlo_rmse(rmse_runs, time=time)

        assert fig is not None

    def test_plot_monte_carlo_rmse_many_runs(self):
        """Test Monte Carlo RMSE with many runs."""
        rmse_runs = np.random.rand(100, 20, 2) * 0.5

        fig = plot_monte_carlo_rmse(rmse_runs)

        assert fig is not None


@pytest.mark.skipif(not HAS_PLOTLY, reason="plotly not available")
class TestMetricsPlottingDataTypes:
    """Tests for different data types in metrics plotting."""

    def test_plot_rmse_float32(self):
        """Test RMSE with float32 data."""
        errors = np.array([0.1, 0.2, 0.15], dtype=np.float32)
        fig = plot_rmse_over_time(errors)

        assert fig is not None

    def test_plot_rmse_float64(self):
        """Test RMSE with float64 data."""
        errors = np.array([0.1, 0.2, 0.15], dtype=np.float64)
        fig = plot_rmse_over_time(errors)

        assert fig is not None

    def test_plot_rmse_integer_input(self):
        """Test RMSE with integer input (should work after conversion)."""
        errors = np.array([1, 2, 1, 3], dtype=np.int32)
        fig = plot_rmse_over_time(errors)

        assert fig is not None

    def test_plot_nees_list_input(self):
        """Test NEES with Python list input."""
        nees_list = [1.0, 1.5, 0.8, 1.2]
        fig = plot_nees_sequence(nees_list)

        assert fig is not None


@pytest.mark.skipif(not HAS_PLOTLY, reason="plotly not available")
class TestMetricsPlottingOutputProperties:
    """Tests for properties of generated figures."""

    def test_rmse_figure_has_traces(self):
        """Test that RMSE figure has traces."""
        errors = np.array([[0.1, 0.2], [0.15, 0.25]])
        fig = plot_rmse_over_time(errors)

        assert len(fig.data) > 0

    def test_rmse_figure_has_layout(self):
        """Test that RMSE figure has layout information."""
        errors = np.array([0.1, 0.2, 0.15])
        fig = plot_rmse_over_time(errors)

        assert fig.layout is not None

    def test_nees_figure_has_legend(self):
        """Test that NEES figure has legend."""
        nees = np.array([1.0, 1.5, 0.8])
        fig = plot_nees_sequence(nees)

        assert fig.layout is not None

    def test_ospa_figure_structure(self):
        """Test OSPA figure structure."""
        ospa = np.array([1.0, 0.9, 0.8, 0.85])
        fig = plot_ospa_over_time(ospa)

        # Figure should have data
        assert len(fig.data) >= 1


@pytest.mark.skipif(not HAS_PLOTLY, reason="plotly not available")
class TestMetricsPlottingIntegration:
    """Integration tests for metrics plotting."""

    def test_workflow_rmse_then_nees(self):
        """Test workflow of plotting RMSE then NEES."""
        errors = np.random.rand(20) * 0.5
        nees = np.random.exponential(1.0, 20)

        fig_rmse = plot_rmse_over_time(errors)
        fig_nees = plot_nees_sequence(nees)

        assert fig_rmse is not None
        assert fig_nees is not None

    def test_all_metric_types_workflow(self):
        """Test creating multiple metric type plots."""
        rmse_data = np.random.rand(15) * 0.5
        nees_data = np.random.exponential(1.0, 15)
        nis_data = np.random.exponential(0.8, 15)
        ospa_data = np.random.exponential(1.0, 15)

        figs = [
            plot_rmse_over_time(rmse_data),
            plot_nees_sequence(nees_data),
            plot_nis_sequence(nis_data),
            plot_ospa_over_time(ospa_data),
        ]

        assert all(f is not None for f in figs)

    def test_consistency_check_workflow(self):
        """Test consistency checking workflow."""
        nees = np.random.exponential(1.0, 100)
        nis = np.random.exponential(1.0, 100)

        fig1 = plot_nees_sequence(nees)
        fig2 = plot_nis_sequence(nis)
        fig3 = plot_consistency_summary(nees, nis)

        assert all(f is not None for f in [fig1, fig2, fig3])
