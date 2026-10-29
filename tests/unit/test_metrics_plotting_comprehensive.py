"""
Tests for the metrics plotting module.

Each test builds a figure from inputs whose expected trace data can be written
down by hand, then compares ``fig.data[i].x`` / ``.y`` and the axis titles.

Oracles:
- running RMSE: sqrt(cumsum(e^2) / k), written from the definition;
- chi-squared 95% bounds for 2 degrees of freedom: the CDF is 1 - exp(-x/2), so
  the quantiles are -2 ln(1 - p);
- chi-squared 95% bounds for 4 degrees of freedom: the CDF is
  1 - exp(-x/2) (1 + x/2), checked by evaluating it at the plotted bounds.
"""

import numpy as np
import pytest

try:
    import plotly.graph_objects as go  # noqa: F401

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

pytestmark = pytest.mark.skipif(not HAS_PLOTLY, reason="plotly not available")

# chi-squared, 2 dof: CDF = 1 - exp(-x/2)  =>  quantile(p) = -2 ln(1 - p)
_CHI2_2_LOWER_95 = -2.0 * np.log(1.0 - 0.025)
_CHI2_2_UPPER_95 = -2.0 * np.log(1.0 - 0.975)


def _chi2_4_cdf(x):
    """Chi-squared CDF for 4 dof in closed form."""
    return 1.0 - np.exp(-x / 2.0) * (1.0 + x / 2.0)


def _by_name(fig):
    return {trace.name: trace for trace in fig.data}


class TestRMSEPlotting:
    def test_1d_errors_running_rmse(self):
        errors = np.array([3.0, 4.0, 0.0, 12.0])
        fig = plot_rmse_over_time(errors)

        assert len(fig.data) == 1
        np.testing.assert_array_equal(fig.data[0].x, np.arange(4))
        expected = np.sqrt(np.array([9.0, 25.0, 25.0, 169.0]) / np.array([1, 2, 3, 4]))
        np.testing.assert_allclose(fig.data[0].y, expected, rtol=1e-14, atol=0)
        assert fig.data[0].name == "RMSE"
        assert fig.layout.xaxis.title.text == "Time"
        assert fig.layout.yaxis.title.text == "RMSE"
        assert fig.layout.title.text == "RMSE Over Time"

    def test_2d_errors_one_trace_per_column(self):
        errors = np.array([[3.0, 1.0], [4.0, 1.0], [0.0, 7.0]])
        fig = plot_rmse_over_time(errors)

        assert [t.name for t in fig.data] == ["Component 0", "Component 1"]
        np.testing.assert_allclose(
            fig.data[0].y,
            np.sqrt(np.array([9.0, 25.0, 25.0]) / np.array([1, 2, 3])),
            rtol=1e-14,
            atol=0,
        )
        np.testing.assert_allclose(
            fig.data[1].y,
            np.sqrt(np.array([1.0, 2.0, 51.0]) / np.array([1, 2, 3])),
            rtol=1e-14,
            atol=0,
        )

    def test_custom_time_names_title_and_label(self):
        errors = np.array([[0.1, 0.2], [0.15, 0.25]])
        time = np.array([10.0, 12.5])
        fig = plot_rmse_over_time(
            errors,
            time=time,
            component_names=["Position Error", "Velocity Error"],
            title="Custom RMSE Plot",
            ylabel="Error (m)",
        )

        assert [t.name for t in fig.data] == ["Position Error", "Velocity Error"]
        for trace in fig.data:
            np.testing.assert_array_equal(trace.x, time)
        assert fig.layout.title.text == "Custom RMSE Plot"
        assert fig.layout.yaxis.title.text == "Error (m)"

    def test_single_value_is_its_absolute_value(self):
        fig = plot_rmse_over_time(np.array([-0.5]))
        np.testing.assert_array_equal(fig.data[0].y, [0.5])

    def test_zero_errors_give_zero_rmse(self):
        fig = plot_rmse_over_time(np.zeros(5))
        np.testing.assert_array_equal(fig.data[0].y, np.zeros(5))

    def test_constant_errors_give_constant_rmse(self):
        fig = plot_rmse_over_time(np.full(10, 0.5))
        np.testing.assert_allclose(fig.data[0].y, np.full(10, 0.5), rtol=1e-14, atol=0)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32])
    def test_dtype_does_not_change_values(self, dtype):
        errors = np.array([3, 4, 0, 12]).astype(dtype)
        fig = plot_rmse_over_time(errors)
        expected = np.sqrt(np.array([9.0, 25.0, 25.0, 169.0]) / np.array([1, 2, 3, 4]))
        np.testing.assert_allclose(fig.data[0].y, expected, rtol=1e-6, atol=0)


class TestNEESPlotting:
    def test_traces_and_chi2_bounds_dof2(self):
        nees = np.array([1.0, 1.5, 0.8, 1.2, 1.1])
        fig = plot_nees_sequence(nees)
        t = np.arange(5)

        band, expected, values = fig.data
        assert band.name == "95% confidence"
        np.testing.assert_array_equal(band.x, np.concatenate([t, t[::-1]]))
        np.testing.assert_allclose(
            band.y,
            np.concatenate(
                [np.full(5, _CHI2_2_UPPER_95), np.full(5, _CHI2_2_LOWER_95)]
            ),
            rtol=1e-12,
            atol=0,
        )
        assert expected.name == "Expected (2)"
        np.testing.assert_array_equal(expected.y, np.full(5, 2))
        np.testing.assert_array_equal(values.x, t)
        np.testing.assert_array_equal(values.y, nees)
        assert values.name == "NEES"
        assert fig.layout.yaxis.title.text == "NEES"
        assert fig.layout.xaxis.title.text == "Time"

    def test_dof4_bounds_are_the_2_5_and_97_5_percentiles(self):
        fig = plot_nees_sequence(np.ones(3), n_dims=4)
        y = np.asarray(fig.data[0].y)
        upper, lower = y[0], y[-1]
        assert _chi2_4_cdf(upper) == pytest.approx(0.975, rel=1e-12, abs=0)
        assert _chi2_4_cdf(lower) == pytest.approx(0.025, rel=1e-12, abs=0)
        assert fig.data[1].name == "Expected (4)"
        np.testing.assert_array_equal(fig.data[1].y, np.full(3, 4))

    def test_confidence_changes_band_and_label(self):
        fig = plot_nees_sequence(np.ones(2), confidence=0.90)
        assert fig.data[0].name == "90% confidence"
        y = np.asarray(fig.data[0].y)
        assert y[0] == pytest.approx(-2.0 * np.log(0.05), rel=1e-12, abs=0)
        assert y[-1] == pytest.approx(-2.0 * np.log(0.95), rel=1e-12, abs=0)

    def test_time_vector_used_on_every_trace(self):
        time = np.array([0.0, 1.0, 2.0, 3.0])
        fig = plot_nees_sequence(np.array([1.0, 1.5, 0.8, 1.2]), time=time)
        np.testing.assert_array_equal(fig.data[1].x, time)
        np.testing.assert_array_equal(fig.data[2].x, time)
        np.testing.assert_array_equal(fig.data[0].x, np.concatenate([time, time[::-1]]))

    def test_y_range_covers_high_values(self):
        fig = plot_nees_sequence(np.array([5.0, 10.0, 15.0, 8.0]))
        # max(15 * 1.1, upper * 1.5): 16.5 beats 7.378 * 1.5
        assert fig.layout.yaxis.range[0] == 0
        assert fig.layout.yaxis.range[1] == pytest.approx(16.5, rel=1e-12, abs=0)

    def test_y_range_floor_is_one_and_a_half_times_upper_bound(self):
        fig = plot_nees_sequence(np.ones(4))
        assert fig.layout.yaxis.range[1] == pytest.approx(
            1.5 * _CHI2_2_UPPER_95, rel=1e-12, abs=0
        )

    def test_list_input_gives_same_series(self):
        fig = plot_nees_sequence([1.0, 1.5, 0.8, 1.2])
        np.testing.assert_array_equal(fig.data[2].y, [1.0, 1.5, 0.8, 1.2])

    def test_custom_title(self):
        fig = plot_nees_sequence(np.ones(3), title="NEES with Bounds")
        assert fig.layout.title.text == "NEES with Bounds"


class TestNISPlotting:
    def test_series_and_bounds(self):
        nis = np.array([0.5, 0.8, 1.2, 0.9, 1.1])
        fig = plot_nis_sequence(nis)
        band, expected, values = fig.data
        np.testing.assert_array_equal(values.y, nis)
        np.testing.assert_array_equal(values.x, np.arange(5))
        np.testing.assert_allclose(
            np.asarray(band.y)[[0, -1]],
            [_CHI2_2_UPPER_95, _CHI2_2_LOWER_95],
            rtol=1e-12,
            atol=0,
        )
        assert expected.name == "Expected (2)"

    def test_n_meas_sets_degrees_of_freedom(self):
        fig = plot_nis_sequence(np.ones(3), n_meas=4)
        y = np.asarray(fig.data[0].y)
        assert _chi2_4_cdf(y[0]) == pytest.approx(0.975, rel=1e-12, abs=0)
        assert _chi2_4_cdf(y[-1]) == pytest.approx(0.025, rel=1e-12, abs=0)

    def test_time_and_title(self):
        time = np.array([0.0, 1.0, 2.0, 3.0])
        fig = plot_nis_sequence(np.array([0.5, 0.8, 1.2, 0.9]), time=time, title="T")
        np.testing.assert_array_equal(fig.data[2].x, time)
        assert fig.layout.title.text == "T"


class TestOSPAPlotting:
    def test_total_only(self):
        ospa = np.array([1.0, 0.8, 1.2, 0.9, 1.1])
        fig = plot_ospa_over_time(ospa)
        assert len(fig.data) == 1
        assert fig.data[0].name == "OSPA (total)"
        np.testing.assert_array_equal(fig.data[0].x, np.arange(5))
        np.testing.assert_array_equal(fig.data[0].y, ospa)
        assert fig.layout.yaxis.title.text == "OSPA"
        assert fig.layout.xaxis.title.text == "Time"

    def test_components_and_time(self):
        ospa = np.array([1.0, 0.8, 1.2, 0.9])
        loc = np.array([0.6, 0.5, 0.7, 0.6])
        card = np.array([0.4, 0.3, 0.5, 0.3])
        time = np.array([0.0, 1.0, 2.0, 3.0])
        fig = plot_ospa_over_time(
            ospa, time=time, localization=loc, cardinality=card, title="OSPA"
        )
        traces = _by_name(fig)
        assert list(traces) == ["OSPA (total)", "Localization", "Cardinality"]
        np.testing.assert_array_equal(traces["Localization"].y, loc)
        np.testing.assert_array_equal(traces["Cardinality"].y, card)
        for trace in fig.data:
            np.testing.assert_array_equal(trace.x, time)
        assert fig.layout.title.text == "OSPA"

    def test_small_values_preserved_exactly(self):
        ospa = np.array([0.001, 0.002, 0.0015, 0.0025])
        fig = plot_ospa_over_time(ospa)
        np.testing.assert_array_equal(fig.data[0].y, ospa)


class TestCardinalityPlotting:
    def test_true_and_estimated_traces(self):
        true_card = np.array([2, 3, 2, 3, 3])
        est_card = np.array([2, 2, 2, 3, 3])
        fig = plot_cardinality_over_time(true_card, est_card)

        assert [t.name for t in fig.data] == ["True", "Estimated"]
        np.testing.assert_array_equal(fig.data[0].y, true_card)
        np.testing.assert_array_equal(fig.data[1].y, est_card)
        np.testing.assert_array_equal(fig.data[0].x, np.arange(5))
        assert fig.layout.yaxis.title.text == "Number of Targets"
        assert fig.layout.xaxis.title.text == "Time"

    def test_time_and_title(self):
        time = np.array([0.0, 1.0, 2.0])
        fig = plot_cardinality_over_time(
            np.array([2, 3, 2]), np.array([2, 2, 2]), time=time, title="Cardinality"
        )
        np.testing.assert_array_equal(fig.data[0].x, time)
        np.testing.assert_array_equal(fig.data[1].x, time)
        assert fig.layout.title.text == "Cardinality"


class TestConsistencySummaryPlotting:
    def test_nees_and_nis_panels(self):
        nees = np.array([3.0, 4.5, 5.0, 3.5])
        nis = np.array([1.0, 2.5, 2.0, 1.5])
        fig = plot_consistency_summary(nees, nis, n_state_dims=4, n_meas_dims=2)
        t = np.arange(4)

        assert len(fig.data) == 6
        nees_band, nees_line, nees_exp, nis_band, nis_line, nis_exp = fig.data
        np.testing.assert_array_equal(nees_line.x, t)
        np.testing.assert_array_equal(nees_line.y, nees)
        np.testing.assert_array_equal(nees_exp.y, np.full(4, 4))
        np.testing.assert_array_equal(nis_line.y, nis)
        np.testing.assert_array_equal(nis_exp.y, np.full(4, 2))

        nees_y = np.asarray(nees_band.y)
        assert _chi2_4_cdf(nees_y[0]) == pytest.approx(0.975, rel=1e-12, abs=0)
        assert _chi2_4_cdf(nees_y[-1]) == pytest.approx(0.025, rel=1e-12, abs=0)
        nis_y = np.asarray(nis_band.y)
        assert nis_y[0] == pytest.approx(_CHI2_2_UPPER_95, rel=1e-12, abs=0)
        assert nis_y[-1] == pytest.approx(_CHI2_2_LOWER_95, rel=1e-12, abs=0)

        titles = [a.text for a in fig.layout.annotations]
        assert titles == ["NEES", "NIS"]

    def test_nees_only(self):
        fig = plot_consistency_summary(np.array([3.0, 4.0]), title="Check")
        assert len(fig.data) == 3
        assert [a.text for a in fig.layout.annotations] == ["NEES"]
        assert fig.layout.title.text == "Check"


class TestErrorHistogramPlotting:
    def test_1d_histogram_and_gaussian_fit(self):
        errors = np.array([-1.0, -0.5, 0.0, 0.5, 1.0, 0.0, 0.25, -0.25])
        fig = plot_error_histogram(errors, n_bins=4)

        hist, fit = fig.data
        np.testing.assert_array_equal(hist.x, errors)
        assert hist.nbinsx == 4

        mean = errors.mean()
        std = errors.std()
        span = errors.max() - errors.min()
        x_fit = np.linspace(errors.min(), errors.max(), 100)
        # Gaussian pdf scaled to counts per bin: n * bin_width * pdf
        expected = (
            errors.size
            * (span / 4)
            * np.exp(-0.5 * ((x_fit - mean) / std) ** 2)
            / (std * np.sqrt(2.0 * np.pi))
        )
        np.testing.assert_allclose(fit.x, x_fit, rtol=1e-14, atol=0)
        np.testing.assert_allclose(fit.y, expected, rtol=1e-13, atol=0)

    def test_fit_can_be_disabled(self):
        fig = plot_error_histogram(np.array([0.0, 1.0, 2.0]), show_gaussian_fit=False)
        assert len(fig.data) == 1

    def test_2d_errors_one_histogram_per_column(self):
        errors = np.array([[0.0, 10.0], [1.0, 11.0], [2.0, 13.0], [1.0, 10.0]])
        labels = ["X Error", "Y Error"]
        fig = plot_error_histogram(
            errors, component_names=labels, show_gaussian_fit=False
        )
        assert [t.name for t in fig.data] == labels
        np.testing.assert_array_equal(fig.data[0].x, errors[:, 0])
        np.testing.assert_array_equal(fig.data[1].x, errors[:, 1])
        assert [a.text for a in fig.layout.annotations] == labels


class TestMonteCarloRMSEPlotting:
    def test_rmse_across_runs(self):
        # runs x steps x dims
        errors = np.array(
            [
                [[3.0, 1.0], [0.0, 2.0], [1.0, 1.0]],
                [[4.0, -1.0], [0.0, -2.0], [-1.0, 3.0]],
            ]
        )
        fig = plot_monte_carlo_rmse(errors)

        assert [t.name for t in fig.data] == ["RMSE Component 0", "RMSE Component 1"]
        np.testing.assert_array_equal(fig.data[0].x, np.arange(3))
        np.testing.assert_allclose(
            fig.data[0].y,
            [np.sqrt(12.5), 0.0, 1.0],
            rtol=1e-14,
            atol=0,
        )
        np.testing.assert_allclose(
            fig.data[1].y, [1.0, 2.0, np.sqrt(5.0)], rtol=1e-14, atol=0
        )

    def test_time_vector(self):
        time = np.linspace(0, 10, 3)
        fig = plot_monte_carlo_rmse(np.ones((2, 3, 1)), time=time)
        np.testing.assert_array_equal(fig.data[0].x, time)
        np.testing.assert_array_equal(fig.data[0].y, np.ones(3))

    def test_show_individual_adds_absolute_error_traces(self):
        errors = np.array([[[3.0], [-4.0]], [[-1.0], [2.0]]])
        fig = plot_monte_carlo_rmse(errors, show_individual=True)
        assert len(fig.data) == 3
        np.testing.assert_array_equal(fig.data[0].y, [3.0, 4.0])
        np.testing.assert_array_equal(fig.data[1].y, [1.0, 2.0])
        np.testing.assert_allclose(
            fig.data[2].y, [np.sqrt(5.0), np.sqrt(10.0)], rtol=1e-14, atol=0
        )


class TestNISChartIsLabelledNIS:
    """A NIS chart must not be labelled NEES.

    ``plot_nis_sequence`` forwards to ``plot_nees_sequence``, which hardcoded
    "NEES" as both the y-axis title and the series name; only ``title`` was
    passed through.
    """

    def test_nis_axis_and_series_say_nis(self):
        fig = plot_nis_sequence([1.0, 2.0, 3.0])
        assert fig.layout.yaxis.title.text == "NIS"
        assert "NIS" in [trace.name for trace in fig.data]
        assert "NEES" not in [trace.name for trace in fig.data]

    def test_nees_chart_is_unchanged(self):
        fig = plot_nees_sequence([1.0, 2.0, 3.0])
        assert fig.layout.yaxis.title.text == "NEES"
        assert "NEES" in [trace.name for trace in fig.data]
