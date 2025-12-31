"""
Performance metric visualization utilities.

This module provides functions for visualizing tracking and estimation
performance metrics such as RMSE, NEES, NIS, and OSPA.
"""

from typing import List, Optional

import numpy as np
from numpy.typing import ArrayLike

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


def plot_rmse_over_time(
    errors: ArrayLike,
    time: Optional[ArrayLike] = None,
    component_names: Optional[List[str]] = None,
    title: str = "RMSE Over Time",
    ylabel: str = "RMSE",
) -> "go.Figure":
    """
    Plot Root Mean Square Error over time.

    Parameters
    ----------
    errors : array_like
        Error trajectory of shape (n_steps, n_dims) or (n_steps,).
    time : array_like, optional
        Time vector. If None, uses step indices.
    component_names : list of str, optional
        Names for each error component.
    title : str, optional
        Figure title.
    ylabel : str, optional
        Y-axis label.

    Returns
    -------
    fig : go.Figure
        Plotly figure.
    """
    if not HAS_PLOTLY:
        raise ImportError("plotly is required for plotting functions")

    errors = np.asarray(errors)
    if errors.ndim == 1:
        errors = errors.reshape(-1, 1)

    n_steps, n_dims = errors.shape

    if time is None:
        time = np.arange(n_steps)

    if component_names is None:
        if n_dims == 1:
            component_names = ["RMSE"]
        else:
            component_names = [f"Component {i}" for i in range(n_dims)]

    fig = go.Figure()

    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
    ]

    for i in range(n_dims):
        rmse = np.sqrt(np.cumsum(errors[:, i] ** 2) / np.arange(1, n_steps + 1))
        fig.add_trace(
            go.Scatter(
                x=time,
                y=rmse,
                mode="lines",
                name=component_names[i],
                line=dict(color=colors[i % len(colors)], width=2),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title=ylabel,
    )

    return fig


def plot_nees_sequence(
    nees_values: ArrayLike,
    time: Optional[ArrayLike] = None,
    n_dims: int = 2,
    confidence: float = 0.95,
    title: str = "NEES Over Time",
) -> "go.Figure":
    """
    Plot Normalized Estimation Error Squared with confidence bounds.

    Parameters
    ----------
    nees_values : array_like
        NEES values over time.
    time : array_like, optional
        Time vector.
    n_dims : int, optional
        Number of dimensions for chi-squared bounds. Default is 2.
    confidence : float, optional
        Confidence level for bounds. Default is 0.95.
    title : str, optional
        Figure title.

    Returns
    -------
    fig : go.Figure
        Plotly figure.
    """
    if not HAS_PLOTLY:
        raise ImportError("plotly is required for plotting functions")

    from scipy import stats

    nees_values = np.asarray(nees_values)
    n_steps = len(nees_values)

    if time is None:
        time = np.arange(n_steps)

    # Chi-squared bounds
    alpha = 1 - confidence
    lower_bound = stats.chi2.ppf(alpha / 2, df=n_dims)
    upper_bound = stats.chi2.ppf(1 - alpha / 2, df=n_dims)

    fig = go.Figure()

    # Confidence region
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([time, time[::-1]]),
            y=np.concatenate(
                [np.full(n_steps, upper_bound), np.full(n_steps, lower_bound)]
            ),
            fill="toself",
            fillcolor="rgba(0, 255, 0, 0.1)",
            line=dict(color="rgba(0,0,0,0)"),
            name=f"{int(confidence * 100)}% confidence",
        )
    )

    # Expected value line
    fig.add_trace(
        go.Scatter(
            x=time,
            y=np.full(n_steps, n_dims),
            mode="lines",
            line=dict(color="green", dash="dash", width=1),
            name=f"Expected ({n_dims})",
        )
    )

    # NEES values
    fig.add_trace(
        go.Scatter(
            x=time,
            y=nees_values,
            mode="lines+markers",
            line=dict(color="blue", width=2),
            marker=dict(size=4),
            name="NEES",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="NEES",
        yaxis=dict(range=[0, max(nees_values.max() * 1.1, upper_bound * 1.5)]),
    )

    return fig


def plot_nis_sequence(
    nis_values: ArrayLike,
    time: Optional[ArrayLike] = None,
    n_meas: int = 2,
    confidence: float = 0.95,
    title: str = "NIS Over Time",
) -> "go.Figure":
    """
    Plot Normalized Innovation Squared with confidence bounds.

    Parameters
    ----------
    nis_values : array_like
        NIS values over time.
    time : array_like, optional
        Time vector.
    n_meas : int, optional
        Measurement dimension for chi-squared bounds. Default is 2.
    confidence : float, optional
        Confidence level for bounds. Default is 0.95.
    title : str, optional
        Figure title.

    Returns
    -------
    fig : go.Figure
        Plotly figure.
    """
    return plot_nees_sequence(
        nis_values,
        time=time,
        n_dims=n_meas,
        confidence=confidence,
        title=title,
    )


def plot_ospa_over_time(
    ospa_values: ArrayLike,
    time: Optional[ArrayLike] = None,
    localization: Optional[ArrayLike] = None,
    cardinality: Optional[ArrayLike] = None,
    title: str = "OSPA Metric Over Time",
) -> "go.Figure":
    """
    Plot OSPA (Optimal SubPattern Assignment) metric over time.

    Parameters
    ----------
    ospa_values : array_like
        Total OSPA values over time.
    time : array_like, optional
        Time vector.
    localization : array_like, optional
        Localization component of OSPA.
    cardinality : array_like, optional
        Cardinality component of OSPA.
    title : str, optional
        Figure title.

    Returns
    -------
    fig : go.Figure
        Plotly figure.
    """
    if not HAS_PLOTLY:
        raise ImportError("plotly is required for plotting functions")

    ospa_values = np.asarray(ospa_values)
    n_steps = len(ospa_values)

    if time is None:
        time = np.arange(n_steps)

    fig = go.Figure()

    # Total OSPA
    fig.add_trace(
        go.Scatter(
            x=time,
            y=ospa_values,
            mode="lines",
            line=dict(color="blue", width=2),
            name="OSPA (total)",
        )
    )

    # Localization component
    if localization is not None:
        fig.add_trace(
            go.Scatter(
                x=time,
                y=localization,
                mode="lines",
                line=dict(color="green", width=2, dash="dash"),
                name="Localization",
            )
        )

    # Cardinality component
    if cardinality is not None:
        fig.add_trace(
            go.Scatter(
                x=time,
                y=cardinality,
                mode="lines",
                line=dict(color="red", width=2, dash="dot"),
                name="Cardinality",
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="OSPA",
    )

    return fig


def plot_cardinality_over_time(
    true_cardinality: ArrayLike,
    estimated_cardinality: ArrayLike,
    time: Optional[ArrayLike] = None,
    title: str = "Cardinality Over Time",
) -> "go.Figure":
    """
    Plot true vs estimated number of targets over time.

    Parameters
    ----------
    true_cardinality : array_like
        True number of targets at each time step.
    estimated_cardinality : array_like
        Estimated number of targets at each time step.
    time : array_like, optional
        Time vector.
    title : str, optional
        Figure title.

    Returns
    -------
    fig : go.Figure
        Plotly figure.
    """
    if not HAS_PLOTLY:
        raise ImportError("plotly is required for plotting functions")

    true_cardinality = np.asarray(true_cardinality)
    estimated_cardinality = np.asarray(estimated_cardinality)
    n_steps = len(true_cardinality)

    if time is None:
        time = np.arange(n_steps)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=time,
            y=true_cardinality,
            mode="lines+markers",
            line=dict(color="green", width=2),
            marker=dict(size=6),
            name="True",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=time,
            y=estimated_cardinality,
            mode="lines+markers",
            line=dict(color="blue", width=2, dash="dash"),
            marker=dict(size=6, symbol="x"),
            name="Estimated",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Number of Targets",
        yaxis=dict(tickmode="linear", tick0=0, dtick=1),
    )

    return fig


def plot_error_histogram(
    errors: ArrayLike,
    n_bins: int = 50,
    component_names: Optional[List[str]] = None,
    title: str = "Error Distribution",
    show_gaussian_fit: bool = True,
) -> "go.Figure":
    """
    Plot histogram of estimation errors.

    Parameters
    ----------
    errors : array_like
        Errors of shape (n_samples, n_dims) or (n_samples,).
    n_bins : int, optional
        Number of histogram bins. Default is 50.
    component_names : list of str, optional
        Names for each error component.
    title : str, optional
        Figure title.
    show_gaussian_fit : bool, optional
        Whether to overlay Gaussian fit. Default is True.

    Returns
    -------
    fig : go.Figure
        Plotly figure.
    """
    if not HAS_PLOTLY:
        raise ImportError("plotly is required for plotting functions")

    errors = np.asarray(errors)
    if errors.ndim == 1:
        errors = errors.reshape(-1, 1)

    n_samples, n_dims = errors.shape

    if component_names is None:
        if n_dims == 1:
            component_names = ["Error"]
        else:
            component_names = [f"Component {i}" for i in range(n_dims)]

    n_cols = min(n_dims, 3)
    n_rows = int(np.ceil(n_dims / n_cols))

    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=component_names)

    for i in range(n_dims):
        row = i // n_cols + 1
        col = i % n_cols + 1

        err = errors[:, i]

        # Histogram
        fig.add_trace(
            go.Histogram(
                x=err,
                nbinsx=n_bins,
                name=component_names[i],
                showlegend=False,
                marker_color="blue",
                opacity=0.7,
            ),
            row=row,
            col=col,
        )

        # Gaussian fit
        if show_gaussian_fit:
            mean = np.mean(err)
            std = np.std(err)
            x_fit = np.linspace(err.min(), err.max(), 100)
            y_fit = (
                n_samples
                * (err.max() - err.min())
                / n_bins
                * np.exp(-0.5 * ((x_fit - mean) / std) ** 2)
                / (std * np.sqrt(2 * np.pi))
            )

            fig.add_trace(
                go.Scatter(
                    x=x_fit,
                    y=y_fit,
                    mode="lines",
                    line=dict(color="red", width=2),
                    name=f"N({mean:.2f}, {std:.2f})",
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

    fig.update_layout(title=title, height=300 * n_rows)

    return fig


def plot_consistency_summary(
    nees_values: ArrayLike,
    nis_values: Optional[ArrayLike] = None,
    n_state_dims: int = 4,
    n_meas_dims: int = 2,
    confidence: float = 0.95,
    title: str = "Filter Consistency Summary",
) -> "go.Figure":
    """
    Create a summary plot of filter consistency metrics.

    Parameters
    ----------
    nees_values : array_like
        NEES values over time.
    nis_values : array_like, optional
        NIS values over time.
    n_state_dims : int, optional
        State dimension for NEES bounds. Default is 4.
    n_meas_dims : int, optional
        Measurement dimension for NIS bounds. Default is 2.
    confidence : float, optional
        Confidence level. Default is 0.95.
    title : str, optional
        Figure title.

    Returns
    -------
    fig : go.Figure
        Plotly figure.
    """
    if not HAS_PLOTLY:
        raise ImportError("plotly is required for plotting functions")

    from scipy import stats

    nees_values = np.asarray(nees_values)

    n_plots = 2 if nis_values is not None else 1
    subplot_titles = ["NEES"] if nis_values is None else ["NEES", "NIS"]

    fig = make_subplots(rows=n_plots, cols=1, subplot_titles=subplot_titles)

    # NEES plot
    n_steps = len(nees_values)
    time = np.arange(n_steps)

    alpha = 1 - confidence
    nees_lower = stats.chi2.ppf(alpha / 2, df=n_state_dims)
    nees_upper = stats.chi2.ppf(1 - alpha / 2, df=n_state_dims)

    # Confidence region
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([time, time[::-1]]),
            y=np.concatenate(
                [np.full(n_steps, nees_upper), np.full(n_steps, nees_lower)]
            ),
            fill="toself",
            fillcolor="rgba(0, 255, 0, 0.1)",
            line=dict(color="rgba(0,0,0,0)"),
            name=f"{int(confidence * 100)}% confidence",
            showlegend=True,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=time,
            y=nees_values,
            mode="lines",
            line=dict(color="blue", width=2),
            name="NEES",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=time,
            y=np.full(n_steps, n_state_dims),
            mode="lines",
            line=dict(color="green", dash="dash"),
            name="Expected",
        ),
        row=1,
        col=1,
    )

    # NIS plot
    if nis_values is not None:
        nis_values = np.asarray(nis_values)
        nis_lower = stats.chi2.ppf(alpha / 2, df=n_meas_dims)
        nis_upper = stats.chi2.ppf(1 - alpha / 2, df=n_meas_dims)

        fig.add_trace(
            go.Scatter(
                x=np.concatenate([time, time[::-1]]),
                y=np.concatenate(
                    [np.full(n_steps, nis_upper), np.full(n_steps, nis_lower)]
                ),
                fill="toself",
                fillcolor="rgba(0, 255, 0, 0.1)",
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=time,
                y=nis_values,
                mode="lines",
                line=dict(color="blue", width=2),
                name="NIS",
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=time,
                y=np.full(n_steps, n_meas_dims),
                mode="lines",
                line=dict(color="green", dash="dash"),
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    fig.update_layout(title=title, height=300 * n_plots)

    return fig


def plot_monte_carlo_rmse(
    monte_carlo_errors: ArrayLike,
    time: Optional[ArrayLike] = None,
    component_names: Optional[List[str]] = None,
    show_individual: bool = False,
    title: str = "Monte Carlo RMSE",
) -> "go.Figure":
    """
    Plot RMSE from Monte Carlo simulations.

    Parameters
    ----------
    monte_carlo_errors : array_like
        Errors of shape (n_runs, n_steps, n_dims).
    time : array_like, optional
        Time vector.
    component_names : list of str, optional
        Names for each error component.
    show_individual : bool, optional
        Whether to show individual run traces. Default is False.
    title : str, optional
        Figure title.

    Returns
    -------
    fig : go.Figure
        Plotly figure.
    """
    if not HAS_PLOTLY:
        raise ImportError("plotly is required for plotting functions")

    errors = np.asarray(monte_carlo_errors)
    n_runs, n_steps, n_dims = errors.shape

    if time is None:
        time = np.arange(n_steps)

    if component_names is None:
        component_names = [f"Component {i}" for i in range(n_dims)]

    # Compute RMSE across runs
    rmse = np.sqrt(np.mean(errors**2, axis=0))

    fig = make_subplots(rows=n_dims, cols=1, subplot_titles=component_names)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for dim in range(n_dims):
        row = dim + 1

        # Individual runs (faded)
        if show_individual:
            for run in range(n_runs):
                run_rmse = np.abs(errors[run, :, dim])
                fig.add_trace(
                    go.Scatter(
                        x=time,
                        y=run_rmse,
                        mode="lines",
                        line=dict(color="gray", width=0.5),
                        opacity=0.3,
                        showlegend=False,
                    ),
                    row=row,
                    col=1,
                )

        # Mean RMSE
        fig.add_trace(
            go.Scatter(
                x=time,
                y=rmse[:, dim],
                mode="lines",
                line=dict(color=colors[dim % len(colors)], width=2),
                name=f"RMSE {component_names[dim]}",
            ),
            row=row,
            col=1,
        )

    fig.update_layout(title=title, height=250 * n_dims)

    return fig


__all__ = [
    "plot_rmse_over_time",
    "plot_nees_sequence",
    "plot_nis_sequence",
    "plot_ospa_over_time",
    "plot_cardinality_over_time",
    "plot_error_histogram",
    "plot_consistency_summary",
    "plot_monte_carlo_rmse",
]
