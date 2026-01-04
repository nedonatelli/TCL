"""
Trajectory and track plotting utilities.

This module provides functions for visualizing trajectories, tracks,
measurements, and estimation results in 2D and 3D.
"""

from typing import Any, Dict, List, Optional

import numpy as np
from numpy.typing import ArrayLike

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from pytcl.plotting.ellipses import covariance_ellipse_points


def plot_trajectory_2d(
    states: ArrayLike,
    x_idx: int = 0,
    y_idx: int = 1,
    mode: str = "lines+markers",
    color: str = "blue",
    name: Optional[str] = None,
    marker_size: int = 4,
    line_width: int = 2,
    showlegend: bool = True,
) -> "go.Scatter":
    """
    Create a Plotly trace for a 2D trajectory.

    Parameters
    ----------
    states : array_like
        State trajectory of shape (n_steps, n_dims).
    x_idx : int, optional
        Index of x coordinate in state vector. Default is 0.
    y_idx : int, optional
        Index of y coordinate in state vector. Default is 1.
    mode : str, optional
        Plotly mode string. Default is "lines+markers".
    color : str, optional
        Color for the trajectory. Default is "blue".
    name : str, optional
        Name for the trace.
    marker_size : int, optional
        Size of markers. Default is 4.
    line_width : int, optional
        Width of lines. Default is 2.
    showlegend : bool, optional
        Whether to show in legend. Default is True.

    Returns
    -------
    trace : go.Scatter
        Plotly scatter trace.
    """
    if not HAS_PLOTLY:
        raise ImportError("plotly is required for plotting functions")

    states = np.asarray(states)

    return go.Scatter(
        x=states[:, x_idx],
        y=states[:, y_idx],
        mode=mode,
        line=dict(color=color, width=line_width),
        marker=dict(size=marker_size, color=color),
        name=name or "Trajectory",
        showlegend=showlegend,
    )


def plot_trajectory_3d(
    states: ArrayLike,
    x_idx: int = 0,
    y_idx: int = 1,
    z_idx: int = 2,
    mode: str = "lines+markers",
    color: str = "blue",
    name: Optional[str] = None,
    marker_size: int = 3,
    line_width: int = 2,
    showlegend: bool = True,
) -> "go.Scatter3d":
    """
    Create a Plotly trace for a 3D trajectory.

    Parameters
    ----------
    states : array_like
        State trajectory of shape (n_steps, n_dims).
    x_idx : int, optional
        Index of x coordinate in state vector. Default is 0.
    y_idx : int, optional
        Index of y coordinate in state vector. Default is 1.
    z_idx : int, optional
        Index of z coordinate in state vector. Default is 2.
    mode : str, optional
        Plotly mode string. Default is "lines+markers".
    color : str, optional
        Color for the trajectory. Default is "blue".
    name : str, optional
        Name for the trace.
    marker_size : int, optional
        Size of markers. Default is 3.
    line_width : int, optional
        Width of lines. Default is 2.
    showlegend : bool, optional
        Whether to show in legend. Default is True.

    Returns
    -------
    trace : go.Scatter3d
        Plotly 3D scatter trace.
    """
    if not HAS_PLOTLY:
        raise ImportError("plotly is required for plotting functions")

    states = np.asarray(states)

    return go.Scatter3d(
        x=states[:, x_idx],
        y=states[:, y_idx],
        z=states[:, z_idx],
        mode=mode,
        line=dict(color=color, width=line_width),
        marker=dict(size=marker_size, color=color),
        name=name or "Trajectory",
        showlegend=showlegend,
    )


def plot_measurements_2d(
    measurements: ArrayLike,
    x_idx: int = 0,
    y_idx: int = 1,
    color: str = "black",
    symbol: str = "x",
    size: int = 6,
    name: Optional[str] = None,
    showlegend: bool = True,
) -> "go.Scatter":
    """
    Create a Plotly trace for 2D measurements.

    Parameters
    ----------
    measurements : array_like
        Measurements of shape (n_meas, n_dims).
    x_idx : int, optional
        Index of x coordinate. Default is 0.
    y_idx : int, optional
        Index of y coordinate. Default is 1.
    color : str, optional
        Color for markers. Default is "black".
    symbol : str, optional
        Marker symbol. Default is "x".
    size : int, optional
        Marker size. Default is 6.
    name : str, optional
        Name for the trace.
    showlegend : bool, optional
        Whether to show in legend. Default is True.

    Returns
    -------
    trace : go.Scatter
        Plotly scatter trace.
    """
    if not HAS_PLOTLY:
        raise ImportError("plotly is required for plotting functions")

    measurements = np.asarray(measurements)

    return go.Scatter(
        x=measurements[:, x_idx],
        y=measurements[:, y_idx],
        mode="markers",
        marker=dict(color=color, size=size, symbol=symbol),
        name=name or "Measurements",
        showlegend=showlegend,
    )


def plot_tracking_result(
    true_states: Optional[ArrayLike] = None,
    estimates: Optional[ArrayLike] = None,
    measurements: Optional[ArrayLike] = None,
    covariances: Optional[List[ArrayLike]] = None,
    x_idx: int = 0,
    y_idx: int = 2,
    cov_xy_idx: tuple[int, int] = (0, 2),
    ellipse_interval: int = 5,
    n_std: float = 2.0,
    title: str = "Tracking Result",
) -> "go.Figure":
    """
    Create a comprehensive tracking result visualization.

    Parameters
    ----------
    true_states : array_like, optional
        True state trajectory of shape (n_steps, n_dims).
    estimates : array_like, optional
        Estimated states of shape (n_steps, n_dims).
    measurements : array_like, optional
        Measurements of shape (n_steps, n_meas_dims).
    covariances : list of array_like, optional
        List of covariance matrices.
    x_idx : int, optional
        Index of x position in state. Default is 0.
    y_idx : int, optional
        Index of y position in state. Default is 2.
    cov_xy_idx : tuple, optional
        Indices for extracting position covariance. Default is (0, 2).
    ellipse_interval : int, optional
        Show ellipse every N steps. Default is 5.
    n_std : float, optional
        Number of standard deviations for ellipses. Default is 2.0.
    title : str, optional
        Figure title.

    Returns
    -------
    fig : go.Figure
        Plotly figure with the tracking result.
    """
    if not HAS_PLOTLY:
        raise ImportError("plotly is required for plotting functions")

    fig = go.Figure()

    # True trajectory
    if true_states is not None:
        true_states = np.asarray(true_states)
        fig.add_trace(
            go.Scatter(
                x=true_states[:, x_idx],
                y=true_states[:, y_idx],
                mode="lines",
                line=dict(color="green", width=3),
                name="True trajectory",
            )
        )

    # Measurements
    if measurements is not None:
        measurements = np.asarray(measurements)
        fig.add_trace(
            go.Scatter(
                x=measurements[:, 0],
                y=measurements[:, 1],
                mode="markers",
                marker=dict(color="black", size=6, symbol="x"),
                name="Measurements",
            )
        )

    # Estimates
    if estimates is not None:
        estimates = np.asarray(estimates)
        fig.add_trace(
            go.Scatter(
                x=estimates[:, x_idx],
                y=estimates[:, y_idx],
                mode="lines+markers",
                line=dict(color="blue", width=2),
                marker=dict(size=4),
                name="Estimate",
            )
        )

    # Covariance ellipses
    if estimates is not None and covariances is not None:
        i_cov, j_cov = cov_xy_idx
        first_ellipse = True
        for i in range(0, len(estimates), ellipse_interval):
            if i < len(covariances):
                pos_mean = np.array([estimates[i, x_idx], estimates[i, y_idx]])
                pos_cov = np.array(
                    [
                        [covariances[i][i_cov, i_cov], covariances[i][i_cov, j_cov]],
                        [covariances[i][j_cov, i_cov], covariances[i][j_cov, j_cov]],
                    ]
                )

                ex, ey = covariance_ellipse_points(pos_mean, pos_cov, n_std=n_std)
                fig.add_trace(
                    go.Scatter(
                        x=ex,
                        y=ey,
                        mode="lines",
                        line=dict(color="rgba(0, 100, 255, 0.3)", width=1),
                        fill="toself",
                        fillcolor="rgba(0, 100, 255, 0.1)",
                        name=f"{n_std}σ covariance" if first_ellipse else None,
                        showlegend=first_ellipse,
                    )
                )
                first_ellipse = False

    fig.update_layout(
        title=title,
        xaxis_title="X Position",
        yaxis_title="Y Position",
        xaxis=dict(scaleanchor="y", scaleratio=1),
        width=1000,
        height=800,
    )

    return fig


def plot_multi_target_tracks(
    tracks: Dict[Any, ArrayLike],
    x_idx: int = 0,
    y_idx: int = 1,
    colors: Optional[Dict[Any, str]] = None,
    title: str = "Multi-Target Tracks",
    show_ids: bool = True,
) -> "go.Figure":
    """
    Plot multiple target tracks with different colors.

    Parameters
    ----------
    tracks : dict
        Dictionary mapping track ID to state trajectory.
    x_idx : int, optional
        Index of x position in state. Default is 0.
    y_idx : int, optional
        Index of y position in state. Default is 1.
    colors : dict, optional
        Dictionary mapping track ID to color.
    title : str, optional
        Figure title.
    show_ids : bool, optional
        Whether to show track IDs at endpoints. Default is True.

    Returns
    -------
    fig : go.Figure
        Plotly figure.
    """
    if not HAS_PLOTLY:
        raise ImportError("plotly is required for plotting functions")

    fig = go.Figure()

    default_colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    for idx, (track_id, states) in enumerate(tracks.items()):
        states = np.asarray(states)
        color = (
            colors.get(track_id)
            if colors
            else default_colors[idx % len(default_colors)]
        )

        fig.add_trace(
            go.Scatter(
                x=states[:, x_idx],
                y=states[:, y_idx],
                mode="lines+markers",
                line=dict(color=color, width=2),
                marker=dict(size=4),
                name=f"Track {track_id}",
            )
        )

        # Add track ID at the end
        if show_ids and len(states) > 0:
            fig.add_trace(
                go.Scatter(
                    x=[states[-1, x_idx]],
                    y=[states[-1, y_idx]],
                    mode="text",
                    text=[str(track_id)],
                    textposition="top right",
                    textfont=dict(size=12, color=color),
                    showlegend=False,
                )
            )

    fig.update_layout(
        title=title,
        xaxis_title="X Position",
        yaxis_title="Y Position",
        xaxis=dict(scaleanchor="y", scaleratio=1),
    )

    return fig


def plot_state_time_series(
    states: ArrayLike,
    time: Optional[ArrayLike] = None,
    state_names: Optional[List[str]] = None,
    title: str = "State Time Series",
    ncols: int = 2,
) -> "go.Figure":
    """
    Plot state components as time series in subplots.

    Parameters
    ----------
    states : array_like
        State trajectory of shape (n_steps, n_dims).
    time : array_like, optional
        Time vector. If None, uses step indices.
    state_names : list of str, optional
        Names for each state component.
    title : str, optional
        Figure title.
    ncols : int, optional
        Number of columns in subplot grid. Default is 2.

    Returns
    -------
    fig : go.Figure
        Plotly figure with subplots.
    """
    if not HAS_PLOTLY:
        raise ImportError("plotly is required for plotting functions")

    states = np.asarray(states)
    n_steps, n_dims = states.shape

    if time is None:
        time = np.arange(n_steps)

    if state_names is None:
        state_names = [f"State {i}" for i in range(n_dims)]

    nrows = int(np.ceil(n_dims / ncols))

    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=state_names)

    for i in range(n_dims):
        row = i // ncols + 1
        col = i % ncols + 1

        fig.add_trace(
            go.Scatter(
                x=time,
                y=states[:, i],
                mode="lines",
                name=state_names[i],
                showlegend=False,
            ),
            row=row,
            col=col,
        )

    fig.update_layout(title=title, height=300 * nrows)

    return fig


def plot_estimation_comparison(
    true_states: ArrayLike,
    estimates: ArrayLike,
    covariances: Optional[List[ArrayLike]] = None,
    time: Optional[ArrayLike] = None,
    state_indices: Optional[List[int]] = None,
    state_names: Optional[List[str]] = None,
    n_std: float = 2.0,
    title: str = "Estimation Comparison",
) -> "go.Figure":
    """
    Plot true states vs estimates with error bounds.

    Parameters
    ----------
    true_states : array_like
        True state trajectory of shape (n_steps, n_dims).
    estimates : array_like
        Estimated states of shape (n_steps, n_dims).
    covariances : list of array_like, optional
        List of covariance matrices for error bounds.
    time : array_like, optional
        Time vector.
    state_indices : list of int, optional
        Indices of states to plot. Default plots all.
    state_names : list of str, optional
        Names for each state component.
    n_std : float, optional
        Number of standard deviations for bounds. Default is 2.0.
    title : str, optional
        Figure title.

    Returns
    -------
    fig : go.Figure
        Plotly figure.
    """
    if not HAS_PLOTLY:
        raise ImportError("plotly is required for plotting functions")

    true_states = np.asarray(true_states)
    estimates = np.asarray(estimates)
    n_steps, n_dims = true_states.shape

    if time is None:
        time = np.arange(n_steps)

    if state_indices is None:
        state_indices = list(range(n_dims))

    if state_names is None:
        state_names = [f"State {i}" for i in state_indices]

    n_plots = len(state_indices)
    fig = make_subplots(rows=n_plots, cols=1, subplot_titles=state_names)

    for plot_idx, state_idx in enumerate(state_indices):
        row = plot_idx + 1

        # Error bounds
        if covariances is not None:
            sigma = n_std * np.array(
                [np.sqrt(P[state_idx, state_idx]) for P in covariances]
            )
            upper = estimates[:, state_idx] + sigma
            lower = estimates[:, state_idx] - sigma

            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([time, time[::-1]]),
                    y=np.concatenate([upper, lower[::-1]]),
                    fill="toself",
                    fillcolor="rgba(0, 100, 255, 0.2)",
                    line=dict(color="rgba(0,0,0,0)"),
                    name=f"±{n_std}σ" if plot_idx == 0 else None,
                    showlegend=(plot_idx == 0),
                ),
                row=row,
                col=1,
            )

        # True state
        fig.add_trace(
            go.Scatter(
                x=time,
                y=true_states[:, state_idx],
                mode="lines",
                line=dict(color="green", width=2),
                name="True" if plot_idx == 0 else None,
                showlegend=(plot_idx == 0),
            ),
            row=row,
            col=1,
        )

        # Estimate
        fig.add_trace(
            go.Scatter(
                x=time,
                y=estimates[:, state_idx],
                mode="lines",
                line=dict(color="blue", width=2, dash="dash"),
                name="Estimate" if plot_idx == 0 else None,
                showlegend=(plot_idx == 0),
            ),
            row=row,
            col=1,
        )

    fig.update_layout(title=title, height=250 * n_plots)

    return fig


def create_animated_tracking(
    true_states: ArrayLike,
    estimates: ArrayLike,
    measurements: ArrayLike,
    covariances: Optional[List[ArrayLike]] = None,
    x_idx: int = 0,
    y_idx: int = 2,
    cov_xy_idx: tuple[int, int] = (0, 2),
    n_std: float = 2.0,
    frame_duration: int = 100,
    title: str = "Animated Tracking",
) -> "go.Figure":
    """
    Create an animated tracking visualization.

    Parameters
    ----------
    true_states : array_like
        True state trajectory of shape (n_steps, n_dims).
    estimates : array_like
        Estimated states of shape (n_steps, n_dims).
    measurements : array_like
        Measurements of shape (n_steps, n_meas_dims).
    covariances : list of array_like, optional
        List of covariance matrices.
    x_idx : int, optional
        Index of x position in state. Default is 0.
    y_idx : int, optional
        Index of y position in state. Default is 2.
    cov_xy_idx : tuple, optional
        Indices for position covariance. Default is (0, 2).
    n_std : float, optional
        Standard deviations for ellipse. Default is 2.0.
    frame_duration : int, optional
        Duration per frame in ms. Default is 100.
    title : str, optional
        Figure title.

    Returns
    -------
    fig : go.Figure
        Plotly figure with animation.
    """
    if not HAS_PLOTLY:
        raise ImportError("plotly is required for plotting functions")

    true_states = np.asarray(true_states)
    estimates = np.asarray(estimates)
    measurements = np.asarray(measurements)
    n_steps = len(measurements)

    frames = []
    i_cov, j_cov = cov_xy_idx

    for k in range(1, n_steps + 1):
        frame_data = [
            # True trajectory
            go.Scatter(
                x=true_states[: k + 1, x_idx],
                y=true_states[: k + 1, y_idx],
                mode="lines",
                line=dict(color="green", width=3),
                name="True",
            ),
            # Measurements
            go.Scatter(
                x=measurements[:k, 0],
                y=measurements[:k, 1],
                mode="markers",
                marker=dict(color="black", size=6, symbol="x"),
                name="Measurements",
            ),
            # Estimates
            go.Scatter(
                x=estimates[: k + 1, x_idx],
                y=estimates[: k + 1, y_idx],
                mode="lines+markers",
                line=dict(color="blue", width=2),
                marker=dict(size=4),
                name="Estimate",
            ),
        ]

        # Add covariance ellipse
        if covariances is not None and k < len(covariances):
            pos_mean = np.array([estimates[k, x_idx], estimates[k, y_idx]])
            pos_cov = np.array(
                [
                    [covariances[k][i_cov, i_cov], covariances[k][i_cov, j_cov]],
                    [covariances[k][j_cov, i_cov], covariances[k][j_cov, j_cov]],
                ]
            )
            ex, ey = covariance_ellipse_points(pos_mean, pos_cov, n_std=n_std)
            frame_data.append(
                go.Scatter(
                    x=ex,
                    y=ey,
                    mode="lines",
                    line=dict(color="rgba(0, 100, 255, 0.5)", width=2),
                    fill="toself",
                    fillcolor="rgba(0, 100, 255, 0.2)",
                    name=f"{n_std}σ covariance",
                )
            )

        frames.append(go.Frame(data=frame_data, name=str(k)))

    # Initial figure
    fig = go.Figure(data=frames[0].data, frames=frames)

    # Compute axis ranges
    x_min = min(true_states[:, x_idx].min(), estimates[:, x_idx].min())
    x_max = max(true_states[:, x_idx].max(), estimates[:, x_idx].max())
    y_min = min(true_states[:, y_idx].min(), estimates[:, y_idx].min())
    y_max = max(true_states[:, y_idx].max(), estimates[:, y_idx].max())
    margin = 10

    fig.update_layout(
        title=title,
        xaxis=dict(
            range=[x_min - margin, x_max + margin],
            title="X Position",
            scaleanchor="y",
            scaleratio=1,
        ),
        yaxis=dict(range=[y_min - margin, y_max + margin], title="Y Position"),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                y=1.15,
                x=0.5,
                xanchor="center",
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=frame_duration, redraw=True),
                                fromcurrent=True,
                                mode="immediate",
                            ),
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            dict(
                                frame=dict(duration=0, redraw=False), mode="immediate"
                            ),
                        ],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                active=0,
                steps=[
                    dict(
                        args=[
                            [str(k)],
                            dict(frame=dict(duration=0, redraw=True), mode="immediate"),
                        ],
                        label=str(k),
                        method="animate",
                    )
                    for k in range(1, n_steps + 1)
                ],
                x=0.1,
                len=0.8,
                xanchor="left",
                y=0,
                yanchor="top",
                currentvalue=dict(prefix="Step: ", visible=True, xanchor="center"),
            )
        ],
        width=1000,
        height=800,
    )

    return fig


__all__ = [
    "plot_trajectory_2d",
    "plot_trajectory_3d",
    "plot_measurements_2d",
    "plot_tracking_result",
    "plot_multi_target_tracks",
    "plot_state_time_series",
    "plot_estimation_comparison",
    "create_animated_tracking",
]
