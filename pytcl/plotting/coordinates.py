"""
Coordinate system visualization utilities.

This module provides functions for visualizing coordinate systems,
rotations, and transformations in 2D and 3D.
"""

from typing import Any, List, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pytcl.core.optional_deps import is_available, requires

# Lazy flag for backward compatibility
HAS_PLOTLY = is_available("plotly")


@requires("plotly", extra="visualization")
def plot_coordinate_axes_3d(
    origin: ArrayLike = (0, 0, 0),
    rotation_matrix: Optional[ArrayLike] = None,
    scale: float = 1.0,
    colors: Tuple[str, str, str] = ("red", "green", "blue"),
    names: Tuple[str, str, str] = ("X", "Y", "Z"),
    line_width: int = 4,
    showlegend: bool = True,
    name_prefix: str = "",
) -> List[Any]:
    """
    Create Plotly traces for 3D coordinate axes.

    Parameters
    ----------
    origin : array_like, optional
        Origin point [x, y, z]. Default is (0, 0, 0).
    rotation_matrix : array_like, optional
        3x3 rotation matrix to apply to axes. Default is identity.
    scale : float, optional
        Length of axes. Default is 1.0.
    colors : tuple of str, optional
        Colors for X, Y, Z axes. Default is ("red", "green", "blue").
    names : tuple of str, optional
        Names for X, Y, Z axes. Default is ("X", "Y", "Z").
    line_width : int, optional
        Line width. Default is 4.
    showlegend : bool, optional
        Whether to show in legend. Default is True.
    name_prefix : str, optional
        Prefix for axis names in legend.

    Returns
    -------
    traces : list of go.Scatter3d
        List of three Plotly traces for the axes.
    """
    import plotly.graph_objects as go

    origin = np.asarray(origin, dtype=np.float64)
    if rotation_matrix is None:
        rotation_matrix = np.eye(3)
    else:
        rotation_matrix = np.asarray(rotation_matrix, dtype=np.float64)

    # Unit vectors
    axes = np.eye(3) * scale

    # Apply rotation
    rotated_axes = rotation_matrix @ axes

    traces = []
    for i, (color, name) in enumerate(zip(colors, names)):
        end_point = origin + rotated_axes[:, i]
        traces.append(
            go.Scatter3d(
                x=[origin[0], end_point[0]],
                y=[origin[1], end_point[1]],
                z=[origin[2], end_point[2]],
                mode="lines+markers",
                line=dict(color=color, width=line_width),
                marker=dict(size=3, color=color),
                name=f"{name_prefix}{name}" if name_prefix else name,
                showlegend=showlegend,
            )
        )

    return traces


@requires("plotly", extra="visualization")
def plot_rotation_comparison(
    R1: ArrayLike,
    R2: ArrayLike,
    labels: Tuple[str, str] = ("Original", "Rotated"),
    title: str = "Rotation Comparison",
) -> Any:
    """
    Compare two rotation matrices by visualizing their coordinate frames.

    Parameters
    ----------
    R1 : array_like
        First 3x3 rotation matrix.
    R2 : array_like
        Second 3x3 rotation matrix.
    labels : tuple of str, optional
        Labels for the two frames.
    title : str, optional
        Figure title.

    Returns
    -------
    fig : go.Figure
        Plotly figure.
    """
    import plotly.graph_objects as go

    fig = go.Figure()

    # First frame (faded)
    traces1 = plot_coordinate_axes_3d(
        rotation_matrix=R1,
        name_prefix=f"{labels[0]} ",
        showlegend=True,
    )
    for trace in traces1:
        trace.opacity = 0.4
        fig.add_trace(trace)

    # Second frame
    traces2 = plot_coordinate_axes_3d(
        rotation_matrix=R2,
        name_prefix=f"{labels[1]} ",
        showlegend=True,
    )
    for trace in traces2:
        fig.add_trace(trace)

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="cube",
        ),
    )

    return fig


@requires("plotly", extra="visualization")
def plot_euler_angles(
    angles: ArrayLike,
    sequence: str = "ZYX",
    title: Optional[str] = None,
) -> Any:
    """
    Visualize Euler angle rotations step by step.

    Parameters
    ----------
    angles : array_like
        Three Euler angles in radians.
    sequence : str, optional
        Euler angle sequence. Default is "ZYX".
    title : str, optional
        Figure title.

    Returns
    -------
    fig : go.Figure
        Plotly figure with subplots showing each rotation step.
    """
    from plotly.subplots import make_subplots

    angles = np.asarray(angles)

    # Create rotation matrices for each axis
    def rotx(a: Any) -> NDArray[np.float64]:
        return np.array(
            [[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]]
        )

    def roty(a: Any) -> NDArray[np.float64]:
        return np.array(
            [[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]
        )

    def rotz(a: Any) -> NDArray[np.float64]:
        return np.array(
            [[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]
        )

    rot_funcs = {"X": rotx, "Y": roty, "Z": rotz}

    # Compute cumulative rotations
    R_cumulative = [np.eye(3)]
    R_current = np.eye(3)
    for i, axis in enumerate(sequence):
        R_step = rot_funcs[axis](angles[i])
        R_current = R_current @ R_step
        R_cumulative.append(R_current.copy())

    # Create subplots
    fig = make_subplots(
        rows=1,
        cols=4,
        specs=[[{"type": "scene"}] * 4],
        subplot_titles=[
            "Initial",
            f"After {sequence[0]} rotation",
            f"After {sequence[0]}{sequence[1]} rotation",
            f"After {sequence} rotation",
        ],
    )

    for col, R in enumerate(R_cumulative):
        # Original axes (faded)
        for trace in plot_coordinate_axes_3d(
            rotation_matrix=np.eye(3),
            name_prefix="Original ",
            showlegend=(col == 0),
        ):
            trace.opacity = 0.3
            fig.add_trace(trace, row=1, col=col + 1)

        # Current axes
        for trace in plot_coordinate_axes_3d(
            rotation_matrix=R,
            name_prefix="Current ",
            showlegend=(col == 0),
        ):
            fig.add_trace(trace, row=1, col=col + 1)

    title_text = title or f"Euler Angle Rotation ({sequence})"
    fig.update_layout(
        title=title_text,
        width=1600,
        height=500,
    )

    # Update each scene
    for i in range(4):
        scene_name = f"scene{i + 1}" if i > 0 else "scene"
        fig.update_layout(
            **{
                scene_name: dict(
                    aspectmode="cube", camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                )
            }
        )

    return fig


@requires("plotly", extra="visualization")
def plot_quaternion_interpolation(
    q_start: ArrayLike,
    q_end: ArrayLike,
    n_steps: int = 10,
    title: str = "Quaternion SLERP Interpolation",
) -> Any:
    """
    Visualize quaternion interpolation (SLERP) between two orientations.

    Parameters
    ----------
    q_start : array_like
        Starting quaternion [w, x, y, z].
    q_end : array_like
        Ending quaternion [w, x, y, z].
    n_steps : int, optional
        Number of interpolation steps. Default is 10.
    title : str, optional
        Figure title.

    Returns
    -------
    fig : go.Figure
        Plotly figure with animation.
    """
    import plotly.graph_objects as go

    q_start = np.asarray(q_start)
    q_end = np.asarray(q_end)

    # Normalize quaternions
    q_start = q_start / np.linalg.norm(q_start)
    q_end = q_end / np.linalg.norm(q_end)

    # SLERP interpolation
    def quat_slerp(q1: Any, q2: Any, t: Any) -> NDArray[np.float64]:
        dot = np.dot(q1, q2)
        if dot < 0:
            q2 = -q2
            dot = -dot
        if dot > 0.9995:
            return q1 + t * (q2 - q1)
        theta = np.arccos(dot)
        return (np.sin((1 - t) * theta) * q1 + np.sin(t * theta) * q2) / np.sin(theta)

    def quat_to_rotmat(q: Any) -> NDArray[np.float64]:
        w, x, y, z = q
        return np.array(
            [
                [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)],
            ]
        )

    # Generate interpolated frames
    t_values = np.linspace(0, 1, n_steps)
    frames = []

    for i, t in enumerate(t_values):
        q_interp = quat_slerp(q_start, q_end, t)
        R = quat_to_rotmat(q_interp)

        frame_traces = []

        # Reference axes
        for trace in plot_coordinate_axes_3d(
            rotation_matrix=np.eye(3),
            name_prefix="Reference ",
            showlegend=(i == 0),
        ):
            trace.opacity = 0.3
            frame_traces.append(trace)

        # Interpolated axes
        for trace in plot_coordinate_axes_3d(
            rotation_matrix=R,
            name_prefix="Interpolated ",
            showlegend=(i == 0),
        ):
            frame_traces.append(trace)

        frames.append(go.Frame(data=frame_traces, name=str(i)))

    # Initial figure
    fig = go.Figure(data=frames[0].data, frames=frames)

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="cube",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
        ),
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
                                frame=dict(duration=200, redraw=True),
                                fromcurrent=True,
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
                            [str(i)],
                            dict(frame=dict(duration=0, redraw=True), mode="immediate"),
                        ],
                        label=f"{t:.1f}",
                        method="animate",
                    )
                    for i, t in enumerate(t_values)
                ],
                x=0.1,
                len=0.8,
                currentvalue=dict(prefix="t = ", visible=True),
            )
        ],
    )

    return fig


@requires("plotly", extra="visualization")
def plot_spherical_grid(
    r: float = 1.0,
    n_lat: int = 10,
    n_lon: int = 20,
    color: str = "lightblue",
    opacity: float = 0.5,
    title: str = "Spherical Coordinate Grid",
) -> Any:
    """
    Plot a spherical coordinate grid.

    Parameters
    ----------
    r : float, optional
        Radius of the sphere. Default is 1.0.
    n_lat : int, optional
        Number of latitude lines. Default is 10.
    n_lon : int, optional
        Number of longitude lines. Default is 20.
    color : str, optional
        Color for the grid. Default is "lightblue".
    opacity : float, optional
        Opacity of the surface. Default is 0.5.
    title : str, optional
        Figure title.

    Returns
    -------
    fig : go.Figure
        Plotly figure.
    """
    import plotly.graph_objects as go

    # Generate sphere surface
    theta = np.linspace(0, 2 * np.pi, n_lon)
    phi = np.linspace(0, np.pi, n_lat)
    theta, phi = np.meshgrid(theta, phi)

    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)

    fig = go.Figure()

    # Sphere surface
    fig.add_trace(
        go.Surface(
            x=x,
            y=y,
            z=z,
            colorscale=[[0, color], [1, color]],
            opacity=opacity,
            showscale=False,
            name="Sphere",
        )
    )

    # Add coordinate axes
    for trace in plot_coordinate_axes_3d(scale=r * 1.3, showlegend=True):
        fig.add_trace(trace)

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="cube",
        ),
    )

    return fig


@requires("plotly", extra="visualization")
def plot_points_spherical(
    points_spherical: ArrayLike,
    r_idx: int = 0,
    theta_idx: int = 1,
    phi_idx: int = 2,
    color: str = "red",
    size: int = 5,
    name: str = "Points",
    title: str = "Points in Spherical Coordinates",
) -> Any:
    """
    Plot points given in spherical coordinates.

    Parameters
    ----------
    points_spherical : array_like
        Points in spherical coordinates (r, theta, phi) of shape (n_points, 3).
    r_idx : int, optional
        Index of radial coordinate. Default is 0.
    theta_idx : int, optional
        Index of azimuthal angle (from x-axis in xy-plane). Default is 1.
    phi_idx : int, optional
        Index of polar angle (from z-axis). Default is 2.
    color : str, optional
        Color for the points. Default is "red".
    size : int, optional
        Marker size. Default is 5.
    name : str, optional
        Name for the trace.
    title : str, optional
        Figure title.

    Returns
    -------
    fig : go.Figure
        Plotly figure.
    """
    import plotly.graph_objects as go

    points = np.asarray(points_spherical)
    r = points[:, r_idx]
    theta = points[:, theta_idx]
    phi = points[:, phi_idx]

    # Convert to Cartesian
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)

    fig = go.Figure()

    # Points
    fig.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers",
            marker=dict(color=color, size=size),
            name=name,
        )
    )

    # Add coordinate axes
    max_r = np.max(r) if len(r) > 0 else 1.0
    for trace in plot_coordinate_axes_3d(scale=max_r * 1.1, showlegend=True):
        fig.add_trace(trace)

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="cube",
        ),
    )

    return fig


@requires("plotly", extra="visualization")
def plot_coordinate_transform(
    points_original: ArrayLike,
    points_transformed: ArrayLike,
    transform_name: str = "Transform",
    title: Optional[str] = None,
) -> Any:
    """
    Visualize a coordinate transformation between two point sets.

    Parameters
    ----------
    points_original : array_like
        Original points of shape (n_points, 3).
    points_transformed : array_like
        Transformed points of shape (n_points, 3).
    transform_name : str, optional
        Name of the transformation.
    title : str, optional
        Figure title.

    Returns
    -------
    fig : go.Figure
        Plotly figure with two subplots.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    points_original = np.asarray(points_original)
    points_transformed = np.asarray(points_transformed)

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=["Original", f"After {transform_name}"],
    )

    # Original points
    fig.add_trace(
        go.Scatter3d(
            x=points_original[:, 0],
            y=points_original[:, 1],
            z=points_original[:, 2],
            mode="markers",
            marker=dict(color="blue", size=4),
            name="Original",
        ),
        row=1,
        col=1,
    )

    # Transformed points
    fig.add_trace(
        go.Scatter3d(
            x=points_transformed[:, 0],
            y=points_transformed[:, 1],
            z=points_transformed[:, 2],
            mode="markers",
            marker=dict(color="red", size=4),
            name="Transformed",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        title=title or f"Coordinate Transformation: {transform_name}",
        width=1200,
        height=600,
    )

    return fig


__all__ = [
    "plot_coordinate_axes_3d",
    "plot_rotation_comparison",
    "plot_euler_angles",
    "plot_quaternion_interpolation",
    "plot_spherical_grid",
    "plot_points_spherical",
    "plot_coordinate_transform",
]
