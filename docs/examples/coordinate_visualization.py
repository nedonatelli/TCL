"""
Interactive 3D Coordinate System Visualization.

This example demonstrates:
1. 3D visualization of coordinate transformations
2. Interactive rotation matrix visualization
3. Quaternion SLERP animation
4. Geodetic to ECEF coordinate plotting

Run with: python examples/coordinate_visualization.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

from pytcl.coordinate_systems import (  # noqa: E402
    euler2rotmat,
    quat2rotmat,
    rotx,
    roty,
    rotz,
    slerp,
    sphere2cart,
)
from pytcl.navigation import geodetic_to_ecef  # noqa: E402


def plot_rotation_axes() -> go.Figure:
    """
    Visualize how rotation matrices transform coordinate axes.

    Creates an interactive 3D plot showing:
    - Original coordinate axes (XYZ)
    - Rotated axes after applying rotx, roty, rotz
    """
    fig = make_subplots(
        rows=1,
        cols=3,
        specs=[[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}]],
        subplot_titles=[
            "Rotation about X (45°)",
            "Rotation about Y (45°)",
            "Rotation about Z (45°)",
        ],
    )

    # Original axes
    _origin = np.array([0, 0, 0])  # noqa: F841
    x_axis = np.array([1, 0, 0])
    y_axis = np.array([0, 1, 0])
    z_axis = np.array([0, 0, 1])

    def add_axes(fig, R, col, original_alpha=0.3):
        """Add original and rotated axes to subplot."""
        # Original axes (faded)
        for axis, color, name in [
            (x_axis, "red", "X"),
            (y_axis, "green", "Y"),
            (z_axis, "blue", "Z"),
        ]:
            fig.add_trace(
                go.Scatter3d(
                    x=[0, axis[0]],
                    y=[0, axis[1]],
                    z=[0, axis[2]],
                    mode="lines",
                    line=dict(color=color, width=3),
                    opacity=original_alpha,
                    name=f"Original {name}",
                    showlegend=(col == 1),
                ),
                row=1,
                col=col,
            )

        # Rotated axes
        for axis, color, name in [
            (x_axis, "red", "X'"),
            (y_axis, "green", "Y'"),
            (z_axis, "blue", "Z'"),
        ]:
            rotated = R @ axis
            fig.add_trace(
                go.Scatter3d(
                    x=[0, rotated[0]],
                    y=[0, rotated[1]],
                    z=[0, rotated[2]],
                    mode="lines+markers",
                    line=dict(color=color, width=5),
                    marker=dict(size=4),
                    name=f"Rotated {name}",
                    showlegend=(col == 1),
                ),
                row=1,
                col=col,
            )

    # Apply rotations
    angle = np.pi / 4  # 45 degrees
    add_axes(fig, rotx(angle), col=1)
    add_axes(fig, roty(angle), col=2)
    add_axes(fig, rotz(angle), col=3)

    # Update layout for each scene
    for i in range(1, 4):
        fig.update_scenes(
            dict(
                xaxis=dict(range=[-1.5, 1.5], title="X"),
                yaxis=dict(range=[-1.5, 1.5], title="Y"),
                zaxis=dict(range=[-1.5, 1.5], title="Z"),
                aspectmode="cube",
            ),
            row=1,
            col=i,
        )

    fig.update_layout(
        title="Rotation Matrix Visualization (45° rotations)",
        width=1400,
        height=500,
    )

    return fig


def plot_euler_rotation_sequence() -> go.Figure:
    """
    Visualize Euler angle rotation sequence (ZYX - aerospace convention).

    Shows how yaw, pitch, roll are applied sequentially.
    """
    fig = make_subplots(
        rows=1,
        cols=4,
        specs=[[{"type": "scene"}] * 4],
        subplot_titles=[
            "Initial",
            "After Yaw (Z)",
            "After Pitch (Y)",
            "After Roll (X)",
        ],
    )

    # Define Euler angles
    yaw = np.radians(30)
    pitch = np.radians(20)
    roll = np.radians(15)

    # Rotation matrices at each stage
    R0 = np.eye(3)
    R1 = rotz(yaw)
    R2 = R1 @ roty(pitch)
    R3 = R2 @ rotx(roll)

    # Also compute full rotation for comparison
    R_full = euler2rotmat([yaw, pitch, roll], "ZYX")
    assert np.allclose(R3, R_full), "Rotation matrices should match"

    # Create a simple airplane shape
    def airplane_points():
        """Generate points representing an airplane."""
        # Fuselage
        fuselage = np.array(
            [
                [1, 0, 0],  # nose
                [-1, 0, 0],  # tail
            ]
        )
        # Wings
        wings = np.array(
            [
                [0, 0.8, 0],  # left wing
                [0, -0.8, 0],  # right wing
            ]
        )
        # Tail fin
        tail = np.array(
            [
                [-0.8, 0, 0],
                [-1, 0, 0.3],
            ]
        )
        return fuselage, wings, tail

    def add_airplane(fig, R, col):
        """Add rotated airplane to subplot."""
        fuselage, wings, tail = airplane_points()

        # Rotate all points
        fuselage_rot = (R @ fuselage.T).T
        wings_rot = (R @ wings.T).T
        tail_rot = (R @ tail.T).T

        # Fuselage
        fig.add_trace(
            go.Scatter3d(
                x=fuselage_rot[:, 0],
                y=fuselage_rot[:, 1],
                z=fuselage_rot[:, 2],
                mode="lines",
                line=dict(color="blue", width=8),
                name="Fuselage",
                showlegend=(col == 1),
            ),
            row=1,
            col=col,
        )

        # Wings
        fig.add_trace(
            go.Scatter3d(
                x=wings_rot[:, 0],
                y=wings_rot[:, 1],
                z=wings_rot[:, 2],
                mode="lines",
                line=dict(color="red", width=6),
                name="Wings",
                showlegend=(col == 1),
            ),
            row=1,
            col=col,
        )

        # Tail
        fig.add_trace(
            go.Scatter3d(
                x=tail_rot[:, 0],
                y=tail_rot[:, 1],
                z=tail_rot[:, 2],
                mode="lines",
                line=dict(color="green", width=4),
                name="Tail",
                showlegend=(col == 1),
            ),
            row=1,
            col=col,
        )

    # Add airplane at each rotation stage
    for i, R in enumerate([R0, R1, R2, R3], 1):
        add_airplane(fig, R, col=i)

    # Update layout
    for i in range(1, 5):
        fig.update_scenes(
            dict(
                xaxis=dict(range=[-1.5, 1.5], title="X"),
                yaxis=dict(range=[-1.5, 1.5], title="Y"),
                zaxis=dict(range=[-1.5, 1.5], title="Z"),
                aspectmode="cube",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
            ),
            row=1,
            col=i,
        )

    fig.update_layout(
        title=f"Euler Rotation Sequence (ZYX): Yaw={np.degrees(yaw):.0f}°, Pitch={np.degrees(pitch):.0f}°, Roll={np.degrees(roll):.0f}°",
        width=1600,
        height=500,
    )

    return fig


def plot_quaternion_slerp() -> go.Figure:
    """
    Visualize quaternion SLERP interpolation.

    Shows smooth interpolation between two orientations.
    """
    fig = go.Figure()

    # Start and end quaternions
    # Identity (no rotation)
    q1 = np.array([1, 0, 0, 0])

    # 90 degree rotation about Z axis
    angle = np.pi / 2
    q2 = np.array([np.cos(angle / 2), 0, 0, np.sin(angle / 2)])

    # Interpolation steps
    n_steps = 20
    t_values = np.linspace(0, 1, n_steps)

    # Colors for interpolation
    colors = [f"rgb({int(255 * (1 - t))}, {int(100 + 155 * t)}, {int(255 * t)})" for t in t_values]

    # Reference vector to rotate
    v = np.array([1, 0, 0])

    # Add interpolated vectors
    for i, t in enumerate(t_values):
        q_interp = slerp(q1, q2, t)
        R = quat2rotmat(q_interp)
        v_rot = R @ v

        fig.add_trace(
            go.Scatter3d(
                x=[0, v_rot[0]],
                y=[0, v_rot[1]],
                z=[0, v_rot[2]],
                mode="lines+markers",
                line=dict(color=colors[i], width=4),
                marker=dict(size=4, color=colors[i]),
                name=f"t={t:.2f}",
                showlegend=(i % 5 == 0),
            )
        )

    # Add arc showing the path
    arc_points = []
    for t in np.linspace(0, 1, 100):
        q_interp = slerp(q1, q2, t)
        R = quat2rotmat(q_interp)
        arc_points.append(R @ v)
    arc_points = np.array(arc_points)

    fig.add_trace(
        go.Scatter3d(
            x=arc_points[:, 0],
            y=arc_points[:, 1],
            z=arc_points[:, 2],
            mode="lines",
            line=dict(color="gray", width=2, dash="dash"),
            name="SLERP path",
        )
    )

    fig.update_layout(
        title="Quaternion SLERP Interpolation (0° to 90° about Z-axis)",
        scene=dict(
            xaxis=dict(range=[-1.5, 1.5], title="X"),
            yaxis=dict(range=[-1.5, 1.5], title="Y"),
            zaxis=dict(range=[-1.5, 1.5], title="Z"),
            aspectmode="cube",
        ),
        width=800,
        height=700,
    )

    return fig


def plot_spherical_coordinates() -> go.Figure:
    """
    Visualize spherical coordinate system.

    Shows the relationship between Cartesian and spherical coordinates.
    """
    fig = go.Figure()

    # Generate a grid of points on a sphere
    n_az = 24
    n_el = 12
    r = 1.0

    # Azimuth lines (constant azimuth, varying elevation)
    for az in np.linspace(0, 2 * np.pi, n_az, endpoint=False):
        el_range = np.linspace(-np.pi / 2, np.pi / 2, 50)
        points = np.array([sphere2cart(r, az, el, system_type="az-el") for el in el_range])
        fig.add_trace(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="lines",
                line=dict(color="lightblue", width=1),
                showlegend=False,
            )
        )

    # Elevation lines (constant elevation, varying azimuth)
    for el in np.linspace(-np.pi / 2 + 0.1, np.pi / 2 - 0.1, n_el):
        az_range = np.linspace(0, 2 * np.pi, 50)
        points = np.array([sphere2cart(r, az, el, system_type="az-el") for az in az_range])
        fig.add_trace(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="lines",
                line=dict(color="lightgreen", width=1),
                showlegend=False,
            )
        )

    # Highlight a specific point
    test_az = np.radians(45)
    test_el = np.radians(30)
    test_point = sphere2cart(r, test_az, test_el, system_type="az-el")

    # Add the point
    fig.add_trace(
        go.Scatter3d(
            x=[test_point[0]],
            y=[test_point[1]],
            z=[test_point[2]],
            mode="markers",
            marker=dict(size=10, color="red"),
            name=f"Point (az={np.degrees(test_az):.0f}°, el={np.degrees(test_el):.0f}°)",
        )
    )

    # Add lines showing the coordinates
    # Line from origin to projection on xy-plane
    proj_xy = np.array([test_point[0], test_point[1], 0])
    fig.add_trace(
        go.Scatter3d(
            x=[0, proj_xy[0]],
            y=[0, proj_xy[1]],
            z=[0, 0],
            mode="lines",
            line=dict(color="blue", width=3, dash="dash"),
            name="XY projection",
        )
    )

    # Line from projection to point (showing elevation)
    fig.add_trace(
        go.Scatter3d(
            x=[proj_xy[0], test_point[0]],
            y=[proj_xy[1], test_point[1]],
            z=[0, test_point[2]],
            mode="lines",
            line=dict(color="green", width=3, dash="dash"),
            name="Elevation",
        )
    )

    # Line from origin to point (range)
    fig.add_trace(
        go.Scatter3d(
            x=[0, test_point[0]],
            y=[0, test_point[1]],
            z=[0, test_point[2]],
            mode="lines",
            line=dict(color="red", width=3),
            name="Range vector",
        )
    )

    # Add coordinate axes
    axis_len = 1.3
    for axis, color, name in [
        ([axis_len, 0, 0], "red", "X"),
        ([0, axis_len, 0], "green", "Y"),
        ([0, 0, axis_len], "blue", "Z"),
    ]:
        fig.add_trace(
            go.Scatter3d(
                x=[0, axis[0]],
                y=[0, axis[1]],
                z=[0, axis[2]],
                mode="lines+text",
                line=dict(color=color, width=4),
                text=["", name],
                textposition="top center",
                name=f"{name}-axis",
                showlegend=False,
            )
        )

    fig.update_layout(
        title="Spherical Coordinate System (az-el convention)",
        scene=dict(
            xaxis=dict(range=[-1.5, 1.5], title="X"),
            yaxis=dict(range=[-1.5, 1.5], title="Y"),
            zaxis=dict(range=[-1.5, 1.5], title="Z"),
            aspectmode="cube",
        ),
        width=900,
        height=800,
    )

    return fig


def plot_earth_coordinates() -> go.Figure:
    """
    Visualize geodetic coordinates on Earth.

    Shows major cities and their ECEF coordinates.
    """
    fig = go.Figure()

    # Create a sphere representing Earth
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 50)
    R_earth = 6371000  # meters (approximate)
    scale = 1e-6  # Scale to make numbers manageable

    x = R_earth * scale * np.outer(np.cos(u), np.sin(v))
    y = R_earth * scale * np.outer(np.sin(u), np.sin(v))
    z = R_earth * scale * np.outer(np.ones(np.size(u)), np.cos(v))

    fig.add_trace(
        go.Surface(
            x=x,
            y=y,
            z=z,
            colorscale=[[0, "lightblue"], [1, "lightblue"]],
            opacity=0.6,
            showscale=False,
            name="Earth",
        )
    )

    # Major cities with their geodetic coordinates
    cities = {
        "New York": (40.7128, -74.0060),
        "London": (51.5074, -0.1278),
        "Tokyo": (35.6762, 139.6503),
        "Sydney": (-33.8688, 151.2093),
        "São Paulo": (-23.5505, -46.6333),
        "Cairo": (30.0444, 31.2357),
    }

    # Convert to ECEF and plot
    city_x, city_y, city_z = [], [], []
    city_names = []

    for name, (lat_deg, lon_deg) in cities.items():
        lat = np.radians(lat_deg)
        lon = np.radians(lon_deg)
        ecef = geodetic_to_ecef(lat, lon, 0)
        city_x.append(ecef[0] * scale)
        city_y.append(ecef[1] * scale)
        city_z.append(ecef[2] * scale)
        city_names.append(name)

    fig.add_trace(
        go.Scatter3d(
            x=city_x,
            y=city_y,
            z=city_z,
            mode="markers+text",
            marker=dict(size=8, color="red"),
            text=city_names,
            textposition="top center",
            name="Cities",
        )
    )

    # Add equator
    eq_lon = np.linspace(0, 2 * np.pi, 100)
    eq_ecef = np.array([geodetic_to_ecef(0, lon, 0) for lon in eq_lon])
    fig.add_trace(
        go.Scatter3d(
            x=eq_ecef[:, 0] * scale,
            y=eq_ecef[:, 1] * scale,
            z=eq_ecef[:, 2] * scale,
            mode="lines",
            line=dict(color="yellow", width=3),
            name="Equator",
        )
    )

    # Add prime meridian
    pm_lat = np.linspace(-np.pi / 2, np.pi / 2, 100)
    pm_ecef = np.array([geodetic_to_ecef(lat, 0, 0) for lat in pm_lat])
    fig.add_trace(
        go.Scatter3d(
            x=pm_ecef[:, 0] * scale,
            y=pm_ecef[:, 1] * scale,
            z=pm_ecef[:, 2] * scale,
            mode="lines",
            line=dict(color="orange", width=3),
            name="Prime Meridian",
        )
    )

    fig.update_layout(
        title="Earth: Geodetic to ECEF Coordinate Conversion",
        scene=dict(
            xaxis=dict(title="X (1000 km)"),
            yaxis=dict(title="Y (1000 km)"),
            zaxis=dict(title="Z (1000 km)"),
            aspectmode="data",
        ),
        width=900,
        height=800,
    )

    return fig


def main():
    """Run coordinate visualization examples."""
    print("Coordinate System Visualization Examples")
    print("=" * 50)

    # 1. Rotation axes
    print("\n1. Generating rotation axes visualization...")
    fig1 = plot_rotation_axes()
    fig1.write_html("coord_viz_rotation_axes.html")
    print("   Saved to coord_viz_rotation_axes.html")

    # 2. Euler rotation sequence
    print("\n2. Generating Euler rotation sequence...")
    fig2 = plot_euler_rotation_sequence()
    fig2.write_html("coord_viz_euler_sequence.html")
    print("   Saved to coord_viz_euler_sequence.html")

    # 3. Quaternion SLERP
    print("\n3. Generating quaternion SLERP visualization...")
    fig3 = plot_quaternion_slerp()
    fig3.write_html("coord_viz_slerp.html")
    print("   Saved to coord_viz_slerp.html")

    # 4. Spherical coordinates
    print("\n4. Generating spherical coordinates visualization...")
    fig4 = plot_spherical_coordinates()
    fig4.write_html("coord_viz_spherical.html")
    print("   Saved to coord_viz_spherical.html")

    # 5. Earth coordinates
    print("\n5. Generating Earth coordinates visualization...")
    fig5 = plot_earth_coordinates()
    fig5.write_html("coord_viz_earth.html")
    print("   Saved to coord_viz_earth.html")

    # Show all figures
    print("\nOpening visualizations in browser...")
    fig1.show()
    fig2.show()
    fig3.show()
    fig4.show()
    fig5.show()

    print("\nDone!")


if __name__ == "__main__":
    main()
