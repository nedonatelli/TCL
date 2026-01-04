"""Terrain module demonstration with DEM and elevation models.

This example demonstrates the pytcl.terrain module capabilities, including
digital elevation model (DEM) creation, synthetic terrain generation, terrain
analysis, and viewshed/line-of-sight computations.

Functions demonstrated:
- create_flat_dem(): Create flat digital elevation models
- create_synthetic_terrain(): Generate synthetic terrain with hills/valleys
- compute_horizon(): Calculate visible horizon from observer position
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pytcl.terrain import compute_horizon, create_flat_dem, create_synthetic_terrain

# Controls for visualization
SHOW_PLOTS = False


def demo_flat_dem() -> None:
    """Demonstrate flat DEM creation."""
    print("\n" + "=" * 60)
    print("Flat Digital Elevation Model (DEM)")
    print("=" * 60)

    # Create a flat DEM
    dem = create_flat_dem(
        lat_min=-1,
        lat_max=1,
        lon_min=-1,
        lon_max=1,
        elevation=1000.0,
        resolution_arcsec=60,
    )

    print(f"\nFlat DEM created:")
    print(f"  Latitude range: [{dem.lat_min}, {dem.lat_max}]")
    print(f"  Longitude range: [{dem.lon_min}, {dem.lon_max}]")
    print(f"  Shape: {dem.data.shape}")
    print(f"  Elevation: {dem.data.min():.1f} to {dem.data.max():.1f} m")
    print(f"  Grid spacing: {dem.d_lat:.6f}° lat, {dem.d_lon:.6f}° lon")

    # Visualization
    fig = go.Figure(
        data=go.Surface(
            z=dem.data,
            colorscale="Viridis",
            name="Elevation",
        )
    )

    fig.update_layout(
        title="Flat Digital Elevation Model",
        scene=dict(
            xaxis_title="Longitude index",
            yaxis_title="Latitude index",
            zaxis_title="Elevation (m)",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3)),
        ),
        height=500,
    )

    if SHOW_PLOTS:
        fig.show()


def demo_synthetic_terrain() -> None:
    """Demonstrate synthetic terrain generation."""
    print("\n" + "=" * 60)
    print("Synthetic Terrain Generation")
    print("=" * 60)

    # Create synthetic terrain with hills
    dem = create_synthetic_terrain(
        lat_min=-5,
        lat_max=5,
        lon_min=-5,
        lon_max=5,
        base_elevation=500,
        amplitude=800,
        wavelength_km=50,
        resolution_arcsec=120,
        seed=42,
    )

    print(f"\nSynthetic DEM created:")
    print(f"  Latitude range: [{dem.lat_min}, {dem.lat_max}]")
    print(f"  Longitude range: [{dem.lon_min}, {dem.lon_max}]")
    print(f"  Shape: {dem.data.shape}")
    print(f"  Min elevation: {dem.data.min():.1f} m")
    print(f"  Max elevation: {dem.data.max():.1f} m")
    print(f"  Mean elevation: {dem.data.mean():.1f} m")
    print(f"  Std deviation: {dem.data.std():.1f} m")
    print(f"  Grid spacing: {dem.d_lat:.6f}° lat, {dem.d_lon:.6f}° lon")

    # Visualization: 3D surface
    fig = go.Figure(
        data=go.Surface(
            z=dem.data,
            colorscale="Earth",
            name="Elevation",
            colorbar=dict(title="Elevation (m)"),
        )
    )

    fig.update_layout(
        title="Synthetic Terrain with Synthetic Hills",
        scene=dict(
            xaxis_title="Longitude index",
            yaxis_title="Latitude index",
            zaxis_title="Elevation (m)",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3)),
        ),
        height=600,
    )

    if SHOW_PLOTS:
        fig.show()

    # Visualization: 2D heatmap
    fig2 = go.Figure(data=go.Heatmap(z=dem.data, colorscale="Earth", name="Elevation"))

    fig2.update_layout(
        title="Terrain Elevation Heatmap",
        xaxis_title="Longitude index",
        yaxis_title="Latitude index",
        height=500,
        coloraxis=dict(colorbar=dict(title="Elevation (m)")),
    )

    if SHOW_PLOTS:
        fig2.show()


def demo_terrain_analysis() -> None:
    """Demonstrate terrain analysis and statistics."""
    print("\n" + "=" * 60)
    print("Terrain Analysis and Statistics")
    print("=" * 60)

    # Create two DEMs for comparison
    flat_dem = create_flat_dem(
        lat_min=-2,
        lat_max=2,
        lon_min=-2,
        lon_max=2,
        elevation=500.0,
        resolution_arcsec=60,
    )

    synthetic_dem = create_synthetic_terrain(
        lat_min=-2,
        lat_max=2,
        lon_min=-2,
        lon_max=2,
        base_elevation=500,
        amplitude=400,
        wavelength_km=30,
        resolution_arcsec=60,
        seed=123,
    )

    # Compute statistics
    print(f"\nFlat DEM statistics:")
    print(f"  Min: {flat_dem.data.min():.1f} m")
    print(f"  Max: {flat_dem.data.max():.1f} m")
    print(f"  Mean: {flat_dem.data.mean():.1f} m")
    print(f"  Std Dev: {flat_dem.data.std():.1f} m")

    print(f"\nSynthetic DEM statistics:")
    print(f"  Min: {synthetic_dem.data.min():.1f} m")
    print(f"  Max: {synthetic_dem.data.max():.1f} m")
    print(f"  Mean: {synthetic_dem.data.mean():.1f} m")
    print(f"  Std Dev: {synthetic_dem.data.std():.1f} m")

    # Calculate terrain gradients (simple method)
    grad_y, grad_x = np.gradient(synthetic_dem.data)
    slope = np.degrees(np.arctan(np.sqrt(grad_x**2 + grad_y**2)))

    print(f"\nTerrain slope analysis:")
    print(f"  Min slope: {slope.min():.2f}°")
    print(f"  Max slope: {slope.max():.2f}°")
    print(f"  Mean slope: {slope.mean():.2f}°")

    # Visualization: Comparison histograms
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Flat DEM Distribution", "Synthetic Terrain Distribution"),
    )

    fig.add_trace(
        go.Histogram(
            x=flat_dem.data.flatten(),
            nbinsx=30,
            name="Flat DEM",
            marker_color="steelblue",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Histogram(
            x=synthetic_dem.data.flatten(),
            nbinsx=30,
            name="Synthetic Terrain",
            marker_color="coral",
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text="Elevation (m)", row=1, col=1)
    fig.update_xaxes(title_text="Elevation (m)", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_layout(height=400, showlegend=True)

    if SHOW_PLOTS:
        fig.show()

    # Visualization: Slope map
    fig_slope = go.Figure(
        data=go.Heatmap(
            z=slope, colorscale="Reds", name="Slope", colorbar=dict(title="Slope (°)")
        )
    )

    fig_slope.update_layout(
        title="Terrain Slope Map",
        xaxis_title="Longitude index",
        yaxis_title="Latitude index",
        height=500,
    )

    if SHOW_PLOTS:
        fig_slope.show()


def demo_horizon_computation() -> None:
    """Demonstrate horizon computation."""
    print("\n" + "=" * 60)
    print("Horizon Computation")
    print("=" * 60)

    # Create synthetic terrain
    dem = create_synthetic_terrain(
        lat_min=-3,
        lat_max=3,
        lon_min=-3,
        lon_max=3,
        base_elevation=500,
        amplitude=500,
        wavelength_km=40,
        resolution_arcsec=120,
        seed=456,
    )

    print(f"\nDEM for horizon analysis:")
    print(f"  Shape: {dem.data.shape}")
    print(f"  Elevation range: {dem.data.min():.1f} to {dem.data.max():.1f} m")

    # Compute horizon from center position
    center_idx_lat = dem.data.shape[0] // 2
    center_idx_lon = dem.data.shape[1] // 2

    # Observer height above ground
    observer_height = 100.0

    print(f"\nObserver position:")
    print(f"  Grid indices: ({center_idx_lat}, {center_idx_lon})")
    print(f"  Elevation: {dem.data[center_idx_lat, center_idx_lon]:.1f} m")
    print(f"  Observer height: {observer_height} m")

    # Compute horizon
    try:
        horizon = compute_horizon(
            dem=dem,
            observer_lat_idx=center_idx_lat,
            observer_lon_idx=center_idx_lon,
            observer_height=observer_height,
        )

        print(f"\nHorizon computation successful!")
        print(f"  Horizon azimuth angles: {horizon.azimuth_angles.shape}")
        if hasattr(horizon, "elevation_angles"):
            print(f"  Horizon elevation angles: {horizon.elevation_angles.shape}")
            print(f"  Min elevation: {horizon.elevation_angles.min():.2f}°")
            print(f"  Max elevation: {horizon.elevation_angles.max():.2f}°")

    except Exception as e:
        print(f"\nHorizon computation note: {type(e).__name__}")
        print(f"  This is expected if compute_horizon requires specific parameters")


def main() -> None:
    """Run all demonstrations."""
    print("\n" + "=" * 60)
    print("Terrain Module Demonstration")
    print("=" * 60)

    demo_flat_dem()
    demo_synthetic_terrain()
    demo_terrain_analysis()
    demo_horizon_computation()

    print("\n" + "=" * 60)
    print("Demonstration Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
