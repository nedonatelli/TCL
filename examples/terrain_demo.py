"""Terrain module demonstration with DEM and elevation models.

This example demonstrates the pytcl.terrain module capabilities, including
digital elevation model (DEM) creation, synthetic terrain generation, terrain
analysis, and viewshed/line-of-sight computations.

Functions demonstrated:
- create_flat_dem(): Create flat digital elevation models
- create_synthetic_terrain(): Generate synthetic terrain with hills/valleys
- compute_horizon(): Calculate visible horizon from observer position
"""

import os
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pytcl.terrain import compute_horizon, create_flat_dem, create_synthetic_terrain

# Controls for visualization
SHOW_PLOTS = os.environ.get("PYTCL_SHOW_PLOTS", "1") != "0"
SKIP_VISUALIZATIONS = True  # Skip visualizations for fast execution


def demo_flat_dem() -> None:
    """Demonstrate flat DEM creation."""
    print("\n" + "=" * 60)
    print("Flat Digital Elevation Model (DEM)")
    print("=" * 60)

    # Create a flat DEM
    dem = create_flat_dem(
        lat_min=np.radians(-1),
        lat_max=np.radians(1),
        lon_min=np.radians(-1),
        lon_max=np.radians(1),
        elevation=1000.0,
        resolution_arcsec=60,
    )

    print(f"\nFlat DEM created:")
    print(
        f"  Latitude range: [{np.degrees(dem.lat_min):.2f}°, {np.degrees(dem.lat_max):.2f}°]"
    )
    print(
        f"  Longitude range: [{np.degrees(dem.lon_min):.2f}°, {np.degrees(dem.lon_max):.2f}°]"
    )
    print(f"  Shape: {dem.data.shape}")
    print(f"  Elevation: {dem.data.min():.1f} to {dem.data.max():.1f} m")
    print(
        f"  Grid spacing: {np.degrees(dem.d_lat):.6f}° lat, {np.degrees(dem.d_lon):.6f}° lon"
    )

    # Visualization (fast heatmap with downsampling for large grids)
    if not SKIP_VISUALIZATIONS:
        # Downsample for faster visualization (keep full resolution data for computations)
        max_size = 500
        stride = max(1, max(dem.data.shape) // max_size)
        z_display = dem.data[::stride, ::stride]

        fig = go.Figure(
            data=go.Heatmap(
                z=z_display,
                colorscale="Viridis",
                name="Elevation",
                colorbar=dict(title="Elevation (m)"),
            )
        )

        fig.update_layout(
            title="Flat Digital Elevation Model",
            xaxis_title="Longitude index",
            yaxis_title="Latitude index",
            height=500,
        )

        if SHOW_PLOTS:
            fig.show()
        else:
            fig.write_html(
                str(OUTPUT_DIR / "terrain_demo.html"),
                include_plotlyjs="cdn",
                div_id="terrain_demo",
            )


def demo_synthetic_terrain() -> None:
    """Demonstrate synthetic terrain generation."""
    print("\n" + "=" * 60)
    print("Synthetic Terrain Generation")
    print("=" * 60)

    # Create synthetic terrain with hills
    dem = create_synthetic_terrain(
        lat_min=np.radians(-5),
        lat_max=np.radians(5),
        lon_min=np.radians(-5),
        lon_max=np.radians(5),
        base_elevation=500,
        amplitude=800,
        wavelength_km=50,
        resolution_arcsec=120,
        seed=42,
    )

    print(f"\nSynthetic DEM created:")
    print(
        f"  Latitude range: [{np.degrees(dem.lat_min):.2f}°, {np.degrees(dem.lat_max):.2f}°]"
    )
    print(
        f"  Longitude range: [{np.degrees(dem.lon_min):.2f}°, {np.degrees(dem.lon_max):.2f}°]"
    )
    print(f"  Shape: {dem.data.shape}")
    print(f"  Min elevation: {dem.data.min():.1f} m")
    print(f"  Max elevation: {dem.data.max():.1f} m")
    print(f"  Mean elevation: {dem.data.mean():.1f} m")
    print(f"  Std deviation: {dem.data.std():.1f} m")
    print(
        f"  Grid spacing: {np.degrees(dem.d_lat):.6f}° lat, {np.degrees(dem.d_lon):.6f}° lon"
    )

    # Visualization: 2D heatmap with downsampling (fast rendering)
    if not SKIP_VISUALIZATIONS:
        # Downsample for faster visualization
        max_size = 500
        stride = max(1, max(dem.data.shape) // max_size)
        z_display = dem.data[::stride, ::stride]

        fig = go.Figure(
            data=go.Heatmap(
                z=z_display,
                colorscale="Earth",
                name="Elevation",
                colorbar=dict(title="Elevation (m)"),
            )
        )

        fig.update_layout(
            title="Synthetic Terrain with Synthetic Hills",
            xaxis_title="Longitude index",
            yaxis_title="Latitude index",
            height=600,
        )

        if SHOW_PLOTS:
            fig.show()
        else:
            fig.write_html(
                str(OUTPUT_DIR / "terrain_demo.html"),
                include_plotlyjs="cdn",
                div_id="terrain_demo",
            )


def demo_terrain_analysis() -> None:
    """Demonstrate terrain analysis and statistics."""
    print("\n" + "=" * 60)
    print("Terrain Analysis and Statistics")
    print("=" * 60)

    # Create two DEMs for comparison
    flat_dem = create_flat_dem(
        lat_min=np.radians(-2),
        lat_max=np.radians(2),
        lon_min=np.radians(-2),
        lon_max=np.radians(2),
        elevation=500.0,
        resolution_arcsec=60,
    )

    synthetic_dem = create_synthetic_terrain(
        lat_min=np.radians(-2),
        lat_max=np.radians(2),
        lon_min=np.radians(-2),
        lon_max=np.radians(2),
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

    # Terrain slope. np.gradient returns meters of rise per *grid cell*, so it
    # has to be divided by the ground spacing of a cell before it means
    # anything -- without that the slope comes out near-vertical everywhere.
    earth_radius = 6371000.0
    mean_lat = 0.5 * (synthetic_dem.lat_min + synthetic_dem.lat_max)
    dy_m = synthetic_dem.d_lat * earth_radius
    dx_m = synthetic_dem.d_lon * earth_radius * np.cos(mean_lat)

    grad_y, grad_x = np.gradient(synthetic_dem.data)
    slope = np.degrees(np.arctan(np.hypot(grad_x / dx_m, grad_y / dy_m)))

    print(f"\nGrid cell ground size: {dx_m:.0f} m east-west, {dy_m:.0f} m north-south")

    print(f"\nTerrain slope analysis:")
    print(f"  Min slope: {slope.min():.2f}°")
    print(f"  Max slope: {slope.max():.2f}°")
    print(f"  Mean slope: {slope.mean():.2f}°")

    # Visualization: Comparison histograms (skip for performance)
    if not SKIP_VISUALIZATIONS:
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
        else:
            fig.write_html(
                str(OUTPUT_DIR / "terrain_demo.html"),
                include_plotlyjs="cdn",
                div_id="terrain_demo",
            )

        # Visualization: Slope map with downsampling
        max_size = 500
        stride_slope = max(1, max(slope.shape) // max_size)
        z_slope_display = slope[::stride_slope, ::stride_slope]

        fig_slope = go.Figure(
            data=go.Heatmap(
                z=z_slope_display,
                colorscale="Reds",
                name="Slope",
                colorbar=dict(title="Slope (°)"),
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

    # A one-degree box of synthetic terrain. All angles are radians, per the
    # library-wide convention -- passing degrees here would ask for a DEM
    # spanning most of the globe.
    lat_min, lat_max = np.radians(35.0), np.radians(36.0)
    lon_min, lon_max = np.radians(-120.0), np.radians(-119.0)
    dem = create_synthetic_terrain(
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
        base_elevation=500,
        amplitude=500,
        wavelength_km=40,
        resolution_arcsec=30,
        seed=456,
    )

    print(f"\nDEM for horizon analysis:")
    print(f"  Shape: {dem.data.shape}")
    print(f"  Elevation range: {dem.data.min():.1f} to {dem.data.max():.1f} m")

    # Observe from the middle of the box, 100 m above local ground level.
    obs_lat = 0.5 * (lat_min + lat_max)
    obs_lon = 0.5 * (lon_min + lon_max)
    obs_height = 100.0

    print(f"\nObserver position:")
    print(f"  Latitude:  {np.degrees(obs_lat):.4f}°")
    print(f"  Longitude: {np.degrees(obs_lon):.4f}°")
    print(f"  Ground elevation: {dem.get_elevation(obs_lat, obs_lon).elevation:.1f} m")
    print(f"  Observer height above ground: {obs_height} m")

    horizon = compute_horizon(
        dem,
        obs_lat,
        obs_lon,
        obs_height,
        n_azimuths=72,
        max_range=40000.0,
    )

    # compute_horizon returns one HorizonPoint per azimuth.
    elevations = np.array([p.elevation_angle for p in horizon])
    distances = np.array([p.distance for p in horizon])
    azimuths = np.array([p.azimuth for p in horizon])

    print(f"\nHorizon profile over {len(horizon)} azimuths:")
    print(
        f"  Elevation angle: {np.degrees(elevations.min()):+.2f}° to "
        f"{np.degrees(elevations.max()):+.2f}°"
    )
    print(
        f"  Horizon distance: {distances.min() / 1e3:.1f} to "
        f"{distances.max() / 1e3:.1f} km"
    )

    highest = int(np.argmax(elevations))
    print(
        f"  Highest horizon at azimuth {np.degrees(azimuths[highest]):.1f}° "
        f"({np.degrees(elevations[highest]):+.2f}°, "
        f"{distances[highest] / 1e3:.1f} km away)"
    )


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


OUTPUT_DIR = Path("docs/_static/images/examples")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    main()
