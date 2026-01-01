"""
Geophysical Models Example
==========================

This example demonstrates the geophysical models in PyTCL:

Gravity Models:
- Normal gravity (Somigliana formula)
- WGS84 and J2 gravity models
- Spherical harmonic expansions
- Geoid height computation
- Gravity anomalies and disturbances
- Tidal effects (solid Earth, ocean loading)

Magnetic Field Models:
- World Magnetic Model (WMM2020)
- International Geomagnetic Reference Field (IGRF-13)
- Enhanced Magnetic Model (EMM)
- Magnetic declination, inclination, and intensity

These models are essential for high-precision navigation, geodesy,
and aerospace applications.
"""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Output directory for generated plots
OUTPUT_DIR = Path(__file__).parent.parent / "docs" / "_static" / "images" / "examples"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Global flag to control plotting
SHOW_PLOTS = True


from pytcl.gravity import (  # Normal gravity; Gravity models; Geoid; Anomalies; Tidal effects; Constants
    GRS80,
    WGS84,
    free_air_anomaly,
    geoid_height,
    geoid_height_j2,
    gravity_anomaly,
    gravity_disturbance,
    gravity_j2,
    gravity_wgs84,
    normal_gravity,
    normal_gravity_somigliana,
    solid_earth_tide_displacement,
    solid_earth_tide_gravity,
    tidal_gravity_correction,
)
from pytcl.magnetism import (  # World Magnetic Model; IGRF
    dipole_moment,
    igrf,
    igrf_declination,
    igrf_inclination,
    magnetic_declination,
    magnetic_field_intensity,
    magnetic_inclination,
    magnetic_north_pole,
    wmm,
)


def demo_normal_gravity():
    """Demonstrate normal gravity computation."""
    print("=" * 70)
    print("Normal Gravity Demo")
    print("=" * 70)

    # Test locations at different latitudes
    locations = [
        ("Equator", 0.0, 0.0, 0.0),
        ("Washington DC", 38.9, -77.0, 0.0),
        ("North Pole", 90.0, 0.0, 0.0),
        ("Mount Everest", 27.99, 86.93, 8848.0),
        ("Dead Sea", 31.5, 35.5, -430.0),
    ]

    print("\nNormal gravity at various locations:")
    print("-" * 60)
    print(f"{'Location':<20} {'Lat':>8} {'Alt (m)':>10} {'g (m/s²)':>12}")
    print("-" * 60)

    for name, lat, lon, alt in locations:
        # Normal gravity at sea level
        g_somigliana = normal_gravity_somigliana(np.radians(lat))

        # Gravity at altitude (free-air correction)
        g_at_alt = normal_gravity(np.radians(lat), alt)

        print(f"{name:<20} {lat:>8.2f} {alt:>10.1f} {g_at_alt:>12.6f}")

    # Show latitude variation
    print("\n--- Latitude Variation ---")
    lats = np.array([0, 15, 30, 45, 60, 75, 90])
    for lat in lats:
        g = normal_gravity_somigliana(np.radians(lat))
        print(f"  {lat:>2}°: {g:.6f} m/s²")

    print("\nNote: Gravity increases from equator to poles due to")
    print("Earth's rotation (centrifugal) and flattening effects.")

    # Plot gravity variation with latitude
    if SHOW_PLOTS:
        # High-resolution latitude array
        lats_fine = np.linspace(0, 90, 181)
        g_values = np.array(
            [normal_gravity_somigliana(np.radians(lat)) for lat in lats_fine]
        )

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(
                "Normal Gravity vs Latitude (Somigliana)",
                "Gravity Increase from Equator",
            ),
        )

        # Left plot: Gravity vs latitude
        fig.add_trace(
            go.Scatter(
                x=lats_fine,
                y=g_values,
                mode="lines",
                name="Gravity",
                line=dict(color="blue", width=2),
            ),
            row=1,
            col=1,
        )
        # Mark equator and pole values
        fig.add_hline(
            y=g_values[0],
            line_dash="dash",
            line_color="red",
            annotation_text=f"Equator: {g_values[0]:.4f}",
            row=1,
            col=1,
        )
        fig.add_hline(
            y=g_values[-1],
            line_dash="dash",
            line_color="green",
            annotation_text=f"Pole: {g_values[-1]:.4f}",
            row=1,
            col=1,
        )

        # Right plot: Gravity difference from equator
        g_diff = (g_values - g_values[0]) * 1000  # mGal
        fig.add_trace(
            go.Scatter(
                x=lats_fine,
                y=g_diff,
                mode="lines",
                name="Δg",
                line=dict(color="blue", width=2),
            ),
            row=1,
            col=2,
        )

        fig.update_xaxes(title_text="Latitude (°)", range=[0, 90], row=1, col=1)
        fig.update_yaxes(title_text="Normal Gravity (m/s²)", row=1, col=1)
        fig.update_xaxes(title_text="Latitude (°)", range=[0, 90], row=1, col=2)
        fig.update_yaxes(title_text="Δg from Equator (mGal)", row=1, col=2)

        fig.update_layout(height=500, width=1200, showlegend=False)
        fig.write_html(str(OUTPUT_DIR / "geophysical_gravity_latitude.html"))
        print("\n  [Plot saved to geophysical_gravity_latitude.html]")


def demo_gravity_models():
    """Demonstrate different gravity models."""
    print("\n" + "=" * 70)
    print("Gravity Models Comparison Demo")
    print("=" * 70)

    # Test point: Washington DC
    lat = np.radians(38.9)
    lon = np.radians(-77.0)
    alt = 0.0  # Sea level

    print(f"\nLocation: Washington DC (38.9°N, 77.0°W)")
    print("-" * 50)

    # WGS84 gravity model
    g_wgs84 = gravity_wgs84(lat, lon, alt)
    print(f"\nWGS84 gravity model:")
    print(f"  Total gravity: {g_wgs84.magnitude:.6f} m/s²")
    print(f"  Down component: {g_wgs84.g_down:.6f} m/s²")
    print(f"  North component: {g_wgs84.g_north:.9f} m/s²")

    # J2 gravity model (simpler, uses only J2 term)
    g_j2 = gravity_j2(lat, lon, alt)
    print(f"\nJ2 gravity model:")
    print(f"  Total gravity: {g_j2.magnitude:.6f} m/s²")
    print(
        f"  Difference from WGS84: {(g_j2.magnitude - g_wgs84.magnitude)*1e6:.2f} µGal"
    )

    # Compare at different altitudes
    print("\n--- Gravity vs Altitude ---")
    altitudes = [0, 1000, 10000, 100000, 400000]  # meters
    for alt in altitudes:
        g = gravity_wgs84(lat, lon, alt)
        print(f"  {alt:>7} m: {g.magnitude:.6f} m/s²")

    print("\nNote: Gravity decreases with altitude approximately as")
    print("g(h) ≈ g₀(1 - 2h/R), about 3.1 mGal per meter at low altitudes.")

    # Plot gravity vs altitude
    if SHOW_PLOTS:
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(
                "Gravity vs Altitude (WGS84)",
                "Gravity Reduction with Altitude",
            ),
        )

        # Altitude range from surface to ISS altitude
        alts = np.linspace(0, 500000, 500)  # 0 to 500 km
        g_values = np.array([gravity_wgs84(lat, lon, a).magnitude for a in alts])

        # Left plot: Gravity vs altitude
        fig.add_trace(
            go.Scatter(
                x=alts / 1000,
                y=g_values,
                mode="lines",
                name="Gravity",
                line=dict(color="blue", width=2),
            ),
            row=1,
            col=1,
        )

        # Right plot: Gravity reduction rate
        g_reduction = (g_values[0] - g_values) / g_values[0] * 100  # percent
        fig.add_trace(
            go.Scatter(
                x=alts / 1000,
                y=g_reduction,
                mode="lines",
                name="Reduction",
                line=dict(color="red", width=2),
            ),
            row=1,
            col=2,
        )

        # Mark ISS at ~400 km
        fig.add_hline(
            y=g_reduction[400],
            line_dash="dash",
            line_color="green",
            annotation_text=f"ISS (~400 km): {g_reduction[400]:.1f}% reduction",
            row=1,
            col=2,
        )

        fig.update_xaxes(title_text="Altitude (km)", row=1, col=1)
        fig.update_yaxes(title_text="Gravity (m/s²)", row=1, col=1)
        fig.update_xaxes(title_text="Altitude (km)", row=1, col=2)
        fig.update_yaxes(title_text="Gravity Reduction (%)", row=1, col=2)

        fig.update_layout(height=500, width=1200, showlegend=False)
        fig.write_html(str(OUTPUT_DIR / "geophysical_gravity_altitude.html"))
        print("\n  [Plot saved to geophysical_gravity_altitude.html]")


def demo_geoid():
    """Demonstrate geoid height computation."""
    print("\n" + "=" * 70)
    print("Geoid Height Demo")
    print("=" * 70)

    # Various latitudes (J2 geoid approximation is zonal - only latitude dependent)
    locations = [
        ("Equator", 0.0),
        ("Mid-latitude (30N)", 30.0),
        ("Mid-latitude (45N)", 45.0),
        ("High latitude (60N)", 60.0),
        ("Pole", 90.0),
    ]

    print("\nJ2 Geoid heights (zonal approximation - latitude dependent only):")
    print("-" * 40)
    print(f"{'Location':<25} {'Lat':>7} {'N (m)':>10}")
    print("-" * 40)

    for name, lat in locations:
        # J2 geoid approximation - only depends on latitude
        N = geoid_height_j2(np.radians(lat))
        print(f"{name:<25} {lat:>7.1f} {N:>10.2f}")

    print("\nNote: The J2 geoid model captures the main flattening effect.")
    print("Real geoid variations are ±100m from the ellipsoid (need EGM96/2008).")


def demo_gravity_anomalies():
    """Demonstrate gravity anomaly computation."""
    print("\n" + "=" * 70)
    print("Gravity Anomalies Demo")
    print("=" * 70)

    # Simulated gravity survey
    np.random.seed(42)

    # Locations along a survey line
    n_points = 5
    lats = np.linspace(38.0, 39.0, n_points)
    lons = np.full(n_points, -77.0)
    alts = np.array([100, 150, 200, 180, 120])  # meters

    # Simulated observed gravity (with anomaly)
    g_normal = np.array(
        [normal_gravity(np.radians(lat), alt) for lat, alt in zip(lats, alts)]
    )
    # Add a gravity anomaly (e.g., from subsurface density variation)
    anomaly_true = np.array([0.0, 10.0, 25.0, 15.0, 5.0]) * 1e-5  # m/s²
    g_observed = g_normal + anomaly_true

    print("\nGravity survey along profile:")
    print("-" * 65)
    print(f"{'Point':>5} {'Lat':>7} {'Alt(m)':>7} {'g_obs':>12} {'FAA':>12}")
    print("-" * 65)

    for i in range(n_points):
        lat_rad = np.radians(lats[i])
        faa = free_air_anomaly(g_observed[i], lat_rad, alts[i])
        print(
            f"{i+1:>5} {lats[i]:>7.2f} {alts[i]:>7.0f} "
            f"{g_observed[i]:>12.6f} {faa*1e5:>12.2f} mGal"
        )

    print("\nNote: Free-air anomaly removes the effect of elevation")
    print("to reveal subsurface density variations.")


def demo_tidal_effects():
    """Demonstrate tidal effects on gravity and position."""
    print("\n" + "=" * 70)
    print("Tidal Effects Demo")
    print("=" * 70)

    # Observer location
    lat = np.radians(38.9)  # Washington DC
    lon = np.radians(-77.0)

    # Time (Julian date) - approximate
    # Let's use a few different times
    times = [
        ("2025-01-01 00:00 UTC", 2460676.5),
        ("2025-01-01 06:00 UTC", 2460676.75),
        ("2025-01-01 12:00 UTC", 2460677.0),
        ("2025-01-01 18:00 UTC", 2460677.25),
    ]

    print(f"\nLocation: Washington DC (38.9°N, 77.0°W)")
    print("\nSolid Earth tide effects (displacement and gravity):")
    print("-" * 60)

    for time_str, jd in times:
        # Solid Earth tide displacement
        disp = solid_earth_tide_displacement(lat, lon, jd)

        # Tidal gravity correction
        dg = tidal_gravity_correction(lat, lon, jd)

        print(f"\n{time_str}:")
        print(
            f"  Displacement: ({disp[0]*1000:.1f}, {disp[1]*1000:.1f}, "
            f"{disp[2]*1000:.1f}) mm (N, E, Up)"
        )
        print(f"  Gravity change: {dg*1e8:.2f} µGal")

    print("\nNote: Solid Earth tides cause displacements of ~30 cm")
    print("and gravity changes of ~300 µGal peak-to-peak.")

    # Plot tidal effects over 24 hours
    if SHOW_PLOTS:
        # Time array: 24 hours at 15-minute intervals
        hours = np.linspace(0, 24, 97)
        jd_start = 2460676.5  # 2025-01-01 00:00 UTC
        jds = jd_start + hours / 24

        # Compute displacements and gravity changes
        disp_n = np.zeros_like(hours)
        disp_e = np.zeros_like(hours)
        disp_u = np.zeros_like(hours)
        dg = np.zeros_like(hours)

        for i, jd in enumerate(jds):
            disp = solid_earth_tide_displacement(lat, lon, jd)
            disp_n[i] = disp[0] * 1000  # mm
            disp_e[i] = disp[1] * 1000
            disp_u[i] = disp[2] * 1000
            dg[i] = tidal_gravity_correction(lat, lon, jd) * 1e8  # µGal

        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=(
                "Solid Earth Tide - Washington DC (2025-01-01)",
                "Tidal Gravity Variation",
            ),
            shared_xaxes=True,
        )

        # Top plot: Displacement components
        fig.add_trace(
            go.Scatter(
                x=hours,
                y=disp_n,
                mode="lines",
                name="North",
                line=dict(color="blue", width=1.5),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=hours,
                y=disp_e,
                mode="lines",
                name="East",
                line=dict(color="green", width=1.5),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=hours,
                y=disp_u,
                mode="lines",
                name="Up",
                line=dict(color="red", width=2),
            ),
            row=1,
            col=1,
        )
        fig.add_hline(y=0, line_color="black", line_width=0.5, row=1, col=1)

        # Bottom plot: Gravity change
        fig.add_trace(
            go.Scatter(
                x=hours,
                y=dg,
                mode="lines",
                name="Gravity",
                line=dict(color="purple", width=2),
                fill="tozeroy",
                fillcolor="rgba(128,0,128,0.3)",
            ),
            row=2,
            col=1,
        )
        fig.add_hline(y=0, line_color="black", line_width=0.5, row=2, col=1)

        fig.update_xaxes(title_text="Hour (UTC)", range=[0, 24], row=2, col=1)
        fig.update_yaxes(title_text="Displacement (mm)", row=1, col=1)
        fig.update_yaxes(title_text="Gravity Change (µGal)", row=2, col=1)

        fig.update_layout(height=600, width=1000)
        fig.write_html(str(OUTPUT_DIR / "geophysical_tides.html"))
        print("\n  [Plot saved to geophysical_tides.html]")


def demo_magnetic_field():
    """Demonstrate magnetic field computation."""
    print("\n" + "=" * 70)
    print("Magnetic Field Models Demo")
    print("=" * 70)

    # Test locations
    locations = [
        ("Washington DC", 38.9, -77.0, 0.0),
        ("London", 51.5, -0.1, 0.0),
        ("Sydney", -33.9, 151.2, 0.0),
        ("Magnetic North", 86.5, -175.3, 0.0),
        ("Magnetic Equator", 0.0, 0.0, 0.0),
        ("South Atlantic Anomaly", -25.0, -50.0, 0.0),
    ]

    # Use 2025 epoch
    decimal_year = 2025.0

    print(f"\nMagnetic field at various locations (WMM2020, {decimal_year}):")
    print("-" * 75)
    print(f"{'Location':<25} {'Dec':>8} {'Inc':>8} {'F (nT)':>10}")
    print("-" * 75)

    for name, lat, lon, alt in locations:
        # Compute magnetic field using WMM
        result = wmm(np.radians(lat), np.radians(lon), alt, decimal_year)

        dec = np.degrees(result.D)  # Declination
        inc = np.degrees(result.I)  # Inclination
        F = result.F  # Total intensity

        print(f"{name:<25} {dec:>8.2f}° {inc:>8.2f}° {F:>10.0f}")

    # Show magnetic declination map concept
    print("\n--- Magnetic Declination Grid ---")
    lats_grid = np.array([-60, -30, 0, 30, 60])
    lons_grid = np.array([-120, -60, 0, 60, 120])

    print(f"{'Lat\\Lon':<8}", end="")
    for lon in lons_grid:
        print(f"{lon:>8}°", end="")
    print()

    for lat in lats_grid:
        print(f"{lat:>6}°  ", end="")
        for lon in lons_grid:
            dec = magnetic_declination(
                np.radians(lat), np.radians(lon), 0.0, decimal_year
            )
            print(f"{np.degrees(dec):>8.1f}", end="")
        print()

    # Plot magnetic declination and field intensity maps
    if SHOW_PLOTS:
        # Create grid for plotting
        lat_grid = np.linspace(-80, 80, 33)
        lon_grid = np.linspace(-180, 180, 73)
        LAT, LON = np.meshgrid(lat_grid, lon_grid)

        # Compute declination and intensity on grid
        DEC = np.zeros_like(LAT)
        F = np.zeros_like(LAT)
        for i in range(LAT.shape[0]):
            for j in range(LAT.shape[1]):
                result = wmm(
                    np.radians(LAT[i, j]), np.radians(LON[i, j]), 0.0, decimal_year
                )
                DEC[i, j] = np.degrees(result.D)
                F[i, j] = result.F

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(
                f"Magnetic Declination (WMM {decimal_year:.0f})",
                f"Magnetic Field Intensity (WMM {decimal_year:.0f})",
            ),
        )

        # Left plot: Magnetic declination
        fig.add_trace(
            go.Contour(
                x=lon_grid,
                y=lat_grid,
                z=DEC.T,
                colorscale="RdBu_r",
                contours=dict(showlines=True),
                colorbar=dict(title="Declination (°)", x=0.45),
            ),
            row=1,
            col=1,
        )

        # Right plot: Total field intensity
        fig.add_trace(
            go.Contour(
                x=lon_grid,
                y=lat_grid,
                z=F.T,
                colorscale="Viridis",
                colorbar=dict(title="Intensity (nT)", x=1.0),
            ),
            row=1,
            col=2,
        )

        # Mark South Atlantic Anomaly region
        fig.add_trace(
            go.Scatter(
                x=[-50],
                y=[-25],
                mode="markers",
                marker=dict(symbol="star", size=15, color="red"),
                name="South Atlantic Anomaly",
                showlegend=True,
            ),
            row=1,
            col=2,
        )

        fig.update_xaxes(title_text="Longitude (°)", range=[-180, 180], row=1, col=1)
        fig.update_yaxes(title_text="Latitude (°)", range=[-80, 80], row=1, col=1)
        fig.update_xaxes(title_text="Longitude (°)", range=[-180, 180], row=1, col=2)
        fig.update_yaxes(title_text="Latitude (°)", range=[-80, 80], row=1, col=2)

        fig.update_layout(height=500, width=1400)
        fig.write_html(str(OUTPUT_DIR / "geophysical_magnetic_field.html"))
        print("\n  [Plot saved to geophysical_magnetic_field.html]")


def demo_magnetic_properties():
    """Demonstrate magnetic field properties and variations."""
    print("\n" + "=" * 70)
    print("Magnetic Field Properties Demo")
    print("=" * 70)

    decimal_year = 2025.0

    # Magnetic pole location
    pole_lat, pole_lon = magnetic_north_pole(decimal_year)
    print(f"\nMagnetic North Pole location ({decimal_year}):")
    print(f"  Latitude: {np.degrees(pole_lat):.2f}°N")
    print(f"  Longitude: {np.degrees(pole_lon):.2f}°E")

    # Earth's dipole moment (using WMM2020 coefficients)
    moment = dipole_moment()  # Uses default WMM2020 coefficients
    print(f"\nEarth's dipole moment (WMM2020): {moment:.4e} A·m²")

    # Compare WMM and IGRF at a test location
    lat, lon, alt = np.radians(40.0), np.radians(-100.0), 0.0

    wmm_result = wmm(lat, lon, alt, decimal_year)
    igrf_result = igrf(lat, lon, alt, decimal_year)

    print(f"\nModel comparison at (40°N, 100°W):")
    print("-" * 40)
    print(f"{'Component':<12} {'WMM':>12} {'IGRF':>12}")
    print("-" * 40)
    print(f"{'X (nT)':<12} {wmm_result.X:>12.1f} {igrf_result.X:>12.1f}")
    print(f"{'Y (nT)':<12} {wmm_result.Y:>12.1f} {igrf_result.Y:>12.1f}")
    print(f"{'Z (nT)':<12} {wmm_result.Z:>12.1f} {igrf_result.Z:>12.1f}")
    print(f"{'F (nT)':<12} {wmm_result.F:>12.1f} {igrf_result.F:>12.1f}")

    # Secular variation
    print("\n--- Secular Variation ---")
    print("Magnetic field changes over time due to core dynamics.")
    years = [2020, 2022, 2024, 2025]
    for year in years:
        dec = magnetic_declination(lat, lon, alt, float(year))
        print(f"  {year}: declination = {np.degrees(dec):.2f}°")


def demo_navigation_application():
    """Demonstrate application to navigation."""
    print("\n" + "=" * 70)
    print("Navigation Application Demo")
    print("=" * 70)

    # Aircraft navigation scenario
    lat = np.radians(40.0)
    lon = np.radians(-74.0)
    alt = 10000.0  # meters (cruise altitude)
    decimal_year = 2025.0

    print("\nAircraft navigation correction scenario:")
    print(f"  Position: 40°N, 74°W")
    print(f"  Altitude: {alt:.0f} m")

    # Magnetic declination for compass correction
    dec = magnetic_declination(lat, lon, alt, decimal_year)
    print(f"\n  Magnetic declination: {np.degrees(dec):.2f}°")
    print(f"  → Add {np.degrees(dec):.2f}° to magnetic heading for true heading")

    # Gravity for inertial navigation
    g_result = gravity_wgs84(lat, lon, alt)
    print(f"\n  Local gravity: {g_result.magnitude:.6f} m/s²")
    print(f"  → Used for vertical channel in INS")

    # Deflection of vertical (simplified)
    print(f"\n  Gravity deflection (N): {g_result.g_north*1e6:.2f} µrad")
    print(f"  → Correction for inertial alignment")

    # Geoid for altitude reference (J2 approximation - latitude only)
    N = geoid_height_j2(lat)
    print(f"\n  Geoid undulation (J2 approx): {N:.1f} m")
    print(f"  → Ellipsoid alt = GPS alt, Orthometric alt = GPS alt - N")


def main():
    """Run all demonstrations."""
    print("\n" + "#" * 70)
    print("# PyTCL Geophysical Models Example")
    print("#" * 70)

    # Gravity models
    demo_normal_gravity()
    demo_gravity_models()
    demo_geoid()
    demo_gravity_anomalies()
    demo_tidal_effects()

    # Magnetic field models
    demo_magnetic_field()
    demo_magnetic_properties()

    # Application
    demo_navigation_application()

    print("\n" + "=" * 70)
    print("Example complete!")
    if SHOW_PLOTS:
        print("Plots saved: geophysical_gravity_latitude.html,")
        print("             geophysical_gravity_altitude.html,")
        print("             geophysical_tides.html,")
        print("             geophysical_magnetic_field.html")
    print("=" * 70)


if __name__ == "__main__":
    main()
