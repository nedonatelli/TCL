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

import matplotlib.pyplot as plt
import numpy as np

# Global flag to control plotting
SHOW_PLOTS = True


def setup_plot_style():
    """Configure matplotlib style for consistent plots."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.figsize": (10, 6),
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
        }
    )


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
        setup_plot_style()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # High-resolution latitude array
        lats_fine = np.linspace(0, 90, 181)
        g_values = np.array(
            [normal_gravity_somigliana(np.radians(lat)) for lat in lats_fine]
        )

        # Left plot: Gravity vs latitude
        ax1.plot(lats_fine, g_values, "b-", linewidth=2)
        ax1.set_xlabel("Latitude (°)")
        ax1.set_ylabel("Normal Gravity (m/s²)")
        ax1.set_title("Normal Gravity vs Latitude (Somigliana)")
        ax1.set_xlim(0, 90)
        ax1.grid(True, alpha=0.3)

        # Mark equator and pole values
        ax1.axhline(
            y=g_values[0],
            color="r",
            linestyle="--",
            alpha=0.5,
            label=f"Equator: {g_values[0]:.4f}",
        )
        ax1.axhline(
            y=g_values[-1],
            color="g",
            linestyle="--",
            alpha=0.5,
            label=f"Pole: {g_values[-1]:.4f}",
        )
        ax1.legend()

        # Right plot: Gravity difference from equator
        g_diff = (g_values - g_values[0]) * 1000  # mGal
        ax2.plot(lats_fine, g_diff, "b-", linewidth=2)
        ax2.set_xlabel("Latitude (°)")
        ax2.set_ylabel("Δg from Equator (mGal)")
        ax2.set_title("Gravity Increase from Equator")
        ax2.set_xlim(0, 90)
        ax2.grid(True, alpha=0.3)

        # Annotate key locations
        for lat_mark in [30, 45, 60]:
            idx = lat_mark * 2
            ax2.annotate(
                f"{lat_mark}°: +{g_diff[idx]:.0f} mGal",
                xy=(lat_mark, g_diff[idx]),
                xytext=(lat_mark + 5, g_diff[idx] - 500),
                fontsize=9,
                arrowprops=dict(arrowstyle="->", alpha=0.5),
            )

        plt.tight_layout()
        plt.savefig("geophysical_gravity_latitude.png", dpi=150, bbox_inches="tight")
        print("\nPlot saved: geophysical_gravity_latitude.png")
        plt.show()


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
        setup_plot_style()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Altitude range from surface to ISS altitude
        alts = np.linspace(0, 500000, 500)  # 0 to 500 km
        g_values = np.array([gravity_wgs84(lat, lon, a).magnitude for a in alts])

        # Left plot: Gravity vs altitude
        ax1.plot(alts / 1000, g_values, "b-", linewidth=2)
        ax1.set_xlabel("Altitude (km)")
        ax1.set_ylabel("Gravity (m/s²)")
        ax1.set_title("Gravity vs Altitude (WGS84)")
        ax1.grid(True, alpha=0.3)

        # Mark key altitudes
        key_alts = [
            (0, "Sea level"),
            (10, "Cruising alt"),
            (100, "Kármán line"),
            (400, "ISS orbit"),
        ]
        for alt_km, label in key_alts:
            idx = int(alt_km * 500 / 500)
            ax1.axvline(x=alt_km, color="gray", linestyle=":", alpha=0.5)
            ax1.annotate(
                label,
                xy=(alt_km, g_values[idx]),
                fontsize=8,
                rotation=90,
                va="bottom",
                ha="right",
            )

        # Right plot: Gravity reduction rate
        g_reduction = (g_values[0] - g_values) / g_values[0] * 100  # percent
        ax2.plot(alts / 1000, g_reduction, "r-", linewidth=2)
        ax2.set_xlabel("Altitude (km)")
        ax2.set_ylabel("Gravity Reduction (%)")
        ax2.set_title("Gravity Reduction with Altitude")
        ax2.grid(True, alpha=0.3)

        # Mark ISS at ~400 km
        ax2.axhline(
            y=g_reduction[400],
            color="g",
            linestyle="--",
            alpha=0.7,
            label=f"ISS (~400 km): {g_reduction[400]:.1f}% reduction",
        )
        ax2.legend()

        plt.tight_layout()
        plt.savefig("geophysical_gravity_altitude.png", dpi=150, bbox_inches="tight")
        print("\nPlot saved: geophysical_gravity_altitude.png")
        plt.show()


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
        setup_plot_style()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

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

        # Top plot: Displacement components
        ax1.plot(hours, disp_n, "b-", linewidth=1.5, label="North")
        ax1.plot(hours, disp_e, "g-", linewidth=1.5, label="East")
        ax1.plot(hours, disp_u, "r-", linewidth=2, label="Up")
        ax1.set_ylabel("Displacement (mm)")
        ax1.set_title("Solid Earth Tide - Washington DC (2025-01-01)")
        ax1.legend(loc="upper right")
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color="k", linestyle="-", alpha=0.3)

        # Bottom plot: Gravity change
        ax2.plot(hours, dg, "m-", linewidth=2)
        ax2.fill_between(hours, 0, dg, alpha=0.3, color="m")
        ax2.set_xlabel("Hour (UTC)")
        ax2.set_ylabel("Gravity Change (µGal)")
        ax2.set_title("Tidal Gravity Variation")
        ax2.set_xlim(0, 24)
        ax2.set_xticks(np.arange(0, 25, 3))
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color="k", linestyle="-", alpha=0.3)

        plt.tight_layout()
        plt.savefig("geophysical_tides.png", dpi=150, bbox_inches="tight")
        print("\nPlot saved: geophysical_tides.png")
        plt.show()


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
        setup_plot_style()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

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

        # Left plot: Magnetic declination
        levels_dec = np.linspace(-30, 30, 13)
        cs1 = ax1.contourf(
            LON, LAT, DEC, levels=levels_dec, cmap="RdBu_r", extend="both"
        )
        ax1.contour(LON, LAT, DEC, levels=[0], colors="k", linewidths=2)
        plt.colorbar(cs1, ax=ax1, label="Declination (°)")
        ax1.set_xlabel("Longitude (°)")
        ax1.set_ylabel("Latitude (°)")
        ax1.set_title(f"Magnetic Declination (WMM {decimal_year:.0f})")
        ax1.set_xlim(-180, 180)
        ax1.set_ylim(-80, 80)

        # Right plot: Total field intensity
        levels_f = np.linspace(20000, 65000, 10)
        cs2 = ax2.contourf(LON, LAT, F, levels=levels_f, cmap="viridis")
        plt.colorbar(cs2, ax=ax2, label="Total Intensity (nT)")
        ax2.set_xlabel("Longitude (°)")
        ax2.set_ylabel("Latitude (°)")
        ax2.set_title(f"Magnetic Field Intensity (WMM {decimal_year:.0f})")
        ax2.set_xlim(-180, 180)
        ax2.set_ylim(-80, 80)

        # Mark South Atlantic Anomaly region
        ax2.plot(-50, -25, "r*", markersize=15, label="South Atlantic Anomaly")
        ax2.legend(loc="lower left")

        plt.tight_layout()
        plt.savefig("geophysical_magnetic_field.png", dpi=150, bbox_inches="tight")
        print("\nPlot saved: geophysical_magnetic_field.png")
        plt.show()


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
    if SHOW_PLOTS:
        setup_plot_style()

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
    print("=" * 70)


if __name__ == "__main__":
    main()
