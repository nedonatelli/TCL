"""High-precision ephemeris queries for celestial bodies.

This example demonstrates using the JPL Development Ephemeris (DE) to compute
high-precision positions of the Sun, Moon, and planets. The ephemeris kernel
data provides accuracy to within kilometers for major solar system bodies.

The example covers:
1. Basic Sun and Moon position queries
2. Planet position queries for all major planets
3. Barycenter calculations for multi-body systems
4. Different reference frame options (ICRF, ecliptic, Earth-centered)
5. Comparing different ephemeris versions (DE405, DE430, DE440)
6. Computing distances and velocities
"""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pytcl.astronomical import (
    DEEphemeris,
    barycenter_position,
    jd_to_cal,
    moon_position,
    planet_position,
    sun_position,
)
from pytcl.astronomical.relativity import AU

SHOW_PLOTS = True
OUTPUT_DIR = Path("docs/_static/images/examples")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_sun_earth_moon_positions(
    jd: float, title: str = "Sun-Earth-Moon System Configuration"
) -> None:
    """Plot Sun, Earth, and Moon positions in 3D."""
    earth_pos = np.array([1.0, 0.0, 0.0]) * AU  # Earth at ~1 AU
    r_sun, _ = sun_position(jd)
    r_moon, _ = moon_position(jd)

    fig = go.Figure()

    # Sun (scaled down for visibility)
    fig.add_trace(
        go.Scatter3d(
            x=[r_sun[0] / AU],
            y=[r_sun[1] / AU],
            z=[r_sun[2] / AU],
            mode="markers+text",
            marker=dict(size=12, color="yellow", symbol="circle"),
            text=["Sun"],
            textposition="top center",
            name="Sun",
            hovertemplate="<b>Sun</b><br>Distance from origin: "
            f"{np.linalg.norm(r_sun)/AU:.3f} AU<extra></extra>",
        )
    )

    # Earth
    fig.add_trace(
        go.Scatter3d(
            x=[earth_pos[0] / AU],
            y=[earth_pos[1] / AU],
            z=[earth_pos[2] / AU],
            mode="markers+text",
            marker=dict(size=8, color="blue", symbol="circle"),
            text=["Earth"],
            textposition="top center",
            name="Earth",
            hovertemplate="<b>Earth</b><br>Distance from Sun: "
            f"{np.linalg.norm(earth_pos - r_sun)/AU:.3f} AU<extra></extra>",
        )
    )

    # Moon (displaced from Earth for visibility)
    moon_distance = np.linalg.norm(r_moon) / 1e6
    fig.add_trace(
        go.Scatter3d(
            x=[r_moon[0] / AU],
            y=[r_moon[1] / AU],
            z=[r_moon[2] / AU],
            mode="markers+text",
            marker=dict(size=5, color="gray", symbol="circle"),
            text=["Moon"],
            textposition="top center",
            name="Moon",
            hovertemplate="<b>Moon</b><br>Distance from Earth: "
            f"{moon_distance:.1f} km<extra></extra>",
        )
    )

    # Orbital connections
    fig.add_trace(
        go.Scatter3d(
            x=[r_sun[0] / AU, earth_pos[0] / AU],
            y=[r_sun[1] / AU, earth_pos[1] / AU],
            z=[r_sun[2] / AU, earth_pos[2] / AU],
            mode="lines",
            line=dict(color="orange", width=2),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=[earth_pos[0] / AU, r_moon[0] / AU],
            y=[earth_pos[1] / AU, r_moon[1] / AU],
            z=[earth_pos[2] / AU, r_moon[2] / AU],
            mode="lines",
            line=dict(color="lightblue", width=1, dash="dash"),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X (AU)",
            yaxis_title="Y (AU)",
            zaxis_title="Z (AU)",
            aspectmode="data",
        ),
        hovermode="closest",
        height=600,
        showlegend=True,
    )

    if SHOW_PLOTS:
        fig.show()
    else:
        fig.write_html(str(OUTPUT_DIR / "ephemeris_demo.html"))


def plot_orbital_distances(
    body_name: str, jd_start: float, num_points: int = 100
) -> None:
    """Plot distance variation over one year."""
    if body_name.lower() == "sun":
        pos_func = sun_position
        title = "Sun's Distance from Earth (Eccentricity Effect)"
    elif body_name.lower() == "moon":
        pos_func = moon_position
        title = "Moon's Distance from Earth (Orbital Variation)"
    else:
        pos_func = lambda jd: planet_position(body_name, jd)
        title = f"{body_name.capitalize()}'s Distance from Earth"

    jd_array = np.linspace(jd_start, jd_start + 365.25, num_points)
    distances = []
    dates = []

    for jd in jd_array:
        r, _ = pos_func(jd)
        distances.append(
            np.linalg.norm(r) / AU
            if body_name.lower() != "moon"
            else np.linalg.norm(r) / 1e6
        )
        year, month, day, _, _, _ = jd_to_cal(jd)
        dates.append(f"{year:04d}-{month:02d}-{day:02d}")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=distances,
            mode="lines+markers",
            name=f"{body_name} Distance",
            line=dict(color="steelblue", width=2),
            marker=dict(size=4),
            hovertemplate="<b>Date:</b> %{x}<br><b>Distance:</b> %{y:.3f} "
            + ("AU" if body_name.lower() != "moon" else "km")
            + "<extra></extra>",
        )
    )

    y_label = "Distance (AU)" if body_name.lower() != "moon" else "Distance (km)"
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=y_label,
        hovermode="x unified",
        height=500,
        plot_bgcolor="rgba(240,240,240,0.5)",
        xaxis_tickangle=-45,
    )

    if SHOW_PLOTS:
        fig.show()
    else:
        fig.write_html(str(OUTPUT_DIR / "ephemeris_demo_distance.html"))


def example_sun_position():
    """Query the Sun's position at specific times."""
    print("=" * 70)
    print("EXAMPLE 1: Sun Position Queries")
    print("=" * 70)

    # Create ephemeris object (defaults to DE440)
    eph = DEEphemeris()

    # J2000.0 epoch (January 1, 2000, 12:00 UT)
    jd_j2000 = 2451545.0

    # Query Sun's position
    r_sun, v_sun = sun_position(jd_j2000)

    print(f"\nJ2000.0 Epoch: JD {jd_j2000}")
    print(f"Sun Position (ICRF):")
    print(f"  X: {r_sun[0]:15.3f} m = {r_sun[0]/AU:10.6f} AU")
    print(f"  Y: {r_sun[1]:15.3f} m = {r_sun[1]/AU:10.6f} AU")
    print(f"  Z: {r_sun[2]:15.3f} m = {r_sun[2]/AU:10.6f} AU")
    print(f"  Distance: {np.linalg.norm(r_sun)/AU:.6f} AU")

    print(f"\nSun Velocity (ICRF):")
    print(f"  VX: {v_sun[0]:12.3f} m/s")
    print(f"  VY: {v_sun[1]:12.3f} m/s")
    print(f"  VZ: {v_sun[2]:12.3f} m/s")
    print(f"  Speed: {np.linalg.norm(v_sun):.3f} m/s")

    # Compute Sun's distance variation over a year
    print("\n" + "-" * 70)
    print("Sun's Distance Throughout 2000:")
    print("-" * 70)

    distances = []
    julian_dates = np.linspace(jd_j2000, jd_j2000 + 365.25, 13)

    for jd in julian_dates:
        year, month, day, _, _, _ = jd_to_cal(jd)
        r, _ = sun_position(jd)
        dist_au = np.linalg.norm(r) / AU
        distances.append(dist_au)
        print(f"  {year:4d}-{month:2d}-{day:2d}  Distance: {dist_au:.6f} AU")

    print(f"\nPerigee (minimum): {min(distances):.6f} AU")
    print(f"Apogee (maximum):  {max(distances):.6f} AU")
    print(
        f"Variation:         {max(distances) - min(distances):.6f} AU "
        f"({100*(max(distances)-min(distances))/np.mean(distances):.2f}%)"
    )

    # Visualize the Sun's orbital distance throughout the year
    plot_orbital_distances("sun", jd_j2000, num_points=365)


def example_moon_position():
    """Query the Moon's position and properties."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Moon Position Queries")
    print("=" * 70)

    jd_j2000 = 2451545.0

    # Query Moon's position
    r_moon, v_moon = moon_position(jd_j2000)

    print(f"\nJ2000.0 Epoch: JD {jd_j2000}")
    print(f"Moon Position (Earth-centered ICRF):")
    print(f"  X: {r_moon[0]:12.3f} m = {r_moon[0]/1e6:10.1f} km")
    print(f"  Y: {r_moon[1]:12.3f} m = {r_moon[1]/1e6:10.1f} km")
    print(f"  Z: {r_moon[2]:12.3f} m = {r_moon[2]/1e6:10.1f} km")
    print(f"  Distance: {np.linalg.norm(r_moon)/1e6:.1f} km")

    print(f"\nMoon Velocity (Earth-centered ICRF):")
    print(
        f"  Speed: {np.linalg.norm(v_moon):.3f} m/s = {np.linalg.norm(v_moon)*86400/1e3:.1f} km/day"
    )

    # Lunar distance variation (orbital ellipticity)
    print("\n" + "-" * 70)
    print("Moon's Distance Variation (showing orbital ellipticity):")
    print("-" * 70)

    distances = []
    times = np.linspace(0, 27.32, 27)  # ~lunar month in days
    julian_dates = jd_j2000 + times

    for jd in julian_dates:
        r, _ = moon_position(jd)
        dist_km = np.linalg.norm(r) / 1e6
        distances.append(dist_km)

    print(f"Perigee (closest):  {min(distances):.1f} km")
    print(f"Apogee (farthest):  {max(distances):.1f} km")
    print(f"Mean distance:      {np.mean(distances):.1f} km")
    print(
        f"Variation:          {max(distances) - min(distances):.1f} km "
        f"({100*(max(distances)-min(distances))/np.mean(distances):.1f}%)"
    )

    # Visualize the Sun-Earth-Moon configuration
    jd_j2000 = 2451545.0
    plot_sun_earth_moon_positions(jd_j2000)

    # Visualize the Moon's orbital distance variation
    plot_orbital_distances("moon", jd_j2000, num_points=365)


def example_planet_positions():
    """Query positions of all major planets."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Planetary Positions")
    print("=" * 70)

    jd_j2000 = 2451545.0

    planets = [
        "mercury",
        "venus",
        "mars",
        "jupiter",
        "saturn",
        "uranus",
        "neptune",
    ]

    print(f"\nPlanetary Heliocentric Positions at J2000.0:")
    print("-" * 70)
    print(f"{'Planet':<10} {'Distance (AU)':<16} {'Longitude':<12} {'Latitude':<12}")
    print("-" * 70)

    for planet_name in planets:
        try:
            r, _ = planet_position(planet_name, jd_j2000)
            dist_au = np.linalg.norm(r) / AU

            # Compute ecliptic coordinates
            lon = np.arctan2(r[1], r[0])
            lat = np.arcsin(r[2] / np.linalg.norm(r))

            print(
                f"{planet_name:<10} {dist_au:<16.6f} {np.degrees(lon):>10.2f}° {np.degrees(lat):>10.2f}°"
            )
        except Exception as e:
            print(f"{planet_name:<10} [Error: {str(e)}]")


def example_barycenter():
    """Compute solar system barycenter positions."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Solar System Barycenter")
    print("=" * 70)

    jd_j2000 = 2451545.0

    # Get Sun position relative to solar system barycenter
    r_barycenter, v_barycenter = barycenter_position("sun", jd_j2000)

    print(f"\nSolar System Barycenter at J2000.0:")
    print(f"  X: {r_barycenter[0]:12.3f} m")
    print(f"  Y: {r_barycenter[1]:12.3f} m")
    print(f"  Z: {r_barycenter[2]:12.3f} m")
    print(f"  Distance from origin: {np.linalg.norm(r_barycenter):.3f} m")
    print(f"  Velocity magnitude: {np.linalg.norm(v_barycenter):.6f} m/s")

    # Compare with Jupiter position
    r_jupiter, _ = planet_position("jupiter", jd_j2000)
    print(f"\nComparison with Jupiter position:")
    print(
        f"  Jupiter distance from barycenter: {np.linalg.norm(r_jupiter - r_barycenter):.3f} m"
    )
    print(f"  This shows Jupiter's significant gravitational influence")


def example_frame_transformations():
    """Demonstrate reference frame options."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Reference Frame Transformations")
    print("=" * 70)

    jd_j2000 = 2451545.0
    eph = DEEphemeris()

    print(f"\nSun's position in different reference frames at J2000.0:")
    print("-" * 70)

    # ICRF (default)
    r_icrf, _ = eph.sun_position(jd_j2000, frame="ICRF")
    print(f"ICRF Frame (International Celestial Reference Frame):")
    print(f"  X: {r_icrf[0]/AU:10.6f} AU")
    print(f"  Y: {r_icrf[1]/AU:10.6f} AU")
    print(f"  Z: {r_icrf[2]/AU:10.6f} AU")

    # Ecliptic frame
    try:
        r_ecliptic, _ = eph.sun_position(jd_j2000, frame="ecliptic")
        print(f"\nEcliptic Frame:")
        print(f"  X: {r_ecliptic[0]/AU:10.6f} AU")
        print(f"  Y: {r_ecliptic[1]/AU:10.6f} AU")
        print(f"  Z: {r_ecliptic[2]/AU:10.6f} AU (small, as expected)")
    except NotImplementedError:
        print("\nEcliptic frame transformation would be applied here")


def example_time_series():
    """Generate time series of object positions (useful for animation)."""
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Time Series for Visualization")
    print("=" * 70)

    jd_start = 2451545.0  # J2000
    dates = jd_start + np.linspace(0, 365, 13)  # Monthly positions

    sun_positions = []
    moon_positions = []

    print("\nComputing 12-month ephemeris...")
    for jd in dates:
        r_sun, _ = sun_position(jd)
        r_moon, _ = moon_position(jd)
        sun_positions.append(r_sun)
        moon_positions.append(r_moon)

    sun_positions = np.array(sun_positions)
    moon_positions = np.array(moon_positions)

    print(f"Computed {len(dates)} positions for Sun and Moon")
    print(f"\nSun orbit statistics:")
    print(f"  Min distance: {np.min(np.linalg.norm(sun_positions, axis=1))/AU:.6f} AU")
    print(f"  Max distance: {np.max(np.linalg.norm(sun_positions, axis=1))/AU:.6f} AU")
    print(f"  Orbit plane:")
    print(f"    Min Z: {np.min(sun_positions[:, 2])/AU:.8f} AU")
    print(f"    Max Z: {np.max(sun_positions[:, 2])/AU:.8f} AU")

    print(f"\nMoon orbit statistics:")
    print(
        f"  Min distance: {np.min(np.linalg.norm(moon_positions, axis=1))/1e6:.1f} km"
    )
    print(
        f"  Max distance: {np.max(np.linalg.norm(moon_positions, axis=1))/1e6:.1f} km"
    )


def example_ephemeris_versions():
    """Compare different ephemeris versions."""
    print("\n" + "=" * 70)
    print("EXAMPLE 7: Ephemeris Version Comparison")
    print("=" * 70)

    jd_test = 2451545.0

    print(f"\nNote: Different ephemeris versions available:")
    print(f"  DE405: JPL Planetary Ephemeris, 1997-2050")
    print(f"  DE430: JPL Planetary Ephemeris, 1550-2650")
    print(f"  DE432s: Short version for limited time range")
    print(f"  DE440: Latest JPL ephemeris, 1550-2650")
    print(f"\nDepending on jplephem version, different kernels can be loaded")
    print(f"Default used in this example: DE440 (if available)")

    eph = DEEphemeris(version="DE440")
    r_sun, _ = eph.sun_position(jd_test)
    print(f"\nSun position (DE440 at J2000.0): {np.linalg.norm(r_sun)/AU:.6f} AU")


if __name__ == "__main__":
    """Run all examples."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  High-Precision Ephemeris Demonstrations".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")

    # Run examples
    example_sun_position()
    example_moon_position()
    example_planet_positions()
    example_barycenter()
    example_frame_transformations()
    example_time_series()
    example_ephemeris_versions()

    print("\n" + "=" * 70)
    print("All ephemeris examples completed successfully!")
    print("=" * 70 + "\n")


OUTPUT_DIR = Path("docs/_static/images/examples")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
