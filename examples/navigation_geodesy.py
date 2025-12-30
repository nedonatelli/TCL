"""
Navigation and Geodesy Example.

This example demonstrates:
1. Geodetic coordinate conversions (WGS84)
2. Local tangent plane transformations (ENU/NED)
3. Geodetic distance calculations
4. Multi-waypoint navigation
5. Sensor placement and coverage analysis

Run with: python examples/navigation_geodesy.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np  # noqa: E402
import plotly.graph_objects as go  # noqa: E402

from pytcl.navigation import (  # noqa: E402
    # Coordinate conversions
    geodetic_to_ecef,
    ecef_to_geodetic,
    ecef_to_enu,
    enu_to_ecef,
    # Geodetic problems
    direct_geodetic,
    inverse_geodetic,
    haversine_distance,
)


def geodetic_basics_demo() -> None:
    """Demonstrate basic geodetic coordinate conversions."""
    print("=" * 60)
    print("1. GEODETIC COORDINATE CONVERSIONS")
    print("=" * 60)

    # Key locations
    locations = {
        "Washington DC": (38.9072, -77.0369, 0.0),
        "New York City": (40.7128, -74.0060, 0.0),
        "Los Angeles": (34.0522, -118.2437, 0.0),
        "GPS Satellite (MEO)": (0.0, -75.0, 20200000.0),  # ~20,200 km altitude
    }

    print("\nGeodetic to ECEF conversions:")
    print("-" * 60)

    for name, (lat_deg, lon_deg, alt) in locations.items():
        lat = np.radians(lat_deg)
        lon = np.radians(lon_deg)

        # Convert to ECEF
        ecef = geodetic_to_ecef(lat, lon, alt)

        print(f"\n{name}:")
        print(f"  Geodetic: {lat_deg:.4f}N, {lon_deg:.4f}E, {alt:.0f} m")
        print(
            f"  ECEF: X={ecef[0]/1000:.1f} km, Y={ecef[1]/1000:.1f} km, "
            f"Z={ecef[2]/1000:.1f} km"
        )

        # Convert back
        lat_r, lon_r, alt_r = ecef_to_geodetic(ecef[0], ecef[1], ecef[2])
        print(
            f"  Roundtrip: {np.degrees(lat_r):.4f}N, {np.degrees(lon_r):.4f}E, "
            f"{alt_r:.0f} m"
        )


def distance_calculations_demo() -> None:
    """Demonstrate geodetic distance calculations."""
    print("\n" + "=" * 60)
    print("2. GEODETIC DISTANCE CALCULATIONS")
    print("=" * 60)

    # City pairs for distance calculation
    city_pairs = [
        ("Washington DC", (38.9072, -77.0369), "New York City", (40.7128, -74.0060)),
        ("Washington DC", (38.9072, -77.0369), "Los Angeles", (34.0522, -118.2437)),
        ("New York City", (40.7128, -74.0060), "London", (51.5074, -0.1278)),
        ("Los Angeles", (34.0522, -118.2437), "Tokyo", (35.6762, 139.6503)),
    ]

    print("\nGreat circle distances:")
    print("-" * 60)

    for city1, (lat1, lon1), city2, (lat2, lon2) in city_pairs:
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)

        # Haversine (approximate, fast)
        dist_haversine = haversine_distance(lat1_rad, lon1_rad, lat2_rad, lon2_rad)

        # Inverse geodetic (accurate)
        dist_geodetic, az_fwd, az_back = inverse_geodetic(
            lat1_rad, lon1_rad, lat2_rad, lon2_rad
        )

        print(f"\n{city1} -> {city2}:")
        print(f"  Haversine distance: {dist_haversine/1000:.1f} km")
        print(f"  Geodetic distance:  {dist_geodetic/1000:.1f} km")
        print(f"  Forward azimuth:    {np.degrees(az_fwd):.1f} deg")
        print(f"  Back azimuth:       {np.degrees(az_back):.1f} deg")


def local_frame_demo() -> None:
    """Demonstrate local tangent plane (ENU) conversions."""
    print("\n" + "=" * 60)
    print("3. LOCAL TANGENT PLANE (ENU) FRAME")
    print("=" * 60)

    # Reference point: Washington DC
    ref_lat = np.radians(38.9072)
    ref_lon = np.radians(-77.0369)
    ref_alt = 0.0

    print("\nReference point: Washington DC")
    print(f"  Lat: {np.degrees(ref_lat):.4f} deg")
    print(f"  Lon: {np.degrees(ref_lon):.4f} deg")

    # Define points relative to reference in ENU
    enu_points = {
        "10 km North": np.array([0.0, 10000.0, 0.0]),
        "10 km East": np.array([10000.0, 0.0, 0.0]),
        "10 km NE, 500m Up": np.array([7071.0, 7071.0, 500.0]),
        "Aircraft overhead (10 km)": np.array([0.0, 0.0, 10000.0]),
    }

    print("\nENU to Geodetic conversions:")
    print("-" * 60)

    for name, enu in enu_points.items():
        # Convert ENU to ECEF (separate components)
        x, y, z = enu_to_ecef(enu[0], enu[1], enu[2], ref_lat, ref_lon, ref_alt)

        # Convert ECEF to geodetic
        lat, lon, alt = ecef_to_geodetic(x, y, z)

        print(f"\n{name}:")
        print(f"  ENU: E={enu[0]:.0f} m, N={enu[1]:.0f} m, U={enu[2]:.0f} m")
        print(
            f"  Geodetic: {np.degrees(lat):.4f}N, {np.degrees(lon):.4f}E, "
            f"{alt:.0f} m"
        )

        # Verify roundtrip
        e_r, n_r, u_r = ecef_to_enu(x, y, z, ref_lat, ref_lon, ref_alt)
        enu_recovered = np.array([e_r, n_r, u_r])
        error = np.linalg.norm(enu - enu_recovered)
        print(f"  Roundtrip error: {error:.2e} m")


def waypoint_navigation_demo() -> None:
    """Demonstrate waypoint-to-waypoint navigation."""
    print("\n" + "=" * 60)
    print("4. WAYPOINT NAVIGATION")
    print("=" * 60)

    # Define a flight path with waypoints
    waypoints = [
        ("DCA (Reagan Airport)", 38.8521, -77.0377),
        ("Waypoint 1", 39.5, -76.5),
        ("Waypoint 2", 40.0, -75.5),
        ("JFK Airport", 40.6413, -73.7781),
    ]

    print("\nFlight path waypoints:")
    print("-" * 60)

    total_distance = 0.0
    for i, (name, lat, lon) in enumerate(waypoints):
        print(f"\n{i+1}. {name}: {lat:.4f}N, {lon:.4f}E")

        if i > 0:
            # Calculate leg distance and heading
            prev_name, prev_lat, prev_lon = waypoints[i - 1]
            lat1_rad = np.radians(prev_lat)
            lon1_rad = np.radians(prev_lon)
            lat2_rad = np.radians(lat)
            lon2_rad = np.radians(lon)

            dist, az_fwd, _ = inverse_geodetic(lat1_rad, lon1_rad, lat2_rad, lon2_rad)
            total_distance += dist

            print(f"   From {prev_name}:")
            heading = np.degrees(az_fwd)
            print(f"   Distance: {dist/1000:.1f} km, Heading: {heading:.1f} deg")

    print(f"\nTotal flight distance: {total_distance/1000:.1f} km")

    # Compute intermediate points along each leg using direct geodetic
    print("\nIntermediate points along first leg (every 10 km):")
    print("-" * 60)

    lat1 = np.radians(waypoints[0][1])
    lon1 = np.radians(waypoints[0][2])
    lat2 = np.radians(waypoints[1][1])
    lon2 = np.radians(waypoints[1][2])

    leg_dist, az_fwd, _ = inverse_geodetic(lat1, lon1, lat2, lon2)

    for d in np.arange(0, leg_dist, 10000):  # Every 10 km
        lat_int, lon_int = direct_geodetic(lat1, lon1, az_fwd, d)
        print(
            f"  {d/1000:.0f} km: {np.degrees(lat_int):.4f}N, "
            f"{np.degrees(lon_int):.4f}E"
        )


def sensor_coverage_demo() -> None:
    """Demonstrate sensor placement and coverage analysis."""
    print("\n" + "=" * 60)
    print("5. SENSOR COVERAGE ANALYSIS")
    print("=" * 60)

    # Radar sensor location (Washington DC)
    sensor_alt = 50.0  # 50m tower

    # Sensor parameters
    max_range = 100000.0  # 100 km
    min_elevation = np.radians(2.0)  # 2 degree minimum elevation

    print("\nRadar sensor location: Washington DC")
    print(f"  Height: {sensor_alt} m")
    print(f"  Max range: {max_range/1000:.0f} km")
    print(f"  Min elevation: {np.degrees(min_elevation):.1f} deg")

    # Calculate coverage at different altitudes
    print("\nCoverage radius at different target altitudes:")
    print("-" * 60)

    target_altitudes_m = [alt * 0.3048 for alt in [1000, 5000, 10000, 20000, 40000]]

    for alt_m in target_altitudes_m:
        # Height difference
        delta_h = alt_m - sensor_alt

        # Maximum slant range is either max_range or limited by min elevation
        # At min elevation, slant range r = delta_h / sin(min_el)
        range_elev_limited = delta_h / np.sin(min_elevation) if delta_h > 0 else 0
        effective_range = min(max_range, range_elev_limited)

        # Ground range
        if effective_range > 0:
            ground_range = np.sqrt(effective_range**2 - delta_h**2)
        else:
            ground_range = 0

        print(f"  Target at {alt_m:.0f} m ({alt_m/0.3048:.0f} ft):")
        print(f"    Effective slant range: {effective_range/1000:.1f} km")
        print(f"    Ground coverage radius: {ground_range/1000:.1f} km")

    # Check if specific targets are in coverage
    print("\nTarget detection check:")
    print("-" * 60)

    targets = [
        ("Aircraft 50km E, 10km alt", 50000.0, 0.0, 10000.0),
        ("Aircraft 80km NE, 5km alt", 56569.0, 56569.0, 5000.0),
        ("Low flyer 30km N, 100m alt", 0.0, 30000.0, 100.0),
        ("High alt 120km W, 20km alt", -120000.0, 0.0, 20000.0),
    ]

    for name, e, n, u in targets:
        enu = np.array([e, n, u - sensor_alt])
        slant_range = np.linalg.norm(enu)
        elevation = np.arcsin(enu[2] / slant_range) if slant_range > 0 else 0

        in_range = slant_range <= max_range
        above_horizon = elevation >= min_elevation
        detectable = in_range and above_horizon

        status = "DETECTABLE" if detectable else "NOT DETECTABLE"
        reason = []
        if not in_range:
            reason.append(f"range {slant_range/1000:.1f} km > {max_range/1000:.0f} km")
        if not above_horizon:
            reason.append(
                f"elev {np.degrees(elevation):.1f} deg < "
                f"{np.degrees(min_elevation):.1f} deg"
            )

        print(f"\n  {name}:")
        print(
            f"    Range: {slant_range/1000:.1f} km, "
            f"Elevation: {np.degrees(elevation):.1f} deg"
        )
        print(f"    Status: {status}")
        if reason:
            print(f"    Reason: {', '.join(reason)}")


def plot_coverage_map() -> None:
    """Create an interactive coverage map."""
    print("\n" + "=" * 60)
    print("6. GENERATING COVERAGE MAP")
    print("=" * 60)

    # Sensor location
    sensor_lat = np.radians(38.9072)
    sensor_lon = np.radians(-77.0369)
    sensor_alt = 50.0
    max_range = 100000.0

    # Generate coverage circle points
    n_points = 72
    azimuths = np.linspace(0, 2 * np.pi, n_points)

    # Coverage at different altitudes
    altitudes = [1000, 5000, 10000]  # meters
    colors = ["green", "blue", "red"]

    fig = go.Figure()

    # Add sensor location
    fig.add_trace(
        go.Scattergeo(
            lon=[np.degrees(sensor_lon)],
            lat=[np.degrees(sensor_lat)],
            mode="markers",
            marker=dict(size=15, color="black", symbol="triangle-up"),
            name="Radar Sensor",
        )
    )

    # Add coverage circles for each altitude
    for alt, color in zip(altitudes, colors):
        # Calculate ground range for this altitude
        delta_h = alt - sensor_alt
        min_elev = np.radians(2.0)
        range_elev_limited = delta_h / np.sin(min_elev)
        effective_range = min(max_range, range_elev_limited)
        ground_range = np.sqrt(max(0, effective_range**2 - delta_h**2))

        # Generate circle points
        lats = []
        lons = []
        for az in azimuths:
            lat, lon = direct_geodetic(sensor_lat, sensor_lon, az, ground_range)
            lats.append(np.degrees(lat))
            lons.append(np.degrees(lon))

        # Close the circle
        lats.append(lats[0])
        lons.append(lons[0])

        fig.add_trace(
            go.Scattergeo(
                lon=lons,
                lat=lats,
                mode="lines",
                line=dict(width=2, color=color),
                name=f"Coverage at {alt}m ({alt*3.28084:.0f}ft)",
            )
        )

    fig.update_layout(
        title="Radar Coverage Map (Washington DC)",
        geo=dict(
            scope="usa",
            center=dict(lat=np.degrees(sensor_lat), lon=np.degrees(sensor_lon)),
            projection_scale=5,
            showland=True,
            landcolor="rgb(243, 243, 243)",
            countrycolor="rgb(204, 204, 204)",
        ),
        width=900,
        height=700,
    )

    fig.write_html("navigation_coverage_map.html")
    print("\nInteractive coverage map saved to navigation_coverage_map.html")
    fig.show()


def main() -> None:
    """Run navigation and geodesy demonstrations."""
    print("\nNavigation and Geodesy Examples")
    print("=" * 60)
    print("Demonstrating pytcl navigation capabilities")

    geodetic_basics_demo()
    distance_calculations_demo()
    local_frame_demo()
    waypoint_navigation_demo()
    sensor_coverage_demo()

    try:
        plot_coverage_map()
    except Exception as e:
        print(f"\nCould not generate coverage map: {e}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
