"""
Advanced Reference Frame Transformations

This example demonstrates advanced reference frame transformations including:
- PEF (Pseudo-Earth Fixed) frame for intermediate processing
- SEZ (South-East-Zenith) horizon-relative frame for observations
- Earth observation and antenna pointing applications

The reference frame transformation chain:
  GCRF (inertial) -> MOD (precession) -> TOD (nutation) -> PEF (rotation)
                                                              |
                                                              v
                                                           ITRF (Earth-fixed)
                                                              ^
                                                              |
                                                        (+ polar motion)

SEZ is useful for:
- Radar and antenna azimuth/elevation calculations
- Line-of-sight observations
- Sensor target tracking from ground stations
"""

import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp

from pytcl.astronomical.reference_frames import (
    gcrf_to_itrf,
    gcrf_to_pef,
    itrf_to_gcrf,
    pef_to_gcrf,
)
from pytcl.astronomical.time_systems import JD_J2000, cal_to_jd
from pytcl.coordinate_systems.conversions.geodetic import (
    ecef2geodetic,
    geodetic2ecef,
    geodetic2sez,
    sez2geodetic,
)


def example_pef_intermediate_frame():
    """
    Demonstrate PEF as an intermediate frame between GCRF and ITRF.
    
    PEF excludes polar motion compared to ITRF, making it useful for:
    - Intermediate calculations
    - Separating Earth rotation from polar motion effects
    - Legacy systems and comparisons
    """
    print("=" * 80)
    print("Example 1: PEF Intermediate Frame")
    print("=" * 80)
    
    # Sample position in GCRF (e.g., geostationary satellite)
    r_gcrf = np.array([42164.0, 0.0, 0.0])  # GEO orbit at Earth equator, km
    
    # Date: 2024-01-01 12:00:00 UTC
    year, month, day, hour, minute, second = 2024, 1, 1, 12, 0, 0
    jd_utc = cal_to_jd(year, month, day, hour, minute, second)
    jd_ut1 = jd_utc  # Simplified (should account for UT1 - UTC offset)
    jd_tt = jd_ut1 + 32.184 / 86400  # TT = UT1 + 32.184 seconds
    
    # Polar motion parameters (for example)
    xp = 0.0001  # ~0.01 arcseconds
    yp = -0.0001
    
    # Transform GCRF -> PEF (no polar motion)
    r_pef = gcrf_to_pef(r_gcrf, jd_ut1, jd_tt)
    
    # Transform GCRF -> ITRF (includes polar motion)
    r_itrf = gcrf_to_itrf(r_gcrf, jd_ut1, jd_tt, xp, yp)
    
    print(f"Position in GCRF:  {r_gcrf}")
    print(f"Position in PEF:   {r_pef}")
    print(f"Position in ITRF:  {r_itrf}")
    
    # Polar motion effect
    polar_motion_effect = np.linalg.norm(r_itrf - r_pef)
    print(f"\nPolar motion effect (PEF vs ITRF difference): {polar_motion_effect:.4f} km")
    print(f"Relative effect: {100 * polar_motion_effect / np.linalg.norm(r_gcrf):.6f}%")
    
    # Verify roundtrip
    r_gcrf_back = pef_to_gcrf(r_pef, jd_ut1, jd_tt)
    roundtrip_error = np.linalg.norm(r_gcrf_back - r_gcrf)
    print(f"\nRoundtrip error (GCRF -> PEF -> GCRF): {roundtrip_error:.2e} km")


def example_sez_radar_observations():
    """
    Demonstrate SEZ frame for radar observations and antenna targeting.
    
    Application: Ground-based radar observing a satellite
    """
    print("\n" + "=" * 80)
    print("Example 2: SEZ Frame for Radar Observations")
    print("=" * 80)
    
    # Ground station (Longitude, Latitude, Altitude)
    station_name = "Tracking Station"
    lat_station = np.radians(40.0)  # 40° N
    lon_station = np.radians(-105.0)  # 105° W (Colorado)
    alt_station = 1655.0  # meters (Denver area)
    
    print(f"\n{station_name}:")
    print(f"  Latitude:  {np.degrees(lat_station):.2f}°")
    print(f"  Longitude: {np.degrees(lon_station):.2f}°")
    print(f"  Altitude:  {alt_station:.0f} m")
    
    # Satellite position (example: LEO satellite)
    # Position in geodetic coordinates
    lat_satellite = np.radians(42.5)
    lon_satellite = np.radians(-103.5)
    alt_satellite = 400_000  # 400 km altitude
    
    # Convert satellite position to SEZ relative to station
    sez_position = geodetic2sez(
        lat_satellite, lon_satellite, alt_satellite,
        lat_station, lon_station, alt_station
    )
    
    print(f"\nSatellite position in SEZ:")
    print(f"  South component:  {sez_position[0]/1000:8.2f} km")
    print(f"  East component:   {sez_position[1]/1000:8.2f} km")
    print(f"  Zenith component: {sez_position[2]/1000:8.2f} km")
    
    # Compute range, azimuth, elevation
    range_km = np.linalg.norm(sez_position) / 1000
    
    # Azimuth: 0° = South, 90° = East, 180° = North, 270° = West
    azimuth = np.degrees(np.arctan2(sez_position[1], sez_position[0]))
    if azimuth < 0:
        azimuth += 360
    
    # Elevation: angle above horizon
    horizontal_distance = np.sqrt(sez_position[0]**2 + sez_position[1]**2)
    elevation = np.degrees(np.arctan2(sez_position[2], horizontal_distance))
    
    print(f"\nRadar observation parameters:")
    print(f"  Range:     {range_km:8.2f} km")
    print(f"  Azimuth:   {azimuth:8.2f}°")
    print(f"  Elevation: {elevation:8.2f}°")
    
    # Check if satellite is above horizon (elevation > 0)
    is_visible = elevation > 0
    print(f"  Visible:   {'Yes' if is_visible else 'No'} (elevation {'above' if is_visible else 'below'} horizon)")


def example_leo_satellite_tracking():
    """
    Demonstrate tracking a LEO satellite through multiple observation passes.
    
    Shows how azimuth/elevation evolve as the satellite passes overhead.
    """
    print("\n" + "=" * 80)
    print("Example 3: LEO Satellite Pass Over Tracking Station")
    print("=" * 80)
    
    # Tracking station (Denver)
    lat_station = np.radians(39.74)  # Denver, CO
    lon_station = np.radians(-104.99)
    alt_station = 1609.0  # meters
    
    # LEO satellite orbital parameters (example)
    # ISS-like orbit: 51.6° inclination, ~400 km altitude
    inclination = 51.6  # degrees
    altitude = 408_000  # meters
    
    # Simulate satellite positions along its ground track
    # (This is simplified; real implementation would propagate orbit)
    orbital_period_minutes = 90  # approximately
    
    # Create a pass: satellite starting from horizon, reaching max elevation, to horizon
    # Use a parametric representation along the pass
    
    num_points = 50
    t_pass = np.linspace(0, orbital_period_minutes, num_points)
    
    # Simplified ground track: satellite moves from SW to NE
    lat_pass = np.degrees(lat_station) + (np.linspace(-5, 5, num_points))
    lon_pass = np.degrees(lon_station) + (np.linspace(-5, 5, num_points))
    
    azimuth_pass = []
    elevation_pass = []
    range_pass = []
    
    for lat_sat, lon_sat in zip(lat_pass, lon_pass):
        lat_sat_rad = np.radians(lat_sat)
        lon_sat_rad = np.radians(lon_sat)
        
        sez = geodetic2sez(
            lat_sat_rad, lon_sat_rad, altitude,
            lat_station, lon_station, alt_station
        )
        
        # Range
        rng = np.linalg.norm(sez) / 1000
        range_pass.append(rng)
        
        # Azimuth
        az = np.degrees(np.arctan2(sez[1], sez[0]))
        if az < 0:
            az += 360
        azimuth_pass.append(az)
        
        # Elevation
        horiz = np.sqrt(sez[0]**2 + sez[1]**2)
        el = np.degrees(np.arctan2(sez[2], horiz))
        elevation_pass.append(el)
    
    # Print pass summary
    max_el_idx = np.argmax(elevation_pass)
    max_elevation = elevation_pass[max_el_idx]
    azimuth_at_max = azimuth_pass[max_el_idx]
    min_range = range_pass[max_el_idx]
    
    print(f"\nSatellite pass over {np.degrees(lat_station):.2f}°, {np.degrees(lon_station):.2f}°:")
    print(f"  Maximum elevation: {max_elevation:.2f}°")
    print(f"  Azimuth at max el: {azimuth_at_max:.2f}°")
    print(f"  Minimum range:     {min_range:.2f} km")
    
    # Determine horizon crossings
    horizon_points = [(el, az, rng) for el, az, rng in zip(elevation_pass, azimuth_pass, range_pass) if abs(el) < 1]
    if horizon_points:
        print(f"  Horizon crossing points: {len(horizon_points)}")
    
    # Plot the pass using Plotly
    fig = sp.make_subplots(
        rows=2, cols=2,
        subplot_titles=("Elevation Angle", "Azimuth Angle", "Slant Range", "Ground Station View"),
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "scatterpolar"}]]
    )
    
    # Elevation vs time
    fig.add_trace(
        go.Scatter(x=t_pass, y=elevation_pass, mode='lines', name='Elevation',
                   line=dict(color='blue', width=2)),
        row=1, col=1
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
    
    # Azimuth vs time
    fig.add_trace(
        go.Scatter(x=t_pass, y=azimuth_pass, mode='lines', name='Azimuth',
                   line=dict(color='red', width=2)),
        row=1, col=2
    )
    
    # Range vs time
    fig.add_trace(
        go.Scatter(x=t_pass, y=range_pass, mode='lines', name='Range',
                   line=dict(color='green', width=2)),
        row=2, col=1
    )
    
    # Azimuth/Elevation polar plot
    fig.add_trace(
        go.Scatterpolar(
            r=90 - np.array(elevation_pass),
            theta=azimuth_pass,
            mode='lines',
            name='Satellite Pass',
            line=dict(color='blue', width=2),
            fill='toself',
            fillcolor='rgba(0, 100, 200, 0.1)'
        ),
        row=2, col=2
    )
    
    # Mark start, peak, and end on polar plot
    fig.add_trace(
        go.Scatterpolar(
            r=[90 - elevation_pass[0]],
            theta=[azimuth_pass[0]],
            mode='markers',
            name='Start',
            marker=dict(size=10, color='green'),
            showlegend=False
        ),
        row=2, col=2
    )
    
    fig.add_trace(
        go.Scatterpolar(
            r=[90 - elevation_pass[max_el_idx]],
            theta=[azimuth_pass[max_el_idx]],
            mode='markers',
            name='Max Elevation',
            marker=dict(size=15, color='red', symbol='star'),
            showlegend=False
        ),
        row=2, col=2
    )
    
    fig.add_trace(
        go.Scatterpolar(
            r=[90 - elevation_pass[-1]],
            theta=[azimuth_pass[-1]],
            mode='markers',
            name='End',
            marker=dict(size=10, color='red', symbol='x'),
            showlegend=False
        ),
        row=2, col=2
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Time in Pass (min)", row=1, col=1)
    fig.update_yaxes(title_text="Elevation (deg)", row=1, col=1)
    
    fig.update_xaxes(title_text="Time in Pass (min)", row=1, col=2)
    fig.update_yaxes(title_text="Azimuth (deg)", row=1, col=2)
    
    fig.update_xaxes(title_text="Time in Pass (min)", row=2, col=1)
    fig.update_yaxes(title_text="Range (km)", row=2, col=1)
    
    # Update polar plot
    fig.update_polars(
        radialaxis=dict(range=[0, 90], ticksuffix="°"),
        angularaxis=dict(tickprefix="", ticksuffix="°"),
        row=2, col=2
    )
    
    fig.update_layout(
        title_text="LEO Satellite Pass Tracking",
        height=800,
        width=1200,
        hovermode='closest'
    )
    
    fig.write_html('leo_satellite_pass.html')
    print(f"\nPlot saved as 'leo_satellite_pass.html'")
    
    return fig


def example_earth_observation():
    """
    Demonstrate Earth observation planning using SEZ frame.
    
    Application: Planning satellite imagery collection from different ground stations
    """
    print("\n" + "=" * 80)
    print("Example 4: Earth Observation Geometry")
    print("=" * 80)
    
    # Ground stations (different locations)
    stations = [
        ("Hawaii", np.radians(20.8), np.radians(-156.5), 3000),
        ("Colorado", np.radians(40.0), np.radians(-105.0), 1500),
        ("Florida", np.radians(28.5), np.radians(-80.5), 0),
    ]
    
    # Target on Earth surface (e.g., geographic point of interest)
    target_lat = np.radians(35.0)  # 35° N
    target_lon = np.radians(-95.0)  # 95° W
    target_alt = 300.0  # meters (ground elevation)
    
    # Observer in space (satellite)
    observer_lat = np.radians(35.5)
    observer_lon = np.radians(-94.5)
    observer_alt = 800_000  # 800 km altitude
    
    print(f"\nObserver satellite:")
    print(f"  Position: {np.degrees(observer_lat):.2f}°N, {np.degrees(observer_lon):.2f}°W")
    print(f"  Altitude: {observer_alt/1000:.0f} km")
    
    print(f"\nTarget location: {np.degrees(target_lat):.2f}°N, {np.degrees(target_lon):.2f}°W")
    
    print(f"\nObservation feasibility from different ground stations:")
    print(f"{'Station':<12} {'Lat':<8} {'Lon':<8} {'El to Sat':<12} {'Visible?':<10}")
    print("-" * 52)
    
    for station_name, station_lat, station_lon, station_alt in stations:
        # Find elevation angle from station to satellite
        sez_to_sat = geodetic2sez(
            observer_lat, observer_lon, observer_alt,
            station_lat, station_lon, station_alt
        )
        
        horiz = np.sqrt(sez_to_sat[0]**2 + sez_to_sat[1]**2)
        elevation_to_sat = np.degrees(np.arctan2(sez_to_sat[2], horiz))
        
        is_visible = elevation_to_sat > 0
        
        print(f"{station_name:<12} {np.degrees(station_lat):>7.2f}° {np.degrees(station_lon):>7.2f}° " +
              f"{elevation_to_sat:>11.2f}° {('Yes' if is_visible else 'No'):<10}")
    
    print("\nConclusion: Ground stations can track satellite if elevation > 0°")


def main():
    """Run all examples."""
    print("\n")
    print("█" * 80)
    print("█ Advanced Reference Frame Transformations - pytcl Examples")
    print("█" * 80)
    
    example_pef_intermediate_frame()
    example_sez_radar_observations()
    example_leo_satellite_tracking()
    example_earth_observation()
    
    print("\n" + "=" * 80)
    print("All examples completed successfully!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
