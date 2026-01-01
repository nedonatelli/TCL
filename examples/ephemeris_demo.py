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

import numpy as np
import matplotlib.pyplot as plt
from pytcl.astronomical import (
    DEEphemeris,
    sun_position,
    moon_position,
    planet_position,
    barycenter_position,
    utc_to_jd,
    jd_to_cal,
    AU,
)


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
    print(f"Variation:         {max(distances) - min(distances):.6f} AU "
          f"({100*(max(distances)-min(distances))/np.mean(distances):.2f}%)")


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
    print(f"  Speed: {np.linalg.norm(v_moon):.3f} m/s = {np.linalg.norm(v_moon)*86400/1e3:.1f} km/day")
    
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
    print(f"Variation:          {max(distances) - min(distances):.1f} km "
          f"({100*(max(distances)-min(distances))/np.mean(distances):.1f}%)")


def example_planet_positions():
    """Query positions of all major planets."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Planetary Positions")
    print("=" * 70)
    
    jd_j2000 = 2451545.0
    
    planets = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']
    planet_ids = [1, 2, 3, 4, 5, 6, 7, 8]
    
    print(f"\nPlanetary Heliocentric Positions at J2000.0:")
    print("-" * 70)
    print(f"{'Planet':<10} {'Distance (AU)':<16} {'Longitude':<12} {'Latitude':<12}")
    print("-" * 70)
    
    for planet_name, planet_id in zip(planets, planet_ids):
        try:
            r, _ = planet_position(jd_j2000, planet_id)
            dist_au = np.linalg.norm(r) / AU
            
            # Compute ecliptic coordinates
            lon = np.arctan2(r[1], r[0])
            lat = np.arcsin(r[2] / np.linalg.norm(r))
            
            print(f"{planet_name:<10} {dist_au:<16.6f} {np.degrees(lon):>10.2f}° {np.degrees(lat):>10.2f}°")
        except Exception as e:
            print(f"{planet_name:<10} [Error: {str(e)}]")


def example_barycenter():
    """Compute solar system barycenter positions."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Solar System Barycenter")
    print("=" * 70)
    
    jd_j2000 = 2451545.0
    
    # Get barycenter positions (body 0 = solar system barycenter)
    r_barycenter, v_barycenter = barycenter_position(jd_j2000, 0)
    
    print(f"\nSolar System Barycenter at J2000.0:")
    print(f"  X: {r_barycenter[0]:12.3f} m")
    print(f"  Y: {r_barycenter[1]:12.3f} m")
    print(f"  Z: {r_barycenter[2]:12.3f} m")
    print(f"  Distance from origin: {np.linalg.norm(r_barycenter):.3f} m")
    print(f"  Velocity magnitude: {np.linalg.norm(v_barycenter):.6f} m/s")
    
    # Compare with Sun position
    r_sun, _ = planet_position(jd_j2000, 11)  # Sun
    print(f"\nComparison with Sun position:")
    print(f"  Sun distance from barycenter: {np.linalg.norm(r_sun - r_barycenter):.3f} m")
    print(f"  This offset represents Jupiter's gravitational influence")


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
    r_icrf, _ = eph.sun_position(jd_j2000, frame='ICRF')
    print(f"ICRF Frame (International Celestial Reference Frame):")
    print(f"  X: {r_icrf[0]/AU:10.6f} AU")
    print(f"  Y: {r_icrf[1]/AU:10.6f} AU")
    print(f"  Z: {r_icrf[2]/AU:10.6f} AU")
    
    # Ecliptic frame
    try:
        r_ecliptic, _ = eph.sun_position(jd_j2000, frame='ecliptic')
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
    print(f"  Min distance: {np.min(np.linalg.norm(moon_positions, axis=1))/1e6:.1f} km")
    print(f"  Max distance: {np.max(np.linalg.norm(moon_positions, axis=1))/1e6:.1f} km")


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
    
    eph = DEEphemeris(version='DE440')
    r_sun, _ = eph.sun_position(jd_test)
    print(f"\nSun position (DE440 at J2000.0): {np.linalg.norm(r_sun)/AU:.6f} AU")


if __name__ == '__main__':
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
