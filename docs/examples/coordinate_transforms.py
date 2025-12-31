#!/usr/bin/env python3
"""
Coordinate Transform Example
============================

Demonstrate geodetic coordinate conversions.
"""

import numpy as np

from pytcl.navigation import (
    WGS84,
    ecef_to_enu,
    ecef_to_geodetic,
    enu_to_ecef,
    geodetic_to_ecef,
    haversine_distance,
    direct_geodetic,
    inverse_geodetic,
)


def main():
    print("Coordinate Transform Examples")
    print("=" * 50)

    # Define a reference point (San Francisco)
    lat_ref = np.radians(37.7749)
    lon_ref = np.radians(-122.4194)
    alt_ref = 0.0

    print(f"\nReference point (San Francisco):")
    print(f"  Latitude:  {np.degrees(lat_ref):.4f}°")
    print(f"  Longitude: {np.degrees(lon_ref):.4f}°")
    print(f"  Altitude:  {alt_ref:.1f} m")

    # Convert to ECEF
    x, y, z = geodetic_to_ecef(lat_ref, lon_ref, alt_ref, WGS84)
    print(f"\nECEF coordinates:")
    print(f"  X: {x / 1e6:.6f} Mm")
    print(f"  Y: {y / 1e6:.6f} Mm")
    print(f"  Z: {z / 1e6:.6f} Mm")

    # Convert back to geodetic
    lat, lon, alt = ecef_to_geodetic(x, y, z, WGS84)
    print(f"\nBack to geodetic:")
    print(f"  Latitude:  {np.degrees(lat):.4f}°")
    print(f"  Longitude: {np.degrees(lon):.4f}°")
    print(f"  Altitude:  {alt:.6f} m")

    # Define a target point (Los Angeles)
    lat_tgt = np.radians(34.0522)
    lon_tgt = np.radians(-118.2437)
    alt_tgt = 0.0

    x_tgt, y_tgt, z_tgt = geodetic_to_ecef(lat_tgt, lon_tgt, alt_tgt, WGS84)

    # Convert to ENU relative to reference
    e, n, u = ecef_to_enu(x_tgt, y_tgt, z_tgt, lat_ref, lon_ref, alt_ref, WGS84)
    print(f"\nLos Angeles in ENU (relative to SF):")
    print(f"  East:  {e / 1e3:.2f} km")
    print(f"  North: {n / 1e3:.2f} km")
    print(f"  Up:    {u:.2f} m")

    # Haversine distance
    dist = haversine_distance(lat_ref, lon_ref, lat_tgt, lon_tgt)
    print(f"\nHaversine distance SF to LA: {dist / 1e3:.2f} km")

    # Inverse geodetic (Vincenty)
    result = inverse_geodetic(lat_ref, lon_ref, lat_tgt, lon_tgt, WGS84)
    print(f"\nVincenty geodesic:")
    print(f"  Distance: {result.distance / 1e3:.3f} km")
    print(f"  Forward azimuth:  {np.degrees(result.azimuth1):.2f}°")
    print(f"  Backward azimuth: {np.degrees(result.azimuth2):.2f}°")

    # Direct geodetic problem
    # Start from SF, travel 500 km at 135° azimuth
    distance = 500e3  # 500 km
    azimuth = np.radians(135)  # Southeast

    result = direct_geodetic(lat_ref, lon_ref, azimuth, distance, WGS84)
    print(f"\nDirect geodetic (500 km at 135°):")
    print(
        f"  Destination: {np.degrees(result.latitude):.4f}°, "
        f"{np.degrees(result.longitude):.4f}°"
    )
    print(f"  Final azimuth: {np.degrees(result.azimuth):.2f}°")


if __name__ == "__main__":
    main()
