"""
Orbital Mechanics Example
=========================

This example demonstrates the orbital mechanics and astronomical
algorithms in PyTCL:

Kepler's Problem:
- Mean, eccentric, and true anomaly conversions
- Orbit propagation
- Orbital elements and state vector conversions

Orbital Quantities:
- Period, mean motion, vis-viva equation
- Specific angular momentum and energy
- Periapsis and apoapsis radii

Lambert's Problem:
- Two-point boundary value orbit determination
- Transfer orbit design
- Hohmann and bi-elliptic transfers

Time Systems:
- Julian date conversions
- UTC, TAI, GPS time conversions
- Sidereal time

Reference Frames:
- GCRF/ITRF transformations
- Precession and nutation

These algorithms are essential for spacecraft trajectory design,
orbit determination, and space situational awareness.
"""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go

# Output directory for generated plots
OUTPUT_DIR = Path(__file__).parent.parent / "docs" / "_static" / "images" / "examples"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Global flag to control plotting
SHOW_PLOTS = True


from pytcl.astronomical import (  # Orbital elements; Kepler's equation; Orbit propagation; Orbital quantities; Lambert problem; Gravitational parameters; Time systems; Reference frames
    GM_EARTH,
    GM_SUN,
    OrbitalElements,
    StateVector,
    apoapsis_radius,
    cal_to_jd,
    circular_velocity,
    eccentric_to_true_anomaly,
    escape_velocity,
    flight_path_angle,
    gcrf_to_itrf,
    gmst,
    hohmann_transfer,
    itrf_to_gcrf,
    jd_to_cal,
    kepler_propagate,
    kepler_propagate_state,
    lambert_izzo,
    lambert_universal,
    mean_motion,
    mean_to_eccentric_anomaly,
    mean_to_true_anomaly,
    minimum_energy_transfer,
    nutation_matrix,
    orbit_radius,
    orbital_elements_to_state,
    orbital_period,
    periapsis_radius,
    precession_matrix_iau76,
    specific_angular_momentum,
    specific_orbital_energy,
    state_to_orbital_elements,
    true_to_eccentric_anomaly,
    utc_to_gps,
    utc_to_tai,
    vis_viva,
)


def demo_orbital_elements():
    """Demonstrate orbital elements and conversions."""
    print("=" * 70)
    print("Orbital Elements Demo")
    print("=" * 70)

    # Define an orbit using classical orbital elements
    # ISS-like orbit
    a = 6778.0  # Semi-major axis (km) - ~400 km altitude
    e = 0.0001  # Eccentricity (nearly circular)
    i = np.radians(51.6)  # Inclination
    raan = np.radians(0.0)  # Right ascension of ascending node
    omega = np.radians(0.0)  # Argument of periapsis
    nu = np.radians(0.0)  # True anomaly

    elements = OrbitalElements(a=a, e=e, i=i, raan=raan, omega=omega, nu=nu)

    print("\nISS-like orbit (orbital elements):")
    print(f"  Semi-major axis: {a:.1f} km")
    print(f"  Eccentricity: {e:.4f}")
    print(f"  Inclination: {np.degrees(i):.1f} deg")
    print(f"  RAAN: {np.degrees(raan):.1f} deg")
    print(f"  Arg. of periapsis: {np.degrees(omega):.1f} deg")
    print(f"  True anomaly: {np.degrees(nu):.1f} deg")

    # Convert to state vector
    state = orbital_elements_to_state(elements, GM_EARTH)

    print("\nState vector (ECI frame):")
    print(f"  Position: ({state.r[0]:.3f}, {state.r[1]:.3f}, {state.r[2]:.3f}) km")
    print(f"  Velocity: ({state.v[0]:.3f}, {state.v[1]:.3f}, {state.v[2]:.3f}) km/s")

    # Compute orbital quantities
    T = orbital_period(a, GM_EARTH)
    n = mean_motion(a, GM_EARTH)
    v_circ = circular_velocity(a, GM_EARTH)
    v_esc = escape_velocity(a, GM_EARTH)

    print("\nOrbital quantities:")
    print(f"  Period: {T:.1f} s ({T/60:.1f} min)")
    print(f"  Mean motion: {n*86400/(2*np.pi):.2f} rev/day")
    print(f"  Circular velocity: {v_circ:.3f} km/s")
    print(f"  Escape velocity: {v_esc:.3f} km/s")

    # Convert back and verify
    elements_back = state_to_orbital_elements(state, GM_EARTH)
    print(f"\nRoundtrip conversion check:")
    print(f"  a difference: {abs(elements_back.a - a):.6f} km")
    print(f"  e difference: {abs(elements_back.e - e):.9f}")


def demo_kepler_equation():
    """Demonstrate Kepler's equation and anomaly conversions."""
    print("\n" + "=" * 70)
    print("Kepler's Equation Demo")
    print("=" * 70)

    # Elliptical orbit
    e = 0.5  # Moderate eccentricity

    print(f"\nAnomaly conversions for e = {e}:")
    print("-" * 50)
    print(f"{'M (deg)':>10} {'E (deg)':>10} {'nu (deg)':>10}")
    print("-" * 50)

    for M_deg in [0, 30, 60, 90, 120, 150, 180]:
        M = np.radians(M_deg)
        E = mean_to_eccentric_anomaly(M, e)
        nu = eccentric_to_true_anomaly(E, e)
        print(f"{M_deg:>10.0f} {np.degrees(E):>10.2f} {np.degrees(nu):>10.2f}")

    # Show the relationship
    print("\nNote: For elliptical orbits:")
    print("  - True anomaly (nu) leads mean anomaly (M) near periapsis")
    print("  - They are equal only at periapsis and apoapsis")

    # Hyperbolic orbit example
    print("\n--- Hyperbolic Orbit ---")
    e_hyp = 1.5  # Hyperbolic

    print(f"Eccentricity: {e_hyp} (hyperbolic trajectory)")
    print("For hyperbolic orbits, only a range of true anomalies is valid:")
    nu_max = np.arccos(-1 / e_hyp)
    print(
        f"  Valid range: -{np.degrees(nu_max):.1f} deg < nu < {np.degrees(nu_max):.1f} deg"
    )


def demo_orbit_propagation():
    """Demonstrate orbit propagation."""
    print("\n" + "=" * 70)
    print("Orbit Propagation Demo")
    print("=" * 70)

    # Initial orbit (GPS satellite-like)
    a = 26560.0  # Semi-major axis (km)
    e = 0.02  # Slight eccentricity
    i = np.radians(55.0)  # Inclination
    raan = np.radians(120.0)
    omega = np.radians(45.0)
    nu0 = np.radians(0.0)

    elements0 = OrbitalElements(a=a, e=e, i=i, raan=raan, omega=omega, nu=nu0)
    state0 = orbital_elements_to_state(elements0, GM_EARTH)

    # Orbital period
    T = orbital_period(a, GM_EARTH)

    print(f"\nGPS satellite orbit:")
    print(f"  Semi-major axis: {a:.0f} km")
    print(f"  Period: {T/3600:.2f} hours (~12 hours)")

    # Propagate for one orbit
    print("\nPropagation around one orbit:")
    print("-" * 60)
    print(f"{'Time (hr)':>10} {'r (km)':>12} {'v (km/s)':>10} {'nu (deg)':>10}")
    print("-" * 60)

    for frac in [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]:
        dt = frac * T
        state = kepler_propagate_state(state0, dt, GM_EARTH)
        elements = state_to_orbital_elements(state, GM_EARTH)

        r_mag = np.linalg.norm(state.r)
        v_mag = np.linalg.norm(state.v)

        print(
            f"{dt/3600:>10.2f} {r_mag:>12.1f} {v_mag:>10.4f} "
            f"{np.degrees(elements.nu):>10.1f}"
        )

    # Verify vis-viva equation
    print("\n--- Vis-Viva Equation Check ---")
    for frac in [0, 0.25, 0.5]:
        dt = frac * T
        state = kepler_propagate_state(state0, dt, GM_EARTH)
        r = np.linalg.norm(state.r)
        v_actual = np.linalg.norm(state.v)
        v_visviva = vis_viva(r, a, GM_EARTH)
        print(
            f"  t={frac*T/3600:.1f}h: v_actual={v_actual:.4f}, "
            f"v_visviva={v_visviva:.4f} km/s"
        )

    # Plot orbit
    if SHOW_PLOTS:
        # Propagate full orbit for plotting
        n_points = 100
        positions = []
        for idx in range(n_points + 1):
            dt = idx * T / n_points
            state = kepler_propagate_state(state0, dt, GM_EARTH)
            positions.append(state.r)
        positions = np.array(positions)

        fig = go.Figure()

        # Plot orbit
        fig.add_trace(
            go.Scatter3d(
                x=positions[:, 0],
                y=positions[:, 1],
                z=positions[:, 2],
                mode="lines",
                line=dict(color="blue", width=4),
                name="Orbit",
            )
        )

        # Plot Earth (scaled for visibility)
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 20)
        earth_r = 6371  # km
        x = earth_r * np.outer(np.cos(u), np.sin(v))
        y = earth_r * np.outer(np.sin(u), np.sin(v))
        z = earth_r * np.outer(np.ones(np.size(u)), np.cos(v))

        fig.add_trace(
            go.Surface(
                x=x,
                y=y,
                z=z,
                colorscale=[[0, "blue"], [1, "blue"]],
                opacity=0.3,
                showscale=False,
                name="Earth",
            )
        )

        # Mark periapsis and apoapsis
        fig.add_trace(
            go.Scatter3d(
                x=[positions[0, 0]],
                y=[positions[0, 1]],
                z=[positions[0, 2]],
                mode="markers",
                marker=dict(color="green", size=10, symbol="circle"),
                name="Periapsis",
            )
        )

        fig.add_trace(
            go.Scatter3d(
                x=[positions[n_points // 2, 0]],
                y=[positions[n_points // 2, 1]],
                z=[positions[n_points // 2, 2]],
                mode="markers",
                marker=dict(color="red", size=10, symbol="square"),
                name="Apoapsis",
            )
        )

        # Equal aspect ratio
        max_range = np.max(np.abs(positions)) * 1.1

        fig.update_layout(
            title="GPS Satellite Orbit",
            scene=dict(
                xaxis=dict(title="X (km)", range=[-max_range, max_range]),
                yaxis=dict(title="Y (km)", range=[-max_range, max_range]),
                zaxis=dict(title="Z (km)", range=[-max_range, max_range]),
                aspectmode="cube",
            ),
            height=700,
            width=800,
            showlegend=True,
        )
        fig.write_html(str(OUTPUT_DIR / "orbital_propagation.html"))
        print("\n  [Plot saved to orbital_propagation.html]")


def demo_lambert_problem():
    """Demonstrate Lambert's problem for orbit determination."""
    print("\n" + "=" * 70)
    print("Lambert's Problem Demo")
    print("=" * 70)

    # Earth to Mars transfer (simplified)
    # Initial position: Earth at 1 AU
    r1 = np.array([1.0, 0.0, 0.0]) * 149597870.7  # km (1 AU)

    # Final position: Mars at 1.52 AU (simplified circular orbit)
    theta_mars = np.radians(135)  # 135 deg ahead
    r2 = np.array([np.cos(theta_mars), np.sin(theta_mars), 0.0]) * 1.52 * 149597870.7

    # Transfer time: approximately 259 days (Hohmann-like)
    tof = 259 * 86400  # seconds

    print("\nEarth-Mars transfer scenario:")
    print(
        f"  Departure: Earth at ({r1[0]/149597870.7:.2f}, "
        f"{r1[1]/149597870.7:.2f}, 0) AU"
    )
    print(
        f"  Arrival: Mars at ({r2[0]/149597870.7:.2f}, "
        f"{r2[1]/149597870.7:.2f}, 0) AU"
    )
    print(f"  Time of flight: {tof/86400:.0f} days")

    # Solve Lambert's problem
    solution = lambert_universal(r1, r2, tof, GM_SUN)

    print("\nLambert solution:")
    print(
        f"  Departure velocity: ({solution.v1[0]:.3f}, {solution.v1[1]:.3f}, "
        f"{solution.v1[2]:.3f}) km/s"
    )
    print(
        f"  Arrival velocity: ({solution.v2[0]:.3f}, {solution.v2[1]:.3f}, "
        f"{solution.v2[2]:.3f}) km/s"
    )
    print(f"  Transfer orbit semi-major axis: {solution.a/149597870.7:.3f} AU")
    print(f"  Transfer orbit eccentricity: {solution.e:.4f}")

    # Delta-v calculations (simplified)
    # Earth's orbital velocity
    v_earth = np.array([0, 29.78, 0])  # km/s (approximately)
    dv_departure = np.linalg.norm(solution.v1 - v_earth)

    print(f"\n  Departure delta-v: {dv_departure:.2f} km/s")


def demo_hohmann_transfer():
    """Demonstrate Hohmann transfer orbit."""
    print("\n" + "=" * 70)
    print("Hohmann Transfer Demo")
    print("=" * 70)

    # LEO to GEO transfer
    r_leo = 6678.0  # km (300 km altitude)
    r_geo = 42164.0  # km (GEO radius)

    print("\nLEO to GEO Hohmann transfer:")
    print(f"  Initial orbit (LEO): r = {r_leo:.0f} km (alt = {r_leo-6378:.0f} km)")
    print(f"  Final orbit (GEO): r = {r_geo:.0f} km (alt = {r_geo-6378:.0f} km)")

    # Velocities in circular orbits
    v_leo = circular_velocity(r_leo, GM_EARTH)
    v_geo = circular_velocity(r_geo, GM_EARTH)

    print(f"\n  LEO circular velocity: {v_leo:.3f} km/s")
    print(f"  GEO circular velocity: {v_geo:.3f} km/s")

    # Hohmann transfer orbit
    a_transfer = (r_leo + r_geo) / 2

    # Velocity at periapsis of transfer orbit (leaving LEO)
    v_transfer_peri = vis_viva(r_leo, a_transfer, GM_EARTH)

    # Velocity at apoapsis of transfer orbit (arriving at GEO)
    v_transfer_apo = vis_viva(r_geo, a_transfer, GM_EARTH)

    # Delta-v's
    dv1 = v_transfer_peri - v_leo  # Burn at LEO
    dv2 = v_geo - v_transfer_apo  # Burn at GEO

    # Transfer time (half the period)
    T_transfer = orbital_period(a_transfer, GM_EARTH)
    tof = T_transfer / 2

    print("\nTransfer orbit:")
    print(f"  Semi-major axis: {a_transfer:.0f} km")
    print(f"  Transfer time: {tof/3600:.2f} hours")

    print("\nDelta-v budget:")
    print(f"  dv1 (LEO departure): {dv1:.3f} km/s")
    print(f"  dv2 (GEO insertion): {dv2:.3f} km/s")
    print(f"  Total dv: {dv1 + dv2:.3f} km/s")


def demo_time_systems():
    """Demonstrate time system conversions."""
    print("\n" + "=" * 70)
    print("Time Systems Demo")
    print("=" * 70)

    # Current epoch (approximate)
    year, month, day = 2025, 1, 1
    hour, minute, second = 12, 0, 0.0

    # Convert to Julian Date
    jd = cal_to_jd(year, month, day, hour, minute, second)

    print(
        f"\nDate: {year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:{second:05.2f} UTC"
    )
    print(f"Julian Date: {jd:.6f}")

    # Convert back
    y, mo, d, h, mi, s = jd_to_cal(jd)
    print(
        f"Roundtrip: {int(y)}-{int(mo):02d}-{int(d):02d} "
        f"{int(h):02d}:{int(mi):02d}:{s:05.2f}"
    )

    # Time scales - use calendar date directly
    tai = utc_to_tai(year, month, day, hour, minute, int(second))
    gps = utc_to_gps(year, month, day, hour, minute, int(second))

    print(f"\nTime scales (as JD):")
    print(f"  UTC: {jd:.6f}")
    print(f"  TAI: {tai:.6f} (UTC + leap seconds)")
    print(f"  GPS: {gps:.6f} (TAI - 19s)")

    # Sidereal time
    gst = gmst(jd)
    print(
        f"\nGreenwich Mean Sidereal Time: {np.degrees(gst):.4f} deg = "
        f"{np.degrees(gst)/15:.4f} hours"
    )


def demo_reference_frames():
    """Demonstrate reference frame transformations."""
    print("\n" + "=" * 70)
    print("Reference Frame Transformations Demo")
    print("=" * 70)

    # J2000 epoch - gcrf_to_itrf needs jd_ut1 and jd_tt
    jd_ut1 = 2451545.0  # J2000.0
    jd_tt = jd_ut1 + 64.184 / 86400  # TT is ~64s ahead of UT1 at J2000

    # Position in GCRF (inertial)
    r_gcrf = np.array([6778.0, 0.0, 0.0])  # km, along x-axis

    print(f"\nPosition in GCRF (inertial): {r_gcrf} km")

    # Transform to ITRF (Earth-fixed)
    r_itrf = gcrf_to_itrf(r_gcrf, jd_ut1, jd_tt)
    print(
        f"Position in ITRF (Earth-fixed): ({r_itrf[0]:.3f}, "
        f"{r_itrf[1]:.3f}, {r_itrf[2]:.3f}) km"
    )

    # Transform back
    r_gcrf_back = itrf_to_gcrf(r_itrf, jd_ut1, jd_tt)
    print(
        f"Back to GCRF: ({r_gcrf_back[0]:.3f}, {r_gcrf_back[1]:.3f}, "
        f"{r_gcrf_back[2]:.3f}) km"
    )

    # Show precession effect
    print("\n--- Precession Effect ---")
    jd_now = 2460676.5  # ~2025
    centuries = (jd_now - 2451545.0) / 36525

    P = precession_matrix_iau76(jd_now)

    # Apply to vernal equinox direction
    equinox_j2000 = np.array([1.0, 0.0, 0.0])
    equinox_now = P @ equinox_j2000

    angle = np.degrees(np.arccos(np.dot(equinox_j2000, equinox_now)))
    print(f"Precession since J2000.0: {angle:.4f} deg")
    print(f"  ({centuries:.2f} Julian centuries)")


def demo_orbit_determination():
    """Demonstrate using orbital mechanics for orbit determination."""
    print("\n" + "=" * 70)
    print("Orbit Determination Application Demo")
    print("=" * 70)

    np.random.seed(42)

    # Simulated radar observations of a satellite
    # Two observations at known times
    jd1 = 2460676.5  # First observation
    jd2 = jd1 + 0.01  # Second observation (~14 minutes later)

    # True orbit
    a_true = 7000.0
    e_true = 0.001
    elements_true = OrbitalElements(
        a=a_true,
        e=e_true,
        i=np.radians(45),
        raan=np.radians(30),
        omega=np.radians(0),
        nu=np.radians(0),
    )

    state1_true = orbital_elements_to_state(elements_true, GM_EARTH)

    # Propagate to second observation
    dt = (jd2 - jd1) * 86400  # seconds
    state2_true = kepler_propagate_state(state1_true, dt, GM_EARTH)

    # Add measurement noise
    noise_pos = 0.05  # km
    r1 = state1_true.r + np.random.randn(3) * noise_pos
    r2 = state2_true.r + np.random.randn(3) * noise_pos

    print(f"\nTwo position observations separated by {dt:.0f} seconds:")
    print(f"  r1 = ({r1[0]:.3f}, {r1[1]:.3f}, {r1[2]:.3f}) km")
    print(f"  r2 = ({r2[0]:.3f}, {r2[1]:.3f}, {r2[2]:.3f}) km")

    # Solve Lambert's problem to determine orbit
    solution = lambert_universal(r1, r2, dt, GM_EARTH)

    print("\nLambert solution (initial orbit determination):")
    print(
        f"  v1 = ({solution.v1[0]:.4f}, {solution.v1[1]:.4f}, "
        f"{solution.v1[2]:.4f}) km/s"
    )
    print(f"  Semi-major axis: {solution.a:.1f} km (true: {a_true:.1f} km)")
    print(f"  Eccentricity: {solution.e:.4f} (true: {e_true:.4f})")

    # Compare with true velocity
    v1_error = np.linalg.norm(solution.v1 - state1_true.v)
    print(f"\n  Velocity error: {v1_error*1000:.1f} m/s")


def main():
    """Run all demonstrations."""
    print("\n" + "#" * 70)
    print("# PyTCL Orbital Mechanics Example")
    print("#" * 70)

    demo_orbital_elements()
    demo_kepler_equation()
    demo_orbit_propagation()
    demo_lambert_problem()
    demo_hohmann_transfer()
    demo_time_systems()
    demo_reference_frames()
    demo_orbit_determination()

    print("\n" + "=" * 70)
    print("Example complete!")
    if SHOW_PLOTS:
        print("Plots saved: orbital_propagation.html")
    print("=" * 70)


if __name__ == "__main__":
    main()
