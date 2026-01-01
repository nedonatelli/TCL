"""Relativistic effects in orbital mechanics and space systems.

This example demonstrates practical applications of general relativity
and special relativity in modern space systems, including:

1. Gravitational time dilation (GPS, atomic clocks)
2. Perihelion precession (Mercury, binary pulsars)
3. Shapiro delay (interplanetary communication)
4. Post-Newtonian orbital corrections
5. Proper time in gravitational fields
6. Lense-Thirring frame-dragging effects

These effects are essential for high-precision positioning, timing,
and fundamental physics tests.
"""

import matplotlib.pyplot as plt
import numpy as np

from pytcl.astronomical.relativity import (
    AU,
    C_LIGHT,
    G_GRAV,
    GM_EARTH,
    GM_SUN,
    geodetic_precession,
    gravitational_time_dilation,
    lense_thirring_precession,
    post_newtonian_acceleration,
    proper_time_rate,
    relativistic_range_correction,
    schwarzschild_precession_per_orbit,
    schwarzschild_radius,
    shapiro_delay,
)


def example_gps_time_dilation():
    """Demonstrate time dilation effects in GPS satellites."""
    print("=" * 70)
    print("EXAMPLE 1: GPS Time Dilation Effects")
    print("=" * 70)
    
    # GPS orbital parameters
    r_gps = 26.56e6  # meters (~20,200 km altitude)
    v_gps = 3870.0   # m/s (circular orbit velocity)
    
    # Compute dilation factors
    dilation = gravitational_time_dilation(r_gps, GM_EARTH)
    
    # Special relativistic effect
    sr_effect = (v_gps ** 2) / (2.0 * C_LIGHT ** 2)
    
    # General relativistic effect
    gr_effect = GM_EARTH / (C_LIGHT ** 2 * r_gps)
    
    # Proper time rate
    rate = proper_time_rate(v_gps, r_gps, GM_EARTH)
    
    print(f"\nGPS Satellite Orbital Parameters:")
    print(f"  Altitude: {(r_gps - 6.371e6)/1e3:.1f} km")
    print(f"  Orbital velocity: {v_gps:.1f} m/s")
    print(f"  Orbital period: ~12 hours")
    
    print(f"\nTime Dilation Effects:")
    print(f"  Gravitational time dilation factor: {dilation:.15f}")
    print(f"    (Time runs slower in gravity field)")
    
    print(f"\nTime Rate Comparison (per day):")
    print(f"  Special relativistic effect: -{sr_effect*86400*1e9:.1f} ns/day")
    print(f"    (Satellite moving fast, slows down time)")
    print(f"  General relativistic effect: +{gr_effect*86400*1e9:.1f} ns/day")
    print(f"    (Weaker gravity field, speeds up time)")
    print(f"  Net effect: {(1-rate)*86400*1e9:.1f} ns/day")
    print(f"    (Net toward weaker field = time speeds up in orbit)")
    
    print(f"\nPractical Impact:")
    total_daily_shift = (1 - rate) * 86400
    print(f"  Without correction, GPS clock would drift: {total_daily_shift:.1f} seconds/day")
    print(f"  This would cause positioning error: {total_daily_shift * C_LIGHT/2:.0f} meters/day")
    print(f"  GPS atomic clocks must be pre-offset by {-total_daily_shift*1e6:.1f} microseconds/day")
    
    # Time dilation vs altitude
    print(f"\n" + "-" * 70)
    print("Time dilation effect at different altitudes:")
    print("-" * 70)
    
    altitudes = [0, 300, 500, 1000, 5000, 20200, 35786]  # km
    
    for alt_km in altitudes:
        r = (6.371e6 + alt_km * 1000)
        # Approximate circular orbit velocity
        v = np.sqrt(GM_EARTH / r)
        time_rate = proper_time_rate(v, r, GM_EARTH)
        daily_shift = (1 - time_rate) * 86400
        
        if alt_km == 0:
            alt_name = "Earth surface"
        elif alt_km == 35786:
            alt_name = "Geostationary"
        elif alt_km == 20200:
            alt_name = "GPS orbit"
        else:
            alt_name = ""
        
        print(f"  {alt_km:>6} km {alt_name:<20} {daily_shift:>8.2f} seconds/day")


def example_mercury_precession():
    """Mercury's perihelion precession: a test of General Relativity."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Mercury's Perihelion Precession")
    print("=" * 70)
    
    # Mercury orbital elements
    a_mercury = 0.38709927 * AU  # Semi-major axis
    e_mercury = 0.20563593       # Eccentricity
    orbital_period = 87.969      # days
    
    # Compute GR precession per orbit
    precession_rad = schwarzschild_precession_per_orbit(a_mercury, e_mercury, GM_SUN)
    precession_arcsec = precession_rad * 206265  # Convert radians to arcseconds
    
    # Compute precession per century
    orbits_per_century = 36525 / orbital_period
    precession_per_century = precession_arcsec * orbits_per_century
    
    print(f"\nMercury Orbital Parameters:")
    print(f"  Semi-major axis: {a_mercury/AU:.8f} AU = {a_mercury/1e9:.3f} Gm")
    print(f"  Eccentricity: {e_mercury:.8f}")
    print(f"  Orbital period: {orbital_period:.3f} days")
    print(f"  Perturbing body: Sun (GM = {GM_SUN:.3e} m³/s²)")
    
    print(f"\nGeneral Relativistic Perihelion Precession:")
    print(f"  Per orbit: {precession_arcsec:.4f} arcseconds")
    print(f"  Per century: {precession_per_century:.2f} arcseconds")
    
    print(f"\nHistorical Context:")
    print(f"  Einstein's prediction (1916): ~43 arcsec/century")
    print(f"  Observed (Le Verrier, 1859): 5600 ± 30 arcsec/century")
    print(f"    (includes Newtonian planetary perturbations)")
    print(f"  Newtonian precession: ~5557 arcsec/century")
    print(f"  GR contribution: ~43 arcsec/century")
    print(f"  Calculated: {precession_per_century:.1f} arcsec/century ✓")
    print(f"\nThis was Einstein's first experimental confirmation of GR!")


def example_shapiro_delay():
    """Shapiro delay in interplanetary communication."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Shapiro Delay in Interplanetary Communication")
    print("=" * 70)
    
    print(f"\nScenario: Earth-Sun-Spacecraft geometry at superior conjunction")
    print(f"(Spacecraft is on opposite side of Sun from Earth)")
    
    # Geometry at superior conjunction
    earth_pos = np.array([1.496e11, 0.0, 0.0])        # 1 AU
    spacecraft_pos = np.array([-(1.496e11 + 0.8e11), 0.0, 0.0])  # ~1.8 AU away
    sun_pos = np.array([0.0, 0.0, 0.0])
    
    # Compute Shapiro delay
    delay = shapiro_delay(earth_pos, spacecraft_pos, sun_pos, GM_SUN)
    
    # Distance from Earth to spacecraft
    distance = np.linalg.norm(earth_pos - spacecraft_pos)
    
    # Nominal light travel time without Shapiro delay
    light_travel = distance / C_LIGHT
    
    print(f"\nParameters:")
    print(f"  Earth distance from Sun: {np.linalg.norm(earth_pos)/AU:.3f} AU")
    print(f"  Spacecraft distance from Sun: {np.linalg.norm(spacecraft_pos)/AU:.3f} AU")
    print(f"  Earth-spacecraft distance: {distance/AU:.3f} AU = {distance/1.496e11:.3f} AU")
    
    print(f"\nRanging Measurement:")
    print(f"  Signal travel time (geometric): {light_travel:.3f} seconds")
    print(f"  Shapiro delay (GR correction): {delay*1e6:.1f} microseconds")
    print(f"  Total propagation time: {light_travel + delay:.3f} seconds")
    print(f"  Error if uncorrected: {delay * C_LIGHT / 2:.0f} meters")
    
    print(f"\nPractical Impact:")
    print(f"  Mariner 10 Venus flybys: Shapiro delay ~50 microseconds")
    print(f"  Cassini Saturn probe: Shapiro delay ~100+ microseconds")
    print(f"  New Horizons: Shapiro delay ~200+ microseconds at aphelion")
    print(f"\nWithout Shapiro delay correction, spacecraft navigation would be off by kilometers!")
    
    # Shapiro delay vs Sun distance
    print(f"\n" + "-" * 70)
    print("Shapiro delay at different distances from Sun:")
    print("-" * 70)
    
    # Earth at various distances
    for au_dist in [0.5, 1.0, 2.0, 5.0]:
        earth_var = np.array([au_dist * AU, 0.0, 0.0])
        craft = np.array([-(au_dist * AU + 0.8*AU), 0.0, 0.0])
        
        delay_var = shapiro_delay(earth_var, craft, sun_pos, GM_SUN)
        print(f"  Earth at {au_dist:.1f} AU: {delay_var*1e6:.1f} microseconds")


def example_post_newtonian_acceleration():
    """Post-Newtonian orbital corrections."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Post-Newtonian Orbital Corrections")
    print("=" * 70)
    
    # Low Earth Orbit satellite
    r = 6.678e6  # ~300 km altitude
    
    # Circular orbit velocity
    v = np.sqrt(GM_EARTH / r)
    
    # Set up position and velocity vectors
    r_vec = np.array([r, 0.0, 0.0])
    v_vec = np.array([0.0, v, 0.0])
    
    # Compute accelerations
    a_newt = -GM_EARTH / r ** 2 * np.array([1.0, 0.0, 0.0])
    a_total = post_newtonian_acceleration(r_vec, v_vec, GM_EARTH)
    a_pn = a_total - a_newt
    
    # Compute relative correction
    correction_magnitude = np.linalg.norm(a_pn)
    relative_correction = correction_magnitude / np.linalg.norm(a_newt)
    
    print(f"\nLEO Satellite Parameters:")
    print(f"  Altitude: {(r - 6.371e6)/1e3:.0f} km")
    print(f"  Orbital velocity: {v:.1f} m/s")
    print(f"  Orbital period: {2*np.pi*r/v/60:.1f} minutes")
    
    print(f"\nAcceleration Comparison:")
    print(f"  Newtonian acceleration: {np.linalg.norm(a_newt):.6f} m/s²")
    print(f"  Post-Newtonian correction: {correction_magnitude:.3e} m/s²")
    print(f"  Relative correction: {relative_correction*1e6:.1f} ppm (parts per million)")
    
    print(f"\nOrbit Impact Over One Day:")
    orbital_period = 2*np.pi*r/v
    daily_orbits = 86400 / orbital_period
    
    # Accumulated error: dv = a*t
    velocity_error = correction_magnitude * 86400
    
    # Approximate range error (v*t / 2)
    range_error = velocity_error * 86400 / 2
    
    print(f"  Orbits per day: {daily_orbits:.1f}")
    print(f"  Velocity error accumulation: {velocity_error:.3e} m/s")
    print(f"  Position error: ~{range_error:.3f} meters")
    
    print(f"\nPractical Impact:")
    print(f"  For LEO satellites (e.g., ISS, TDRSS):")
    print(f"    - PN effects are measurable but small (ppm level)")
    print(f"    - Other perturbations (gravity harmonics, drag) dominate")
    print(f"    - PN corrections important for ultra-precise orbit determination")
    print(f"  For high-precision applications (LAGEOS, GPS):")
    print(f"    - PN corrections must be included in force models")


def example_geodetic_precession():
    """Geodetic (de Sitter) precession of orbital plane."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Geodetic Precession")
    print("=" * 70)
    
    # Different orbits
    orbits = [
        ("ISS", 6.678e6, 0.0, np.radians(51.6)),
        ("LAGEOS", 12.27e6, 0.0045, np.radians(109.9)),
        ("Polar", 6.678e6, 0.0, np.radians(90.0)),
    ]
    
    print(f"\nGeodetic Precession (causes orbital plane to rotate):")
    print(f"  Formula: Ω_geodetic = -GM/(c² a³(1-e²)²) cos(i)")
    print(f"  (Negative = retrograde precession)")
    print("-" * 70)
    print(f"{'Orbit':<12} {'Altitude':<12} {'Inclination':<16} {'Precession':<20}")
    print("-" * 70)
    
    for name, a, e, inc in orbits:
        prec = geodetic_precession(a, e, inc, GM_EARTH)
        alt_km = (a - 6.371e6) / 1e3
        inc_deg = np.degrees(inc)
        
        # Convert to degrees per year
        orbital_period = 2*np.pi*np.sqrt(a**3/GM_EARTH)
        orbits_per_year = 365.25 * 86400 / orbital_period
        prec_per_year = prec * orbits_per_year * 206265  # arcsec/year
        
        print(f"{name:<12} {alt_km:>7.0f} km   {inc_deg:>6.1f}°         {prec_per_year:>8.2f} arcsec/year")
    
    print(f"\nPhysical Interpretation:")
    print(f"  - Geodetic precession arises from parallel transport of velocity")
    print(f"  - Also called de Sitter precession (discovered 1916)")
    print(f"  - Related to Lense-Thirring effect (frame dragging)")
    print(f"  - At i=90°, geodetic effect vanishes (observer aligned with angular momentum)")


def example_lense_thirring_precession():
    """Lense-Thirring (frame-dragging) effect on orbital node."""
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Lense-Thirring Effect (Frame-Dragging)")
    print("=" * 70)
    
    # LAGEOS satellite parameters
    a = 12.27e6  # Semi-major axis
    e = 0.0045
    i = np.radians(109.9)
    L_earth = 7.05e33  # Earth's angular momentum (kg·m²/s)
    
    # Compute Lense-Thirring precession
    precession = lense_thirring_precession(a, e, i, L_earth, GM_EARTH)
    
    # Convert to observable rates
    orbital_period = 2*np.pi*np.sqrt(a**3/GM_EARTH)
    orbits_per_year = 365.25 * 86400 / orbital_period
    precession_per_year = precession * orbits_per_year * 206265  # arcsec/year
    
    print(f"\nLAGEOS Satellite (Test of General Relativity):")
    print(f"  Semi-major axis: {a/1e6:.2f} Mm = {(a-6.371e6)/1e3:.0f} km altitude")
    print(f"  Eccentricity: {e:.4f}")
    print(f"  Inclination: {np.degrees(i):.2f}°")
    print(f"  Orbital period: {orbital_period/60:.0f} minutes = {orbital_period/3600:.2f} hours")
    
    print(f"\nLense-Thirring Effect:")
    print(f"  Precession per orbit: {precession*206265:.3f} milliarcseconds")
    print(f"  Precession per year: {precession_per_year:.3f} arcsecond")
    print(f"  Detection method: Laser ranging (~mm precision on altitude)")
    
    print(f"\nHistorical Context:")
    print(f"  - Predicted by Lense & Thirring (1918)")
    print(f"  - Represents frame-dragging effect of rotating body")
    print(f"  - LAGEOS confirmed at ~20% accuracy (1998)")
    print(f"  - GRAVITY Probe B tested at higher precision (~0.5%)")
    
    print(f"\nPhysical Interpretation:")
    print(f"  - Earth's rotation 'drags' spacetime around it")
    print(f"  - Causes orbits to precess even though not purely axisymmetric")
    print(f"  - Similar to electromagnetic induction but for gravity")


def example_relativistic_range_correction():
    """Relativistic corrections to ranging measurements."""
    print("\n" + "=" * 70)
    print("EXAMPLE 7: Relativistic Range Corrections")
    print("=" * 70)
    
    print(f"\nRanging measurement technique:")
    print(f"  - Send light signal to reflector (satellite or corner cube)")
    print(f"  - Measure round-trip travel time: t = 2d/c")
    print(f"  - Compute distance: d = ct/2")
    print(f"\nRelativistic corrections modify measured distance:")
    
    # Lunar laser ranging
    d_moon = 3.84e8  # meters
    r_corr_moon = relativistic_range_correction(d_moon, 0.0, GM_EARTH)
    
    print(f"\nLunar Laser Ranging (LLR):")
    print(f"  Target: Apollo 11, 14, 15 retroreflectors on Moon")
    print(f"  Distance: {d_moon/1e6:.0f} km")
    print(f"  Relativistic correction: {r_corr_moon:.1f} meters")
    print(f"  Precision of LLR: ~2-3 cm")
    print(f"  Relativistic correction: {r_corr_moon*100:.0f}% of measurement error")
    
    print(f"\n" + "-" * 70)
    print("Relativistic range corrections at various distances:")
    print("-" * 70)
    print(f"{'Distance':<20} {'Correction (m)':<20} {'Relative':<15}")
    print("-" * 70)
    
    distances = {
        'TDRSS (42,000 km)': 42.16e6,
        'GPS (26,600 km)': 26.56e6,
        'ISS (400 km)': 6.771e6,
        'Moon (384,400 km)': 3.84e8,
        'Sun (150 M km)': 1.5e11,
    }
    
    for name, dist in distances.items():
        corr = relativistic_range_correction(dist, 0.0, GM_EARTH if 'Sun' not in name else GM_SUN)
        # Nominal range error at accuracy of 1 cm
        relative = corr / 0.01
        print(f"{name:<20} {corr:>15.2f}    {relative:>12.0f}× cm accuracy")


if __name__ == '__main__':
    """Run all relativity examples."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  Relativistic Effects in Space Systems".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")
    
    # Run examples
    example_gps_time_dilation()
    example_mercury_precession()
    example_shapiro_delay()
    example_post_newtonian_acceleration()
    example_geodetic_precession()
    example_lense_thirring_precession()
    example_relativistic_range_correction()
    
    print("\n" + "=" * 70)
    print("All relativity examples completed successfully!")
    print("=" * 70 + "\n")
