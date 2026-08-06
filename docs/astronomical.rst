Astronomical & Celestial Mechanics
===================================

Overview
--------

The Tracker Component Library provides **orbital mechanics, ephemeris, and celestial reference frame** functionality for satellite operations, space vehicle navigation, and astronomical calculations.

**Key Modules:**

- ``astronomical.orbital_mechanics`` - Keplerian elements, anomaly conversions, propagation
- ``astronomical.reference_frames`` - ECEF/ECI, GCRF/ITRF/TEME, precession, nutation, polar motion
- ``astronomical.ephemerides`` - Sun, Moon, planet positions (JPL DE ephemeris via ``jplephem``)
- ``astronomical.sgp4`` - SGP4/SDP4 propagator (wraps the ``sgp4`` package)
- ``astronomical.tle`` - Two-Line Element parsing and formatting
- ``astronomical.lambert`` - Lambert's problem and orbit transfers
- ``astronomical.relativity`` - Relativistic corrections
- ``astronomical.time_systems`` - Julian dates, GMST, UTC/TAI/TT/GPS conversions

All functions are re-exported at the ``pytcl.astronomical`` package level.

.. note::

   The orbital mechanics, Lambert, and SGP4 APIs work in **kilometers** and
   km/s (the default gravitational parameter is
   ``mu = 398600.4418 km^3/s^2``), unlike the coordinate-conversion modules,
   which work in meters.

Orbital Elements and Representations
-------------------------------------

**Keplerian Elements** (6 parameters describing an orbit) are held in the
``OrbitalElements`` named tuple:

.. code-block:: python

   import numpy as np
   from pytcl.astronomical import OrbitalElements, orbital_elements_to_state

   elements = OrbitalElements(
       a=6700.0,                # semi-major axis (km)
       e=0.001,                 # eccentricity (0 = circular)
       i=np.radians(98.2),      # inclination (rad, sun-sync here)
       raan=np.radians(45.0),   # right ascension of ascending node (rad)
       omega=np.radians(30.0),  # argument of perigee (rad)
       nu=0.0,                  # true anomaly (rad, 0 = perigee)
   )

   # Convert to a Cartesian state (StateVector with .r and .v)
   state = orbital_elements_to_state(elements)
   print(f"Position (km):   {state.r}")
   print(f"Velocity (km/s): {state.v}")

Output::

   Position (km):   [4436.31508215 3761.26976554 3312.43462317]
   Velocity (km/s): [-2.05538115 -3.40409261  6.61810164]

**Anomaly Conversions**

In orbital mechanics, there are three ways to measure position along the orbit:

1. **True Anomaly (nu)**: Actual angle from focus
2. **Eccentric Anomaly (E)**: Auxiliary angle (used in equations)
3. **Mean Anomaly (M)**: Time-based angle (increases uniformly with time)

.. code-block:: python

   from pytcl.astronomical import (
       true_to_eccentric_anomaly, eccentric_to_true_anomaly,
       eccentric_to_mean_anomaly, mean_to_eccentric_anomaly,
   )

   nu = 0.1   # true anomaly (radians)
   e = 0.001  # eccentricity

   # True -> Eccentric
   E = true_to_eccentric_anomaly(nu, e)

   # Eccentric -> Mean
   M = eccentric_to_mean_anomaly(E, e)

   # Mean -> Eccentric (Newton-Raphson iteration)
   E_recovered = mean_to_eccentric_anomaly(M, e)

   # Eccentric -> True
   nu_recovered = eccentric_to_true_anomaly(E_recovered, e)

   print(f"Original nu={nu:.6f}, recovered nu={nu_recovered:.6f}")

Hyperbolic and parabolic counterparts exist as well
(``mean_to_hyperbolic_anomaly``, ``true_anomaly_to_parabolic_anomaly``, etc.).

**State Conversions**

.. code-block:: python

   from pytcl.astronomical import StateVector, state_to_orbital_elements

   # Cartesian state to Keplerian elements
   state = StateVector(
       r=np.array([6600.0, 0.0, 0.0]),  # km
       v=np.array([0.0, 7.8, 0.0]),     # km/s
   )
   elements = state_to_orbital_elements(state)
   print(f"a={elements.a:.1f} km, e={elements.e:.4f}")

   # Keplerian back to Cartesian
   state_recovered = orbital_elements_to_state(elements)

Keplerian Propagation
---------------------

**Basic Circular Orbit**

.. code-block:: python

   from pytcl.astronomical import kepler_propagate, orbital_period

   # Initial orbital elements: LEO polar orbit
   elements = OrbitalElements(a=6700.0, e=0.0, i=np.pi/2,
                              raan=0.0, omega=0.0, nu=0.0)

   period = orbital_period(elements.a)  # seconds
   print(f"Orbital period: {period / 60:.1f} minutes")

   # Propagate 100 seconds (two-body motion, no perturbations)
   elements_prop = kepler_propagate(elements, 100.0)

   # Check: semi-major axis is unchanged
   print(f"a unchanged: {np.isclose(elements.a, elements_prop.a)}")

Output::

   Orbital period: 91.0 minutes
   a unchanged: True

``kepler_propagate_state`` does the same starting from a ``StateVector``.

**With Mean Anomaly Propagation**

.. code-block:: python

   from pytcl.astronomical import mean_motion

   # For efficient long-term propagation, advance the mean anomaly
   a = 6700.0
   e = 0.001
   n = mean_motion(a)  # rad/s

   # Initial state
   nu0 = 0.0
   E0 = true_to_eccentric_anomaly(nu0, e)
   M0 = eccentric_to_mean_anomaly(E0, e)

   # Propagate mean anomaly
   t_prop = 3600.0  # 1 hour
   M_prop = M0 + n * t_prop

   # Convert back to true anomaly
   E_prop = mean_to_eccentric_anomaly(M_prop, e)
   nu_prop = eccentric_to_true_anomaly(E_prop, e)

   print(f"True anomaly after 1 hour: {np.degrees(nu_prop):.2f} deg")

Output::

   True anomaly after 1 hour: 237.36 deg

Perturbations
-------------

The Keplerian propagator is a pure two-body model. For perturbed motion the
library offers SGP4 (below), which models J2-J4, drag, and resonance effects
for TLE-based satellites. The secular J2 rates themselves are a one-line
formula if you need them directly, as in the sun-synchronous design example
at the end of this page.

**Atmospheric Drag** (for low Earth orbits)

``pytcl.atmosphere.simplified_thermosphere`` provides upper-atmosphere
density. It is a **simplified barometric model** with solar-flux and
geomagnetic coupling -- not NRLMSISE-00 -- and is only meaningful above
roughly 200 km.

.. code-block:: python

   from pytcl.atmosphere import simplified_thermosphere

   # Density at 400 km altitude (angles in radians, altitude in meters)
   state_atm = simplified_thermosphere(
       latitude=np.radians(40.0),
       longitude=np.radians(-74.0),
       altitude=400e3,
       year=2026,
       day_of_year=57,
       seconds_in_day=43200.0,
       f107=150.0,   # 10.7 cm solar flux (SFU)
       ap=4.0,       # geomagnetic index
   )
   print(f"Density at 400 km: {state_atm.density:.3e} kg/m^3")

   def drag_acceleration(rho, vel, area, mass, cd=2.2):
       """Cannonball drag: a = -0.5 (cd A / m) rho |v| v."""
       v_mag = np.linalg.norm(vel)
       return -0.5 * (cd * area / mass) * rho * v_mag * vel

   vel = np.array([7500.0, 0.0, 0.0])  # m/s
   a_drag = drag_acceleration(state_atm.density, vel, area=10.0, mass=500.0)

Output::

   Density at 400 km: 2.886e-12 kg/m^3

Reference Frames: ECEF and ECI
-------------------------------

**Earth-Centered Inertial (ECI) vs Earth-Centered Earth-Fixed (ECEF)**

- **ECI**: Fixed to distant stars (inertial). Doesn't rotate with Earth.
- **ECEF**: Rotates with Earth. Fixed to ground stations.

The simple conversion is a rotation about the z-axis by the Greenwich Mean
Sidereal Time (GMST):

.. code-block:: python

   import numpy as np
   from pytcl.astronomical import cal_to_jd, gmst
   from pytcl.astronomical.reference_frames import ecef_to_eci, eci_to_ecef

   # Satellite position in ECI (km); positions only, shape (3,)
   sat_eci = np.array([6600.0, 0.0, 0.0])

   # Time of observation -> Julian date -> GMST (radians)
   jd = cal_to_jd(2026, 2, 26, 12, 0, 0.0)
   theta = gmst(jd)

   # Transform to ECEF (what ground stations see)
   sat_ecef = eci_to_ecef(sat_eci, theta)

   print(f"ECI position:  {sat_eci}")
   print(f"ECEF position: {sat_ecef}")

   # Transform back
   sat_eci_recovered = ecef_to_eci(sat_ecef, theta)

Output::

   ECI position:  [6600.    0.    0.]
   ECEF position: [6045.68286597 2647.58733268    0.        ]

**Components of the Full Transformation**

A high-precision ECI/ECEF transformation includes:

1. **Precession** - Long-term wobble of Earth's spin axis (~26,000 year period)
2. **Nutation** - Short-term oscillation (~18.6 year period)
3. **Earth Rotation (GMST/GAST)** - Daily rotation
4. **Polar Motion** - Shift of the rotation axis (~1 meter, from EOP data)

Each component is available as a matrix:

.. code-block:: python

   from pytcl.astronomical import (
       precession_matrix_iau76, nutation_matrix, polar_motion_matrix,
   )

   jd = cal_to_jd(2026, 2, 26, 12, 0, 0.0)

   theta = gmst(jd)
   print(f"GMST: {theta:.4f} rad = {np.degrees(theta):.2f} deg")

   P = precession_matrix_iau76(jd)      # IAU 1976 precession
   N = nutation_matrix(jd)              # IAU 1980 nutation
   W = polar_motion_matrix(xp=1e-6, yp=2e-6)  # pole coordinates (rad)

Output::

   GMST: 5.8704 rad = 336.35 deg

**High-Precision Transformations (GCRF/ITRF)**

The full chain (precession + nutation + sidereal rotation + polar motion) is
packaged as ``gcrf_to_itrf`` / ``itrf_to_gcrf``. Supply UT1 and TT Julian
dates and, if available, the pole coordinates from IERS EOP data:

.. code-block:: python

   from pytcl.astronomical import gcrf_to_itrf, itrf_to_gcrf

   r_gcrf = np.array([6600.0, 0.0, 0.0])  # km
   jd_ut1 = cal_to_jd(2026, 2, 26, 12, 0, 0.0)
   jd_tt = jd_ut1 + 69.184 / 86400  # TT - UTC as of 2026

   r_itrf = gcrf_to_itrf(r_gcrf, jd_ut1=jd_ut1, jd_tt=jd_tt)
   r_back = itrf_to_gcrf(r_itrf, jd_ut1=jd_ut1, jd_tt=jd_tt)

``teme_to_itrf`` / ``itrf_to_teme`` convert between the TEME frame that SGP4
outputs and the Earth-fixed ITRF frame.

Ephemeris Functions
-------------------

The ephemeris functions use a JPL Development Ephemeris kernel through the
``jplephem`` package (``pip install nrl-tracker[astronomy]``). Positions are
in **AU** and, in the ``'icrf'`` frame, relative to the **solar system
barycenter (SSB)**, not to Earth.

**Sun and Moon Positions**

.. code-block:: python

   from pytcl.astronomical import cal_to_jd, sun_position, moon_position

   jd = cal_to_jd(2026, 2, 26, 12, 0, 0.0)  # Julian date (TT)

   # Sun position relative to the solar system barycenter (AU, ICRF)
   sun_pos, sun_vel = sun_position(jd)
   print(f"Sun position (AU, barycentric ICRF): {sun_pos}")

   # Moon position relative to Earth's center
   moon_pos, moon_vel = moon_position(jd, frame='earth_centered')
   AU_KM = 149597870.7
   print(f"Earth-Moon distance: {np.linalg.norm(moon_pos) * AU_KM:.0f} km")

Output::

   Sun position (AU, barycentric ICRF): [-0.00266542 -0.00510309 -0.00207743]
   Earth-Moon distance: 366214 km

**Planet Positions**

.. code-block:: python

   from pytcl.astronomical import planet_position

   # Available: mercury, venus, mars, jupiter, saturn, uranus, neptune
   for planet in ['mars', 'jupiter']:
       pos, vel = planet_position(planet, jd)
       print(f"{planet.capitalize()}: {np.linalg.norm(pos):.2f} AU from the SSB")

Output::

   Mars: 1.39 AU from the SSB
   Jupiter: 5.23 AU from the SSB

For repeated queries, instantiate ``DEEphemeris`` once and call its methods;
the convenience functions above create and cache one internally.

SGP4 Propagator (TLE-Based)
---------------------------

**From Two-Line Element (TLE)**

``parse_tle`` returns a ``TLE`` named tuple (angles in radians, mean motion
in radians/minute); ``sgp4_propagate`` takes the TLE and the time since
epoch in minutes and returns position/velocity in the TEME frame:

.. code-block:: python

   from pytcl.astronomical import (
       parse_tle, sgp4_propagate, tle_epoch_to_datetime,
   )

   # Example TLE (International Space Station)
   tle_line1 = "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9997"
   tle_line2 = "2 25544  51.6400 247.4627 0006703 130.5360 325.0288 15.49815350479003"

   tle = parse_tle(tle_line1, tle_line2, name="ISS (ZARYA)")

   print(f"Catalog number: {tle.catalog_number}")
   print(f"Epoch: {tle_epoch_to_datetime(tle)}")
   rev_per_day = tle.mean_motion * 1440 / (2 * np.pi)
   print(f"Mean motion: {rev_per_day:.4f} revolutions/day")

   # Propagate 10 minutes past epoch
   state = sgp4_propagate(tle, 10.0)
   print(f"Position (TEME, km):   {state.r}")
   print(f"Velocity (TEME, km/s): {state.v}")

Output::

   Catalog number: 25544
   Epoch: 2024-01-01 12:00:00+00:00
   Mean motion: 15.4982 revolutions/day
   Position (TEME, km):   [4602.28257877 3229.53092379 3800.56614424]
   Velocity (TEME, km/s): [-0.96603157  6.34040043 -4.2050613 ]

.. note::

   ``parse_tle`` verifies the TLE line checksums by default, so hand-edited
   lines are rejected (pass ``verify_checksum=False`` to skip the check).

**Batch Propagation**

.. code-block:: python

   from pytcl.astronomical import sgp4_propagate_batch

   times = np.arange(0.0, 90.0, 10.0)  # minutes since epoch
   positions, velocities = sgp4_propagate_batch(tle, times)
   # positions: (9, 3) km, velocities: (9, 3) km/s

Lambert's Problem
-----------------

**Orbit Transfers** (from one orbit to another)

.. code-block:: python

   from pytcl.astronomical import lambert_universal, hohmann_transfer

   # Initial orbit: LEO (km); final orbit: GEO
   r1 = 6700.0
   r2 = 42164.0

   # Hohmann transfer delta-v and time of flight
   dv1, dv2, tof = hohmann_transfer(r1, r2)
   print(f"Hohmann: dv1={dv1:.3f} km/s, dv2={dv2:.3f} km/s, "
         f"tof={tof / 3600:.2f} hours")

   # Solve Lambert's problem for a 90-degree transfer in the same time
   r_initial = np.array([r1, 0.0, 0.0])
   r_final = np.array([0.0, r2, 0.0])

   sol = lambert_universal(r_initial, r_final, tof)
   print(f"Departure velocity: {sol.v1} km/s")
   print(f"Arrival velocity:   {sol.v2} km/s")

Output::

   Hohmann: dv1=2.420 km/s, dv2=1.465 km/s, tof=5.28 hours
   Departure velocity: [6.73474339 7.50479386 0.        ] km/s
   Arrival velocity:   [-1.19253673 -0.42248626 -0.        ] km/s

``lambert_izzo`` implements the same interface with Izzo's algorithm, and
``bi_elliptic_transfer`` / ``minimum_energy_transfer`` cover the other
classical transfer computations.

Relativistic Corrections
------------------------

**Clock Rate (Proper Time)** (main relativistic correction for GPS/timing)

.. code-block:: python

   from pytcl.astronomical.relativity import proper_time_rate

   # GPS satellite: orbital speed (m/s) and geocentric radius (m)
   rate_sat = proper_time_rate(v=3874.0, r=26560e3)

   # Ground clock on the rotating equator
   rate_ground = proper_time_rate(v=465.0, r=6378e3)

   # GPS satellite clocks run ~38 microseconds/day faster than ground clocks
   delta = rate_sat - rate_ground
   print(f"Fractional rate difference: {delta:.2e}")
   print(f"Clock offset: {delta * 86400 * 1e6:.1f} microseconds/day")

Output::

   Fractional rate difference: 4.46e-10
   Clock offset: 38.5 microseconds/day

**Shapiro Delay** (photon traveling through spacetime curvature)

.. code-block:: python

   from pytcl.astronomical.relativity import shapiro_delay, GM_EARTH

   # Ground station position (ECEF, equator)
   station = np.array([6378e3, 0.0, 0.0])  # meters

   # GPS satellite position
   sat = np.array([26560e3, 0.0, 0.0])  # meters

   # Signal delay due to Earth's spacetime curvature
   # (observer, light source, gravitating body position, GM)
   delay = shapiro_delay(station, sat, np.zeros(3), GM_EARTH)

   print(f"Shapiro delay: {delay * 1e12:.1f} picoseconds")

Output::

   Shapiro delay: 42.2 picoseconds

**Relativistic Range Correction** (Shapiro delay as an equivalent range)

.. code-block:: python

   from pytcl.astronomical.relativity import relativistic_range_correction

   r1 = 6378e3          # Station geocentric radius (m)
   r2 = 26560e3         # Satellite geocentric radius (m)
   rho = r2 - r1        # Station-satellite range (m)

   dr = relativistic_range_correction(r1, r2, rho, GM_EARTH)
   print(f"Range correction: {dr * 1000:.1f} mm")

Output::

   Range correction: 12.7 mm

Complete Satellite Tracking Example
-----------------------------------

**Track a Satellite from a Ground Station**

.. code-block:: python

   import datetime
   import numpy as np
   from pytcl.astronomical import (
       cal_to_jd, parse_tle, sgp4_propagate, teme_to_itrf,
       tle_epoch_to_datetime,
   )
   from pytcl.coordinate_systems.conversions import ecef2enu, geodetic2ecef

   class GroundTracker:
       def __init__(self, tle, station_lat, station_lon, station_alt):
           """
           Args:
               tle: parsed TLE (from parse_tle)
               station_lat, station_lon: ground station (radians)
               station_alt: ground station altitude (meters)
           """
           self.tle = tle
           self.epoch = tle_epoch_to_datetime(tle)
           self.lat = station_lat
           self.lon = station_lon
           self.station_ecef = geodetic2ecef(station_lat, station_lon,
                                             station_alt)

       def get_topocentric(self, time):
           """Satellite azimuth/elevation/range as seen from the station."""
           minutes = (time - self.epoch).total_seconds() / 60.0
           state = sgp4_propagate(self.tle, minutes)

           # SGP4 output is TEME; rotate to Earth-fixed and convert to m
           jd = cal_to_jd(time.year, time.month, time.day,
                          time.hour, time.minute, time.second)
           pos_ecef = teme_to_itrf(state.r, jd) * 1000.0

           enu = ecef2enu(pos_ecef, self.lat, self.lon, self.station_ecef)
           east, north, up = enu
           range_m = np.linalg.norm(enu)
           azimuth = np.arctan2(east, north)  # from north
           elevation = np.arctan2(up, np.hypot(east, north))
           return np.degrees(azimuth), np.degrees(elevation), range_m

   # Example: Track the ISS from NYC
   tle1 = "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9997"
   tle2 = "2 25544  51.6400 247.4627 0006703 130.5360 325.0288 15.49815350479003"
   tle = parse_tle(tle1, tle2)

   tracker = GroundTracker(tle, np.radians(40.7128), np.radians(-74.0060),
                           10.0)

   # Sample one orbit starting at the TLE epoch
   t_start = tle_epoch_to_datetime(tle)
   for i in range(0, 90, 15):
       t = t_start + datetime.timedelta(minutes=i)
       az, el, rng = tracker.get_topocentric(t)
       print(f"{i:3d} min: Az={az:7.1f} deg, El={el:6.1f} deg, "
             f"Range={rng / 1000:8.1f} km")

Output (negative elevation means the satellite is below the horizon)::

     0 min: Az=   24.1 deg, El= -38.6 deg, Range=  8608.8 km
    15 min: Az=  -21.0 deg, El= -56.9 deg, Range= 11188.4 km
    30 min: Az=  -87.5 deg, El= -65.3 deg, Range= 12041.4 km
    45 min: Az= -145.7 deg, El= -51.8 deg, Range= 10528.0 km
    60 min: Az=  175.5 deg, El= -29.8 deg, Range=  7100.4 km
    75 min: Az=  109.9 deg, El= -14.1 deg, Range=  4373.8 km

Performance Considerations
--------------------------

**Propagation Speed**

.. code-block:: python

   import time

   # Keplerian propagation: very fast (two-body only)
   elements = OrbitalElements(a=6700.0, e=0.001, i=1.0,
                              raan=0.0, omega=0.0, nu=0.0)
   t0 = time.perf_counter()
   for _ in range(10000):
       kepler_propagate(elements, 100.0)
   print(f"Keplerian: {time.perf_counter() - t0:.3f} s for 10k propagations")

   # SGP4: slower but includes perturbations
   t0 = time.perf_counter()
   for _ in range(10000):
       sgp4_propagate(tle, 1.0)
   print(f"SGP4: {time.perf_counter() - t0:.3f} s for 10k propagations")

For many time steps of a single satellite, ``sgp4_propagate_batch``
amortizes the per-call overhead.

**Accuracy vs Speed Tradeoff**

.. code-block:: text

   Method              Accuracy     Speed      Best For

   Keplerian           ~1 km        Fastest    Circular orbits, quick estimates
   SGP4                ~1-3 km      Medium     TLE-based (public data)
   Numerical + drag    Best         Slow       Precise long-term (GPS, research)

Common Orbital Scenarios
------------------------

**Sun-Synchronous Orbit Design**

For a sun-synchronous orbit the J2-driven RAAN precession must equal
360 degrees per year. Solve the secular J2 rate equation for inclination:

.. code-block:: python

   from scipy.optimize import fsolve

   J2 = 1.08262668e-3
   Re = 6378.137        # km
   mu = 398600.4418     # km^3/s^2

   def design_sun_sync_orbit(altitude_km):
       """Find the inclination whose J2 nodal precession tracks the Sun."""
       a = Re + altitude_km
       n = np.sqrt(mu / a**3)
       target = 2 * np.pi / (365.25 * 86400)  # rad/s

       def sun_sync_condition(i):
           # Secular RAAN rate for a circular orbit:
           # dRAAN/dt = -1.5 n J2 (Re/a)^2 cos(i)
           return -1.5 * n * J2 * (Re / a) ** 2 * np.cos(i) - target

       return np.degrees(fsolve(sun_sync_condition, np.radians(98.0))[0])

   print(f"Sun-sync inclination at 700 km: {design_sun_sync_orbit(700.0):.2f} deg")

Output::

   Sun-sync inclination at 700 km: 98.19 deg

See Also
~~~~~~~~

- :doc:`architecture` - Module organization
- :doc:`coordinate_systems` - Coordinate transformations
- :doc:`api_navigation` - Finding astronomical functions
- :doc:`kalman_filter_tuning` - Filtering satellite orbits
- ``examples/orbital_mechanics.py`` - Orbit propagation examples
- ``examples/ephemeris_demo.py`` - Ephemeris usage
- ``examples/reference_frame_advanced.py`` - Frame transformations
