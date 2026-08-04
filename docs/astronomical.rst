Astronomical & Celestial Mechanics
===================================

Overview
--------

The Tracker Component Library provides comprehensive **orbital mechanics, ephemeris, and celestial reference frame** functionality for satellite operations, space vehicle navigation, and astronomical calculations.

**Key Modules:**

- ``astronomical.orbital_mechanics`` - Keplerian elements, anomaly conversions, propagation
- ``astronomical.reference_frames`` - ECEF↔ECI, precession, nutation, polar motion
- ``astronomical.ephemerides`` - Sun, Moon, planet positions (JPL ephemeris)
- ``astronomical.sgp4`` - SGP4 propagator with perturbations
- ``astronomical.lambert`` - Lambert's problem (bi-elliptic transfers)
- ``astronomical.relativity`` - Relativistic corrections

Orbital Elements and Representations
-------------------------------------

**Keplerian Elements** (6 parameters describing orbit)

.. code-block:: python

   import numpy as np
   
   # Keplerian orbital elements
   a = 6700e3      # Semi-major axis (meters)
   e = 0.001       # Eccentricity (0 = circular, 1 = parabolic)
   i = 98.2        # Inclination (degrees, for sun-sync orbit)
   Omega = 45.0    # Right ascension of ascending node (degrees)
   omega = 30.0    # Argument of perigee (degrees)
   nu = 0.0        # True anomaly (degrees, 0 = perigee, 180 = apogee)
   
   kep = np.array([a, e, np.radians(i), np.radians(Omega), 
                   np.radians(omega), np.radians(nu)])
   
   from pytcl.astronomical.orbital_mechanics import orbital_elements_to_state
   
   # Convert to Cartesian state [x, y, z, vx, vy, vz]
   state_eci = orbital_elements_to_state(kep)
   print(f"Position: {state_eci[:3]}")
   print(f"Velocity: {state_eci[3:]}")

**Anomaly Conversions**

In orbital mechanics, there are three ways to measure position along the orbit:

1. **True Anomaly (ν)**: Actual angle from focus
2. **Eccentric Anomaly (E)**: Auxiliary angle (used in equations)
3. **Mean Anomaly (M)**: Time-based angle (increases uniformly with time)

.. code-block:: python

   from pytcl.astronomical.orbital_mechanics import (
       true2eccentric, eccentric2true, eccentric2mean, mean2eccentric
   )
   
   # Convert between anomalies
   nu = 0.1  # True anomaly (radians)
   e = 0.001  # Eccentricity
   
   # True → Eccentric
   E = true2eccentric(nu, e)
   
   # Eccentric → Mean
   M = eccentric2mean(E, e)
   
   # Mean → Eccentric (Newton-Raphson iteration)
   E_recovered = mean2eccentric(M, e)
   
   # Eccentric → True
   nu_recovered = eccentric2true(E_recovered, e)
   
   print(f"Original ν={nu:.4f}, Recovered ν={nu_recovered:.4f}")

**State Conversions**

.. code-block:: python

   from pytcl.astronomical.orbital_mechanics import (
       state2kep, orbital_elements_to_state, state2rate
   )
   
   # Cartesian state to Keplerian
   state = np.array([
       6600e3, 0, 0,    # Position [m]
       0, 7500, 0       # Velocity [m/s]
   ])
   
   kep = state2kep(state)  # [a, e, i, Ω, ω, ν]
   
   # Keplerian back to Cartesian
   state_recovered = orbital_elements_to_state(kep)
   
   # Mean motion and derivatives
   dkep_dt = state2rate(state)  # Perturbation rates

Keplerian Propagation
---------------------

**Basic Circular Orbit**

.. code-block:: python

   from pytcl.astronomical.orbital_mechanics import kepler_propagate
   
   # Initial orbital elements
   a = 6700e3       # LEO at 300 km altitude
   e = 0.0          # Circular
   i = 90.0 * np.pi/180  # Polar orbit
   Omega = 0.0
   omega = 0.0
   nu = 0.0
   
   kep0 = np.array([a, e, i, Omega, omega, nu])
   state0 = orbital_elements_to_state(kep0)
   
   # Propagate for one orbital period
   period = 2 * np.pi * np.sqrt(a**3 / 3.986004418e14)  # seconds
   print(f"Orbital period: {period/60:.1f} minutes")
   
   # Propagate 100 seconds
   state_prop = kepler_propagate(state0, t=100)
   kep_prop = state2kep(state_prop)
   
   # Check: semi-major axis should be unchanged
   print(f"a unchanged: {np.isclose(kep0[0], kep_prop[0])}")

**With Mean Anomaly Propagation**

.. code-block:: python

   from pytcl.astronomical.orbital_mechanics import (
       mean_anomaly_propagation, eccentric2mean, mean2eccentric
   )
   
   # For efficient long-term propagation, use mean anomaly
   a = 6700e3
   e = 0.001
   n = np.sqrt(3.986004418e14 / a**3)  # Mean motion
   
   # Initial state
   nu0 = 0.0
   E0 = true2eccentric(nu0, e)
   M0 = eccentric2mean(E0, e)
   
   # Propagate mean anomaly
   t_prop = 3600  # 1 hour
   M_prop = M0 + n * t_prop
   
   # Convert back to true anomaly
   E_prop = mean2eccentric(M_prop, e)
   nu_prop = eccentric2true(E_prop, e)
   
   print(f"True anomaly after 1 hour: {nu_prop * 180/np.pi:.2f}°")

Perturbations
-------------

**J2 Oblateness Perturbation** (most significant)

.. code-block:: python

   from pytcl.astronomical.orbital_mechanics import (
       j2_perturbation, propagate_perturbed_kepler
   )
   
   # J2 causes precession of orbital elements
   
   # Initial orbit
   a = 6700e3
   e = 0.001
   i = 98.2 * np.pi/180  # Inclination
   Omega = 0.0
   omega = 0.0
   nu = 0.0
   
   kep = np.array([a, e, i, Omega, omega, nu])
   
   # J2 perturbation rates
   dOmega_dt, domega_dt = j2_perturbation(kep)
   
   print(f"RAAN precession: {dOmega_dt * 86400:.2f}°/day")
   print(f"Arg of perigee precession: {domega_dt * 86400:.2f}°/day")
   
   # Propagate with perturbations
   state = orbital_elements_to_state(kep)
   t_steps = np.linspace(0, 86400, 100)  # 1 day
   
   trajectory = []
   state_current = state
   for t in t_steps:
       state_current = propagate_perturbed_kepler(state_current, t)
       trajectory.append(state_current)
   
   trajectory = np.array(trajectory)

**Atmospheric Drag** (for low Earth orbits)

.. code-block:: python

   from pytcl.atmosphere.thermosphere import simplified_thermosphere
   from pytcl.astronomical.reference_frames import ecef_to_eci
   
   def drag_perturbation(state_eci, time, area, mass, cd=2.2):
       """
       Simple drag model for LEO satellite
       
       Args:
           state_eci: [x, y, z, vx, vy, vz] in meters, m/s
           time: datetime object
           area: cross-sectional area (m²)
           mass: satellite mass (kg)
           cd: drag coefficient
       """
       from pytcl.astronomical.reference_frames import eci_to_ecef
       from pytcl.coordinate_systems.conversions import ecef2geodetic
       
       # Convert to ECEF for atmosphere model
       state_ecef = eci_to_ecef(state_eci, time)
       pos_ecef = state_ecef[:3]
       vel_eci = state_eci[3:]
       
       # Get position in geodetic
       geodetic = ecef2geodetic(pos_ecef)
       lat, lon, alt = geodetic
       
       # Atmospheric density at satellite altitude
       rho = simplified_thermosphere(np.degrees(lat), np.degrees(lon), alt)
       
       # Drag acceleration
       # a_drag = -0.5 * (cd * area / mass) * rho * v²
       v_magnitude = np.linalg.norm(vel_eci)
       if v_magnitude > 0:
           accel_drag = -0.5 * (cd * area / mass) * rho * v_magnitude * vel_eci
       else:
           accel_drag = np.zeros(3)
       
       return accel_drag

Reference Frames: ECEF ↔ ECI
-----------------------------

**Earth-Centered Inertial (ECI) vs Earth-Centered Earth-Fixed (ECEF)**

- **ECI**: Fixed to distant stars (inertial). Doesn't rotate with Earth.
- **ECEF**: Rotates with Earth. Fixed to ground stations.

.. code-block:: python

   import numpy as np
   import datetime
   from pytcl.astronomical.reference_frames import ecef_to_eci, eci_to_ecef
   
   # Satellite in ECI (inertial frame)
   sat_eci = np.array([6600e3, 0, 0, 0, 7500, 0])  # 6600 km, 7.5 km/s
   
   # Time of observation
   time = datetime.datetime(2026, 2, 26, 12, 0, 0, tzinfo=datetime.timezone.utc)
   
   # Transform to ECEF (what ground stations see)
   sat_ecef = eci_to_ecef(sat_eci, time)
   
   print(f"ECI position: {sat_eci[:3]}")
   print(f"ECEF position: {sat_ecef[:3]}")
   
   # Transform back
   sat_eci_recovered = ecef_to_eci(sat_ecef, time)
   print(f"Recovered ECI: {sat_eci_recovered[:3]}")

**Components of the Transformation**

The ECI↔ECEF transformation includes:

1. **Precession** - Long-term wobble of Earth's spin axis (~26,000 year period)
2. **Nutation** - Short-term oscillation (~18.6 year period)
3. **Earth Rotation (GMST)** - Daily rotation
4. **Polar Motion** - Shift of rotation axis (~1 meter)

.. code-block:: python

   from pytcl.astronomical.reference_frames import (
       apply_precession, apply_nutation, apply_polar_motion, gmst
   )
   
   # Compute GMST (Greenwich Mean Sidereal Time)
   t = datetime.datetime(2026, 2, 26, 12, 0, 0)
   gst = gmst(t)
   print(f"GMST: {gst:.4f} rad = {gst * 180/np.pi:.2f}°")
   
   # Precession matrix
   prec_matrix = apply_precession(t)
   
   # Nutation matrix
   nut_matrix = apply_nutation(t)
   
   # Polar motion matrix
   pm_matrix = apply_polar_motion(t)

**High-Precision Transformations**

.. code-block:: python

   from pytcl.astronomical.reference_frames import (
       eci2ecef_high_precision, ecef2eci_high_precision
   )
   
   # Use high-precision version for demanding applications
   # (includes more effects: ut1-utc, x/y pole coordinates, etc.)
   
   sat_eci = np.array([6600e3, 0, 0, 0, 7500, 0])
   time = datetime.datetime(2026, 2, 26, 12, 0, 0)
   
   # High precision (requires EOP data)
   sat_ecef = eci2ecef_high_precision(sat_eci, time)

Ephemeris Functions
-------------------

**Sun and Moon Positions**

.. code-block:: python

   from pytcl.astronomical.ephemerides import (
       sun_position, get_moon_position, get_sun_earth_distance
   )
   
   time = datetime.datetime(2026, 2, 26, 12, 0, 0)
   
   # Sun position in ECI (ecliptic coordinates)
   sun_eci = sun_position(time)
   print(f"Sun position: {sun_eci}")
   
   # Moon position in ECI
   moon_eci = get_moon_position(time)
   print(f"Moon position: {moon_eci}")
   
   # Sun-Earth distance (for solar radiation pressure)
   sun_dist = get_sun_earth_distance(time)
   print(f"Sun distance: {sun_dist / 1.496e11:.3f} AU")

**Planet Positions**

.. code-block:: python

   from pytcl.astronomical.ephemerides import planet_position
   
   time = datetime.datetime(2026, 2, 26, 12, 0, 0)
   
   # Available planets: mercury, venus, mars, jupiter, saturn, uranus, neptune
   for planet in ['mars', 'jupiter']:
       pos = planet_position(planet, time)
       print(f"{planet.capitalize()}: {pos}")

**Illumination Angle** (for spacecraft thermal modeling)

.. code-block:: python

   def sun_illumination_angle(sat_ecef, time):
       """
       Compute angle between satellite and Sun (for solar panel orientation)
       
       Returns:
           angle: 0° = directly facing sun, 90° = tangent, 180° = in shadow
       """
       from pytcl.astronomical.ephemerides import sun_position
       from pytcl.astronomical.reference_frames import eci_to_ecef
       
       sun_eci = sun_position(time)
       sun_ecef = eci_to_ecef(sun_eci, time)
       
       # Vector from Earth to satellite
       r_sat = sat_ecef[:3]
       
       # Vector from Earth to sun
       r_sun = sun_ecef[:3]
       
       # Angle
       cos_angle = np.dot(r_sat, r_sun) / (np.linalg.norm(r_sat) * np.linalg.norm(r_sun))
       angle = np.arccos(np.clip(cos_angle, -1, 1))
       
       return angle

SGP4 Propagator (TLE-Based)
---------------------------

**From Two-Line Element (TLE)**

.. code-block:: python

   from pytcl.astronomical import sgp4_propagate, parse_tle
   
   # Example TLE (International Space Station)
   tle_line1 = "1 25544U 98067A   26057.50000000  .00008823  00000-0  15247-3 0  9990"
   tle_line2 = "2 25544  51.6445 121.8140 0003144 208.6915 333.3321 15.53806289 30474"
   
   # Parse TLE
   sat_num, epoch, inclination, raan, eccentricity, arg_perigee, mean_anomaly, mean_motion = parse_tle(tle_line1, tle_line2)
   
   print(f"Satellite number: {sat_num}")
   print(f"Epoch: {epoch}")
   print(f"Mean motion: {mean_motion:.4f} revolutions/day")
   
   # Create propagator
   propagator = sgp4_propagate(tle_line1, tle_line2)
   
   # Propagate from epoch
   minutes_after_epoch = 10
   position, velocity = propagator.propagate_minutes(minutes_after_epoch)
   
   print(f"Position: {position} km")
   print(f"Velocity: {velocity} km/s")

**Batch Propagation of TLE**

.. code-block:: python

   def propagate_tle_constellation(tle_list, times, output_frame='ecef'):
       """
       Propagate constellation of satellites
       
       Args:
           tle_list: List of (tle_line1, tle_line2) tuples
           times: List of datetime objects
           output_frame: 'eci' or 'ecef'
       
       Returns:
           positions: (N_sats, N_times, 3) array
       """
       from pytcl.astronomical.reference_frames import eci_to_ecef
       
       positions = []
       
       for tle_line1, tle_line2 in tle_list:
           propagator = sgp4_propagate(tle_line1, tle_line2)
           
           # Get reference epoch
           epoch_datetime = propagator.epoch
           
           sat_trajectory = []
           for t in times:
               # Minutes from epoch
               td = (t - epoch_datetime).total_seconds() / 60
               
               pos_eci, vel_eci = propagator.propagate_minutes(td)
               
               if output_frame == 'ecef':
                   # Convert to ECEF
                   state_eci = np.concatenate([pos_eci * 1000, vel_eci * 1000])
                   state_ecef = eci_to_ecef(state_eci, t)
                   pos = state_ecef[:3]
               else:
                   pos = pos_eci * 1000  # Convert km to m
               
               sat_trajectory.append(pos)
           
           positions.append(np.array(sat_trajectory))
       
       return np.array(positions)

Lambert's Problem
-----------------

**Bi-Elliptic Transfers** (from one orbit to another)

.. code-block:: python

   from pytcl.astronomical.lambert import lambert_universal
   
   # Initial orbit: LEO
   r1 = 6700e3  # 300 km altitude
   
   # Final orbit: GEO
   r2 = 42164e3  # 35,786 km altitude
   
   # Time of flight for Hohmann transfer
   a_transfer = (r1 + r2) / 2
   mu = 3.986004418e14  # Earth's gravitational parameter
   
   tof_hohmann = np.pi * np.sqrt(a_transfer**3 / mu)
   print(f"Hohmann transfer time: {tof_hohmann / 3600:.2f} hours")
   
   # Solve Lambert's problem
   # Need: position at t0, position at t1, time of flight
   r_initial = np.array([r1, 0, 0])  # Initial position
   r_final = np.array([0, r2, 0])    # Final position (90° transfer)
   
   # Get transfer velocity
   v_initial, v_final = lambert_universal(r_initial, r_final, tof_hohmann, mu=mu)
   
   print(f"Initial velocity: {v_initial}")
   print(f"Final velocity: {v_final}")

Relativistic Corrections
------------------------

**Clock Rate (Proper Time)** (main relativistic correction for GPS/timing)

.. code-block:: python

   from pytcl.astronomical.relativity import proper_time_rate

   # GPS satellite: orbital radius and speed
   rate_sat = proper_time_rate(v=3874.0, r=26560e3)

   # Ground clock on the rotating equator
   rate_ground = proper_time_rate(v=465.0, r=6378e3)

   # GPS satellite clocks run ~38 microseconds/day faster than ground clocks
   delta = rate_sat - rate_ground
   print(f"Fractional rate difference: {delta:.2e}")
   print(f"Clock offset: {delta * 86400 * 1e6:.1f} microseconds/day")

**Shapiro Delay** (photon traveling through spacetime curvature)

.. code-block:: python

   from pytcl.astronomical.relativity import shapiro_delay, GM_EARTH

   # Ground station position (ECEF, equator)
   station = np.array([6378e3, 0, 0])  # meters

   # GPS satellite position
   sat = np.array([26560e3, 0, 0])  # meters

   # Signal delay due to Earth's spacetime curvature
   # (observer, light source, gravitating body position, GM)
   delay = shapiro_delay(station, sat, np.zeros(3), GM_EARTH)

   print(f"Shapiro delay: {delay * 1e12:.1f} picoseconds")

**Relativistic Range Correction** (Shapiro delay as an equivalent range)

.. code-block:: python

   from pytcl.astronomical.relativity import relativistic_range_correction, GM_EARTH

   r1 = 6378e3          # Station geocentric radius (m)
   r2 = 26560e3         # Satellite geocentric radius (m)
   rho = r2 - r1        # Station-satellite range (m)

   dr = relativistic_range_correction(r1, r2, rho, GM_EARTH)
   print(f"Range correction: {dr * 1000:.1f} mm")

Complete Satellite Tracking Example
-----------------------------------

**Track a Satellite from Ground Station**

.. code-block:: python

   import numpy as np
   import datetime
   from pytcl.astronomical.sgp4 import sgp4_propagate
   from pytcl.astronomical.reference_frames import eci_to_ecef
   from pytcl.coordinate_systems.conversions import ecef2enu, ecef2geodetic
   
   class GroundTracker:
       def __init__(self, tle_line1, tle_line2, station_position):
           """
           Initialize ground tracker for a satellite
           
           Args:
               tle_line1, tle_line2: TLE lines
               station_position: [lat, lon, alt] of ground station
           """
           self.propagator = sgp4_propagate(tle_line1, tle_line2)
           self.station_pos = station_position  # Geodetic
           
           # Convert to ECEF
           from pytcl.coordinate_systems.conversions import geodetic2ecef
           self.station_ecef = geodetic2ecef(station_position)
       
       def get_position(self, time):
           """Get satellite position at given time"""
           td = (time - self.propagator.epoch).total_seconds() / 60
           pos_eci_km, vel_eci_km = self.propagator.propagate_minutes(td)
           
           # Convert to ECEF and m
           state_eci = np.concatenate([pos_eci_km * 1000, vel_eci_km * 1000])
           state_ecef = eci_to_ecef(state_eci, time)
           
           return state_ecef
       
       def get_topocentric(self, time):
           """Get satellite position in topocentric (ENU) frame"""
           state_ecef = self.get_position(time)
           pos_ecef = state_ecef[:3]
           
           # Convert to ENU (relative to station)
           enu = ecef2enu(pos_ecef - self.station_ecef, self.station_pos)
           
           # Compute azimuth, elevation, range
           east, north, up = enu
           range_m = np.linalg.norm(enu)
           azimuth = np.arctan2(east, north)  # From north
           elevation = np.arctan2(up, np.sqrt(east**2 + north**2))
           
           return {
               'azimuth': np.degrees(azimuth),
               'elevation': np.degrees(elevation),
               'range': range_m,
               'enu': enu
           }
   
   # Example: Track ISS from NYC
   tle1 = "1 25544U 98067A   26057.50000000  .00008823  00000-0  15247-3 0  9990"
   tle2 = "2 25544  51.6445 121.8140 0003144 208.6915 333.3321 15.53806289 30474"
   
   nyc = np.array([40.7128, -74.0060, 10.0])  # NYC
   tracker = GroundTracker(tle1, tle2, nyc)
   
   # Track for next 90 minutes
   t_start = datetime.datetime(2026, 2, 26, 12, 0, 0)
   for i in range(90):
       t = t_start + datetime.timedelta(minutes=i)
       topo = tracker.get_topocentric(t)
       
       print(f"{i:3d} min: Az={topo['azimuth']:6.1f}°, "
             f"El={topo['elevation']:6.1f}°, Range={topo['range']/1000:7.1f}km")

Performance Considerations
--------------------------

**Propagation Speed**

.. code-block:: python

   import time
   
   # Keplerian propagation: Very fast
   state = orbital_elements_to_state(kep)
   t_start = time.time()
   for _ in range(10000):
       state_prop = kepler_propagate(state, 100)
   t_kepler = time.time() - t_start
   print(f"Keplerian: {t_kepler:.3f}s for 10k propagations")
   
   # SGP4: Slower but includes perturbations
   propagator = sgp4_propagate(tle1, tle2)
   t_start = time.time()
   for _ in range(10000):
       pos, vel = propagator.propagate_minutes(1)
   t_sgp4 = time.time() - t_start
   print(f"SGP4: {t_sgp4:.3f}s for 10k propagations")

**Accuracy vs Speed Tradeoff**

.. code-block:: text

   Method              Accuracy     Speed      Best For
   
   Keplerian           ~1 km        Fastest    Circular orbits, quick estimates
   J2 Perturbation     ~10 km       Fast       Sun-sync, early predictions
   SGP4                ~100 m       Medium     TLE-based (public data)
   N-body with Drag    Best         Slow       Precise long-term (GPS, research)

Common Orbital Scenarios
------------------------

**Sun-Synchronous Orbit Design**

.. code-block:: python

   def design_sun_sync_orbit(altitude_km):
       """Design sun-synchronous orbit"""
       # For sun-sync: RAAN precesses 1°/day to track sun
       
       a = (altitude_km + 6371) * 1000  # meters
       
       # J2 perturbation rate (RAAN precession)
       # dOmega/dt = -1.5 * J2 * (Re/p)² * (5*cos²(i) - 1) / (1-e²)²
       
       # For sun-sync: dOmega/dt = 2π / (365.25 * 86400)
       # Solve for inclination
       
       from scipy.optimize import fsolve
       
       J2 = 1.081874e-3
       Re = 6371e3
       mu = 3.986004418e14
       
       def sun_sync_condition(i):
           e = 0.0  # Assume circular
           p = a * (1 - e**2)
           n = np.sqrt(mu / a**3)
           dOmega_dt = -1.5 * J2 * (Re/p)**2 * (5*np.cos(i)**2 - 1) * n / 2
           
           # Target: 1°/day in radians
           target = 2 * np.pi / (365.25 * 86400)
           return dOmega_dt - target
       
       i_initial = 98.0 * np.pi/180
       i_solution = fsolve(sun_sync_condition, i_initial)[0]
       
       return np.degrees(i_solution)

See Also
~~~~~~~~

- :doc:`architecture` - Module organization
- :doc:`coordinate_systems` - Coordinate transformations
- :doc:`api_navigation` - Finding astronomical functions
- :doc:`kalman_filter_tuning` - Filtering satellite orbits
- ``examples/orbital_mechanics.py`` - Orbit propagation examples
- ``examples/tracking_3d.py`` - 3D tracking examples
