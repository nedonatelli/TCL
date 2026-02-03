Astronomical Computations
==========================

The Tracker Component Library provides comprehensive astronomical and
orbital mechanics functions, including high-precision ephemeris queries,
relativistic corrections, and orbital dynamics.

JPL Development Ephemeris
--------------------------

The ephemeris module provides access to JPL's high-precision Development
Ephemeris (DE) files for computing accurate positions and velocities of
celestial bodies.

Basic Usage
^^^^^^^^^^^

.. code-block:: python

   from pytcl.astronomical.ephemerides import DEEphemeris
   import numpy as np

   # Create ephemeris object (auto-downloads kernel if needed)
   eph = DEEphemeris(version='DE440')

   # Query Sun position at J2000.0
   jd = 2451545.0  # Julian Date
   r_sun, v_sun = eph.sun_position(jd)

   print(f"Sun distance: {np.linalg.norm(r_sun):.6f} AU")
   print(f"Sun velocity: {np.linalg.norm(v_sun):.9f} AU/day")

Available Bodies
^^^^^^^^^^^^^^^^

The ephemeris supports queries for:

* **Sun** - Position relative to Solar System Barycenter (0.007 AU offset)
* **Moon** - Position relative to SSB or Earth-centered
* **Planets** - Mercury, Venus, Mars, Jupiter, Saturn, Uranus, Neptune
* **Barycenters** - For any supported body

.. code-block:: python

   # Query planet positions
   r_mars, v_mars = eph.planet_position('mars', jd)

   # Moon geocentric position
   r_moon_ec, v_moon_ec = eph.moon_position(jd, frame='earth_centered')

   # Any body relative to SSB
   r_body, v_body = eph.barycenter_position('mars', jd)

Ephemeris Versions
^^^^^^^^^^^^^^^^^^

Supported DE versions with coverage:

* **DE440** (latest, 2020) - Covers 1550-2650, highest precision
* **DE432s** (2013) - Covers 1350-3000, high precision for long-term
* **DE430** (2013) - Covers 1550-2650
* **DE405** (1998) - Covers 1600-2200, compact size

.. code-block:: python

   # Use a specific ephemeris version
   eph_432 = DEEphemeris(version='DE432s')
   r, v = eph_432.sun_position(jd)

Frame Support
^^^^^^^^^^^^^

Positions can be returned in different frames:

.. code-block:: python

   # ICRF (default) - International Celestial Reference Frame
   r_icrf, v_icrf = eph.sun_position(jd, frame='icrf')

   # Ecliptic - J2000.0 ecliptic plane
   r_ecliptic, v_ecliptic = eph.sun_position(jd, frame='ecliptic')

   # Earth-centered (Moon only)
   r_ec, v_ec = eph.moon_position(jd, frame='earth_centered')

Kernel Files
^^^^^^^^^^^^

The library automatically downloads JPL ephemeris kernels (~100 MB) on first use.
They are cached in ``~/.jplephem/`` for subsequent accesses.

To manually download kernels:

.. code-block:: bash

   # Download a specific kernel
   mkdir -p ~/.jplephem
   cd ~/.jplephem
   wget https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440.bsp

Module-Level Functions
^^^^^^^^^^^^^^^^^^^^^^^

Convenience functions are available for quick queries:

.. code-block:: python

   from pytcl.astronomical.ephemerides import (
       sun_position,
       moon_position,
       planet_position,
       barycenter_position,
   )

   # Use default DE440 ephemeris
   r_sun, v_sun = sun_position(jd)
   r_mars, v_mars = planet_position('mars', jd)

Relativistic Corrections
-------------------------

The relativity module provides functions for computing relativistic effects
in orbital mechanics and space-time geometry.

Basic Schwarzschild Metrics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from pytcl.astronomical.relativity import (
       schwarzschild_radius,
       gravitational_time_dilation,
   )

   # Schwarzschild radius (event horizon)
   M_sun = 1.989e30  # kg
   Rs = schwarzschild_radius(M_sun)
   print(f"Sun's Schwarzschild radius: {Rs:.2f} m")

   # Time dilation for object at distance r from mass M
   M = 1e24  # kg
   r = 7e6   # meters
   time_dilation = gravitational_time_dilation(M, r)
   print(f"Time dilation factor: {time_dilation:.10f}")

Orbital Precession
^^^^^^^^^^^^^^^^^^^

Relativistic perihelion precession calculations:

.. code-block:: python

   from pytcl.astronomical.relativity import schwarzschild_precession_per_orbit

   # Mercury around Sun
   M_sun = 1.989e30  # kg
   a = 5.79e10       # semi-major axis (m)
   e = 0.2056        # eccentricity

   precession = schwarzschild_precession_per_orbit(M_sun, a, e)
   print(f"Precession: {precession:.6f} rad/orbit")
   # For Mercury: ~5.03e-7 rad/orbit ≈ 43 arcsec/century

GPS Time Effects
^^^^^^^^^^^^^^^^

Relativistic effects affecting GPS:

.. code-block:: python

   from pytcl.astronomical.relativity import proper_time_rate

   # GPS satellite velocity and orbital parameters
   v_orbit = 3874.0  # m/s
   M_earth = 5.972e24  # kg
   r_orbit = 2.66e7   # meters (GPS altitude)

   # Proper time rate (compared to Earth surface)
   rate = proper_time_rate(v_orbit, M_earth, r_orbit)

   # Time offset per day
   seconds_per_day = 86400
   offset = (rate - 1.0) * seconds_per_day
   print(f"Time offset: {offset*1e6:.2f} microseconds/day")
   # GPS: ~21.6 microseconds/day (7.2 µs from SR, 46 µs from GR)

Light Propagation Effects
^^^^^^^^^^^^^^^^^^^^^^^^^^

Shapiro delay - light travel delay in gravitational fields:

.. code-block:: python

   from pytcl.astronomical.relativity import shapiro_delay

   # Light traveling near the Sun
   M_sun = 1.989e30  # kg
   r_min = 1e11      # closest approach (meters)

   delay = shapiro_delay(M_sun, r_min)
   print(f"Shapiro delay: {delay*1e6:.2f} microseconds")

Post-Newtonian Effects
^^^^^^^^^^^^^^^^^^^^^^

First-order post-Newtonian orbital accelerations:

.. code-block:: python

   from pytcl.astronomical.relativity import post_newtonian_acceleration

   # Orbital parameters
   r = np.array([1e11, 0, 0])  # position (m)
   v = np.array([0, 30000, 0])  # velocity (m/s)
   M = 1.989e30  # mass (kg)

   a_pn = post_newtonian_acceleration(r, v, M)
   print(f"1PN acceleration: {a_pn}")

Geodetic Precession (De Sitter Effect)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from pytcl.astronomical.relativity import geodetic_precession

   # Orbital parameters
   M = 1.989e30  # Sun mass (kg)
   a = 5.79e10   # semi-major axis (m)

   # Precession rate
   rate = geodetic_precession(M, a)
   print(f"Geodetic precession rate: {rate:.9f} rad/s")

Frame-Dragging (Lense-Thirring Effect)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from pytcl.astronomical.relativity import lense_thirring_precession

   # Spinning body parameters
   M = 5.972e24     # Earth mass (kg)
   J = 8.04e37      # angular momentum (kg⋅m²/s)
   r = 4.22e7       # orbital radius (m)

   # Frame-dragging precession
   rate = lense_thirring_precession(M, J, r)
   print(f"Lense-Thirring precession: {rate:.12f} rad/s")

Orbital Mechanics
-----------------

The astronomical module provides comprehensive orbital mechanics functions:

.. code-block:: python

   from pytcl.astronomical.orbital_mechanics import (
       orbital_elements_from_state,
       state_from_orbital_elements,
       kepler_propagation,
   )

   # Convert state to orbital elements
   r = np.array([6.378e6, 0, 0])      # position (m)
   v = np.array([0, 7560, 0])          # velocity (m/s)
   mu = 3.986e14                       # Earth's GM

   a, e, i, Omega, omega, M = orbital_elements_from_state(r, v, mu)

   # Propagate orbit
   times = np.linspace(0, 3600, 100)   # 1 hour
   states = kepler_propagation(r, v, mu, times)

Reference Frame Transformations
--------------------------------

The reference frames module provides coordinate system conversions:

.. code-block:: python

   from pytcl.astronomical.reference_frames import (
       equatorial_to_ecliptic,
       gcrf_to_itrf,
       precessional_matrix,
   )

   # Convert between coordinate systems
   r_equatorial = np.array([1, 0, 0])
   r_ecliptic = equatorial_to_ecliptic(r_equatorial)

   # Precession effects
   jd1 = 2451545.0
   jd2 = 2451545.0 + 36525.0  # 100 years later
   P = precessional_matrix(jd1, jd2)
   r_precessed = P @ r_equatorial

Time Systems
------------

Astronomical time conversions and calculations:

.. code-block:: python

   from pytcl.astronomical.time_systems import (
       jd_to_mjd,
       mjd_to_jd,
       utc_to_jd,
       jd_to_utc,
   )

   # Julian Date conversions
   jd = 2451545.0  # J2000.0 epoch
   mjd = jd_to_mjd(jd)

   # UTC conversions
   from datetime import datetime
   dt = datetime(2000, 1, 1, 12, 0, 0)
   jd = utc_to_jd(dt)

Lambert Problem
---------------

Solve for orbits connecting two positions:

.. code-block:: python

   from pytcl.astronomical.lambert import lambert_universal, lambert_battin

   # Initial and final positions
   r1 = np.array([6.378e6, 0, 0])
   r2 = np.array([0, 6.378e6, 0])

   # Transfer time
   tof = 2700.0  # seconds
   mu = 3.986e14

   # Solve Lambert problem
   v1, v2 = lambert_universal(r1, r2, tof, mu)
   print(f"Initial velocity: {v1}")
   print(f"Final velocity: {v2}")

See Also
--------

* :doc:`../api/astronomical` - Complete API reference
* `JPL Ephemeris Documentation <https://ssd.jpl.nasa.gov/?ephemerides>`_
* `NAIF SPICE Toolkit <https://naif.jpl.nasa.gov/naif/>`_
