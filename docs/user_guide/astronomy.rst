Astronomical Computations
==========================

The Tracker Component Library provides astronomical and orbital mechanics
functions, including JPL ephemeris queries, relativistic corrections, and
orbital dynamics.

.. note::

   The ephemeris functions require the ``astronomy`` extra
   (``pip install nrl-tracker[astronomy]``), which provides ``jplephem``
   and ``astropy``. The relativity, orbital mechanics, reference frame,
   time system, and Lambert functions need only the core install.

JPL Development Ephemeris
--------------------------

The ephemeris module provides access to JPL's high-precision Development
Ephemeris (DE) files for computing accurate positions and velocities of
celestial bodies. Positions are returned in AU and velocities in AU/day,
relative to the Solar System Barycenter (SSB) unless stated otherwise.

Basic Usage
^^^^^^^^^^^

.. code-block:: python

   from pytcl.astronomical.ephemerides import DEEphemeris
   import numpy as np

   # Create ephemeris object (auto-downloads kernel if needed)
   eph = DEEphemeris(version='DE440')

   # Query Sun position at J2000.0.  The Sun orbits the SSB within
   # a couple of solar radii, so its SSB distance is small.
   jd = 2451545.0  # Julian Date (TT)
   r_sun, v_sun = eph.sun_position(jd)

   print(f"Sun distance from SSB: {np.linalg.norm(r_sun):.6f} AU")
   print(f"Sun speed: {np.linalg.norm(v_sun):.9f} AU/day")

Output:

.. code-block:: text

   Sun distance from SSB: 0.007668 AU
   Sun speed: 0.000009154 AU/day

Available Bodies
^^^^^^^^^^^^^^^^

The ephemeris supports queries for:

* **Sun** - Position relative to Solar System Barycenter (~0.007 AU offset)
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
   eph_430 = DEEphemeris(version='DE430')
   r, v = eph_430.sun_position(jd)

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

The library automatically downloads JPL ephemeris kernels (~100-120 MB)
on first use. They are cached in ``~/.jplephem/`` for subsequent accesses.

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
in orbital mechanics. The functions take a gravitational parameter
:math:`GM` (in m^3/s^2) rather than a mass; ``GM_EARTH`` and ``GM_SUN``
constants are provided, and ``GM_EARTH`` is the default for most functions.

Basic Schwarzschild Metrics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from pytcl.astronomical.relativity import (
       schwarzschild_radius,
       gravitational_time_dilation,
   )

   # Schwarzschild radius (event horizon) takes a mass in kg
   M_sun = 1.989e30  # kg
   Rs = schwarzschild_radius(M_sun)
   print(f"Sun's Schwarzschild radius: {Rs:.2f} m")

   # Time dilation factor at distance r from Earth's center
   r = 7e6  # meters
   factor = gravitational_time_dilation(r)  # gm defaults to GM_EARTH
   print(f"Time dilation factor: {factor:.12f}")

Output:

.. code-block:: text

   Sun's Schwarzschild radius: 2954.13 m
   Time dilation factor: 0.999999999366

Orbital Precession
^^^^^^^^^^^^^^^^^^^

Relativistic perihelion precession calculations:

.. code-block:: python

   from pytcl.astronomical.relativity import (
       GM_SUN,
       schwarzschild_precession_per_orbit,
   )

   # Mercury around the Sun
   a = 5.79e10  # semi-major axis (m)
   e = 0.2056   # eccentricity

   precession = schwarzschild_precession_per_orbit(a, e, gm=GM_SUN)
   print(f"Precession: {precession:.6e} rad/orbit")

Output:

.. code-block:: text

   Precession: 5.019383e-07 rad/orbit

For Mercury this accumulates to the famous ~43 arcsec/century.

GPS Time Effects
^^^^^^^^^^^^^^^^

``proper_time_rate(v, r)`` combines special relativistic (velocity) and
general relativistic (gravity) time dilation, relative to a distant
observer at rest. The familiar GPS number is the *difference* between the
satellite clock and a clock on the ground:

.. code-block:: python

   from pytcl.astronomical.relativity import proper_time_rate

   # GPS satellite: ~3.87 km/s at r ~ 26,600 km
   rate_sat = proper_time_rate(3874.0, 2.66e7)

   # Ground clock: equatorial rotation speed at Earth's surface
   rate_ground = proper_time_rate(465.0, 6.371e6)

   seconds_per_day = 86400
   offset = (rate_sat / rate_ground - 1.0) * seconds_per_day
   print(f"GPS clock gain: {offset*1e6:.1f} microseconds/day")

Output:

.. code-block:: text

   GPS clock gain: 38.6 microseconds/day

(The satellite clock runs fast by ~45 us/day from gravity and slow by
~7 us/day from velocity, for a net gain of ~38.6 us/day.)

Light Propagation Effects
^^^^^^^^^^^^^^^^^^^^^^^^^^

Shapiro delay - the extra light travel time in a gravitational field.
The function takes observer, source, and gravitating-body positions:

.. code-block:: python

   from pytcl.astronomical.relativity import shapiro_delay

   # Earth-to-Mars signal passing near the Sun (superior conjunction)
   AU = 1.495978707e11
   observer = np.array([AU, 1e9, 0.0])         # Earth (m)
   source = np.array([-1.52 * AU, -1e9, 0.0])  # Mars, opposite side (m)
   sun = np.zeros(3)

   delay = shapiro_delay(observer, source, sun)  # gm defaults to GM_SUN
   print(f"Shapiro delay: {delay*1e6:.1f} microseconds")

Output:

.. code-block:: text

   Shapiro delay: 147.5 microseconds

Post-Newtonian Effects
^^^^^^^^^^^^^^^^^^^^^^

``post_newtonian_acceleration`` returns the *total* acceleration
(Newtonian plus the first post-Newtonian Schwarzschild correction):

.. code-block:: python

   from pytcl.astronomical.relativity import post_newtonian_acceleration

   r_vec = np.array([1e11, 0.0, 0.0])     # heliocentric position (m)
   v_vec = np.array([0.0, 30000.0, 0.0])  # velocity (m/s)

   a_total = post_newtonian_acceleration(r_vec, v_vec, gm=GM_SUN)
   a_newt = -GM_SUN / np.linalg.norm(r_vec) ** 3 * r_vec

   print(f"Total acceleration: {a_total} m/s^2")
   print(f"1PN correction magnitude: {np.linalg.norm(a_total - a_newt):.3e} m/s^2")

Output:

.. code-block:: text

   Total acceleration: [-0.01327124  0.          0.        ] m/s^2
   1PN correction magnitude: 6.510e-10 m/s^2

Geodetic Precession (De Sitter Effect)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Precession per orbit of a gyroscope transported around a central mass:

.. code-block:: python

   from pytcl.astronomical.relativity import geodetic_precession

   # GPS-like orbit around Earth
   a_orbit = 2.66e7  # semi-major axis (m)
   e_orbit = 0.01
   inc = np.deg2rad(55.0)

   dw = geodetic_precession(a_orbit, e_orbit, inc)  # gm defaults to GM_EARTH
   print(f"Geodetic precession: {dw:.3e} rad/orbit")

Output:

.. code-block:: text

   Geodetic precession: 1.572e-09 rad/orbit

Frame-Dragging (Lense-Thirring Effect)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Nodal precession rate caused by the spin of the central body:

.. code-block:: python

   from pytcl.astronomical.relativity import lense_thirring_precession

   J_earth = 5.86e33  # Earth's spin angular momentum (kg m^2/s)
   rate = lense_thirring_precession(a_orbit, e_orbit, inc, J_earth)
   print(f"Lense-Thirring nodal rate: {rate:.3e} rad/s")

Output:

.. code-block:: text

   Lense-Thirring nodal rate: 4.625e-16 rad/s

Orbital Mechanics
-----------------

The orbital mechanics module works with ``StateVector`` (position/velocity)
and ``OrbitalElements`` named tuples. Distances are in km, velocities in
km/s, and the gravitational parameter defaults to Earth's
(398600.4418 km^3/s^2):

.. code-block:: python

   from pytcl.astronomical.orbital_mechanics import (
       StateVector,
       state_to_orbital_elements,
       orbital_elements_to_state,
       kepler_propagate_state,
   )

   r0 = np.array([7000.0, 0.0, 0.0])  # position (km)
   v0 = np.array([0.0, 7.546, 0.0])   # velocity (km/s)
   state = StateVector(r=r0, v=v0)

   # Convert state to classical orbital elements (a, e, i, raan, omega, nu)
   elements = state_to_orbital_elements(state)
   print(f"a = {elements.a:.1f} km, e = {elements.e:.4f}")

   # Two-body propagation via Kepler's equation
   state_1h = kepler_propagate_state(state, 3600.0)
   print(f"Position after 1 h: {state_1h.r.round(1)} km")

Output:

.. code-block:: text

   a = 6999.9 km, e = 0.0000
   Position after 1 h: [-5172.3 -4716.5     0. ] km

``kepler_propagate`` propagates ``OrbitalElements`` directly, and
``orbital_elements_to_state`` converts elements back to a state vector.

Reference Frame Transformations
--------------------------------

The reference frames module provides coordinate system conversions:

.. code-block:: python

   from pytcl.astronomical.reference_frames import (
       equatorial_to_ecliptic,
       mean_obliquity_iau80,
       precession_matrix_iau76,
   )

   # Equatorial to ecliptic requires the obliquity of the ecliptic
   jd = 2451545.0
   eps = mean_obliquity_iau80(jd)
   r_equatorial = np.array([1.0, 0.0, 0.0])
   r_ecliptic = equatorial_to_ecliptic(r_equatorial, eps)

   # IAU 1976 precession from J2000.0 to a target epoch
   jd_future = jd + 36525.0  # 100 years later
   P = precession_matrix_iau76(jd_future)
   r_precessed = P @ r_equatorial

Additional transformations include ``gcrf_to_itrf`` / ``itrf_to_gcrf``,
``teme_to_itrf``, ``eci_to_ecef`` / ``ecef_to_eci``, nutation and polar
motion matrices, and sidereal time functions (``gmst_iau82``,
``gast_iau82``).

Time Systems
------------

Astronomical time conversions and calculations:

.. code-block:: python

   from pytcl.astronomical.time_systems import (
       jd_to_mjd,
       mjd_to_jd,
       cal_to_jd,
       jd_to_cal,
   )

   # Julian Date conversions
   jd = 2451545.0  # J2000.0 epoch
   mjd = jd_to_mjd(jd)
   print(f"MJD: {mjd}")

   # Calendar conversions
   jd2 = cal_to_jd(2000, 1, 1, 12, 0, 0.0)
   print(f"JD for 2000-01-01 12:00: {jd2}")
   print(f"Back to calendar: {jd_to_cal(jd2)}")

Output:

.. code-block:: text

   MJD: 51544.5
   JD for 2000-01-01 12:00: 2451545.0
   Back to calendar: (2000, 1, 1, 12, 0, 0.0)

Time scale conversions between UTC, TAI, TT, GPS, and Unix time are also
available (``utc_to_tai``, ``tai_to_tt``, ``utc_to_gps``, ``unix_to_jd``,
and friends), along with sidereal time (``gmst``, ``gast``).

Lambert Problem
---------------

Solve for orbits connecting two positions. Positions are in km, the time
of flight in seconds, and the solvers return a ``LambertSolution`` named
tuple with fields ``v1``, ``v2``, ``a``, ``e``, and ``tof``:

.. code-block:: python

   from pytcl.astronomical.lambert import lambert_universal, lambert_izzo

   r1 = np.array([7000.0, 0.0, 0.0])  # km
   r2 = np.array([0.0, 7000.0, 0.0])  # km
   tof = 2700.0                       # seconds

   sol = lambert_universal(r1, r2, tof)  # mu defaults to Earth's, km^3/s^2
   print(f"Initial velocity: {sol.v1.round(3)} km/s")
   print(f"Final velocity: {sol.v2.round(3)} km/s")

Output:

.. code-block:: text

   Initial velocity: [3.585 5.964 0.   ] km/s
   Final velocity: [-5.964 -3.585 -0.   ] km/s

See Also
--------

* :doc:`../api/astronomical` - Complete API reference
* `JPL Ephemeris Documentation <https://ssd.jpl.nasa.gov/?ephemerides>`_
* `NAIF SPICE Toolkit <https://naif.jpl.nasa.gov/naif/>`_
