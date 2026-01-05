High-Precision Ephemeris
========================

This example demonstrates using the JPL Development Ephemeris (DE) to compute high-precision positions of the Sun, Moon, and planets.

.. raw:: html

   <div class="plotly-container">
       <iframe class="plotly-iframe" src="../_static/images/examples/ephemeris_demo.html"></iframe>
   </div>

Overview
--------

Planetary ephemerides provide high-accuracy positions essential for:

- **Deep space navigation**: Spacecraft trajectory planning
- **Astronomy**: Telescope pointing and observation scheduling
- **Satellite operations**: Eclipse and conjunction predictions
- **Time systems**: Planetary aberration corrections

Ephemeris Versions
------------------

**DE405** (1997)
   JPL Planetary Ephemeris covering 1997-2050

**DE430** (2013)
   Extended coverage from 1550-2650

**DE440** (2020)
   Latest JPL ephemeris with improved accuracy

Examples Covered
----------------

**Sun Position**
   - Heliocentric distance variation (Earth's orbital eccentricity)
   - Perihelion and aphelion distances
   - Position in ICRF (International Celestial Reference Frame)

**Moon Position**
   - Earth-centered position and velocity
   - Perigee and apogee variations
   - Lunar orbital ellipticity

**Planetary Positions**
   - All major planets: Mercury through Neptune
   - Heliocentric coordinates
   - Ecliptic longitude and latitude

**Solar System Barycenter**
   - Center of mass of the solar system
   - Jupiter's gravitational influence
   - Reference for high-precision astrometry

**Reference Frames**
   - ICRF (default inertial frame)
   - Ecliptic frame transformations
   - Earth-centered coordinates

Code Highlights
---------------

The example demonstrates:

- Sun position queries with ``sun_position()``
- Moon position queries with ``moon_position()``
- Planet positions with ``planet_position()``
- Barycenter calculations with ``barycenter_position()``
- Julian date conversions with ``jd_to_cal()``
- Ephemeris version selection with ``DEEphemeris()``

Source Code
-----------

.. literalinclude:: ../../examples/ephemeris_demo.py
   :language: python
   :linenos:

Running the Example
-------------------

.. code-block:: bash

   python examples/ephemeris_demo.py

See Also
--------

- :doc:`orbital_mechanics` - Orbital propagation
- :doc:`relativity_demo` - Relativistic corrections
- :doc:`coordinate_systems` - Coordinate transformations
