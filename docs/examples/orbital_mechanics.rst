Orbital Mechanics
=================

This example demonstrates orbit propagation, Kepler's equation, and Lambert's problem.

.. raw:: html

   <div class="plotly-container aspect-square">
       <iframe class="plotly-iframe" src="../_static/images/examples/orbital_propagation.html"></iframe>
   </div>

Overview
--------

Orbital mechanics for satellite tracking and space applications:

- **Two-body problem**: Keplerian orbits
- **Orbit propagation**: State evolution over time
- **SGP4/SDP4**: TLE-based propagation
- **Orbital maneuvers**: Hohmann, Lambert transfers

Key Concepts
------------

- **Orbital elements**: Semi-major axis, eccentricity, inclination
- **Kepler's equation**: Mean anomaly to eccentric anomaly
- **State vectors**: Position and velocity in inertial frame
- **Perturbations**: J2, drag, solar radiation pressure

Algorithms
----------

**Kepler's Equation**
   - Iterative solution (Newton-Raphson)
   - Universal variable formulation
   - Handles all orbit types

**Lambert's Problem**
   - Find orbit connecting two points
   - Given transfer time
   - Used for rendezvous planning

**SGP4/SDP4**
   - NORAD propagator for TLEs
   - Includes perturbations
   - Standard for satellite tracking

Code Highlights
---------------

The example demonstrates:

- State vector to orbital elements conversion
- Kepler equation solving with ``solve_kepler()``
- Two-body propagation with ``propagate_twobody()``
- Lambert solver with ``solve_lambert()``
- TLE parsing and SGP4 propagation

Source Code
-----------

.. literalinclude:: ../../examples/orbital_mechanics.py
   :language: python
   :linenos:

Running the Example
-------------------

.. code-block:: bash

   python examples/orbital_mechanics.py

See Also
--------

- :doc:`ephemeris_demo` - Planetary ephemeris
- :doc:`relativity_demo` - Relativistic corrections
