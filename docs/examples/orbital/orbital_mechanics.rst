Orbital Mechanics
=================

This example demonstrates orbit propagation, Kepler's equation, and Lambert's problem.

.. raw:: html

   <div class="plotly-container aspect-square">
       <iframe class="plotly-iframe" src="../../_static/images/examples/orbital_propagation.html"></iframe>
   </div>

Overview
--------

Orbital mechanics for satellite tracking and space applications:

- **Two-body problem**: Keplerian orbits
- **Orbit propagation**: State evolution over time
- **Reference frames**: GCRF/ITRF conversions
- **Orbital maneuvers**: Hohmann, Lambert transfers

Key Concepts
------------

- **Orbital elements**: Semi-major axis, eccentricity, inclination
- **Kepler's equation**: Mean anomaly to eccentric anomaly
- **State vectors**: Position and velocity in inertial frame
- **Time systems**: UTC/TAI/GPS conversions and Julian dates

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

**Transfer Orbits**
   - Hohmann two-impulse transfer
   - Minimum-energy transfer
   - Delta-v budgeting

Code Highlights
---------------

The example demonstrates:

- State vector to orbital elements conversion with
  ``state_to_orbital_elements()``
- Kepler equation solving with ``mean_to_eccentric_anomaly()``
- Two-body propagation with ``kepler_propagate()``
- Lambert solvers ``lambert_universal()`` and ``lambert_izzo()``
- GCRF/ITRF frame conversions and Hohmann transfer design

Source Code
-----------

.. literalinclude:: ../../../examples/orbital_mechanics.py
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
