Relativistic Effects
====================

This example demonstrates relativistic corrections for precision applications.

.. raw:: html

   <div class="plotly-container">
       <iframe class="plotly-iframe" src="../../_static/images/examples/relativity_effects.html"></iframe>
   </div>

Overview
--------

Relativistic effects are significant for:

- **GNSS timing**: Satellite clock corrections
- **Deep space navigation**: Signal propagation delays
- **Precision timing**: Gravitational time dilation
- **Astrometry**: Light deflection near Sun

Effects Covered
---------------

**Gravitational Time Dilation**
   - Clocks run slower in stronger gravity
   - GPS satellite clocks run ~45 μs/day fast
   - Essential for GNSS accuracy

**Special Relativistic Time Dilation**
   - Moving clocks run slower
   - GPS satellites: ~7 μs/day slow
   - Combined effect: ~38 μs/day fast

.. raw:: html

   <div class="plotly-container">
       <iframe class="plotly-iframe" src="../../_static/images/examples/geophysical_gravity_altitude.html"></iframe>
   </div>

**Gravitational Potential**: Gravity decreases with altitude, affecting time dilation for satellites at different orbital heights.

**Geodetic Precession**
   - Spin axis precession in curved spacetime
   - De Sitter precession: ~1.9 arcsec/year
   - Lense-Thirring: frame dragging

.. raw:: html

   <div class="plotly-container aspect-square">
       <iframe class="plotly-iframe" src="../../_static/images/examples/orbital_propagation.html"></iframe>
   </div>

**Orbital Motion**: Relativistic effects accumulate over orbital periods, requiring corrections for precision applications.

**Shapiro Delay**
   - Light delay in gravitational field
   - Solar conjunction corrections
   - Affects planetary radar

.. raw:: html

   <div class="plotly-container aspect-square">
       <iframe class="plotly-iframe" src="../../_static/images/examples/ephemeris_demo.html"></iframe>
   </div>

**Planetary Positions**: Shapiro delay corrections are essential for ranging to planets, especially during solar conjunction.

Code Highlights
---------------

The example demonstrates:

- Time dilation computation with ``gravitational_time_dilation()``
- Shapiro delay with ``shapiro_delay()``
- Geodetic precession with ``geodetic_precession()``
- Combined corrections for satellite clocks

Source Code
-----------

.. literalinclude:: ../../../examples/relativity_demo.py
   :language: python
   :linenos:

Running the Example
-------------------

.. code-block:: bash

   python examples/relativity_demo.py

See Also
--------

- :doc:`orbital_mechanics` - Orbital propagation
- :doc:`ephemeris_demo` - Planetary positions
- :doc:`ins_gnss_navigation` - GNSS applications
