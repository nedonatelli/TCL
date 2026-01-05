Atmospheric Modeling
====================

This example demonstrates NRLMSISE-00 high-fidelity atmosphere model and drag calculations for analyzing satellite orbital decay and atmospheric interactions.

.. raw:: html

   <div class="plotly-container">
       <iframe class="plotly-iframe" src="../../_static/images/examples/geophysical_gravity_altitude.html"></iframe>
   </div>

Overview
--------

Atmospheric density models are essential for:

- **LEO satellite operations**: Predicting orbital decay and drag
- **Reentry analysis**: Determining vehicle heating and trajectory
- **Space weather effects**: Understanding solar activity impacts
- **Mission planning**: Fuel requirements for station-keeping

Key Scenarios
-------------

**ISS Altitude Profile**
   Atmospheric density varies significantly with solar activity level,
   affecting ISS orbital decay rate and reboost requirements.

**Satellite Drag**
   Drag coefficients depend on satellite geometry and atmospheric
   composition at orbital altitudes.

**Temperature Profiles**
   The thermosphere temperature increases dramatically with altitude
   and varies with solar flux (F10.7) and geomagnetic activity (Ap).

**Composition Transitions**
   The atmosphere transitions from molecular (N2, O2) to atomic (O, He, H)
   species with increasing altitude.

.. raw:: html

   <div class="plotly-container aspect-square">
       <iframe class="plotly-iframe" src="../../_static/images/examples/orbital_propagation.html"></iframe>
   </div>

**Orbital Decay**: Atmospheric drag causes LEO satellites to gradually lose altitude, with decay rate depending on solar activity levels.

Models Covered
--------------

**NRLMSISE-00**
   - Naval Research Laboratory Mass Spectrometer and Incoherent Scatter Radar
   - High-fidelity empirical model from sea level to 1000+ km
   - Accounts for solar activity (F10.7) and geomagnetic storms (Ap)

**US Standard Atmosphere 1976**
   - Reference atmosphere up to 85 km
   - Useful for comparison and validation

Code Highlights
---------------

The example demonstrates:

- Density profiles across altitude with ``NRLMSISE00()``
- Species composition (N2, O2, O, He, H, Ar, N) vs altitude
- Temperature profiles under various solar activity levels
- Solar flux (F10.7) effects on ISS-altitude conditions
- Comparison with US Standard Atmosphere 1976

Source Code
-----------

.. literalinclude:: ../../../examples/atmospheric_modeling.py
   :language: python
   :linenos:

Running the Example
-------------------

.. code-block:: bash

   python examples/atmospheric_modeling.py

See Also
--------

- :doc:`orbital_mechanics` - Orbital propagation
- :doc:`relativity_demo` - Relativistic corrections
- :doc:`ephemeris_demo` - Planetary positions
