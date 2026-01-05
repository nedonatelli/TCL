Geophysical Models
==================

This example demonstrates gravity and magnetic field models essential for high-precision navigation, geodesy, and aerospace applications.

.. raw:: html

   <div class="plotly-container">
       <iframe class="plotly-iframe" src="../_static/images/examples/geophysical_models.html"></iframe>
   </div>

Overview
--------

Geophysical models are critical for:

- **Inertial navigation**: Gravity compensation in INS
- **Geodesy**: Height reference and surveying
- **Aerospace**: Satellite orbit determination
- **Geophysics**: Subsurface exploration

Gravity Models
--------------

**Normal Gravity (Somigliana)**
   - Gravity variation with latitude
   - ~0.5% increase from equator to poles
   - Due to Earth's rotation and flattening

**WGS84 Gravity Model**
   - Full gravity vector computation
   - Includes deflection of vertical
   - Essential for inertial navigation

**J2 Gravity Model**
   - Simplified model using J2 oblateness term
   - Adequate for many applications
   - Faster computation than full models

**Geoid Height**
   - Separation between geoid and ellipsoid
   - J2 approximation captures main flattening
   - Full models (EGM96/2008) for precision

**Gravity Anomalies**
   - Free-air anomaly
   - Gravity disturbance
   - Used for geophysical exploration

**Tidal Effects**
   - Solid Earth tide displacement (~30 cm)
   - Tidal gravity variations (~300 uGal)
   - Essential for precision gravimetry

Magnetic Field Models
---------------------

**World Magnetic Model (WMM2020)**
   - Standard model for navigation
   - Updated every 5 years
   - Valid 2020-2025

**IGRF-13**
   - International Geomagnetic Reference Field
   - Historical and predictive coefficients
   - Used for scientific applications

**Key Parameters**
   - Declination: angle between true and magnetic north
   - Inclination: dip angle of field lines
   - Total intensity: field strength in nT

**Notable Features**
   - South Atlantic Anomaly: weak field region
   - Magnetic poles: ~11° offset from geographic
   - Secular variation: field changes over time

Code Highlights
---------------

The example demonstrates:

- Normal gravity with ``normal_gravity_somigliana()``
- WGS84 gravity with ``gravity_wgs84()``
- Geoid height with ``geoid_height_j2()``
- Free-air anomaly with ``free_air_anomaly()``
- Solid Earth tides with ``solid_earth_tide_displacement()``
- Magnetic field with ``wmm()`` and ``igrf()``
- Magnetic declination with ``magnetic_declination()``

Source Code
-----------

.. literalinclude:: ../../examples/geophysical_models.py
   :language: python
   :linenos:

Running the Example
-------------------

.. code-block:: bash

   python examples/geophysical_models.py

See Also
--------

- :doc:`ins_gnss_navigation` - INS/GNSS integration
- :doc:`navigation_geodesy` - Geodetic calculations
- :doc:`magnetism_demo` - Magnetic field details
