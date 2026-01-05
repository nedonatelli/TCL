Advanced Reference Frames
=========================

This example demonstrates advanced reference frame transformations including PEF (Pseudo-Earth Fixed) and SEZ (South-East-Zenith) frames for satellite tracking and Earth observation.

Overview
--------

Reference frame transformations are essential for:

- **Satellite tracking**: Ground station to satellite geometry
- **Radar observations**: Azimuth and elevation calculations
- **Navigation**: Inertial to Earth-fixed conversions
- **Astrometry**: Precise position measurements

Transformation Chain
--------------------

The complete transformation from inertial to Earth-fixed coordinates::

    GCRF (inertial)
        |
        v (precession)
    MOD (Mean of Date)
        |
        v (nutation)
    TOD (True of Date)
        |
        v (Earth rotation)
    PEF (Pseudo-Earth Fixed)
        |
        v (polar motion)
    ITRF (International Terrestrial Reference Frame)

**GCRF**: Geocentric Celestial Reference Frame (inertial)

**PEF**: Excludes polar motion, useful for intermediate calculations

**ITRF**: Standard Earth-fixed frame for geodetic coordinates

SEZ Frame
---------

The South-East-Zenith frame is horizon-relative:

**South (S)**: Points toward geographic south
**East (E)**: Points toward geographic east
**Zenith (Z)**: Points away from Earth center (up)

Applications:

- Radar and antenna azimuth/elevation
- Line-of-sight observations
- Ground station to satellite geometry
- Horizon crossing calculations

Examples Demonstrated
---------------------

**PEF Intermediate Frame**
   - GCRF to PEF transformation
   - Polar motion effects (PEF vs ITRF)
   - Roundtrip verification

**SEZ Radar Observations**
   - Ground station coordinates
   - Satellite position in SEZ
   - Range, azimuth, elevation computation
   - Visibility determination

**LEO Satellite Tracking**
   - Satellite pass over ground station
   - Time evolution of azimuth/elevation
   - Polar plot of satellite track
   - Maximum elevation and range

**Earth Observation Geometry**
   - Multiple ground stations
   - Satellite visibility analysis
   - Observation feasibility

Code Highlights
---------------

The example demonstrates:

- GCRF to ITRF with ``gcrf_to_itrf()``
- GCRF to PEF with ``gcrf_to_pef()``
- Geodetic to SEZ with ``geodetic2sez()``
- Julian date computation with ``cal_to_jd()``
- Polar motion corrections

Source Code
-----------

.. literalinclude:: ../../examples/reference_frame_advanced.py
   :language: python
   :linenos:

Running the Example
-------------------

.. code-block:: bash

   python examples/reference_frame_advanced.py

See Also
--------

- :doc:`coordinate_systems` - Basic coordinate transformations
- :doc:`coordinate_visualization` - 3D frame visualizations
- :doc:`ephemeris_demo` - Planetary positions
