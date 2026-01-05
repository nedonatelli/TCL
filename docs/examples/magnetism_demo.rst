Magnetism Models
================

This example demonstrates the pytcl.magnetism module capabilities, including World Magnetic Model (WMM2020) coefficients, dipole moment calculations, and geomagnetic field properties.

Overview
--------

Earth's magnetic field models are essential for:

- **Navigation**: Compass corrections and heading reference
- **Aerospace**: Attitude determination using magnetometers
- **Geophysics**: Understanding Earth's core dynamics
- **Space weather**: Radiation environment modeling

WMM2020 Coefficients
--------------------

The World Magnetic Model uses spherical harmonic coefficients:

**Gauss Coefficients (g, h)**
   - g[n,m]: coefficients for cos(m*lambda) terms
   - h[n,m]: coefficients for sin(m*lambda) terms
   - Units: nanoTesla (nT)

**Secular Variation (g_dot, h_dot)**
   - Rate of change of coefficients
   - Units: nT/year
   - Used for temporal extrapolation

**Epoch and Validity**
   - WMM2020 epoch: 2020.0
   - Valid period: 2020-2025
   - Maximum order: n_max = 12

Dipole Properties
-----------------

**Magnetic Dipole Moment**
   - Earth's main field approximated as dipole
   - Moment: ~8 x 10^22 A*m^2
   - Decreasing ~5% per century

**Dipole Axis**
   - Tilted ~11° from rotation axis
   - Magnetic poles vs geographic poles
   - Axis drifts over time (secular variation)

**Harmonic Strength by Order**
   - n=1: Dipole (dominant)
   - n=2-4: Quadrupole, octupole terms
   - Higher orders: smaller contributions

Code Highlights
---------------

The example demonstrates:

- WMM2020 coefficient creation with ``create_wmm2020_coefficients()``
- Dipole moment calculation with ``dipole_moment()``
- Dipole axis orientation with ``dipole_axis()``
- Coefficient structure and magnitudes

Source Code
-----------

.. literalinclude:: ../../examples/magnetism_demo.py
   :language: python
   :linenos:

Running the Example
-------------------

.. code-block:: bash

   python examples/magnetism_demo.py

See Also
--------

- :doc:`geophysical_models` - Gravity and magnetic field models
- :doc:`ins_gnss_navigation` - Navigation applications
- :doc:`coordinate_systems` - Coordinate transformations
