Coordinate Systems
==================

.. automodule:: pytcl.coordinate_systems
   :no-members:
   :no-undoc-members:

Conversions
-----------

.. automodule:: pytcl.coordinate_systems.conversions
   :no-members:
   :no-undoc-members:

Spherical Coordinates
^^^^^^^^^^^^^^^^^^^^^

.. automodule:: pytcl.coordinate_systems.conversions.spherical
   :members:
   :undoc-members:
   :show-inheritance:

u-v Direction Cosines
^^^^^^^^^^^^^^^^^^^^^

The angle-only u-v(-w) measurement system of planar phased arrays, and
the full bistatic r-u-v conversions.

.. automodule:: pytcl.coordinate_systems.conversions.uv
   :members:
   :undoc-members:
   :show-inheritance:

Geodetic Coordinates
^^^^^^^^^^^^^^^^^^^^

.. automodule:: pytcl.coordinate_systems.conversions.geodetic
   :members:
   :undoc-members:
   :show-inheritance:

Rotations
---------

.. automodule:: pytcl.coordinate_systems.rotations
   :members:
   :undoc-members:
   :show-inheritance:

Jacobians
---------

Coordinate-transformation Jacobians, the component gradients of
bistatic measurement functions (range, range rate, spherical/polar
angles, direction cosines, TDOA) and the full measurement Jacobians
(spherical, polar, r-u-v, with range-rate and converted variants)
composed from them.

.. automodule:: pytcl.coordinate_systems.jacobians
   :members:
   :undoc-members:
   :show-inheritance:

Hessians
--------

Second derivatives of measurement components (bistatic range,
spherical angles, direction cosines), the composed spherical
measurement Hessians, chain-rule helpers and the cross
gradients/Hessians between angular parameterizations.

.. automodule:: pytcl.coordinate_systems.hessians
   :members:
   :undoc-members:
   :show-inheritance:

Projections
-----------

.. automodule:: pytcl.coordinate_systems.projections.projections
   :members:
   :undoc-members:
   :show-inheritance:
