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

.. automodule:: pytcl.coordinate_systems.jacobians
   :members:
   :undoc-members:
   :show-inheritance:

Component Gradients
-------------------

Gradients of individual bistatic measurement components (range, range
rate, spherical/polar angles, direction cosines, TDOA) with respect
to Cartesian position or state.

.. automodule:: pytcl.coordinate_systems.jacobians.component_gradients
   :members:
   :undoc-members:
   :show-inheritance:

Measurement Jacobians
---------------------

Full bistatic measurement Jacobians (spherical, polar, r-u-v, with
range-rate and converted variants) composed from the component
gradients.

.. automodule:: pytcl.coordinate_systems.jacobians.measurement_jacobians
   :members:
   :undoc-members:
   :show-inheritance:

Projections
-----------

.. automodule:: pytcl.coordinate_systems.projections.projections
   :members:
   :undoc-members:
   :show-inheritance:
