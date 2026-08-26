Magnetism Models
================

Earth magnetic field models and computations.

.. automodule:: pytcl.magnetism
   :no-members:
   :no-undoc-members:

World Magnetic Model (WMM)
--------------------------

World Magnetic Model implementation (WMM2020 epoch, valid 2020-2025).

.. automodule:: pytcl.magnetism.wmm
   :members:
   :undoc-members:
   :show-inheritance:

IGRF Model
----------

International Geomagnetic Reference Field implementation: IGRF-14 (the
default, with the full 1900.0-2025.0 epoch tables and 2025-30 secular
variation) plus the superseded IGRF-13 for reproducibility.

.. automodule:: pytcl.magnetism.igrf
   :members:
   :undoc-members:
   :show-inheritance:

Enhanced Magnetic Model (EMM) and WMMHR2025
--------------------------------------------

High-resolution magnetic field models: EMM2017 (degree 790) and WMMHR2025
(degree 133). Both support scalar and array inputs for lat/lon/height.

.. automodule:: pytcl.magnetism.emm
   :members:
   :undoc-members:
   :show-inheritance:

Package-level Aliases
---------------------

``pytcl.magnetism`` re-exports two EMM helpers under names that make their
scope explicit at the package level.

.. autofunction:: pytcl.magnetism.create_emm_test_coefficients
.. autofunction:: pytcl.magnetism.get_emm_data_dir
