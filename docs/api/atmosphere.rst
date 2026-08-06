Atmosphere
==========

Atmospheric models for propagation and refraction.

.. automodule:: pytcl.atmosphere
   :no-members:
   :no-undoc-members:

Atmospheric Models
------------------

Standard atmosphere and refraction models.

.. automodule:: pytcl.atmosphere.models
   :members:
   :undoc-members:
   :show-inheritance:

.. _thermosphere-model:

Thermosphere Model
------------------

Simplified barometric thermosphere density, temperature and composition
model with solar-activity and geomagnetic inputs. Not NRLMSISE-00: usable
above ~200 km (within ~2x of published NRLMSISE-00 values), up to 50x wrong
below ~86 km where ``us_standard_atmosphere_1976`` should be used. Limits
are documented in the module and pinned by validation tests (gh-79).

.. automodule:: pytcl.atmosphere.thermosphere
   :members:
   :undoc-members:
   :show-inheritance:

Ionosphere
----------

.. automodule:: pytcl.atmosphere.ionosphere
   :members:
   :undoc-members:
   :show-inheritance:
