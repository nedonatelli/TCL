Atmosphere
==========

Standard-atmosphere, thermosphere and ionosphere models for propagation
work, plus humidity conversions and refractivity helpers. The remainder of
MATLAB's ``Atmosphere_and_Refraction`` refraction suite (astronomical
refraction, standard-refraction ray tracing) is unported -- see
:doc:`../matlab_parity_inventory`.

.. automodule:: pytcl.atmosphere
   :no-members:
   :no-undoc-members:

Atmospheric Models
------------------

US Standard Atmosphere 1976 and ISA density/temperature/pressure models, plus
the pressure-altitude, Mach and true-airspeed conversions built on them.

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

Humidity and Dew Point
----------------------

.. automodule:: pytcl.atmosphere.humidity
   :members:
   :undoc-members:
   :show-inheritance:

Refractivity
------------

.. automodule:: pytcl.atmosphere.refraction
   :members:
   :undoc-members:
   :show-inheritance:

Ionosphere
----------

.. automodule:: pytcl.atmosphere.ionosphere
   :members:
   :undoc-members:
   :show-inheritance:
