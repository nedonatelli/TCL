Thermosphere Density Modeling
=============================

Overview
--------

``pytcl`` ships NRLMSISE-00, the standard empirical model of
upper-atmosphere density, temperature, and composition, as
:func:`~pytcl.atmosphere.nrlmsise00` (since v2.10.0). It is a transcription
of the NRL reference C implementation that the MATLAB TCL wraps, compiled
into a C extension with a machine-precision pure-Python fallback
(:func:`~pytcl.atmosphere.uses_compiled_backend` reports which is active),
and is validated against that reference code. The older
:func:`~pytcl.atmosphere.simplified_thermosphere` barometric model is
deprecated and will be removed in v3.0.0; it was never NRLMSISE-00.

Unlike that model, :func:`~pytcl.atmosphere.nrlmsise00` is scalar, takes
degrees and kilometers, and has no ``year`` argument; the examples below
loop over altitudes instead of passing arrays.

Key Parameters
~~~~~~~~~~~~~~

- **F10.7**: 10.7 cm solar radio flux (SFU, Solar Flux Units)
  - Range: 65-300 SFU (typically 70-150 SFU during moderate conditions)
  - Indicator of solar activity and EUV radiation

- **F10.7A**: 81-day centered average of F10.7
  - Smoother than daily F10.7
  - Better represents baseline thermospheric conditions

- **Kp Index**: Geomagnetic activity index
  - Range: 0-9 (0=quiet, 9=severe storm)
  - Derived from ground magnetometers
  - Drives high-latitude thermospheric heating

- **Ap Index**: Magnetic planetary three-hour index
  - Linear-scale counterpart of Kp (Kp 3 corresponds to Ap 15)
  - This is the geomagnetic input of :func:`~pytcl.atmosphere.nrlmsise00` (``ap`` parameter)

Applications
~~~~~~~~~~~~

1. **Satellite Drag Modeling**
   - Atmospheric density affects satellite orbital decay

2. **Radar Cross-Section (RCS) Prediction**
   - Ionospheric scattering depends on electron density
   - Composition affects scattering characteristics

3. **Communications**
   - Ionospheric refraction and attenuation
   - Density gradients affect HF radio propagation

4. **Space Weather**
   - Track thermospheric responses to solar storms
   - Validate space weather models

Basic Usage
-----------

.. code-block:: python

   import numpy as np
   from pytcl.atmosphere import nrlmsise00

   # Get atmospheric density at ISS altitude
   out = nrlmsise00(
       doy=100,                # April 9
       sec=12 * 3600.0,        # 12:00 UT
       alt_km=400.0,
       g_lat_deg=51.6,         # ISS inclination
       g_long_deg=0.0,
       lst=12.0,               # local solar time, hours
       f107a=130.0,            # 81-day average
       f107=150.0,             # Moderate solar activity
       ap=15.0,                # Quiet geomagnetic conditions (Kp ~ 3)
   )

   print(f"ISS density: {out.d[5]:.3e} kg/m³")
   print(f"Temperature: {out.t[1]:.0f} K")
   # ISS density: 5.649e-12 kg/m³
   # Temperature: 1134 K

Density Calculation
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # The model is scalar: evaluate it at each altitude
   altitudes_km = np.linspace(200, 500, 50)

   densities = np.array([
       nrlmsise00(1, 12 * 3600.0, float(alt), 0.0, 0.0, 12.0, 120.0, 120.0, 4.0).d[5]
       for alt in altitudes_km
   ])

   # Plot density profile
   import matplotlib.pyplot as plt
   plt.semilogy(densities, altitudes_km)
   plt.xlabel('Density (kg/m³)')
   plt.ylabel('Altitude (km)')
   plt.title('Thermosphere Density Profile')
   plt.show()

Atmospheric Composition
-----------------------

Get individual species densities (number/m³):

.. code-block:: python

   import numpy as np
   from pytcl.atmosphere import nrlmsise00

   out = nrlmsise00(100, 12 * 3600.0, 400.0, 0.0, 0.0, 12.0, 130.0, 150.0, 15.0)

   # NRLMSISEOutput.d, indexed as the reference implementation orders it
   # (number densities in m^-3, d[5] in kg/m^3):
   #   d[0] He   Helium (light, escapes slowly)
   #   d[1] O    Atomic O (dominant at ~400 km)
   #   d[2] N2   N2 (dominant at low altitude)
   #   d[3] O2   O2
   #   d[4] Ar   Argon
   #   d[5]      Total mass density
   #   d[6] H    Hydrogen (very light)
   #   d[7] N    Atomic nitrogen
   #   d[8]      Anomalous oxygen
   # out.t[0] is the exospheric temperature, out.t[1] the temperature at
   # altitude (K).

   total_number_density = out.d[[0, 1, 2, 3, 4, 6, 7]].sum()
   print(f"Total number density: {total_number_density:.3e} particles/m³")
   # Total number density: 2.124e+14 particles/m³

Real-World Example: LEO Satellite Drag
---------------------------------------

Calculate drag force on a satellite:

.. code-block:: python

   import numpy as np
   from pytcl.atmosphere import nrlmsise00

   # Satellite parameters
   altitude_km = 400.0
   latitude_deg = 51.6
   longitude_deg = -100.0
   velocity_kmps = 7.66  # Orbital velocity (~27,600 km/h)

   # Satellite aerodynamic properties
   cross_section_m2 = 10.0  # Projected area
   drag_coefficient = 2.2   # Typical value
   mass_kg = 1000.0

   # Solar activity (check NOAA for current values)
   f107 = 150.0     # 10.7 cm solar flux
   f107a = 130.0    # 81-day average
   ap = 7.0         # Planetary magnetic index (Kp ~ 2)

   # Calculate atmospheric density
   sec = 12 * 3600.0
   lst = sec / 3600.0 + longitude_deg / 15.0
   rho = nrlmsise00(
       100, sec, altitude_km, latitude_deg, longitude_deg, lst, f107a, f107, ap
   ).d[5]

   # Drag force: F = 0.5 * rho * v² * Cd * A
   v_ms = velocity_kmps * 1000.0  # Convert to m/s
   drag_force = 0.5 * rho * v_ms**2 * drag_coefficient * cross_section_m2

   # Drag acceleration: a = F/m
   drag_accel = drag_force / mass_kg  # m/s²

   # Orbital decay: ΔV/orbit ≈ a * period
   orbital_period = 2 * np.pi * np.sqrt((altitude_km + 6371)**3 / 398600.0)  # seconds
   dv_per_orbit = drag_accel * orbital_period  # m/s

   print(f"Density: {rho:.3e} kg/m³")
   print(f"Drag force: {drag_force:.3e} N")
   print(f"Drag acceleration: {drag_accel:.3e} m/s²")
   print(f"ΔV per orbit: {dv_per_orbit:.6f} m/s")
   print(f"Orbital decay (~0.5 years): {0.5 * 365 * dv_per_orbit / orbital_period:.1f} km")

Validation
----------

``nrlmsise00`` is checked record-for-record against the vendored NRL C
reference, and the pure-Python fallback is cross-checked against the
compiled path; see ``tests/validation`` and the module docstring of
``pytcl.atmosphere.nrlmsise00``.

Effects of Solar Activity
--------------------------

Observe density changes with different solar activity levels:

.. code-block:: python

   import numpy as np
   from pytcl.atmosphere import nrlmsise00
   import matplotlib.pyplot as plt

   # Fixed parameters
   alt_km = 400.0
   doy, sec, lat, lon, lst = 100, 12 * 3600.0, 0.0, 0.0, 12.0

   # Vary F10.7 (quiet to active)
   f107_values = np.linspace(70, 200, 20)
   densities_solar = [
       nrlmsise00(doy, sec, alt_km, lat, lon, lst, f107, f107, 4.0).d[5]
       for f107 in f107_values  # same value for the daily and 81-day flux
   ]

   # Vary Ap (quiet to severe storm)
   ap_values = np.linspace(0, 200, 20)
   densities_geomag = [
       nrlmsise00(doy, sec, alt_km, lat, lon, lst, 130.0, 150.0, ap).d[5]
       for ap in ap_values
   ]

   # Plot effects
   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

   ax1.semilogy(f107_values, densities_solar, 'b-o')
   ax1.set_xlabel('F10.7 Solar Flux (SFU)')
   ax1.set_ylabel('Density (kg/m³)')
   ax1.set_title('Density vs Solar Activity')
   ax1.grid()

   ax2.semilogy(ap_values, densities_geomag, 'r-o')
   ax2.set_xlabel('Ap Geomagnetic Index')
   ax2.set_ylabel('Density (kg/m³)')
   ax2.set_title('Density vs Geomagnetic Activity')
   ax2.grid()

   plt.tight_layout()
   plt.show()

Model Limitations
-----------------

**Applicable Range**
  - Altitude: from the ground into the exosphere; for the lower
    atmosphere :func:`~pytcl.atmosphere.us_standard_atmosphere_1976`
    is the standard-atmosphere reference
  - Solar and geomagnetic inputs (``f107``, ``f107a``, ``ap``) must be
    supplied by the caller; there is no built-in index lookup

**When NOT to Use**
  - High-precision drag modeling (use HASDM or JB2008 with in-situ
    corrections; NRLMSISE-00 is an empirical climatological model)

Integration with Orbit Propagation
-----------------------------------

Evaluate the density model inside a propagation loop:

.. code-block:: python

   import numpy as np
   from pytcl.atmosphere import nrlmsise00

   # Inside a propagation loop, evaluate density at the current
   # geodetic position and epoch of the satellite state:
   for i in range(100):
       # ... propagate to step i, then convert the position to
       # geodetic latitude/longitude/altitude and the epoch to
       # (day_of_year, seconds_in_day, local solar time) ...

       rho = nrlmsise00(
           doy,                # day of year
           sec,                # second of the UT day
           alt_km,             # kilometers
           lat,                # degrees
           lon,                # degrees
           lst,                # local solar time, hours
           130.0,              # f107a: 81-day average F10.7
           150.0,              # f107: latest NOAA value
           15.0,               # ap: latest Ap from NOAA
       ).d[5]

       # Apply drag acceleration a = -0.5 * rho * |v| * (Cd*A/m) * v
       # to the propagated state.

See Also
~~~~~~~~

- **NOAA Space Weather Prediction Center**: https://www.swpc.noaa.gov
  - Daily F10.7 and Kp index values
- **SPDF/NSSDC**: NRLMSISE-00 Original Publication
- :doc:`getting_started` - Basic atmospheric modeling
- :doc:`api/dynamic_models` - Orbit propagation with drag
- :ref:`Thermosphere Model <thermosphere-model>` - API Reference

References
----------

1. Picone, J.M., et al. (2002), "NRLMSISE-00 Empirical Model of the
   Atmosphere" -- the reference paper for
   :func:`~pytcl.atmosphere.nrlmsise00`
2. COSPAR International Reference Atmosphere (CIRA)
