Atmospheric Modeling with NRLMSISE-00
======================================

Overview
--------

The NRLMSISE-00 (Naval Research Laboratory Mass Spectrometer Incoherent Scatter
Radar Exosphere) model is the standard empirical model of Earth's upper atmosphere.
It provides atmospheric density, temperature, and composition from ground level to
1000 km altitude, with dependencies on solar activity and geomagnetic conditions.

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
  - Average of Ap values over previous 12 hours
  - Alternative to Kp for averaging

Applications
~~~~~~~~~~~~

1. **Satellite Drag Modeling**
   - Atmospheric density affects satellite orbital decay
   - NRLMSISE-00 is standard for orbit propagation

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
   density = nrlmsise00.get_density(
       altitude_km=400.0,
       latitude_deg=51.6,      # ISS inclination
       longitude_deg=0.0,
       year=2024,
       day_of_year=100,        # April 9, 2024
       hour_utc=12.0,
       f107=150.0,             # Moderate solar activity
       f107a=130.0,            # 81-day average
       kp=3.0,                 # Quiet geomagnetic conditions
   )

   print(f"ISS density: {density:.3e} kg/m³")
   # ISS density: 3.165e-12 kg/m³ (typical value)

Density Calculation
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Density varies with altitude
   altitudes_km = np.linspace(200, 500, 50)
   densities = np.zeros_like(altitudes_km)

   for i, alt in enumerate(altitudes_km):
       densities[i] = nrlmsise00.get_density(
           altitude_km=alt,
           latitude_deg=0.0,
           longitude_deg=0.0,
           year=2024,
           day_of_year=1,
           hour_utc=12.0,
           f107=120.0,
           f107a=120.0,
           kp=0.0,
       )

   # Plot density profile
   import matplotlib.pyplot as plt
   plt.semilogy(densities, altitudes_km)
   plt.xlabel('Density (kg/m³)')
   plt.ylabel('Altitude (km)')
   plt.title('NRLMSISE-00 Density Profile')
   plt.show()

Atmospheric Composition
-----------------------

Get individual species densities (number/cm³):

.. code-block:: python

   from pytcl.atmosphere import nrlmsise00

   composition = nrlmsise00.get_composition(
       altitude_km=400.0,
       latitude_deg=0.0,
       longitude_deg=0.0,
       year=2024,
       day_of_year=100,
       hour_utc=12.0,
       f107=150.0,
       f107a=130.0,
       kp=3.0,
   )

   # Return structure (dict-like):
   # {
   #     'n2': number_density_N2,      # N2 (dominant, ~78%)
   #     'o2': number_density_O2,      # O2 (~21%)
   #     'o': number_density_O,        # O (significant at alt>200km)
   #     'he': number_density_He,      # Helium (light, escapes slowly)
   #     'ar': number_density_Ar,      # Argon (~1%)
   #     'h': number_density_H,        # Hydrogen (very light)
   #     'n': number_density_N,        # Nitrogen atoms
   #     'temperature': temperature_kelvin,
   #     'exospheric_temp': T_exo,     # Temperature at infinity
   # }

   # Verify composition sums correctly
   total_density = (
       composition['n2'] +
       composition['o2'] +
       composition['o'] +
       composition['he'] +
       composition['ar'] +
       composition['h'] +
       composition['n']
   )
   print(f"Total density: {total_density:.3e} particles/cm³")

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
   kp = 2.0         # Geomagnetic index

   # Calculate atmospheric density
   rho = nrlmsise00.get_density(
       altitude_km=altitude_km,
       latitude_deg=latitude_deg,
       longitude_deg=longitude_deg,
       year=2024,
       day_of_year=100,
       hour_utc=12.0,
       f107=f107,
       f107a=f107a,
       kp=kp,
   )

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

Validation Against Data
------------------------

Comparison with in-situ measurements:

.. code-block:: python

   import numpy as np
   from pytcl.atmosphere import nrlmsise00

   # ISS accelerometer-derived density measurements (example)
   measured_data = [
       {"alt": 400, "density": 3.165e-12, "year": 2024, "doy": 100},
       {"alt": 405, "density": 2.850e-12, "year": 2024, "doy": 100},
       {"alt": 410, "density": 2.560e-12, "year": 2024, "doy": 100},
   ]

   # NRLMSISE-00 predictions
   model_params = {
       "latitude_deg": 51.6,
       "longitude_deg": 0.0,
       "year": 2024,
       "day_of_year": 100,
       "hour_utc": 12.0,
       "f107": 150.0,
       "f107a": 130.0,
       "kp": 3.0,
   }

   errors = []
   for data in measured_data:
       model_density = nrlmsise00.get_density(
           altitude_km=data["alt"],
           **model_params
       )
       relative_error = abs(model_density - data["density"]) / data["density"]
       errors.append(relative_error)
       print(f"Alt {data['alt']} km: Measured {data['density']:.3e}, "
             f"Model {model_density:.3e} (+{relative_error*100:.1f}%)")

   mean_error = np.mean(errors)
   print(f"\nMean relative error: {mean_error*100:.1f}%")
   # Typical accuracy: 10-30% depending on space weather conditions

Effects of Solar Activity
--------------------------

Observe density changes with different solar activity levels:

.. code-block:: python

   import numpy as np
   from pytcl.atmosphere import nrlmsise00
   import matplotlib.pyplot as plt

   # Fixed parameters
   alt = 400.0
   lat, lon = 0.0, 0.0
   year, doy, hour = 2024, 100, 12.0

   # Vary F10.7 (quiet to active)
   f107_values = np.linspace(70, 200, 20)
   densities_solar = []

   for f107 in f107_values:
       rho = nrlmsise00.get_density(
           altitude_km=alt,
           latitude_deg=lat,
           longitude_deg=lon,
           year=year,
           day_of_year=doy,
           hour_utc=hour,
           f107=f107,
           f107a=f107,  # Assume same as daily
           kp=0.0,      # Quiet conditions
       )
       densities_solar.append(rho)

   # Vary Kp (quiet to severe storm)
   kp_values = np.linspace(0, 8, 20)
   densities_geomag = []

   for kp in kp_values:
       rho = nrlmsise00.get_density(
           altitude_km=alt,
           latitude_deg=lat,
           longitude_deg=lon,
           year=year,
           day_of_year=doy,
           hour_utc=hour,
           f107=150.0,
           f107a=130.0,
           kp=kp,
       )
       densities_geomag.append(rho)

   # Plot effects
   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

   ax1.semilogy(f107_values, densities_solar, 'b-o')
   ax1.set_xlabel('F10.7 Solar Flux (SFU)')
   ax1.set_ylabel('Density (kg/m³)')
   ax1.set_title('Density vs Solar Activity')
   ax1.grid()

   ax2.semilogy(kp_values, densities_geomag, 'r-o')
   ax2.set_xlabel('Kp Geomagnetic Index')
   ax2.set_ylabel('Density (kg/m³)')
   ax2.set_title('Density vs Geomagnetic Activity')
   ax2.grid()

   plt.tight_layout()
   plt.show()

Model Limitations
-----------------

**Applicable Range**
  - Altitude: 0-1000 km (poor accuracy below 85 km)
  - Latitude: -90 to 90 degrees
  - Longitude: 0 to 360 degrees
  - Year: 1961-2100 (by Jacchia-based extrapolation)

**Known Uncertainties**

1. **Low Solar Activity**: Density can be underestimated by 20-30%
2. **Disturbed Times**: Kp-dependent heating not exact (+30% during storms)
3. **Seasonal Anomalies**: Hemispheric differences not fully captured
4. **Microvariations**: Hour-to-hour variations up to ±10%

**When NOT to Use**
  - Stratospheric studies (use MSISE-90 or COSPAR)
  - High-precision drag modeling (use in-situ data + correction)
  - Real-time forecasting (use HASDM or JB2008 instead)

Integration with Orbit Propagation
-----------------------------------

Use NRLMSISE-00 in SGP4 propagator:

.. code-block:: python

   import numpy as np
   from pytcl.atmosphere import nrlmsise00
   from pytcl.dynamic_models import sgp4_propagator

   # TLE for ISS
   tle_line1 = "1 25544U 98067A   24100.50000000  .00016717  00000-0  29800-3 0  9990"
   tle_line2 = "2 25544  51.6400 339.8014 0002006  34.5857  120.4689 15.50582022450261"

   # Initial conditions from TLE
   state0 = sgp4_propagator.tle_to_state(tle_line1, tle_line2)

   # Propagate with NRLMSISE-00 drag
   for i in range(100):
       time_utc = /* compute UTC time at step i */
       doy = /* day of year */

       # Get current density
       altitude_km = state0['position'].magnitude / 1000.0
       rho = nrlmsise00.get_density(
           altitude_km=altitude_km,
           latitude_deg=state0['latitude'],
           longitude_deg=state0['longitude'],
           year=time_utc.year,
           day_of_year=doy,
           hour_utc=time_utc.hour + time_utc.minute/60,
           f107=150.0,  # Use latest NOAA value
           f107a=130.0,
           kp=kp,  # Use latest Kp from NOAA
       )

       # Apply drag force to state0
       # (implementation depends on propagator interface)

See Also
--------

- **NOAA Space Weather Prediction Center**: https://www.swpc.noaa.gov
  - Daily F10.7 and Kp index values
- **SPDF/NSSDC**: NRLMSISE-00 Original Publication
- :doc:`getting_started` - Basic atmospheric modeling
- :doc:`dynamic_models` - Orbit propagation with drag
- :ref:`api/atmosphere:NRLMSISE-00 Model` - API Reference

References
----------

1. Picone, J.M., et al. (2002), "NRLMSISE-00 Empirical Model of the Atmosphere"
2. NASA Technical Reports Server (NTRS)
3. COSPAR International Reference Atmosphere (CIRA)
