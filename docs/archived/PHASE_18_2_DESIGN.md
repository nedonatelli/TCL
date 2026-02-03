"""
Phase 18.2: NRLMSISE-00 Implementation Design

Research Notes and Implementation Strategy
"""

# ==============================================================================
# NRLMSISE-00 Model Specification Research
# ==============================================================================

## Overview
NRLMSISE-00 (Naval Research Laboratory Mass Spectrometer and Incoherent Scatter
Extended Model) is a high-fidelity atmosphere model developed by the U.S. Naval
Research Laboratory. It provides accurate density, temperature, and composition
data from -5 km to 1000 km altitude.

Key Features:
- Altitude range: -5 km to 1000 km (troposphere through thermosphere)
- Composition: 8 major species (N2, O2, O, He, H, Ar, N)
- Solar activity: F10.7 flux and Ap magnetic index coupling
- Temperature: Exosphere temperature variable with solar activity
- Validation: Satellite density measurements, upper atmosphere sensors

## Model Structure

### 1. Altitude Layers
- Troposphere (0-12 km): Fixed composition, temperature decreases
- Stratosphere (12-50 km): Ozone maximum, temperature increases
- Mesosphere (50-85 km): Temperature minimum, atomic species emerge
- Thermosphere (85-600 km): O/O2 transition, temperature rises with Kp
- Exosphere (>600 km): Dominated by H/He, diffusive equilibrium

### 2. Physical Parameters Influencing Density

**Solar Activity (F10.7):**
- Values: Typical 70-200 SFU (Solar Flux Units)
- Quiet: 70-100 SFU
- Active: 150-200 SFU
- Storm: >200 SFU
- Effect: Increases exosphere temperature (Texo) and density at high altitudes
- Time lag: 81-day average (F107a) used for better correlation

**Magnetic Activity (Ap index):**
- Values: 0-400+ (linear scale)
- Quiet: 0-20
- Moderate: 20-50
- Active: 50-100
- Storm: >100
- Effect: Primarily affects thermosphere density through Joule heating
- Coupling: Different altitudes respond differently

**Geographic Location:**
- Latitude: Affects density distribution (equatorial bulge effects)
- Longitude: Longitudinal variations in thermosphere
- Local time: Diurnal bulge (day-night asymmetry)

### 3. Composition Species in NRLMSISE-00

The model outputs number densities (m⁻³) for:
1. N2 (Molecular nitrogen) - dominant to ~85 km
2. O2 (Molecular oxygen) - dominant to ~100 km
3. O (Atomic oxygen) - increases above ~100 km, dominant >200 km
4. He (Helium) - increases above ~100 km
5. H (Atomic hydrogen) - exosphere, >500 km
6. Ar (Argon) - trace, constant ratio
7. N (Atomic nitrogen) - trace, photochemically produced

Total density ρ = Σ(ni × Mi) where ni is number density, Mi is mass

### 4. Temperature Model

**Exosphere Temperature (Texo):**
- Base value: ~1000 K (quiet conditions)
- Solar activity variation: +0.5 K/SFU
- Magnetic activity variation: +30-100 K per Ap unit
- Used as boundary condition for thermosphere

**Lower Atmosphere Temperature:**
- Troposphere: Fixed ICAO ISA profile
- Stratosphere: Polynomial fits with latitude variation
- Mesosphere: Complex fitting with seasonal variation
- Thermosphere: Interpolation between surface and Texo

### 5. Key Model Parameters

For implementation, need to define:

```
Temperature(z, lat, lon, LST, Texo, season) → T(z)
  where LST = Local Solar Time
        season = Day-of-year effect

N2_density(z, lat, lon, T, Texo, F107a, Ap) → n_N2
O2_density(z, lat, lon, T, Texo, F107a, Ap) → n_O2
O_density(z, lat, lon, T, Texo, F107a, Ap) → n_O
He_density(z, lat, lon, T, Texo, F107a, Ap) → n_He
H_density(z, lat, lon, T, Texo, F107a, Ap) → n_H
Ar_density(z, lat, lon, T, Texo, F107a, Ap) → n_Ar
N_density(z, lat, lon, T, Texo, F107a, Ap) → n_N
```

## Implementation Strategy

### Phase 1: Core Structure (Current)
✅ Create module stub with class signatures
✅ Define input/output data structures
✅ Plan integration with existing atmosphere module

### Phase 2: Algorithm Implementation
1. Download NRLMSISE-00 reference data/coefficients
2. Implement temperature profile calculation
3. Implement species density calculations
4. Handle altitude interpolation
5. Integrate solar/magnetic activity coupling

### Phase 3: Testing
1. Reference implementation comparisons (NASA/NOAA)
2. Edge case validation (altitude limits, solar extremes)
3. Roundtrip consistency checks
4. Performance benchmarking

### Phase 4: Documentation
1. Example usage and integration
2. Orbit decay simulations
3. Drag coefficient database reference
4. Atmospheric model comparison tables

## Reference Data Sources

### NOAA Space Weather
- F10.7 daily and 81-day average values
- Ap index historical data
- Solar activity forecasts

### NASA GSFC
- NRLMSISE-00 original algorithm documentation
- Reference coefficient tables
- Validation against satellite data

### Related Models
- HWM (Horizontal Wind Model) for wind velocities
- IRI (International Reference Ionosphere) for ion density
- MSIS (related predecessor model)

## Integration with pytcl

### Module Location
`pytcl/atmosphere/nrlmsise00.py`

### Class Structure
```python
class NRLMSISE00:
    """High-fidelity atmosphere model (-5 to 1000 km)"""

    def __call__(self, latitude, longitude, altitude,
                 year, day_of_year, seconds_in_day,
                 f107, f107a, ap):
        # Returns NRLMSISE00Output with 8 species densities
```

### API Consistency
- Match existing `us_standard_atmosphere_1976()` signature where possible
- Use same coordinate conventions (radians, meters)
- Return NamedTuple with typed outputs

### Unit Tests
- Compare against reference implementations
- Test altitude extremes (-5 km, 1000 km)
- Test solar activity variation (70-250 SFU)
- Test magnetic activity variation (Ap 0-400)
- Benchmark against satellite density data

## Implementation Roadmap

**Week 1:** Research complete, basic structure
**Week 2:** Core algorithm implementation
**Week 3:** Testing and validation
**Week 4:** Documentation and examples

## Success Criteria

- ✅ Module loads without errors
- ✅ API matches project standards
- ✅ Handles all altitude ranges (-5 to 1000 km)
- ✅ Solar/magnetic activity variations implemented
- ✅ Numerical accuracy ±10% vs reference
- ✅ 15+ unit tests, all passing
- ✅ Example with orbit decay simulation
- ✅ mypy/isort/black/flake8 compliant
- ✅ All 1930+ project tests still passing
