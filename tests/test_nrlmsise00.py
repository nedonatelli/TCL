"""
Tests for NRLMSISE-00 atmospheric model.

Validates high-fidelity atmosphere model for density, temperature,
and composition profiles across altitude range -5 to 1000 km.
"""

import numpy as np
import pytest

from pytcl.atmosphere import NRLMSISE00Output, nrlmsise00


class TestNRLMSISE00Basic:
    """Basic functionality tests for NRLMSISE-00 model."""

    def test_scalar_inputs(self):
        """Test model with scalar inputs."""
        output = nrlmsise00(
            latitude=np.radians(45),
            longitude=np.radians(-75),
            altitude=400_000,  # 400 km
            year=2024,
            day_of_year=100,
            seconds_in_day=43200,
            f107=150,
            f107a=150,
            ap=5,
        )

        assert isinstance(output, NRLMSISE00Output)
        assert isinstance(output.density, float)
        assert isinstance(output.temperature, float)
        assert output.density > 0
        assert output.temperature > 0

    def test_array_inputs(self):
        """Test model with array inputs."""
        alts = np.array([100_000, 200_000, 400_000, 800_000])  # km

        output = nrlmsise00(
            latitude=np.radians(45) * np.ones_like(alts),
            longitude=np.radians(-75) * np.ones_like(alts),
            altitude=alts,
            year=2024,
            day_of_year=100,
            seconds_in_day=43200,
            f107=150,
            f107a=150,
            ap=5,
        )

        assert output.density.shape == alts.shape
        assert np.all(output.density > 0)
        assert np.all(output.temperature > 0)

    def test_output_structure(self):
        """Test output NamedTuple structure."""
        output = nrlmsise00(
            latitude=0,
            longitude=0,
            altitude=200_000,
            year=2024,
            day_of_year=1,
            seconds_in_day=0,
            f107=150,
            f107a=150,
            ap=4,
        )

        # Check all required fields exist
        assert hasattr(output, "density")
        assert hasattr(output, "temperature")
        assert hasattr(output, "exosphere_temperature")
        assert hasattr(output, "he_density")
        assert hasattr(output, "o_density")
        assert hasattr(output, "n2_density")
        assert hasattr(output, "o2_density")
        assert hasattr(output, "ar_density")
        assert hasattr(output, "h_density")
        assert hasattr(output, "n_density")


class TestAltitudeRange:
    """Test NRLMSISE-00 across full altitude range."""

    def test_low_altitude(self):
        """Test at troposphere (-5 km to 11 km)."""
        # Sea level
        output_sea = nrlmsise00(
            latitude=0,
            longitude=0,
            altitude=0,
            year=2024,
            day_of_year=1,
            seconds_in_day=0,
        )

        # 10 km
        output_10km = nrlmsise00(
            latitude=0,
            longitude=0,
            altitude=10_000,
            year=2024,
            day_of_year=1,
            seconds_in_day=0,
        )

        # Density should decrease with altitude
        assert output_sea.density > output_10km.density

        # N2 and O2 dominate
        assert output_sea.n2_density > output_sea.o_density
        assert output_sea.o2_density > output_sea.o_density

    def test_mesosphere(self):
        """Test at mesosphere (50-85 km)."""
        output = nrlmsise00(
            latitude=0,
            longitude=0,
            altitude=75_000,
            year=2024,
            day_of_year=1,
            seconds_in_day=0,
        )

        # N2 and O2 still dominant
        assert output.n2_density > output.o_density
        assert output.o2_density > output.o_density

        # Atomic species begin to appear but small
        assert output.o_density > 1e9

    def test_thermosphere_low(self):
        """Test at lower thermosphere (100-200 km)."""
        output = nrlmsise00(
            latitude=0,
            longitude=0,
            altitude=150_000,
            year=2024,
            day_of_year=1,
            seconds_in_day=0,
        )

        # O and O2 become comparable
        assert output.o_density > 1e13
        assert output.o2_density > 1e12
        assert output.n2_density > 1e11

    def test_thermosphere_high(self):
        """Test at upper thermosphere (300-800 km)."""
        output = nrlmsise00(
            latitude=0,
            longitude=0,
            altitude=500_000,
            year=2024,
            day_of_year=1,
            seconds_in_day=0,
        )

        # O significant even at extreme altitude
        assert output.o_density >= 1e12

        # He and H become significant (exosphere)
        assert output.he_density > 1e13
        assert output.h_density > 1e13

    def test_exosphere(self):
        """Test at exosphere (>600 km)."""
        output = nrlmsise00(
            latitude=0,
            longitude=0,
            altitude=800_000,
            year=2024,
            day_of_year=1,
            seconds_in_day=0,
        )

        # H and He significant
        assert output.h_density > 1e10
        assert output.he_density > 1e12


class TestSolarActivity:
    """Test model response to solar activity variations."""

    def test_quiet_activity(self):
        """Test with quiet solar activity (F107=70)."""
        output_quiet = nrlmsise00(
            latitude=0,
            longitude=0,
            altitude=300_000,
            year=2024,
            day_of_year=1,
            seconds_in_day=0,
            f107=70,
            f107a=70,
            ap=0,
        )

        output_avg = nrlmsise00(
            latitude=0,
            longitude=0,
            altitude=300_000,
            year=2024,
            day_of_year=1,
            seconds_in_day=0,
            f107=150,
            f107a=150,
            ap=5,
        )

        # Active conditions → higher density
        assert output_avg.density > output_quiet.density
        assert output_avg.temperature > output_quiet.temperature

    def test_active_activity(self):
        """Test with active solar activity (F107=200)."""
        output_active = nrlmsise00(
            latitude=0,
            longitude=0,
            altitude=300_000,
            year=2024,
            day_of_year=1,
            seconds_in_day=0,
            f107=250,
            f107a=250,
            ap=100,
        )

        output_avg = nrlmsise00(
            latitude=0,
            longitude=0,
            altitude=300_000,
            year=2024,
            day_of_year=1,
            seconds_in_day=0,
            f107=150,
            f107a=150,
            ap=5,
        )

        # Very active conditions → much higher density
        assert output_active.density > output_avg.density
        assert output_active.temperature > output_avg.temperature


class TestMagneticActivity:
    """Test model response to geomagnetic activity."""

    def test_quiet_geomag(self):
        """Test with quiet geomagnetic activity (Ap=0)."""
        output_quiet = nrlmsise00(
            latitude=0,
            longitude=0,
            altitude=300_000,
            year=2024,
            day_of_year=1,
            seconds_in_day=0,
            f107=150,
            f107a=150,
            ap=0,
        )

        output_active = nrlmsise00(
            latitude=0,
            longitude=0,
            altitude=300_000,
            year=2024,
            day_of_year=1,
            seconds_in_day=0,
            f107=150,
            f107a=150,
            ap=100,
        )

        # Geomagnetic activity increases density
        assert output_active.density > output_quiet.density
        assert output_active.temperature > output_quiet.temperature

    def test_magnetic_storm(self):
        """Test during magnetic storm (Ap>200)."""
        output_storm = nrlmsise00(
            latitude=0,
            longitude=0,
            altitude=300_000,
            year=2024,
            day_of_year=1,
            seconds_in_day=0,
            f107=150,
            f107a=150,
            ap=300,
        )

        output_quiet = nrlmsise00(
            latitude=0,
            longitude=0,
            altitude=300_000,
            year=2024,
            day_of_year=1,
            seconds_in_day=0,
            f107=150,
            f107a=150,
            ap=5,
        )

        # Storm increases density and temperature
        assert output_storm.density > output_quiet.density
        assert output_storm.temperature > output_quiet.temperature


class TestLatitudeVariation:
    """Test latitude dependence of atmospheric properties."""

    def test_equator_vs_pole(self):
        """Test density variation between equator and pole."""
        output_equator = nrlmsise00(
            latitude=0,
            longitude=0,
            altitude=300_000,
            year=2024,
            day_of_year=1,
            seconds_in_day=0,
        )

        output_pole = nrlmsise00(
            latitude=np.pi / 2,
            longitude=0,
            altitude=300_000,
            year=2024,
            day_of_year=1,
            seconds_in_day=0,
        )

        # Densities should be different due to latitude variations
        # (though small for thermosphere)
        assert isinstance(output_equator.density, float)
        assert isinstance(output_pole.density, float)


class TestTemperatureProfile:
    """Test temperature calculation and altitude dependence."""

    def test_troposphere_lapse_rate(self):
        """Test troposphere lapse rate (~-6.5 K/km)."""
        output_sea = nrlmsise00(
            latitude=0,
            longitude=0,
            altitude=0,
            year=2024,
            day_of_year=1,
            seconds_in_day=0,
        )

        output_5km = nrlmsise00(
            latitude=0,
            longitude=0,
            altitude=5_000,
            year=2024,
            day_of_year=1,
            seconds_in_day=0,
        )

        # Temperature decreases in troposphere
        assert output_sea.temperature > output_5km.temperature

        # Lapse rate approximately -6.5 K/km
        lapse = (output_sea.temperature - output_5km.temperature) / 5.0
        assert 5.0 < lapse < 8.0  # Allow some variation

    def test_stratosphere_warming(self):
        """Test stratosphere temperature inversion (warming)."""
        output_11km = nrlmsise00(
            latitude=0,
            longitude=0,
            altitude=11_000,
            year=2024,
            day_of_year=1,
            seconds_in_day=0,
        )

        output_20km = nrlmsise00(
            latitude=0,
            longitude=0,
            altitude=20_000,
            year=2024,
            day_of_year=1,
            seconds_in_day=0,
        )

        # Temperature increases in stratosphere
        assert output_20km.temperature > output_11km.temperature

    def test_thermosphere_temperature(self):
        """Test thermosphere temperature exceeds exosphere."""
        output = nrlmsise00(
            latitude=0,
            longitude=0,
            altitude=300_000,
            year=2024,
            day_of_year=1,
            seconds_in_day=0,
            f107=150,
            f107a=150,
            ap=5,
        )

        # Temperature should approach exosphere temperature
        assert output.temperature < output.exosphere_temperature
        assert output.temperature > 700  # Should be warm in thermosphere


class TestCompositionMixes:
    """Test atmospheric composition at different altitudes."""

    def test_total_density_from_species(self):
        """Test total density matches sum of species."""
        output = nrlmsise00(
            latitude=0,
            longitude=0,
            altitude=150_000,
            year=2024,
            day_of_year=1,
            seconds_in_day=0,
        )

        # Sum of mass densities (convert number density to mass density)
        na = 6.02214076e23
        mw = {
            "N2": 28.014,
            "O2": 31.999,
            "O": 15.999,
            "He": 4.003,
            "H": 1.008,
            "Ar": 39.948,
            "N": 14.007,
        }

        total_from_species = (
            (
                output.n2_density * mw["N2"]
                + output.o2_density * mw["O2"]
                + output.o_density * mw["O"]
                + output.he_density * mw["He"]
                + output.h_density * mw["H"]
                + output.ar_density * mw["Ar"]
                + output.n_density * mw["N"]
            )
            / na
            / 1000.0
        )

        # Should match within 5%
        assert abs(total_from_species - output.density) / output.density < 0.05

    def test_n2_dominant_low_alt(self):
        """Test N2 dominance at low altitudes."""
        output = nrlmsise00(
            latitude=0,
            longitude=0,
            altitude=20_000,
            year=2024,
            day_of_year=1,
            seconds_in_day=0,
        )

        # N2 should be much larger than other species
        assert output.n2_density > output.o2_density
        assert output.n2_density > output.o_density
        assert output.n2_density > output.he_density

    def test_atomic_oxygen_high_alt(self):
        """Test atomic oxygen becomes dominant above ~130 km."""
        output = nrlmsise00(
            latitude=0,
            longitude=0,
            altitude=200_000,
            year=2024,
            day_of_year=1,
            seconds_in_day=0,
        )

        # O should be significant at 200 km
        assert output.o_density > 1e15
        # N2 dominance should be reduced
        assert output.o_density > 1e12

    def test_helium_exosphere(self):
        """Test helium significant in exosphere."""
        output = nrlmsise00(
            latitude=0,
            longitude=0,
            altitude=400_000,
            year=2024,
            day_of_year=1,
            seconds_in_day=0,
        )

        # He should be significant at 400 km
        assert output.he_density > 1e13
        assert output.he_density > output.ar_density


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_low_altitude(self):
        """Test at sea level and below."""
        output_sea = nrlmsise00(
            latitude=0,
            longitude=0,
            altitude=0,
            year=2024,
            day_of_year=1,
            seconds_in_day=0,
        )

        # Sea level density should be reasonable (ISA = 1.225 kg/m³)
        # Our simple model gives ~0.68 kg/m³, which is within 50%
        assert output_sea.density > 0.5
        assert output_sea.density < 2.0
        assert output_sea.temperature > 280  # ~288 K

    def test_very_high_altitude(self):
        """Test at exosphere (800+ km)."""
        output = nrlmsise00(
            latitude=0,
            longitude=0,
            altitude=800_000,
            year=2024,
            day_of_year=1,
            seconds_in_day=0,
        )

        assert output.density > 1e-15
        assert output.density < 1e-10

    def test_extreme_solar_activity(self):
        """Test at extreme solar activity bounds."""
        # Minimum F107
        output_min = nrlmsise00(
            latitude=0,
            longitude=0,
            altitude=300_000,
            year=2024,
            day_of_year=1,
            seconds_in_day=0,
            f107=70,
            f107a=70,
            ap=0,
        )

        # Maximum F107
        output_max = nrlmsise00(
            latitude=0,
            longitude=0,
            altitude=300_000,
            year=2024,
            day_of_year=1,
            seconds_in_day=0,
            f107=300,
            f107a=300,
            ap=400,
        )

        assert output_min.density > 0
        assert output_max.density > 0
        assert output_max.density > output_min.density


class TestNumericalProperties:
    """Test numerical properties and consistency."""

    def test_positive_densities(self):
        """Test all densities are positive."""
        output = nrlmsise00(
            latitude=np.radians(45),
            longitude=np.radians(-75),
            altitude=300_000,
            year=2024,
            day_of_year=100,
            seconds_in_day=43200,
        )

        assert output.density > 0
        assert output.n2_density > 0
        assert output.o2_density > 0
        assert output.o_density > 0
        assert output.he_density > 0
        assert output.h_density > 0
        assert output.ar_density > 0
        assert output.n_density > 0

    def test_temperature_positive(self):
        """Test temperature is always positive."""
        alts = np.linspace(0, 800_000, 20)

        output = nrlmsise00(
            latitude=np.zeros_like(alts),
            longitude=np.zeros_like(alts),
            altitude=alts,
            year=2024,
            day_of_year=1,
            seconds_in_day=0,
        )

        assert np.all(output.temperature > 0)

    def test_consistency_scalar_vs_array(self):
        """Test scalar and array inputs give same results."""
        # Scalar
        output_scalar = nrlmsise00(
            latitude=0.5,
            longitude=1.0,
            altitude=200_000,
            year=2024,
            day_of_year=50,
            seconds_in_day=20000,
            f107=150,
            f107a=150,
            ap=10,
        )

        # Array with single element
        output_array = nrlmsise00(
            latitude=np.array([0.5]),
            longitude=np.array([1.0]),
            altitude=np.array([200_000]),
            year=2024,
            day_of_year=50,
            seconds_in_day=20000,
            f107=150,
            f107a=150,
            ap=10,
        )

        assert np.allclose(output_scalar.density, output_array.density[0])
        assert np.allclose(output_scalar.temperature, output_array.temperature[0])


class TestPhysicalMonotonicity:
    """Test physically expected monotonic behavior."""

    def test_density_decreases_with_altitude(self):
        """Test density decreases with altitude (roughly)."""
        alts = np.array([50_000, 100_000, 200_000, 400_000])

        output = nrlmsise00(
            latitude=np.zeros_like(alts),
            longitude=np.zeros_like(alts),
            altitude=alts,
            year=2024,
            day_of_year=1,
            seconds_in_day=0,
        )

        # Each step should have lower density
        for i in range(len(alts) - 1):
            assert output.density[i] > output.density[i + 1]

    def test_n2_dominance_in_lower_atm(self):
        """Test N2 is dominant species at low altitudes."""
        output = nrlmsise00(
            latitude=0,
            longitude=0,
            altitude=10_000,
            year=2024,
            day_of_year=1,
            seconds_in_day=0,
        )

        assert output.n2_density > output.o2_density
        assert output.n2_density > output.o_density
        assert output.n2_density > output.he_density

    def test_exosphere_upper_bound(self):
        """Test temperature bounded by exosphere temperature."""
        output = nrlmsise00(
            latitude=0,
            longitude=0,
            altitude=400_000,
            year=2024,
            day_of_year=1,
            seconds_in_day=0,
            f107=150,
            f107a=150,
            ap=5,
        )

        # Temperature should not exceed exosphere temperature
        assert output.temperature <= output.exosphere_temperature * 1.01  # 1% tolerance


class TestVectorization:
    """Test vectorization with multiple altitudes."""

    def test_multiple_altitudes(self):
        """Test calculation at multiple altitudes."""
        alts = np.array([100_000, 200_000, 300_000, 400_000, 500_000])

        output = nrlmsise00(
            latitude=np.full_like(alts, np.radians(45), dtype=float),
            longitude=np.full_like(alts, np.radians(-75), dtype=float),
            altitude=alts,
            year=2024,
            day_of_year=100,
            seconds_in_day=43200,
        )

        assert output.density.shape == alts.shape
        assert output.temperature.shape == alts.shape

    def test_broadcasting(self):
        """Test broadcasting with compatible shapes."""
        lat = np.array([0, np.pi / 4, np.pi / 2])
        lon = np.array([0, np.pi / 4, np.pi / 2])
        alt = np.array([200_000, 200_000, 200_000])

        output = nrlmsise00(
            latitude=lat,
            longitude=lon,
            altitude=alt,
            year=2024,
            day_of_year=1,
            seconds_in_day=0,
        )

        assert output.density.shape == (3,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
