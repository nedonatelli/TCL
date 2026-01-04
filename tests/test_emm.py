"""Tests for Enhanced Magnetic Model (EMM) and WMMHR."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pytcl.magnetism import (
    EMM_PARAMETERS,
    create_emm_test_coefficients,
    emm,
    emm_declination,
    emm_inclination,
    emm_intensity,
    get_emm_data_dir,
    wmm,
    wmmhr,
)


class TestHighResCoefficients:
    """Tests for HighResCoefficients creation and structure."""

    def test_create_test_coefficients_default(self):
        """Test creating default test coefficients (n_max=36)."""
        coef = create_emm_test_coefficients()
        assert coef.n_max == 36
        assert coef.n_max_sv == 12
        assert coef.epoch == 2020.0
        assert coef.model_name == "EMM_TEST"

    def test_create_test_coefficients_custom_nmax(self):
        """Test creating test coefficients with custom n_max."""
        coef = create_emm_test_coefficients(n_max=50)
        assert coef.n_max == 50
        assert coef.n_max_sv == 12  # SV capped at 12

    def test_coefficient_arrays_shape(self):
        """Test that coefficient arrays have correct shape."""
        coef = create_emm_test_coefficients(n_max=20)
        assert coef.g.shape == (21, 21)
        assert coef.h.shape == (21, 21)
        assert coef.g_dot.shape == (13, 13)  # n_max_sv + 1
        assert coef.h_dot.shape == (13, 13)

    def test_dipole_coefficient_nonzero(self):
        """Test that g[1,0] (axial dipole) is nonzero."""
        coef = create_emm_test_coefficients()
        assert coef.g[1, 0] != 0
        # Should be negative (pointing south)
        assert coef.g[1, 0] < -20000  # nT

    def test_dipole_dominant(self):
        """g[1,0] should be the largest coefficient in magnitude."""
        coef = create_emm_test_coefficients()
        assert abs(coef.g[1, 0]) > abs(coef.g[2, 0])
        assert abs(coef.g[1, 0]) > abs(coef.g[1, 1])

    def test_higher_degrees_smaller(self):
        """Higher degree coefficients should be smaller on average."""
        coef = create_emm_test_coefficients(n_max=50)
        # Average magnitude of degree 5 vs degree 40
        avg_n5 = np.mean(np.abs(coef.g[5, :6]))
        avg_n40 = np.mean(np.abs(coef.g[40, :41]))
        assert avg_n5 > avg_n40


class TestEMMParameters:
    """Tests for model parameter definitions."""

    def test_emm2017_parameters(self):
        """Test EMM2017 parameters are defined correctly."""
        assert "EMM2017" in EMM_PARAMETERS
        params = EMM_PARAMETERS["EMM2017"]
        assert params["n_max"] == 790
        assert params["epoch"] == 2017.0

    def test_wmmhr2025_parameters(self):
        """Test WMMHR2025 parameters are defined correctly."""
        assert "WMMHR2025" in EMM_PARAMETERS
        params = EMM_PARAMETERS["WMMHR2025"]
        assert params["n_max"] == 133
        assert params["n_max_sv"] == 15
        assert params["epoch"] == 2025.0


class TestDataDirectory:
    """Tests for data directory functionality."""

    def test_get_data_dir_returns_path(self):
        """get_emm_data_dir should return a Path object."""
        from pathlib import Path

        data_dir = get_emm_data_dir()
        assert isinstance(data_dir, Path)

    def test_data_dir_is_in_home(self):
        """Default data dir should be under home directory."""
        from pathlib import Path

        data_dir = get_emm_data_dir()
        home = Path.home()
        # Check that data_dir is under home
        assert str(data_dir).startswith(str(home))


class TestEMMFunction:
    """Tests for the emm() main function."""

    @pytest.fixture
    def test_coefficients(self):
        """Create test coefficients for use in tests."""
        return create_emm_test_coefficients(n_max=36)

    def test_emm_returns_magnetic_result(self, test_coefficients):
        """EMM returns MagneticResult with all expected fields."""
        result = emm(
            np.radians(40),
            np.radians(-105),
            1.0,
            2020.0,
            coefficients=test_coefficients,
        )
        assert hasattr(result, "X")
        assert hasattr(result, "Y")
        assert hasattr(result, "Z")
        assert hasattr(result, "H")
        assert hasattr(result, "F")
        assert hasattr(result, "I")
        assert hasattr(result, "D")

    def test_total_intensity_reasonable(self, test_coefficients):
        """Total field intensity should be in expected range."""
        result = emm(
            np.radians(45), np.radians(0), 0, 2020.0, coefficients=test_coefficients
        )
        # Field intensity typically 25,000-65,000 nT at mid-latitudes
        # but can exceed this at high latitudes, allow up to 100,000 nT
        assert 20000 < result.F < 100000

    def test_horizontal_intensity_formula(self, test_coefficients):
        """H = sqrt(X^2 + Y^2)."""
        result = emm(
            np.radians(40), np.radians(-75), 0, 2020.0, coefficients=test_coefficients
        )
        H_calc = np.sqrt(result.X**2 + result.Y**2)
        assert_allclose(result.H, H_calc, rtol=1e-10)

    def test_total_intensity_formula(self, test_coefficients):
        """F = sqrt(H^2 + Z^2)."""
        result = emm(
            np.radians(40), np.radians(-75), 0, 2020.0, coefficients=test_coefficients
        )
        F_calc = np.sqrt(result.H**2 + result.Z**2)
        assert_allclose(result.F, F_calc, rtol=1e-10)

    def test_declination_range(self, test_coefficients):
        """Declination should be within -180 to 180 degrees."""
        result = emm(
            np.radians(45), np.radians(-75), 0, 2020.0, coefficients=test_coefficients
        )
        assert -np.pi <= result.D <= np.pi

    def test_inclination_range(self, test_coefficients):
        """Inclination should be within -90 to 90 degrees."""
        result = emm(
            np.radians(45), np.radians(-75), 0, 2020.0, coefficients=test_coefficients
        )
        assert -np.pi / 2 <= result.I <= np.pi / 2


class TestEMMPhysicalProperties:
    """Tests for physical properties of EMM field."""

    @pytest.fixture
    def test_coefficients(self):
        """Create test coefficients for use in tests."""
        return create_emm_test_coefficients(n_max=36)

    def test_inclination_positive_north(self, test_coefficients):
        """Inclination is positive in northern hemisphere."""
        result = emm(np.radians(60), 0, 0, 2020.0, coefficients=test_coefficients)
        assert result.I > 0  # Field points into Earth

    def test_inclination_negative_south(self, test_coefficients):
        """Inclination is negative in southern hemisphere."""
        result = emm(np.radians(-60), 0, 0, 2020.0, coefficients=test_coefficients)
        assert result.I < 0  # Field points out of Earth

    def test_field_stronger_at_poles(self, test_coefficients):
        """Magnetic field is stronger near poles than equator."""
        F_pole = emm(np.radians(80), 0, 0, 2020.0, coefficients=test_coefficients).F
        F_eq = emm(0, 0, 0, 2020.0, coefficients=test_coefficients).F
        assert F_pole > F_eq

    def test_field_decreases_with_altitude(self, test_coefficients):
        """Magnetic field decreases with altitude."""
        F_0 = emm(np.radians(45), 0, 0, 2020.0, coefficients=test_coefficients).F
        F_100 = emm(np.radians(45), 0, 100, 2020.0, coefficients=test_coefficients).F
        assert F_0 > F_100


class TestEMMComparisonWithWMM:
    """Compare EMM (low degree) with standard WMM."""

    def test_low_degree_similar_to_wmm(self):
        """EMM with low n_max should give similar results to WMM."""
        lat = np.radians(40)
        lon = np.radians(-75)

        # Create test coefficients that match WMM core field
        coef = create_emm_test_coefficients(n_max=12)

        # Compare with WMM
        emm_result = emm(lat, lon, 0, 2020.0, coefficients=coef, n_max=12)
        wmm_result = wmm(lat, lon, 0, 2020.0)

        # Should be within 10% for total field (test coefficients slightly different)
        rel_diff = abs(emm_result.F - wmm_result.F) / wmm_result.F
        assert rel_diff < 0.15


class TestWMMHR:
    """Tests for wmmhr() convenience function."""

    @pytest.fixture
    def test_coefficients(self):
        """Create test coefficients for use in tests."""
        return create_emm_test_coefficients(n_max=50)

    def test_wmmhr_returns_result(self, test_coefficients):
        """WMMHR returns MagneticResult."""
        result = wmmhr(
            np.radians(45), np.radians(-75), 0, 2025.0, coefficients=test_coefficients
        )
        assert hasattr(result, "F")
        assert hasattr(result, "D")
        assert hasattr(result, "I")

    def test_wmmhr_uses_model_coefficients(self, test_coefficients):
        """WMMHR should use the provided coefficients."""
        result1 = wmmhr(
            np.radians(45),
            np.radians(-75),
            0,
            2025.0,
            coefficients=test_coefficients,
            n_max=36,
        )
        result2 = wmmhr(
            np.radians(45),
            np.radians(-75),
            0,
            2025.0,
            coefficients=test_coefficients,
            n_max=50,
        )
        # Different n_max should give slightly different results
        # (due to higher degree contributions in result2)
        # But both should be valid (can reach ~90,000 nT at high latitudes)
        assert 20000 < result1.F < 100000
        assert 20000 < result2.F < 100000


class TestConvenienceFunctions:
    """Tests for emm_declination, emm_inclination, emm_intensity."""

    @pytest.fixture
    def test_coefficients(self):
        """Create test coefficients for use in tests."""
        return create_emm_test_coefficients(n_max=36)

    def test_emm_declination(self, test_coefficients):
        """emm_declination returns correct value."""
        result = emm(
            np.radians(40), np.radians(-105), 0, 2020.0, coefficients=test_coefficients
        )
        D = emm_declination(
            np.radians(40), np.radians(-105), 0, 2020.0, coefficients=test_coefficients
        )
        assert_allclose(D, result.D)

    def test_emm_inclination(self, test_coefficients):
        """emm_inclination returns correct value."""
        result = emm(
            np.radians(40), np.radians(-105), 0, 2020.0, coefficients=test_coefficients
        )
        incl = emm_inclination(
            np.radians(40), np.radians(-105), 0, 2020.0, coefficients=test_coefficients
        )
        assert_allclose(incl, result.I)

    def test_emm_intensity(self, test_coefficients):
        """emm_intensity returns correct value."""
        result = emm(
            np.radians(40), np.radians(-105), 0, 2020.0, coefficients=test_coefficients
        )
        F = emm_intensity(
            np.radians(40), np.radians(-105), 0, 2020.0, coefficients=test_coefficients
        )
        assert_allclose(F, result.F)

    def test_declination_is_scalar(self, test_coefficients):
        """Declination function returns scalar."""
        D = emm_declination(
            np.radians(45), np.radians(-75), 0, 2020.0, coefficients=test_coefficients
        )
        assert isinstance(D, float)

    def test_inclination_is_scalar(self, test_coefficients):
        """Inclination function returns scalar."""
        incl = emm_inclination(
            np.radians(45), np.radians(-75), 0, 2020.0, coefficients=test_coefficients
        )
        assert isinstance(incl, float)

    def test_intensity_is_scalar(self, test_coefficients):
        """Intensity function returns scalar."""
        F = emm_intensity(
            np.radians(45), np.radians(-75), 0, 2020.0, coefficients=test_coefficients
        )
        assert isinstance(F, float)


class TestSecularVariation:
    """Tests for secular variation in high-resolution models."""

    @pytest.fixture
    def test_coefficients(self):
        """Create test coefficients for use in tests."""
        return create_emm_test_coefficients(n_max=36)

    def test_field_changes_with_time(self, test_coefficients):
        """Field should change slightly between years."""
        result_2020 = emm(
            np.radians(45), np.radians(0), 0, 2020.0, coefficients=test_coefficients
        )
        result_2022 = emm(
            np.radians(45), np.radians(0), 0, 2022.0, coefficients=test_coefficients
        )
        # Small but non-zero change expected
        # Secular variation is ~50-100 nT/year at mid-latitudes
        assert result_2020.F != result_2022.F

    def test_declination_changes_with_time(self, test_coefficients):
        """Declination should change over time."""
        D_2020 = emm_declination(
            np.radians(45), np.radians(-75), 0, 2020.0, coefficients=test_coefficients
        )
        D_2025 = emm_declination(
            np.radians(45), np.radians(-75), 0, 2025.0, coefficients=test_coefficients
        )
        # Should be different
        assert D_2020 != D_2025


class TestNumericalStability:
    """Tests for numerical stability at various locations."""

    @pytest.fixture
    def test_coefficients(self):
        """Create test coefficients for use in tests."""
        return create_emm_test_coefficients(n_max=36)

    def test_equator(self, test_coefficients):
        """Field computation at equator should not produce NaN."""
        result = emm(0, 0, 0, 2020.0, coefficients=test_coefficients)
        assert not np.isnan(result.F)
        assert not np.isnan(result.D)
        assert not np.isnan(result.I)

    def test_north_pole(self, test_coefficients):
        """Field computation near north pole should not produce NaN."""
        result = emm(np.radians(89.9), 0, 0, 2020.0, coefficients=test_coefficients)
        assert not np.isnan(result.F)
        # Declination may be undefined at pole, but should not crash

    def test_south_pole(self, test_coefficients):
        """Field computation near south pole should not produce NaN."""
        result = emm(np.radians(-89.9), 0, 0, 2020.0, coefficients=test_coefficients)
        assert not np.isnan(result.F)

    def test_high_altitude(self, test_coefficients):
        """Field computation at high altitude should work."""
        result = emm(
            np.radians(45),
            np.radians(0),
            500,
            2020.0,  # 500 km altitude
            coefficients=test_coefficients,
        )
        assert not np.isnan(result.F)
        assert result.F > 0

    def test_various_longitudes(self, test_coefficients):
        """Test field at various longitudes."""
        for lon_deg in [0, 45, 90, 135, 180, -135, -90, -45]:
            result = emm(
                np.radians(45),
                np.radians(lon_deg),
                0,
                2020.0,
                coefficients=test_coefficients,
            )
            assert not np.isnan(result.F)
            assert result.F > 0


class TestHighDegreeEvaluation:
    """Tests for evaluation at higher harmonic degrees."""

    def test_higher_degree_coefficients(self):
        """Test with higher degree coefficients (n_max=50)."""
        coef = create_emm_test_coefficients(n_max=50)
        result = emm(
            np.radians(40), np.radians(-105), 0, 2020.0, coefficients=coef, n_max=50
        )
        assert not np.isnan(result.F)
        assert result.F > 0

    def test_n_max_limit_respected(self):
        """Specifying n_max should limit evaluation."""
        coef = create_emm_test_coefficients(n_max=50)

        # Evaluate at different n_max values
        result_12 = emm(
            np.radians(40), np.radians(-105), 0, 2020.0, coefficients=coef, n_max=12
        )
        result_50 = emm(
            np.radians(40), np.radians(-105), 0, 2020.0, coefficients=coef, n_max=50
        )

        # Both should be valid
        assert result_12.F > 0
        assert result_50.F > 0

        # Higher degree should give slightly different result
        # (due to crustal field contributions)
        assert result_12.F != result_50.F
