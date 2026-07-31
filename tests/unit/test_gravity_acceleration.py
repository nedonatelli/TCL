"""``gravity_acceleration`` against closed-form fields.

The primary entry point of the gravity package, and until gh-47 no test
referenced it. Spherical-harmonic synthesis is exactly where this library has
been wrong before without anything noticing: `associated_legendre_scaled` was
off by 1e199 at degree 2000 while the suite stayed green, and the ordinary
synthesis returned 3.6e20 where the potential is 6.25e7 at EGM2008's degree.

Two fields have closed forms and are used as the reference here: a point mass,
and a point mass plus J2. Both are exact, so the comparisons are to machine
precision rather than to a tolerance chosen to make them pass.

This sits in unit/ rather than validation/ because the expected values are
derived from the mathematics in this file, not obtained from an outside
implementation -- PROPERTY class in the CONTRIBUTING taxonomy, not REFERENCE.

Coefficients are 4pi (geodesy) normalized, which is what the function expects:
supplying an unnormalized ``C[2, 0] = -J2`` disagrees with the closed form by
2e-3 relative, while ``-J2 / sqrt(5)`` matches to 3e-15.
"""

import numpy as np
import pytest

from pytcl.gravity import gravity_acceleration

GM = 3.986004418e14  # WGS84
R = 6378137.0  # WGS84 equatorial radius
J2 = 1.08262668e-3  # WGS84


def _monopole():
    C = np.zeros((3, 3))
    S = np.zeros((3, 3))
    C[0, 0] = 1.0
    return C, S


def _monopole_plus_j2():
    C, S = _monopole()
    C[2, 0] = -J2 / np.sqrt(5.0)  # 4pi-normalized zonal term
    return C, S


class TestPointMass:
    """A single C[0,0] term is Newtonian gravity, which has an exact answer."""

    @pytest.mark.parametrize("height_km", [0.0, 400.0, 20_000.0, 35_786.0])
    def test_radial_component_is_minus_gm_over_r_squared(self, height_km):
        C, S = _monopole()
        h = height_km * 1e3
        g_r, _, _ = gravity_acceleration(0.0, 0.0, h, C, S, R, GM)
        assert g_r == pytest.approx(-GM / (R + h) ** 2, rel=1e-14)

    @pytest.mark.parametrize("lat_deg", [-90.0, -45.0, 0.0, 30.0, 90.0])
    def test_no_tangential_component(self, lat_deg):
        """A spherically symmetric field pulls straight down, everywhere."""
        C, S = _monopole()
        _, g_lat, g_lon = gravity_acceleration(
            np.radians(lat_deg), 1.0, 0.0, C, S, R, GM
        )
        assert g_lat == pytest.approx(0.0, abs=1e-12)
        assert g_lon == pytest.approx(0.0, abs=1e-12)

    def test_field_is_isotropic(self):
        """Same magnitude at every latitude and longitude, at fixed radius."""
        C, S = _monopole()
        values = [
            gravity_acceleration(np.radians(lat), np.radians(lon), 0.0, C, S, R, GM)[0]
            for lat in (-60.0, 0.0, 17.0, 60.0)
            for lon in (0.0, 90.0, 271.0)
        ]
        assert np.allclose(values, values[0], rtol=1e-14)

    def test_obeys_the_inverse_square_law(self):
        """Doubling the radius must quarter the acceleration."""
        C, S = _monopole()
        g_at_r, _, _ = gravity_acceleration(0.0, 0.0, 0.0, C, S, R, GM)
        g_at_2r, _, _ = gravity_acceleration(0.0, 0.0, R, C, S, R, GM)
        assert g_at_2r == pytest.approx(g_at_r / 4.0, rel=1e-14)


class TestJ2Oblateness:
    """C[2,0] reproduces the closed-form J2 field.

    g_r   = -GM/r^2 [1 - (3/2) J2 (R/r)^2 (3 sin^2(phi) - 1)]
    g_phi = -GM/r^2 (3 J2) (R/r)^2 sin(phi) cos(phi)
    """

    @staticmethod
    def _closed_form(lat, r):
        s, c = np.sin(lat), np.cos(lat)
        scale = -GM / r**2
        g_r = scale * (1.0 - 1.5 * J2 * (R / r) ** 2 * (3.0 * s * s - 1.0))
        g_lat = scale * 3.0 * J2 * (R / r) ** 2 * s * c
        return g_r, g_lat

    @pytest.mark.parametrize("lat_deg", [-90.0, -45.0, 0.0, 23.5, 45.0, 90.0])
    def test_radial_component_matches_closed_form(self, lat_deg):
        C, S = _monopole_plus_j2()
        lat = np.radians(lat_deg)
        g_r, _, _ = gravity_acceleration(lat, 0.0, 0.0, C, S, R, GM)
        expected, _ = self._closed_form(lat, R)
        assert g_r == pytest.approx(expected, rel=1e-12)

    @pytest.mark.parametrize("lat_deg", [-45.0, 20.0, 45.0, 70.0])
    def test_meridional_component_matches_closed_form(self, lat_deg):
        C, S = _monopole_plus_j2()
        lat = np.radians(lat_deg)
        _, g_lat, _ = gravity_acceleration(lat, 0.0, 0.0, C, S, R, GM)
        _, expected = self._closed_form(lat, R)
        assert g_lat == pytest.approx(expected, rel=1e-10)

    def test_poles_are_pulled_harder_than_the_equator(self):
        """The physical signature of positive J2: mass bulges at the equator."""
        C, S = _monopole_plus_j2()
        g_equator, _, _ = gravity_acceleration(0.0, 0.0, 0.0, C, S, R, GM)
        g_pole, _, _ = gravity_acceleration(np.pi / 2, 0.0, 0.0, C, S, R, GM)
        assert abs(g_pole) < abs(g_equator)

    def test_zonal_term_is_independent_of_longitude(self):
        """A zonal harmonic has no longitude dependence, by definition."""
        C, S = _monopole_plus_j2()
        lat = np.radians(35.0)
        values = [
            gravity_acceleration(lat, np.radians(lon), 0.0, C, S, R, GM)
            for lon in (0.0, 45.0, 180.0, 300.0)
        ]
        for component in range(3):
            column = [v[component] for v in values]
            assert np.allclose(column, column[0], rtol=1e-12, atol=1e-14)

    def test_no_tangential_component_at_the_equator_or_poles(self):
        """sin(phi) cos(phi) vanishes at both, so the meridional pull does too."""
        C, S = _monopole_plus_j2()
        for lat in (0.0, np.pi / 2, -np.pi / 2):
            _, g_lat, _ = gravity_acceleration(lat, 0.0, 0.0, C, S, R, GM)
            assert g_lat == pytest.approx(0.0, abs=1e-9)


class TestSectoralTerms:
    """A tesseral/sectoral term must introduce longitude dependence."""

    def test_c22_varies_with_longitude(self):
        C, S = _monopole()
        C[2, 2] = 1e-6
        lat = np.radians(10.0)
        values = [
            gravity_acceleration(lat, np.radians(lon), 0.0, C, S, R, GM)[0]
            for lon in (0.0, 45.0, 90.0)
        ]
        assert not np.allclose(values, values[0], rtol=1e-12), (
            "a sectoral coefficient produced no longitude dependence"
        )

    def test_s22_shifts_the_pattern_a_quarter_period_in_longitude(self):
        """The sine coefficient must act, and be a quarter period out of phase.

        Compared over a profile rather than at one longitude: C and S agree
        exactly wherever cos(m*lon) == sin(m*lon), which for m = 2 happens at
        lon = 22.5 degrees. A single sample there proves nothing.
        """
        C_cos, S_cos = _monopole()
        C_cos[2, 2] = 1e-6
        C_sin, S_sin = _monopole()
        S_sin[2, 2] = 1e-6

        lat = np.radians(10.0)
        longitudes = np.radians(np.arange(0.0, 180.0, 15.0))
        cos_profile = np.array(
            [
                gravity_acceleration(lat, lon, 0.0, C_cos, S_cos, R, GM)[0]
                for lon in longitudes
            ]
        )
        sin_profile = np.array(
            [
                gravity_acceleration(lat, lon, 0.0, C_sin, S_sin, R, GM)[0]
                for lon in longitudes
            ]
        )

        assert not np.allclose(cos_profile, sin_profile, rtol=1e-12), (
            "the S coefficient produced the same field as the C coefficient, "
            "so the sine term is being dropped"
        )

        # The cosine term peaks at lon = 0; the sine term is zero there.
        at_zero_cos = gravity_acceleration(lat, 0.0, 0.0, C_cos, S_cos, R, GM)[0]
        at_zero_sin = gravity_acceleration(lat, 0.0, 0.0, C_sin, S_sin, R, GM)[0]
        monopole_C, monopole_S = _monopole()
        plain = gravity_acceleration(lat, 0.0, 0.0, monopole_C, monopole_S, R, GM)[0]

        assert at_zero_sin == pytest.approx(plain, rel=1e-12)
        assert at_zero_cos != pytest.approx(plain, rel=1e-12)


def test_n_max_truncates_the_expansion():
    """Passing n_max must actually drop the higher-degree terms."""
    C, S = _monopole_plus_j2()
    full = gravity_acceleration(np.radians(45.0), 0.0, 0.0, C, S, R, GM)[0]
    truncated = gravity_acceleration(np.radians(45.0), 0.0, 0.0, C, S, R, GM, n_max=0)[
        0
    ]

    monopole_C, monopole_S = _monopole()
    monopole = gravity_acceleration(
        np.radians(45.0), 0.0, 0.0, monopole_C, monopole_S, R, GM
    )[0]

    assert truncated == pytest.approx(monopole, rel=1e-14)
    assert truncated != pytest.approx(full, rel=1e-9)
