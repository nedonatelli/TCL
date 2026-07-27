"""Correctness audit tests for pytcl.gravity.

Reference-validates the gravity module against published values and
independent implementations:

- Somigliana / WGS84 normal gravity vs NIMA TR8350.2 values
- Free-air gradient vs the canonical 0.3086 mGal/m
- J2 gravity vs the numerical gradient of the J2 + centrifugal potential
- Fully normalized associated Legendre functions vs scipy.special.lpmv
- Clenshaw summation vs direct Legendre sums (machine precision)
- EGM coefficient parsing on synthetic files (incl. Fortran D exponents)
- Geoid height vs an independent fully-normalized harmonic sum
- Solid Earth tides: M2 spectral peak at 12.42 h, physical bounds
- Fundamental arguments vs the IERS 2010 polynomial expressions
- Pole tide vs the IERS 2010 (-33 mm / 9 mm per arcsec) coefficients
"""

from math import factorial, lgamma

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.special import lpmv

from pytcl.gravity import (
    WGS84,
    associated_legendre,
    associated_legendre_scaled,
    bouguer_anomaly,
    clenshaw_gravity,
    clenshaw_potential,
    clenshaw_sum_order,
    clenshaw_sum_order_derivative,
    create_test_coefficients,
    deflection_of_vertical,
    free_air_anomaly,
    geoid_height,
    geoid_height_j2,
    geoid_heights,
    gravitational_potential,
    gravity_disturbance,
    gravity_j2,
    gravity_wgs84,
    normal_gravity,
    normal_gravity_somigliana,
    spherical_harmonic_sum,
)
from pytcl.gravity.clenshaw import clenshaw_geoid
from pytcl.gravity.egm import EGMCoefficients, _subtract_reference_field, parse_egm_file
from pytcl.gravity.spherical_harmonics import associated_legendre_derivative
from pytcl.gravity.tides import (
    GRAVIMETRIC_FACTOR,
    LOVE_H2,
    LOVE_K2,
    SHIDA_L2,
    atmospheric_pressure_loading,
    fundamental_arguments,
    julian_centuries_j2000,
    moon_position_approximate,
    ocean_tide_loading_displacement,
    pole_tide_displacement,
    solid_earth_tide_displacement,
    solid_earth_tide_gravity,
    sun_position_approximate,
    tidal_gravity_correction,
    total_tidal_displacement,
)


def full_norm_legendre(n: int, m: int, x: float) -> float:
    """Geodesy fully normalized P̄nm via scipy (no Condon-Shortley phase)."""
    p = lpmv(m, n, x) * (-1.0) ** m
    norm = np.sqrt((2 - (m == 0)) * (2 * n + 1) * factorial(n - m) / factorial(n + m))
    return norm * p


class TestNormalGravity:
    """Normal gravity vs NIMA TR8350.2 published values."""

    def test_somigliana_equator(self):
        # NIMA TR8350.2: gamma_e = 9.7803253359 m/s^2
        assert_allclose(normal_gravity_somigliana(0.0), 9.7803253359, atol=1e-6)

    def test_somigliana_pole(self):
        # NIMA TR8350.2: gamma_p = 9.8321849379 m/s^2
        assert_allclose(normal_gravity_somigliana(np.pi / 2), 9.8321849379, atol=1e-6)

    def test_somigliana_45deg(self):
        assert_allclose(normal_gravity_somigliana(np.radians(45)), 9.806199, atol=5e-6)

    def test_free_air_gradient(self):
        """The free-air gradient should be the canonical 0.3086 mGal/m."""
        lat = np.radians(45)
        grad = (normal_gravity(lat, 0.0) - normal_gravity(lat, 1.0)) * 1e5
        assert_allclose(grad, 0.3086, atol=3e-4)

    def test_height_decreases_gravity(self):
        lat = np.radians(30)
        assert normal_gravity(lat, 5000.0) < normal_gravity(lat, 0.0)

    def test_gravity_wgs84_matches_normal_gravity(self):
        lat = np.radians(52.0)
        res = gravity_wgs84(lat, 0.1, 250.0)
        assert_allclose(res.magnitude, normal_gravity(lat, 250.0), rtol=1e-12)
        assert res.g_down == res.magnitude
        assert res.g_north == 0.0 and res.g_east == 0.0


class TestGravityJ2:
    """J2 gravity vs numerical gradient of the J2 + centrifugal potential."""

    @staticmethod
    def _potential(r, lat):
        GM, a, J2, om = WGS84.GM, WGS84.a, WGS84.J2, WGS84.omega
        P2 = 0.5 * (3 * np.sin(lat) ** 2 - 1)
        return GM / r * (1 - J2 * (a / r) ** 2 * P2) + 0.5 * om**2 * r**2 * (
            np.cos(lat) ** 2
        )

    @pytest.mark.parametrize("lat_deg", [0.0, 30.0, 45.0, 60.0, 89.0, -37.0])
    def test_components_match_gradient(self, lat_deg):
        lat = np.radians(lat_deg)
        r = WGS84.a
        eps_r, eps_l = 1.0, 1e-6
        g_up = (self._potential(r + eps_r, lat) - self._potential(r - eps_r, lat)) / (
            2 * eps_r
        )
        g_north = (
            (self._potential(r, lat + eps_l) - self._potential(r, lat - eps_l))
            / (2 * eps_l)
            / r
        )
        res = gravity_j2(lat, 0.0, 0.0)
        assert_allclose(res.g_down, -g_up, rtol=1e-9)
        assert_allclose(res.g_north, g_north, atol=1e-7)

    def test_magnitude_includes_north(self):
        res = gravity_j2(np.radians(45), 0.0, 0.0)
        assert_allclose(res.magnitude, np.hypot(res.g_down, res.g_north), rtol=1e-12)


class TestGeoidHeightJ2:
    """geoid_height_j2 returns the documented -a*J2*P2 surface."""

    @pytest.mark.parametrize("lat_deg", [0.0, 45.0, 90.0, -60.0])
    def test_analytic(self, lat_deg):
        lat = np.radians(lat_deg)
        P2 = 0.5 * (3 * np.sin(lat) ** 2 - 1)
        assert_allclose(geoid_height_j2(lat), -WGS84.a * WGS84.J2 * P2, rtol=1e-12)


class TestGravitationalPotential:
    """gravitational_potential vs the closed-form J2 potential."""

    @pytest.mark.parametrize(
        "lat_deg,lon_deg,r",
        [(0.0, 0.0, 6.4e6), (45.0, 30.0, 6.5e6), (-60.0, 100.0, 7.0e6)],
    )
    def test_closed_form(self, lat_deg, lon_deg, r):
        lat, lon = np.radians(lat_deg), np.radians(lon_deg)
        GM, a, J2 = WGS84.GM, WGS84.a, WGS84.J2
        P2 = 0.5 * (3 * np.sin(lat) ** 2 - 1)
        expected = GM / r * (1 - J2 * (a / r) ** 2 * P2)
        assert_allclose(gravitational_potential(lat, lon, r), expected, rtol=1e-10)


class TestAnomalies:
    def test_free_air_definition(self):
        lat = np.radians(45)
        fa = free_air_anomaly(9.81, lat, 100.0)
        assert_allclose(fa, 9.81 - normal_gravity(lat, 100.0), rtol=1e-12)

    def test_bouguer_plate_correction(self):
        """The Bouguer plate for 1 km of 2670 kg/m^3 rock is ~112 mGal."""
        lat = np.radians(45)
        fa = free_air_anomaly(9.81, lat, 1000.0)
        ba = bouguer_anomaly(9.81, lat, 1000.0)
        plate_mgal = (fa - ba) * 1e5
        assert_allclose(plate_mgal, 111.9, atol=0.2)


class TestAssociatedLegendre:
    """Fully normalized Legendre functions vs scipy.special.lpmv."""

    def test_fully_normalized_vs_scipy(self):
        x = 0.37
        n_max = 10
        P = associated_legendre(n_max, n_max, x, normalized=True)
        for n in range(n_max + 1):
            for m in range(n + 1):
                assert_allclose(
                    P[n, m],
                    full_norm_legendre(n, m, x),
                    rtol=1e-10,
                    atol=1e-12,
                    err_msg=f"n={n}, m={m}",
                )

    def test_unnormalized_vs_scipy(self):
        x = -0.42
        n_max = 8
        P = associated_legendre(n_max, n_max, x, normalized=False)
        for n in range(n_max + 1):
            for m in range(n + 1):
                # No Condon-Shortley phase in the geodesy convention
                assert_allclose(
                    P[n, m],
                    lpmv(m, n, x) * (-1.0) ** m,
                    rtol=1e-10,
                    atol=1e-12,
                    err_msg=f"n={n}, m={m}",
                )

    @pytest.mark.parametrize("n", [2, 5, 10, 30])
    def test_addition_theorem(self, n):
        """sum_m P̄nm(x)^2 = 2n+1 for the geodesy full normalization."""
        x = np.cos(np.radians(40.0))
        P = associated_legendre(n, n, x, normalized=True)
        assert_allclose(np.sum(P[n, : n + 1] ** 2), 2 * n + 1, rtol=1e-12)

    def test_derivative_vs_numerical(self):
        """dP/dx consistent with central differences of P."""
        x = 0.3
        eps = 1e-6
        n_max = 8
        dP = associated_legendre_derivative(n_max, n_max, x)
        Pp = associated_legendre(n_max, n_max, x + eps)
        Pm = associated_legendre(n_max, n_max, x - eps)
        num = (Pp - Pm) / (2 * eps)
        assert_allclose(dP, num, atol=5e-5)


class TestScaledLegendre:
    def test_matches_unscaled_low_degree(self):
        x = np.cos(np.radians(40.0))
        P_scaled, exp = associated_legendre_scaled(100, 100, x)
        P = associated_legendre(100, 100, x)
        assert np.all(exp == 0)
        # The unscaled path quantizes x to 1e-12 for caching, which bounds
        # the achievable agreement at degree 100 to ~n^2 * 5e-13
        assert_allclose(P_scaled, P, rtol=1e-8, atol=1e-10)

    @pytest.mark.parametrize("n_max", [200, 500])
    def test_addition_theorem_high_degree(self, n_max):
        """Reconstructed values satisfy sum_m P̄nm^2 = 2n+1 at n = n_max."""
        x = np.cos(np.radians(40.0))
        P_scaled, exp = associated_legendre_scaled(n_max, n_max, x)
        row = P_scaled[n_max, :]
        mx = np.max(np.abs(row))
        assert mx > 0
        s_scaled = np.sum((row / mx) ** 2)
        log10_sum = np.log10(s_scaled) + 2 * np.log10(mx) + 2 * exp[n_max]
        assert_allclose(log10_sum, np.log10(2 * n_max + 1), atol=1e-9)

    def test_sectoral_seeds_do_not_underflow(self):
        """Regression: scale ratios must be applied before absolute scales."""
        x = np.cos(np.radians(40.0))
        P_scaled, _ = associated_legendre_scaled(200, 200, x)
        diag = np.diagonal(P_scaled)
        assert np.count_nonzero(diag) == 201


class TestClenshaw:
    """Clenshaw summation vs direct Legendre sums (must agree to ~machine)."""

    def setup_method(self):
        rng = np.random.default_rng(42)
        self.n_max = 30
        self.C = rng.standard_normal((self.n_max + 1, self.n_max + 1)) * 1e-6
        self.S = rng.standard_normal((self.n_max + 1, self.n_max + 1)) * 1e-6

    @pytest.mark.parametrize("m", [0, 1, 5, 17, 30])
    def test_sum_order_vs_direct(self, m):
        theta = np.radians(37.0)
        ct, st = np.cos(theta), np.sin(theta)
        sc, ss = clenshaw_sum_order(m, ct, st, self.C, self.S, self.n_max)
        sc_ref = sum(
            self.C[n, m] * full_norm_legendre(n, m, ct)
            for n in range(m, self.n_max + 1)
        )
        ss_ref = sum(
            self.S[n, m] * full_norm_legendre(n, m, ct)
            for n in range(m, self.n_max + 1)
        )
        assert_allclose(sc, sc_ref, rtol=1e-10, atol=1e-18)
        assert_allclose(ss, ss_ref, rtol=1e-10, atol=1e-18)

    @pytest.mark.parametrize("m", [0, 3, 12])
    def test_sum_order_derivative_vs_numerical(self, m):
        theta = np.radians(37.0)
        eps = 1e-7
        _, _, dsc, dss = clenshaw_sum_order_derivative(
            m, np.cos(theta), np.sin(theta), self.C, self.S, self.n_max
        )
        scp, ssp = clenshaw_sum_order(
            m, np.cos(theta + eps), np.sin(theta + eps), self.C, self.S, self.n_max
        )
        scm, ssm = clenshaw_sum_order(
            m, np.cos(theta - eps), np.sin(theta - eps), self.C, self.S, self.n_max
        )
        assert_allclose(dsc, (scp - scm) / (2 * eps), rtol=1e-6, atol=1e-12)
        assert_allclose(dss, (ssp - ssm) / (2 * eps), rtol=1e-6, atol=1e-12)

    def test_potential_vs_spherical_harmonic_sum(self):
        lat, lon = np.radians(53.0), np.radians(-40.0)
        r, R, GM = 6.6e6, 6378136.3, 3.986004415e14
        Vc = clenshaw_potential(lat, lon, r, self.C, self.S, R, GM, self.n_max)
        Vd, dVr, dVlat = spherical_harmonic_sum(
            lat, lon, r, self.C, self.S, R, GM, self.n_max
        )
        # Direct sum quantizes x to 1e-8 in its cache, hence the tolerance
        assert_allclose(Vc, Vd, rtol=1e-7)
        g_r, g_lat, _ = clenshaw_gravity(lat, lon, r, self.C, self.S, R, GM, self.n_max)
        assert_allclose(dVr, g_r, rtol=1e-7)
        assert_allclose(dVlat, g_lat, rtol=1e-7)

    def test_potential_central_term(self):
        C = np.zeros((3, 3))
        S = np.zeros((3, 3))
        C[0, 0] = 1.0
        R, GM = 6.378e6, 3.986e14
        V = clenshaw_potential(0.0, 0.0, R, C, S, R, GM)
        assert_allclose(V, GM / R, rtol=1e-12)

    def test_geoid_excludes_reference_terms(self):
        """clenshaw_geoid must exclude n=0,1 as documented."""
        C = np.zeros((5, 5))
        S = np.zeros((5, 5))
        C[0, 0] = 1.0
        C[1, 0] = 1e-6
        N = clenshaw_geoid(0.0, 0.0, C, S, 6.378e6, 3.986e14, 9.81)
        assert N == 0.0

    def test_geoid_bruns_formula(self):
        """A single C20 disturbing term follows Bruns' formula."""
        C = np.zeros((5, 5))
        S = np.zeros((5, 5))
        C[2, 0] = 1e-8
        R, GM, gamma = 6.378e6, 3.986e14, 9.81
        lat = np.radians(30.0)
        N = clenshaw_geoid(lat, 0.0, C, S, R, GM, gamma)
        expected = GM / (R * gamma) * 1e-8 * full_norm_legendre(2, 0, np.sin(lat))
        assert_allclose(N, expected, rtol=1e-9)


class TestSphericalHarmonicSum:
    def test_j2_only_closed_form(self):
        """A normalized C20 = -J2/sqrt(5) reproduces the J2 potential."""
        GM, a, J2 = WGS84.GM, WGS84.a, WGS84.J2
        C = np.zeros((3, 3))
        S = np.zeros((3, 3))
        C[0, 0] = 1.0
        C[2, 0] = -J2 / np.sqrt(5)
        lat, lon, r = np.radians(37.0), np.radians(12.0), 6.6e6
        V, dV_r, dV_lat = spherical_harmonic_sum(lat, lon, r, C, S, a, GM, 2)
        P2 = 0.5 * (3 * np.sin(lat) ** 2 - 1)
        V_ref = GM / r * (1 - J2 * (a / r) ** 2 * P2)
        assert_allclose(V, V_ref, rtol=1e-10)
        # Radial derivative of the closed form
        dV_ref = -GM / r**2 * (1 - 3 * J2 * (a / r) ** 2 * P2)
        assert_allclose(dV_r, dV_ref, rtol=1e-10)


class TestEGMParsing:
    def _write(self, tmp_path, text):
        p = tmp_path / "test.cof"
        p.write_text(text)
        return p

    def test_parse_basic(self, tmp_path):
        p = self._write(
            tmp_path,
            "2 0 -4.84e-4 0.0\n2 1 1.0e-9 2.0e-9\n2 2 2.4e-6 -1.4e-6\n",
        )
        C, S = parse_egm_file(p)
        assert C.shape == (3, 3)
        assert C[0, 0] == 1.0  # Central term convention
        assert_allclose(C[2, 0], -4.84e-4)
        assert_allclose(S[2, 1], 2.0e-9)
        assert_allclose(S[2, 2], -1.4e-6)

    def test_parse_fortran_d_exponents(self, tmp_path):
        """NGA files use Fortran D exponents; they must parse correctly."""
        p = self._write(
            tmp_path,
            "    2    0 -0.484165143790815D-03  0.000000000000000D+00\n"
            "    2    2  0.243938357328313D-05 -0.140027370385934D-05\n",
        )
        C, S = parse_egm_file(p)
        assert_allclose(C[2, 0], -0.484165143790815e-3, rtol=1e-15)
        assert_allclose(S[2, 2], -0.140027370385934e-5, rtol=1e-15)

    def test_parse_skips_headers_and_comments(self, tmp_path):
        p = self._write(
            tmp_path,
            "# comment line\nEGM MODEL HEADER WITH FOUR TOKENS\n2 0 -4.84e-4 0.0\n",
        )
        C, S = parse_egm_file(p)
        assert C.shape == (3, 3)
        assert_allclose(C[2, 0], -4.84e-4)

    def test_parse_n_max_truncation(self, tmp_path):
        p = self._write(
            tmp_path,
            "2 0 1.0e-6 0.0\n3 0 2.0e-6 0.0\n4 0 3.0e-6 0.0\n",
        )
        C, S = parse_egm_file(p, n_max=3)
        assert C.shape == (4, 4)
        assert_allclose(C[3, 0], 2.0e-6)


class TestGeoidHeightEGM:
    def test_c20_only_analytic(self):
        """With C20 = C20_ref + delta, N follows Bruns' formula for delta."""
        n_max = 4
        C = np.zeros((n_max + 1, n_max + 1))
        S = np.zeros((n_max + 1, n_max + 1))
        # Recover the reference C20 subtracted internally
        ref = np.zeros((n_max + 1, n_max + 1))
        _subtract_reference_field(ref, n_max)
        c20_ref = -ref[2, 0]
        delta = 1.0e-8
        C[0, 0] = 1.0
        C[2, 0] = c20_ref + delta
        coef = EGMCoefficients(
            C=C, S=S, GM=3.986004415e14, R=6378136.3, n_max=n_max, model_name="TEST"
        )
        for lat_deg in [0.0, 30.0, -60.0]:
            lat = np.radians(lat_deg)
            N = geoid_height(lat, 0.0, coefficients=coef)
            gamma = normal_gravity_somigliana(lat, WGS84)
            # Higher even zonal reference terms (J4...) remain in the
            # disturbing field; include them in the expectation.
            # After internal subtraction: C_dist[2,0] = delta and
            # C_dist[4,0] = ref[4,0] (the negated J4 reference term)
            expected = (
                coef.GM
                / (coef.R * gamma)
                * sum(
                    (delta if n == 2 else ref[n, 0])
                    * full_norm_legendre(n, 0, np.sin(lat))
                    for n in (2, 4)
                )
            )
            assert_allclose(N, expected, rtol=1e-6, err_msg=f"lat={lat_deg}")

    def test_low_degree_egm2008_vs_independent_sum(self):
        """geoid_height matches an independent fully normalized sum."""
        coef = create_test_coefficients(n_max=6)
        for lat_deg, lon_deg in [(0.0, 0.0), (45.0, 10.0), (-30.0, 120.0)]:
            lat, lon = np.radians(lat_deg), np.radians(lon_deg)
            Cd = coef.C.copy()
            Sd = coef.S.copy()
            Cd[0, 0] = 0.0
            Cd[1, :] = 0.0
            Sd[1, :] = 0.0
            _subtract_reference_field(Cd, coef.n_max)
            x = np.sin(lat)
            T = 0.0
            for n in range(2, coef.n_max + 1):
                for m in range(n + 1):
                    T += full_norm_legendre(n, m, x) * (
                        Cd[n, m] * np.cos(m * lon) + Sd[n, m] * np.sin(m * lon)
                    )
            T *= coef.GM / coef.R
            expected = T / normal_gravity_somigliana(lat, WGS84)
            N = geoid_height(lat, lon, coefficients=coef)
            assert_allclose(N, expected, rtol=1e-6)

    def test_low_degree_geoid_in_physical_range(self):
        """With the normal field removed, low-degree geoid is within ±110 m."""
        coef = create_test_coefficients(n_max=6)
        lats = np.radians(np.linspace(-85, 85, 9))
        lons = np.radians(np.linspace(-180, 160, 9))
        heights = geoid_heights(lats, lons, coefficients=coef)
        assert np.all(np.abs(heights) < 110.0)

    def test_geoid_heights_matches_scalar(self):
        coef = create_test_coefficients(n_max=6)
        lats = np.radians(np.array([10.0, -45.0]))
        lons = np.radians(np.array([20.0, 100.0]))
        hs = geoid_heights(lats, lons, coefficients=coef)
        for i in range(2):
            assert_allclose(
                hs[i], geoid_height(lats[i], lons[i], coefficients=coef), rtol=1e-12
            )


class TestGravityDisturbanceEGM:
    def test_magnitude_physical(self):
        """Low-degree gravity disturbance is well below 1e-3 m/s^2."""
        coef = create_test_coefficients(n_max=6)
        d = gravity_disturbance(np.radians(45), np.radians(10), coefficients=coef)
        assert 0 < d.magnitude < 1e-3
        assert_allclose(
            d.magnitude,
            np.sqrt(d.delta_g_r**2 + d.delta_g_lat**2 + d.delta_g_lon**2),
            rtol=1e-12,
        )

    def test_radial_component_vs_potential_derivative(self):
        """delta_g_r equals the radial derivative of the disturbing potential."""
        coef = create_test_coefficients(n_max=6)
        lat, lon = np.radians(20.0), np.radians(-70.0)
        Cd = coef.C.copy()
        Sd = coef.S.copy()
        Cd[0, 0] = 0.0
        Cd[1, :] = 0.0
        Sd[1, :] = 0.0
        _subtract_reference_field(Cd, coef.n_max)
        r = coef.R
        eps = 1.0
        Tp = clenshaw_potential(lat, lon, r + eps, Cd, Sd, coef.R, coef.GM, coef.n_max)
        Tm = clenshaw_potential(lat, lon, r - eps, Cd, Sd, coef.R, coef.GM, coef.n_max)
        d = gravity_disturbance(lat, lon, h=0.0, coefficients=coef)
        assert_allclose(d.delta_g_r, (Tp - Tm) / (2 * eps), rtol=1e-5)


class TestDeflectionOfVertical:
    def test_zonal_field_analytic(self):
        """For a purely zonal disturbing field, eta = 0 and xi = -dN/ds."""
        n_max = 4
        C = np.zeros((n_max + 1, n_max + 1))
        S = np.zeros((n_max + 1, n_max + 1))
        ref = np.zeros((n_max + 1, n_max + 1))
        _subtract_reference_field(ref, n_max)
        delta = 1.0e-7
        C[0, 0] = 1.0
        C[2, 0] = -ref[2, 0] + delta  # Reference C20 plus a zonal disturbance
        C[4, 0] = -ref[4, 0]  # Cancel J4 residual
        coef = EGMCoefficients(
            C=C, S=S, GM=3.986004415e14, R=6378136.3, n_max=n_max, model_name="TEST"
        )
        lat = np.radians(30.0)
        xi, eta = deflection_of_vertical(lat, 0.0, coefficients=coef)
        assert abs(eta) < 1e-4  # Zonal field: no east-west deflection

        # Analytic: N(phi) ~ k * P̄20(sin phi), xi = -dN/dphi / R
        gamma = normal_gravity_somigliana(lat, WGS84)
        k = coef.GM / (coef.R * gamma) * delta
        dP20_dphi = np.sqrt(5.0) * 3.0 * np.sin(lat) * np.cos(lat)
        xi_expected_arcsec = (-k * dP20_dphi / coef.R) * (3600.0 * 180.0 / np.pi)
        assert_allclose(xi, xi_expected_arcsec, rtol=5e-2)


class TestFundamentalArguments:
    """IERS 2010 Delaunay arguments vs independently coded polynomials."""

    REF_J2000 = {
        "l": 134.96340251,
        "lp": 357.52910918,
        "F": 93.27209062,
        "D": 297.85019547,
        "Om": 125.04455501,
    }
    POLY = {
        "l": (134.96340251, 1717915923.2178, 31.8792, 0.051635, -0.00024470),
        "lp": (357.52910918, 129596581.0481, -0.5532, 0.000136, -0.00001149),
        "F": (93.27209062, 1739527262.8478, -12.7512, -0.001037, 0.00000417),
        "D": (297.85019547, 1602961601.2090, -6.3706, 0.006593, -0.00003169),
        "Om": (125.04455501, -6962890.5431, 7.4722, 0.007702, -0.00005939),
    }

    def test_at_j2000(self):
        vals = fundamental_arguments(0.0)
        for ref, v in zip(self.REF_J2000.values(), vals):
            assert_allclose(np.degrees(v), ref, atol=1e-9)

    @pytest.mark.parametrize("T", [0.1, -0.2, 0.25])
    def test_at_other_epochs(self, T):
        vals = fundamental_arguments(T)
        for coeffs, v in zip(self.POLY.values(), vals):
            a0, a1, a2, a3, a4 = coeffs
            ref = (a0 + (a1 * T + a2 * T**2 + a3 * T**3 + a4 * T**4) / 3600.0) % 360.0
            assert_allclose(np.degrees(v), ref, atol=1e-8)

    def test_julian_centuries(self):
        assert_allclose(julian_centuries_j2000(51544.5), 0.0, atol=1e-12)
        assert_allclose(julian_centuries_j2000(51544.5 + 36525.0), 1.0, atol=1e-12)


class TestBodyPositions:
    def test_moon_distance_range(self):
        for mjd in np.linspace(57000, 60000, 25):
            r, lat, lon = moon_position_approximate(mjd)
            assert 350000e3 < r < 420000e3
            assert abs(lat) < np.radians(6.0)

    def test_sun_distance_range(self):
        for mjd in np.linspace(57000, 60000, 25):
            r, lat, lon = sun_position_approximate(mjd)
            assert 1.45e11 < r < 1.53e11
            assert lat == 0.0

    def test_positions_vs_astropy(self):
        """Ecliptic positions vs astropy ephemerides (if available)."""
        astropy_time = pytest.importorskip("astropy.time")
        from astropy.coordinates import GeocentricMeanEcliptic, get_body

        t = astropy_time.Time(58000.0, format="mjd")
        r, lat, lon = moon_position_approximate(58000.0)
        moon = get_body("moon", t).transform_to(GeocentricMeanEcliptic(equinox=t))
        assert_allclose(np.degrees(lon), moon.lon.deg % 360.0, atol=0.2)
        assert_allclose(np.degrees(lat), moon.lat.deg, atol=0.2)
        assert_allclose(r, moon.distance.to_value("m"), rtol=0.01)

        r, _, lon = sun_position_approximate(58000.0)
        sun = get_body("sun", t).transform_to(GeocentricMeanEcliptic(equinox=t))
        assert_allclose(np.degrees(lon), sun.lon.deg % 360.0, atol=0.05)
        assert_allclose(r, sun.distance.to_value("m"), rtol=0.01)


class TestSolidEarthTides:
    """Physical bounds and spectral content of the solid Earth tide."""

    @staticmethod
    def _hourly_series(lat_deg, lon_deg, days=30):
        lat, lon = np.radians(lat_deg), np.radians(lon_deg)
        mjds = 58000.0 + np.arange(days * 24) / 24.0
        rad = np.array(
            [solid_earth_tide_displacement(lat, lon, m).radial for m in mjds]
        )
        return mjds, rad

    def test_radial_bounds(self):
        _, rad = self._hourly_series(0.0, 0.0)
        assert np.all(np.abs(rad) < 0.6)
        assert np.max(np.abs(rad)) > 0.15  # Reaches decimeter level

    def test_m2_spectral_peak(self):
        """The dominant spectral peak must be semidiurnal (M2, ~12.42 h)."""
        _, rad = self._hourly_series(45.0, 7.0)
        freqs = np.fft.rfftfreq(len(rad), d=1.0 / 24.0)  # cycles/day
        amps = np.abs(np.fft.rfft(rad - rad.mean()))
        peak_freq = freqs[np.argmax(amps)]
        # M2 at 1.9323 cpd; 30 days of hourly data resolves it to ~0.033 cpd
        assert_allclose(peak_freq, 1.9323, atol=0.05)

    def test_diurnal_band_present(self):
        _, rad = self._hourly_series(45.0, 7.0)
        freqs = np.fft.rfftfreq(len(rad), d=1.0 / 24.0)
        amps = np.abs(np.fft.rfft(rad - rad.mean()))
        diurnal = amps[(freqs > 0.85) & (freqs < 1.1)].max()
        assert diurnal > 0.1 * amps.max()

    def test_smooth_over_a_day(self):
        """No discontinuities: consecutive 60 s samples move < 2 mm.

        The physical tidal rate peaks near 1.2 mm/min; a frame or phase
        discontinuity would produce centimeter-level jumps.
        """
        lat, lon = np.radians(45.0), np.radians(7.0)
        mjds = 58000.0 + np.arange(0, 1440) / 1440.0
        rad = np.array(
            [solid_earth_tide_displacement(lat, lon, m).radial for m in mjds]
        )
        assert np.max(np.abs(np.diff(rad))) < 2e-3

    def test_horizontal_smaller_than_radial(self):
        lat, lon = np.radians(45.0), np.radians(7.0)
        mjds = 58000.0 + np.arange(0, 3 * 24) / 24.0
        for m in mjds:
            d = solid_earth_tide_displacement(lat, lon, m)
            assert np.hypot(d.north, d.east) < 0.2

    def test_gravity_bounds(self):
        lat, lon = np.radians(0.0), np.radians(0.0)
        mjds = 58000.0 + np.arange(0, 30 * 24) / 24.0
        dg = np.array([solid_earth_tide_gravity(lat, lon, m).delta_g for m in mjds])
        assert np.all(np.abs(dg) < 3e-6)  # < 300 microGal always
        assert np.max(np.abs(dg)) > 5e-7  # Reaches tens of microGal

    def test_gravity_anticorrelated_with_uplift(self):
        """Gravity decreases when the surface is tidally uplifted."""
        lat, lon = np.radians(30.0), np.radians(50.0)
        mjds = 58000.0 + np.arange(0, 10 * 24) / 24.0
        rad = np.array(
            [solid_earth_tide_displacement(lat, lon, m).radial for m in mjds]
        )
        dg = np.array([solid_earth_tide_gravity(lat, lon, m).delta_g for m in mjds])
        corr = np.corrcoef(rad, dg)[0, 1]
        assert corr < -0.99

    def test_correction_is_negative_effect(self):
        lat, lon = np.radians(45.0), 0.0
        grav = solid_earth_tide_gravity(lat, lon, 58000.0)
        corr = tidal_gravity_correction(lat, lon, 58000.0)
        assert_allclose(corr, -grav.delta_g, rtol=1e-12)


class TestPoleTide:
    """Pole tide vs IERS Conventions (2010) Sec. 7.1.4."""

    def test_iers_reference_values(self):
        """Radial ~ -33 mm/arcsec * sin(2 theta), horizontals ~ 9 mm/arcsec."""
        lat, lon = np.radians(45.0), np.radians(30.0)
        xp, yp = 0.1, 0.05  # arcsec
        d = pole_tide_displacement(lat, lon, xp, yp)
        m1, m2 = xp, -yp  # IERS wobble convention
        theta = np.pi / 2 - lat
        # IERS mm-per-arcsec coefficients (33 and 9); the Love-number
        # parametrization used in the code gives 32.5 / 9.06.
        ur = -0.033 * np.sin(2 * theta) * (m1 * np.cos(lon) + m2 * np.sin(lon))
        un = 0.009 * np.cos(2 * theta) * (m1 * np.cos(lon) + m2 * np.sin(lon))
        ue = 0.009 * np.cos(theta) * (m1 * np.sin(lon) - m2 * np.cos(lon))
        assert_allclose(d.radial, ur, rtol=0.03)
        assert_allclose(d.north, un, atol=1e-4)
        assert_allclose(d.east, ue, rtol=0.03)

    def test_magnitude_bound(self):
        d = pole_tide_displacement(np.radians(45), 0.0, 0.3, 0.3)
        assert abs(d.radial) < 0.03
        assert abs(d.north) < 0.01
        assert abs(d.east) < 0.01

    def test_proportional_to_wobble(self):
        d1 = pole_tide_displacement(np.radians(40), 0.5, 0.1, 0.0)
        d2 = pole_tide_displacement(np.radians(40), 0.5, 0.2, 0.0)
        assert_allclose(d2.radial, 2 * d1.radial, rtol=1e-12)
        assert_allclose(d2.east, 2 * d1.east, rtol=1e-12)

    def test_mean_pole_subtraction(self):
        d = pole_tide_displacement(np.radians(40), 0.5, 0.1, 0.2, 0.1, 0.2)
        assert d.radial == 0.0 and d.north == 0.0 and d.east == 0.0


class TestAtmosphericLoading:
    def test_admittance_value(self):
        """+10 hPa must depress the surface ~3.5 mm (-0.35 mm/hPa)."""
        d = atmospheric_pressure_loading(np.radians(45), 0.0, 101325.0 + 1000.0)
        assert_allclose(d.radial, -3.5e-3, rtol=1e-6)

    def test_zero_at_reference(self):
        d = atmospheric_pressure_loading(np.radians(45), 0.0, 101325.0)
        assert d.radial == 0.0


class TestOceanTideLoading:
    def test_s2_constituent_analytic(self):
        """S2 has chi = 0, so the phase is exactly 4*pi*days - phase0."""
        amp = np.array([[0.01], [0.0], [0.0]])
        phase = np.array([[0.3], [0.0], [0.0]])
        mjd = 58123.375
        d = ocean_tide_loading_displacement(mjd, amp, phase, ("S2",))
        days = mjd - 51544.5
        expected = 0.01 * np.cos(2 * np.pi * 2.0 * days - 0.3)
        assert_allclose(d.radial, expected, rtol=1e-10)

    def test_bounded_by_amplitudes(self):
        rng = np.random.default_rng(7)
        amp = np.abs(rng.standard_normal((3, 8))) * 0.01
        phase = rng.uniform(0, 2 * np.pi, (3, 8))
        consts = ("M2", "S2", "N2", "K2", "K1", "O1", "P1", "Q1")
        for mjd in 58000.0 + np.arange(0, 2, 0.13):
            d = ocean_tide_loading_displacement(mjd, amp, phase, consts)
            assert abs(d.radial) <= amp[0].sum() + 1e-12
            assert abs(d.north) <= amp[1].sum() + 1e-12
            assert abs(d.east) <= amp[2].sum() + 1e-12


class TestTotalTidalDisplacement:
    def test_composition(self):
        lat, lon, mjd = np.radians(45.0), np.radians(7.0), 58000.25
        solid = solid_earth_tide_displacement(lat, lon, mjd)
        atm = atmospheric_pressure_loading(lat, lon, 102000.0)
        pole = pole_tide_displacement(lat, lon, 0.2, 0.1)
        total = total_tidal_displacement(
            lat, lon, mjd, pressure=102000.0, xp=0.2, yp=0.1
        )
        assert_allclose(total.radial, solid.radial + atm.radial + pole.radial)
        assert_allclose(total.north, solid.north + atm.north + pole.north)
        assert_allclose(total.east, solid.east + atm.east + pole.east)


class TestTidalConstants:
    def test_gravimetric_factor(self):
        assert_allclose(GRAVIMETRIC_FACTOR, 1.0 + LOVE_H2 - 1.5 * LOVE_K2, rtol=1e-12)
        assert 0.5 < LOVE_H2 < 0.7
        assert 0.2 < LOVE_K2 < 0.4
        assert 0.05 < SHIDA_L2 < 0.12


def test_full_norm_helper_self_check():
    """The scipy-based reference agrees with a log-space evaluation."""
    x = 0.37
    for n, m in [(3, 2), (8, 5)]:
        lognorm = 0.5 * (
            np.log((2 - (m == 0)) * (2 * n + 1)) + lgamma(n - m + 1) - lgamma(n + m + 1)
        )
        p = lpmv(m, n, x) * (-1.0) ** m
        ref = np.sign(p) * np.exp(np.log(abs(p)) + lognorm)
        assert_allclose(full_norm_legendre(n, m, x), ref, rtol=1e-12)
