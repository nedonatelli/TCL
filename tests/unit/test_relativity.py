"""Test suite for relativistic corrections module.

Every expected value is a closed form written out independently of the
module, a numerical quadrature of the underlying integral, or a published
figure with its source. The module's own output is never an oracle.
"""

import numpy as np
import pytest
from scipy.integrate import quad

from pytcl.astronomical.relativity import (
    AU,
    C_LIGHT,
    G_GRAV,
    GM_EARTH,
    GM_SUN,
    geodetic_precession,
    gravitational_time_dilation,
    lense_thirring_precession,
    post_newtonian_acceleration,
    proper_time_rate,
    relativistic_range_correction,
    schwarzschild_precession_per_orbit,
    schwarzschild_radius,
    shapiro_delay,
)

C2 = C_LIGHT**2
ARCSEC_PER_RAD = 206264.806


def _shapiro_by_quadrature(obs, src, body, gm):
    """Shapiro delay (2GM/c^3) * integral of ds/r along the straight path."""
    obs, src, body = (np.asarray(x, dtype=float) for x in (obs, src, body))
    length = np.linalg.norm(src - obs)

    def inv_r(s):
        point = obs + (src - obs) * (s / length)
        return 1.0 / np.linalg.norm(point - body)

    integral, _ = quad(inv_r, 0.0, length, epsabs=0, epsrel=1e-13, limit=200)
    return 2.0 * gm / C_LIGHT**3 * integral


class TestSchwarzchildRadius:
    """r_s = 2GM/c^2."""

    @pytest.mark.parametrize("mass", [5.972e24, 1.989e30, 1e24])
    def test_closed_form(self, mass):
        assert schwarzschild_radius(mass) == pytest.approx(
            2.0 * G_GRAV * mass / C2, rel=1e-12, abs=0
        )

    def test_earth_is_8_87_mm(self):
        """Standard textbook figure, 8.87 mm, quoted to three digits."""
        assert schwarzschild_radius(5.972e24) == pytest.approx(8.87e-3, rel=1e-3, abs=0)

    def test_sun_is_2_95_km(self):
        """Standard textbook figure, 2.95 km, quoted to three digits."""
        assert schwarzschild_radius(1.989e30) == pytest.approx(2.95e3, rel=2e-3, abs=0)

    def test_zero_mass(self):
        assert schwarzschild_radius(0.0) == 0.0

    def test_radius_is_linear_in_mass(self):
        assert schwarzschild_radius(2e24) == pytest.approx(
            2.0 * schwarzschild_radius(1e24), rel=1e-12, abs=0
        )


class TestGravitationalTimeDilation:
    """Schwarzschild g_00: sqrt(1 - 2GM/(r c^2))."""

    @pytest.mark.parametrize(
        "r,gm",
        [
            (1e20, GM_EARTH),
            (6.371e6, GM_EARTH),
            (1e7, GM_EARTH),
            (1e8, GM_EARTH),
            (1e7, GM_SUN),
            (26.56e6, GM_EARTH),
        ],
    )
    def test_closed_form(self, r, gm):
        assert gravitational_time_dilation(r, gm) == pytest.approx(
            np.sqrt(1.0 - 2.0 * gm / (C2 * r)), rel=1e-12, abs=0
        )

    def test_weak_field_limit(self):
        r = 1e12
        assert gravitational_time_dilation(r, GM_EARTH) == pytest.approx(
            1.0 - GM_EARTH / (C2 * r), rel=1e-12, abs=0
        )

    def test_gps_orbit_loses_14427_ns_per_day(self):
        """Earth-clock rate deficit at GPS radius (the module docstring's
        figure). 1 - sqrt(1 - 2x) with x ~ 1.7e-10 cancels to ~6 digits."""
        deficit_ns = (1 - gravitational_time_dilation(26.56e6, GM_EARTH)) * 86400e9
        assert deficit_ns == pytest.approx(
            GM_EARTH / (C2 * 26.56e6) * 86400e9, rel=1e-6, abs=0
        )

    def test_at_or_inside_schwarzschild_radius(self):
        r_s = schwarzschild_radius(5.972e24)
        with pytest.raises(ValueError, match="Schwarzschild radius"):
            gravitational_time_dilation(r_s - 1e-3, GM_EARTH)


class TestProperTimeRate:
    """dtau/dt = 1 - v^2/(2c^2) - GM/(c^2 r)."""

    @pytest.mark.parametrize(
        "v,r,gm",
        [
            (0.0, 1e20, GM_EARTH),
            (3870.0, 26.56e6, GM_EARTH),
            (0.1 * C_LIGHT, 1e7, GM_EARTH),
            (0.0, 6.371e6, GM_EARTH),
        ],
    )
    def test_closed_form(self, v, r, gm):
        assert proper_time_rate(v, r, gm) == pytest.approx(
            1.0 - v**2 / (2.0 * C2) - gm / (C2 * r), rel=1e-12, abs=0
        )

    def test_gps_net_fractional_frequency_offset(self):
        """GPS clock offset against the geoid: 4.4647e-10.

        Ashby, "Relativity in the Global Positioning System", Living Rev.
        Relativity 6 (2003) 1: for a circular orbit of a = 26561.75 km the
        satellite clock runs fast by 4.4647e-10 relative to a clock on the
        geoid, whose potential is Phi_0/c^2 = 6.969290134e-10. The five
        quoted digits bound the agreement.
        """
        a = 26561.75e3
        v = np.sqrt(GM_EARTH / a)
        satellite = proper_time_rate(v, a, GM_EARTH) - 1.0

        assert satellite + 6.969290134e-10 == pytest.approx(4.4647e-10, rel=2e-5, abs=0)

    @pytest.mark.filterwarnings("ignore::scipy.integrate.IntegrationWarning")
    def test_periodic_term_matches_is_gps_200(self):
        """Eccentric-orbit clock term F*e*sqrt(a)*sin(E), IS-GPS-200 20.3.3.3.3.1.

        F = -4.442807633e-10 s/sqrt(m) is the IS-GPS-200 constant (it is
        -2 sqrt(mu)/c^2 for mu = 3.986005e14; here GM_EARTH differs by 1.5e-7
        relative). The clock correction is the time integral of the periodic
        part of dtau/dt, integrated here over eccentric anomaly with
        dt = (1 - e cos E) dE / n.
        """
        e, a = 0.2, 26559.8e3
        n = np.sqrt(GM_EARTH / a**3)
        mean_offset = -1.5 * GM_EARTH / (a * C2)

        def periodic_rate(anomaly):
            r = a * (1.0 - e * np.cos(anomaly))
            v = np.sqrt(GM_EARTH * (2.0 / r - 1.0 / a))
            return (
                (proper_time_rate(v, r, GM_EARTH) - 1.0 - mean_offset)
                * (1.0 - e * np.cos(anomaly))
                / n
            )

        f_is_gps_200 = -4.442807633e-10
        for anomaly in (np.pi / 2, 2.0):
            accumulated, _ = quad(periodic_rate, 0.0, anomaly, epsabs=0, epsrel=1e-8)
            assert accumulated == pytest.approx(
                f_is_gps_200 * e * np.sqrt(a) * np.sin(anomaly), rel=1e-5, abs=0
            )


class TestShapiroDelay:
    """Shapiro delay (2GM/c^3) ln((r_o + r_s + r_os) / (r_o + r_s - r_os))."""

    BODY = np.zeros(3)

    @pytest.mark.parametrize(
        "obs,src",
        [
            ([2.0e11, 0.0, 0.0], [0.0, 2.0e11, 0.0]),
            ([1.496e11, 0.5e11, 0.0], [-1.496e11, 0.5e11, 0.0]),
            ([1.496e11, 0.1e11, 0.0], [-1.496e11, 0.1e11, 0.0]),
            ([1.0e11, 0.2e10, 0.0], [-1.0e11, 0.2e10, 0.0]),
            ([1.0e11, 0.2e10, 3e9], [-2.0e11, 4e9, -1e10]),
        ],
    )
    def test_matches_line_integral_of_inverse_distance(self, obs, src):
        delay = shapiro_delay(np.array(obs), np.array(src), self.BODY, GM_SUN)

        assert delay == pytest.approx(
            _shapiro_by_quadrature(obs, src, self.BODY, GM_SUN), rel=1e-9, abs=0
        )

    def test_closer_passage_gives_larger_delay(self):
        wide = shapiro_delay(
            np.array([1.496e11, 0.5e11, 0.0]),
            np.array([-1.496e11, 0.5e11, 0.0]),
            self.BODY,
            GM_SUN,
        )
        close = shapiro_delay(
            np.array([1.496e11, 0.1e11, 0.0]),
            np.array([-1.496e11, 0.1e11, 0.0]),
            self.BODY,
            GM_SUN,
        )

        assert close == pytest.approx(
            2.0
            * GM_SUN
            / C_LIGHT**3
            * np.log(
                (2 * 1.496e11 + 2 * np.hypot(1.496e11, 0.1e11))
                / (2 * np.hypot(1.496e11, 0.1e11) - 2 * 1.496e11)
            ),
            rel=1e-12,
            abs=0,
        )
        assert wide == pytest.approx(
            2.0
            * GM_SUN
            / C_LIGHT**3
            * np.log(
                (2 * 1.496e11 + 2 * np.hypot(1.496e11, 0.5e11))
                / (2 * np.hypot(1.496e11, 0.5e11) - 2 * 1.496e11)
            ),
            rel=1e-12,
            abs=0,
        )

    def test_far_body_matches_quadrature(self):
        obs = np.array([1.0, 0.0, 0.0])
        src = np.array([0.0, 1.0, 0.0])
        far = np.array([1000.0, 1000.0, 0.0])

        assert shapiro_delay(obs, src, far, GM_SUN) == pytest.approx(
            _shapiro_by_quadrature(obs, src, far, GM_SUN), rel=1e-9, abs=0
        )

    def test_source_between_body_and_observer(self):
        """Collinear with the body: the log argument is exactly 4."""
        obs = np.array([2.0e11, 0.0, 0.0])
        src = np.array([0.5e11, 0.0, 0.0])

        delay = shapiro_delay(obs, src, self.BODY, GM_SUN)

        assert delay == pytest.approx(
            2.0 * GM_SUN / C_LIGHT**3 * np.log(4.0), rel=1e-12, abs=0
        )


class TestSchwarzchildPrecession:
    """Perihelion advance 6 pi GM / (c^2 a (1 - e^2)) per orbit."""

    def test_mercury_matches_published_43_arcsec_per_century(self):
        """GR contribution to Mercury's perihelion advance, 42.98 arcsec/century
        (Will, "The Confrontation between General Relativity and Experiment",
        Living Rev. Relativity 17 (2014) 4), with a, e from Standish and the
        87.969 d sidereal period."""
        per_orbit = schwarzschild_precession_per_orbit(
            0.38709927 * AU, 0.20563593, GM_SUN
        )
        per_century = per_orbit * ARCSEC_PER_RAD * 36525.0 / 87.969

        assert per_century == pytest.approx(42.98, rel=5e-4, abs=0)

    @pytest.mark.parametrize("e", [0.0, 0.1, 0.5, 0.3])
    def test_closed_form(self, e):
        a = 1.0e7
        assert schwarzschild_precession_per_orbit(a, e, GM_EARTH) == pytest.approx(
            6.0 * np.pi * GM_EARTH / (C2 * a * (1.0 - e**2)), rel=1e-12, abs=0
        )

    @pytest.mark.parametrize("e", [1.5, -0.1, 1.0])
    def test_invalid_eccentricity(self, e):
        with pytest.raises(ValueError, match="must be in"):
            schwarzschild_precession_per_orbit(1e7, e, GM_SUN)


class TestPostNewtonianAcceleration:
    """1PN: GM/(c^2 r^3) [(4GM/r - v.v) r + 4 (r.v) v] added to -GM r / r^3."""

    @staticmethod
    def _expected_correction(r_vec, v_vec, gm=GM_EARTH):
        r = np.linalg.norm(r_vec)
        return (gm / (C2 * r**3)) * (
            (4.0 * gm / r - np.dot(v_vec, v_vec)) * r_vec
            + 4.0 * np.dot(r_vec, v_vec) * v_vec
        )

    @pytest.mark.parametrize(
        "r_vec,v_vec",
        [
            ([6.678e6, 0.0, 0.0], [0.0, 7.7e3, 0.0]),
            ([1e7, 0.0, 0.0], [0.0, 0.0, 0.0]),
            ([1e7, 0.0, 0.0], [1e3, 0.0, 0.0]),
            ([1e7, 0.0, 0.0], [0.0, 1e3, 0.0]),
            ([4.2e6, -3.1e6, 2.5e6], [-2.1e3, 4.4e3, 3.3e3]),
        ],
    )
    def test_correction_matches_formula(self, r_vec, v_vec):
        r_vec, v_vec = np.array(r_vec), np.array(v_vec)
        newtonian = -GM_EARTH / np.linalg.norm(r_vec) ** 3 * r_vec

        total = post_newtonian_acceleration(r_vec, v_vec, GM_EARTH)

        assert total == pytest.approx(
            newtonian + self._expected_correction(r_vec, v_vec), rel=1e-12, abs=0
        )
        # The correction itself, resolved through the 1e-9 cancellation
        assert total - newtonian == pytest.approx(
            self._expected_correction(r_vec, v_vec), rel=1e-6, abs=0
        )

    def test_circular_leo_ratio_is_three_gm_over_c2_r(self):
        """Circular orbit: (4GM/r - v^2)/c^2 = 3GM/(c^2 r)."""
        r = 6.678e6
        r_vec = np.array([r, 0.0, 0.0])
        v_vec = np.array([0.0, np.sqrt(GM_EARTH / r), 0.0])

        a_total = post_newtonian_acceleration(r_vec, v_vec, GM_EARTH)
        a_newt = -GM_EARTH / r**3 * r_vec
        ratio = np.linalg.norm(a_total - a_newt) / np.linalg.norm(a_newt)

        assert ratio == pytest.approx(3.0 * GM_EARTH / (C2 * r), rel=1e-6, abs=0)

    def test_circular_correction_points_radially_outward(self):
        r = 6.678e6
        r_vec = np.array([r, 0.0, 0.0])
        v_vec = np.array([0.0, np.sqrt(GM_EARTH / r), 0.0])

        correction = post_newtonian_acceleration(r_vec, v_vec, GM_EARTH) - (
            -GM_EARTH / r**3 * r_vec
        )

        assert correction[0] == pytest.approx(
            3.0 * GM_EARTH**2 / (C2 * r**3), rel=1e-6, abs=0
        )
        assert correction[1] == 0.0
        assert correction[2] == 0.0


class TestGeodeticPrecession:
    """de Sitter precession 3 pi GM / (c^2 a (1 - e^2)) per orbit."""

    @pytest.mark.parametrize("a,e", [(6.678e6, 0.0), (42.164e6, 0.0), (2.0e7, 0.3)])
    def test_closed_form_and_inclination_independence(self, a, e):
        expected = 3.0 * np.pi * GM_EARTH / (C2 * a * (1.0 - e**2))
        for inclination in (0.0, np.radians(51.6), np.pi / 2):
            assert geodetic_precession(a, e, inclination, GM_EARTH) == pytest.approx(
                expected, rel=1e-12, abs=0
            )

    def test_earth_sun_de_sitter_is_1_92_arcsec_per_century(self):
        """19.2 mas/yr geodetic precession of the Earth-Moon system
        (Will 2014, Living Rev. Relativity 17, sec. 3.6)."""
        per_orbit = geodetic_precession(AU, 0.0167, 0.0, GM_SUN)
        per_century = per_orbit * (36525.0 / 365.25636) * ARCSEC_PER_RAD

        assert per_century == pytest.approx(1.92, rel=3e-3, abs=0)

    def test_gravity_probe_b_geodetic_rate(self):
        """Gravity Probe B predicted 6606 mas/yr (Everitt et al., Phys. Rev.
        Lett. 106, 221101 (2011)) for a polar orbit at 642 km; the semi-major
        axis a = 7027 km is rounded, which bounds agreement at ~0.1%."""
        a = 7027e3
        period = 2.0 * np.pi * np.sqrt(a**3 / GM_EARTH)
        per_year = geodetic_precession(a, 0.0, np.pi / 2, GM_EARTH) * (
            365.25 * 86400.0 / period
        )

        assert per_year * ARCSEC_PER_RAD * 1e3 == pytest.approx(6606.0, rel=1e-3, abs=0)


class TestLenseThirringPrecession:
    """Nodal rate 2 G J / (c^2 a^3 (1 - e^2)^(3/2))."""

    J_EARTH = 5.86e33  # Earth's spin angular momentum (kg m^2/s)

    @pytest.mark.parametrize(
        "a,e,i_deg",
        [(12.27e6, 0.0045, 109.9), (6.678e6, 0.0, 51.6), (42.164e6, 0.0, 51.6)],
    )
    def test_closed_form(self, a, e, i_deg):
        rate = lense_thirring_precession(
            a, e, np.radians(i_deg), self.J_EARTH, GM_EARTH
        )

        assert rate == pytest.approx(
            2.0 * G_GRAV * self.J_EARTH / (C2 * a**3 * (1.0 - e**2) ** 1.5),
            rel=1e-12,
            abs=0,
        )

    def test_lageos_nodal_drag_is_about_31_mas_per_year(self):
        """Ciufolini and Pavlis, Nature 431, 958 (2004): the LAGEOS nodal
        Lense-Thirring drift is quoted as 31 mas/yr (two digits)."""
        rate = lense_thirring_precession(
            12.27e6, 0.0045, np.radians(109.9), self.J_EARTH, GM_EARTH
        )

        assert rate * 86400 * 365.25 * ARCSEC_PER_RAD * 1e3 == pytest.approx(
            31.0, rel=0.02, abs=0
        )

    def test_no_rotation_no_precession(self):
        rate = lense_thirring_precession(6.678e6, 0.0, np.radians(51.6), 0.0, GM_EARTH)
        assert rate == 0.0


class TestRelativisticsRangeCorrection:
    """Shapiro range correction (2GM/c^2) ln((r1 + r2 + rho)/(r1 + r2 - rho))."""

    @pytest.mark.parametrize(
        "r1,r2", [(6.371e6, 3.84e8), (6.371e6, 26.561e6), (7.0e6, 4.2164e7)]
    )
    def test_radial_path_is_log_of_radius_ratio(self, r1, r2):
        """Along a radial path the integral of ds/r is ln(r2/r1) exactly."""
        correction = relativistic_range_correction(r1, r2, r2 - r1, GM_EARTH)

        assert correction == pytest.approx(
            2.0 * GM_EARTH / C2 * np.log(r2 / r1), rel=1e-12, abs=0
        )

    def test_gps_zenith_range_correction(self):
        """2GM/c^2 = 8.870 mm times ln(26.561/6.371) gives 12.66 mm."""
        correction = relativistic_range_correction(
            6.371e6, 26.561e6, 26.561e6 - 6.371e6, GM_EARTH
        )

        assert correction == pytest.approx(
            8.870056e-3 * np.log(26.561 / 6.371), rel=1e-6, abs=0
        )

    def test_oblique_path_matches_quadrature(self):
        r1, r2 = 6.371e6, 26.561e6
        rho = np.sqrt(r2**2 - r1**2)
        obs = np.array([r1, 0.0, 0.0])
        src = np.array([r1, rho, 0.0])
        assert np.linalg.norm(src) == pytest.approx(np.hypot(r1, rho), rel=1e-12, abs=0)

        correction = relativistic_range_correction(
            r1, np.linalg.norm(src), rho, GM_EARTH
        )

        assert correction == pytest.approx(
            C_LIGHT * _shapiro_by_quadrature(obs, src, np.zeros(3), GM_EARTH),
            rel=1e-9,
            abs=0,
        )

    def test_consistent_with_shapiro_delay(self):
        obs = np.array([6.371e6, 0.0, 0.0])
        src = np.array([0.0, 26.561e6, 0.0])
        body = np.zeros(3)

        delay = shapiro_delay(obs, src, body, GM_EARTH)
        correction = relativistic_range_correction(
            np.linalg.norm(obs),
            np.linalg.norm(src),
            np.linalg.norm(obs - src),
            GM_EARTH,
        )

        assert correction == pytest.approx(C_LIGHT * delay, rel=1e-12, abs=0)
