"""Tests for astronomical module (orbital mechanics, Lambert, reference frames)."""

import numpy as np
from numpy.testing import assert_allclose

from pytcl.astronomical import (  # Orbital mechanics; Lambert; Reference frames; Constants
    JD_J2000,
    OrbitalElements,
    StateVector,
    apoapsis_radius,
    bi_elliptic_transfer,
    circular_velocity,
    eccentric_to_mean_anomaly,
    eccentric_to_true_anomaly,
    ecef_to_eci,
    eci_to_ecef,
    ecliptic_to_equatorial,
    equatorial_to_ecliptic,
    escape_velocity,
    gcrf_to_itrf,
    gmst_iau82,
    hohmann_transfer,
    itrf_to_gcrf,
    julian_centuries_j2000,
    kepler_propagate,
    kepler_propagate_state,
    lambert_universal,
    mean_motion,
    mean_obliquity_iau80,
    mean_to_eccentric_anomaly,
    mean_to_hyperbolic_anomaly,
    mean_to_true_anomaly,
    nutation_matrix,
    orbital_elements_to_state,
    orbital_period,
    periapsis_radius,
    precession_matrix_iau76,
    sidereal_rotation_matrix,
    state_to_orbital_elements,
    true_to_eccentric_anomaly,
    true_to_mean_anomaly,
    vis_viva,
)


class TestKeplerEquation:
    """Tests for Kepler's equation solvers."""

    def test_mean_to_eccentric_circular(self):
        """For circular orbit, E = M."""
        M = np.pi / 4
        E = mean_to_eccentric_anomaly(M, 0.0)
        assert_allclose(E, M, rtol=1e-10)

    def test_mean_to_eccentric_low_ecc(self):
        """Test with low eccentricity."""
        M = np.pi / 3
        e = 0.1
        E = mean_to_eccentric_anomaly(M, e)
        # Verify by computing M back
        M_check = E - e * np.sin(E)
        assert_allclose(M_check, M, rtol=1e-10)

    def test_mean_to_eccentric_high_ecc(self):
        """Test with high eccentricity."""
        M = np.pi / 2
        e = 0.9
        E = mean_to_eccentric_anomaly(M, e)
        M_check = E - e * np.sin(E)
        assert_allclose(M_check, M, rtol=1e-10)

    def test_eccentric_to_mean_roundtrip(self):
        """Test E -> M -> E roundtrip."""
        E_orig = 1.5
        e = 0.5
        M = eccentric_to_mean_anomaly(E_orig, e)
        E_back = mean_to_eccentric_anomaly(M, e)
        assert_allclose(E_back, E_orig, rtol=1e-10)

    def test_mean_to_hyperbolic(self):
        """Test hyperbolic Kepler's equation."""
        M = 2.0
        e = 1.5
        H = mean_to_hyperbolic_anomaly(M, e)
        M_check = e * np.sinh(H) - H
        assert_allclose(M_check, M, rtol=1e-8)


class TestAnomalyConversions:
    """Tests for anomaly conversions."""

    def test_eccentric_true_roundtrip(self):
        """Test E <-> nu roundtrip."""
        E_orig = np.pi / 3
        e = 0.3
        nu = eccentric_to_true_anomaly(E_orig, e)
        E_back = true_to_eccentric_anomaly(nu, e)
        assert_allclose(E_back, E_orig, rtol=1e-10)

    def test_mean_true_roundtrip(self):
        """Test M <-> nu roundtrip."""
        M_orig = np.pi / 4
        e = 0.2
        nu = mean_to_true_anomaly(M_orig, e)
        M_back = true_to_mean_anomaly(nu, e)
        assert_allclose(M_back, M_orig, rtol=1e-10)

    def test_circular_orbit_anomalies_equal(self):
        """For circular orbit, M = E = nu."""
        M = np.pi / 6
        e = 0.0
        E = mean_to_eccentric_anomaly(M, e)
        nu = eccentric_to_true_anomaly(E, e)
        assert_allclose(E, M, rtol=1e-10)
        assert_allclose(nu, M, rtol=1e-10)


class TestOrbitalElementConversions:
    """Tests for orbital element conversions."""

    def test_elements_to_state_circular_equatorial(self):
        """Test circular equatorial orbit."""
        a = 7000  # km
        elements = OrbitalElements(a=a, e=0.0, i=0.0, raan=0.0, omega=0.0, nu=0.0)
        state = orbital_elements_to_state(elements)

        # At nu=0, should be at periapsis along x-axis
        assert_allclose(state.r[0], a, rtol=1e-10)
        assert_allclose(state.r[1], 0, atol=1e-10)
        assert_allclose(state.r[2], 0, atol=1e-10)

        # Velocity should be along y-axis
        v_expected = circular_velocity(a)
        assert_allclose(state.v[0], 0, atol=1e-10)
        assert_allclose(state.v[1], v_expected, rtol=1e-10)
        assert_allclose(state.v[2], 0, atol=1e-10)

    def test_state_to_elements_circular(self):
        """Test state to elements for circular orbit."""
        r = np.array([7000.0, 0.0, 0.0])
        v_circ = circular_velocity(7000.0)
        v = np.array([0.0, v_circ, 0.0])
        state = StateVector(r=r, v=v)

        elements = state_to_orbital_elements(state)

        assert_allclose(elements.a, 7000.0, rtol=1e-6)
        assert_allclose(elements.e, 0.0, atol=1e-6)
        assert_allclose(elements.i, 0.0, atol=1e-6)

    def test_roundtrip_elements_state(self):
        """Test elements -> state -> elements roundtrip."""
        elements_orig = OrbitalElements(
            a=8000, e=0.2, i=0.5, raan=0.3, omega=0.8, nu=1.0
        )
        state = orbital_elements_to_state(elements_orig)
        elements_back = state_to_orbital_elements(state)

        assert_allclose(elements_back.a, elements_orig.a, rtol=1e-8)
        assert_allclose(elements_back.e, elements_orig.e, rtol=1e-8)
        assert_allclose(elements_back.i, elements_orig.i, rtol=1e-8)
        assert_allclose(elements_back.raan, elements_orig.raan, rtol=1e-8)
        assert_allclose(elements_back.omega, elements_orig.omega, rtol=1e-8)
        assert_allclose(elements_back.nu, elements_orig.nu, rtol=1e-8)


class TestKeplerPropagation:
    """Tests for Kepler propagation."""

    def test_propagate_one_period(self):
        """After one orbital period, should return to start."""
        elements = OrbitalElements(a=7000, e=0.1, i=0.5, raan=0.2, omega=0.3, nu=0.5)
        T = orbital_period(elements.a)

        new_elements = kepler_propagate(elements, T)

        # True anomaly should be back to original (mod 2*pi)
        assert_allclose(new_elements.nu, elements.nu, rtol=1e-6)
        # Other elements unchanged
        assert_allclose(new_elements.a, elements.a)
        assert_allclose(new_elements.e, elements.e)
        assert_allclose(new_elements.i, elements.i)

    def test_propagate_half_period(self):
        """After half period, should be at opposite anomaly."""
        elements = OrbitalElements(a=7000, e=0.0, i=0.0, raan=0.0, omega=0.0, nu=0.0)
        T = orbital_period(elements.a)

        new_elements = kepler_propagate(elements, T / 2)

        # For circular orbit starting at nu=0, should be at nu=pi
        assert_allclose(new_elements.nu, np.pi, rtol=1e-6)

    def test_propagate_state_consistency(self):
        """State propagation should match element propagation."""
        elements = OrbitalElements(a=7500, e=0.15, i=0.4, raan=0.1, omega=0.2, nu=0.3)
        state = orbital_elements_to_state(elements)

        dt = 1800  # 30 minutes

        new_elements = kepler_propagate(elements, dt)
        new_state = kepler_propagate_state(state, dt)

        expected_state = orbital_elements_to_state(new_elements)

        assert_allclose(new_state.r, expected_state.r, rtol=1e-6)
        assert_allclose(new_state.v, expected_state.v, rtol=1e-6)


class TestOrbitalQuantities:
    """Tests for orbital quantity calculations."""

    def test_orbital_period_leo(self):
        """LEO orbital period should be ~90 minutes."""
        a = 6678  # ~300 km altitude
        T = orbital_period(a)
        assert 5400 < T < 5500  # ~90 minutes in seconds

    def test_orbital_period_geo(self):
        """GEO orbital period should be ~24 hours."""
        a = 42164  # GEO
        T = orbital_period(a)
        assert 86100 < T < 86500  # ~24 hours

    def test_mean_motion(self):
        """Mean motion should be 2*pi/T."""
        a = 7000
        n = mean_motion(a)
        T = orbital_period(a)
        assert_allclose(n, 2 * np.pi / T, rtol=1e-10)

    def test_vis_viva_circular(self):
        """Vis-viva for circular orbit equals circular velocity."""
        r = 7000
        a = r  # circular
        v = vis_viva(r, a)
        v_circ = circular_velocity(r)
        assert_allclose(v, v_circ, rtol=1e-10)

    def test_periapsis_apoapsis(self):
        """Test periapsis and apoapsis radius."""
        a = 8000
        e = 0.2
        r_p = periapsis_radius(a, e)
        r_a = apoapsis_radius(a, e)

        assert_allclose(r_p, a * (1 - e))
        assert_allclose(r_a, a * (1 + e))
        assert_allclose((r_p + r_a) / 2, a)

    def test_escape_velocity(self):
        """Escape velocity should be sqrt(2) times circular velocity."""
        r = 7000
        v_esc = escape_velocity(r)
        v_circ = circular_velocity(r)
        assert_allclose(v_esc, v_circ * np.sqrt(2), rtol=1e-10)


class TestLambert:
    """Tests for Lambert problem solver."""

    def test_lambert_basic(self):
        """Test basic Lambert solution."""
        r1 = np.array([5000.0, 10000.0, 2100.0])
        r2 = np.array([-14600.0, 2500.0, 7000.0])
        tof = 3600  # 1 hour

        sol = lambert_universal(r1, r2, tof)

        # Verify velocities produce correct end states
        assert sol.v1 is not None
        assert sol.v2 is not None
        assert len(sol.v1) == 3
        assert len(sol.v2) == 3

    def test_lambert_short_transfer(self):
        """Test Lambert with short transfer."""
        # Simple coplanar transfer
        r1 = np.array([7000.0, 0.0, 0.0])
        r2 = np.array([0.0, 8000.0, 0.0])
        tof = 1500

        sol = lambert_universal(r1, r2, tof)

        # Initial velocity should have positive y component
        assert sol.v1[1] > 0

    def test_hohmann_transfer(self):
        """Test Hohmann transfer calculation."""
        r1 = 6678  # LEO
        r2 = 42164  # GEO

        dv1, dv2, tof = hohmann_transfer(r1, r2)

        # Total delta-v for LEO to GEO should be ~3.9 km/s
        total_dv = dv1 + dv2
        assert 3.8 < total_dv < 4.1

        # Transfer time should be about 5 hours
        assert 17000 < tof < 20000

    def test_bi_elliptic_transfer(self):
        """Test bi-elliptic transfer."""
        r1 = 7000
        r2 = 70000
        r_int = 100000

        dv1, dv2, dv3, tof = bi_elliptic_transfer(r1, r2, r_int)

        # All delta-v's should be positive
        assert dv1 > 0
        assert dv2 > 0
        assert dv3 > 0

        # Transfer time should be longer than Hohmann
        _, _, tof_hohmann = hohmann_transfer(r1, r2)
        assert tof > tof_hohmann


class TestReferenceFrames:
    """Tests for reference frame transformations."""

    def test_julian_centuries_j2000(self):
        """Test Julian centuries calculation."""
        T = julian_centuries_j2000(JD_J2000)
        assert_allclose(T, 0.0)

        # One century later
        T = julian_centuries_j2000(JD_J2000 + 36525)
        assert_allclose(T, 1.0)

    def test_precession_matrix_j2000(self):
        """Precession matrix at J2000 should be identity."""
        P = precession_matrix_iau76(JD_J2000)
        assert_allclose(P, np.eye(3), atol=1e-10)

    def test_precession_matrix_orthogonal(self):
        """Precession matrix should be orthogonal."""
        jd = JD_J2000 + 3650  # 10 years later
        P = precession_matrix_iau76(jd)

        # P * P.T should be identity
        assert_allclose(P @ P.T, np.eye(3), atol=1e-10)
        # det(P) should be 1
        assert_allclose(np.linalg.det(P), 1.0, atol=1e-10)

    def test_nutation_matrix_orthogonal(self):
        """Nutation matrix should be orthogonal."""
        jd = JD_J2000 + 1000
        N = nutation_matrix(jd)

        assert_allclose(N @ N.T, np.eye(3), atol=1e-10)
        assert_allclose(np.linalg.det(N), 1.0, atol=1e-10)

    def test_mean_obliquity_j2000(self):
        """Mean obliquity at J2000 should be ~23.4 degrees."""
        eps0 = mean_obliquity_iau80(JD_J2000)
        eps0_deg = np.degrees(eps0)
        assert 23.4 < eps0_deg < 23.5

    def test_gmst_varies_with_time(self):
        """GMST should increase with time."""
        jd1 = JD_J2000
        jd2 = JD_J2000 + 1  # One day later

        gmst1 = gmst_iau82(jd1)
        gmst2 = gmst_iau82(jd2)

        # After one day, GMST should have advanced
        # (not exactly 2*pi due to sidereal vs solar day)
        assert gmst2 != gmst1

    def test_sidereal_rotation_matrix_orthogonal(self):
        """Sidereal rotation matrix should be orthogonal."""
        theta = np.pi / 4
        R = sidereal_rotation_matrix(theta)

        assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
        assert_allclose(np.linalg.det(R), 1.0, atol=1e-10)

    def test_gcrf_itrf_roundtrip(self):
        """Test GCRF <-> ITRF roundtrip."""
        r_gcrf = np.array([5102.5096, 6123.01152, 6378.1363])
        jd_ut1 = JD_J2000 + 100
        jd_tt = jd_ut1 + 32.184 / 86400  # Approximate TT

        r_itrf = gcrf_to_itrf(r_gcrf, jd_ut1, jd_tt)
        r_gcrf_back = itrf_to_gcrf(r_itrf, jd_ut1, jd_tt)

        assert_allclose(r_gcrf_back, r_gcrf, rtol=1e-10)

    def test_eci_ecef_roundtrip(self):
        """Test simple ECI <-> ECEF roundtrip."""
        r_eci = np.array([7000.0, 0.0, 0.0])
        gmst = np.pi / 4

        r_ecef = eci_to_ecef(r_eci, gmst)
        r_eci_back = ecef_to_eci(r_ecef, gmst)

        assert_allclose(r_eci_back, r_eci, atol=1e-10)

    def test_ecliptic_equatorial_roundtrip(self):
        """Test ecliptic <-> equatorial roundtrip."""
        r_ecl = np.array([1.0, 0.5, 0.2])
        obliquity = np.radians(23.4)

        r_eq = ecliptic_to_equatorial(r_ecl, obliquity)
        r_ecl_back = equatorial_to_ecliptic(r_eq, obliquity)

        assert_allclose(r_ecl_back, r_ecl, rtol=1e-10)

    def test_ecliptic_equatorial_x_unchanged(self):
        """X component should be unchanged by ecliptic/equatorial rotation."""
        r_ecl = np.array([5.0, 3.0, 2.0])
        obliquity = np.radians(23.4)

        r_eq = ecliptic_to_equatorial(r_ecl, obliquity)

        assert_allclose(r_eq[0], r_ecl[0], rtol=1e-10)

    def test_gcrf_pef_roundtrip(self):
        """Test GCRF <-> PEF roundtrip."""
        from pytcl.astronomical.reference_frames import gcrf_to_pef, pef_to_gcrf

        r_gcrf = np.array([5102.5096, 6123.01152, 6378.1363])
        jd_ut1 = JD_J2000 + 100
        jd_tt = jd_ut1 + 32.184 / 86400

        r_pef = gcrf_to_pef(r_gcrf, jd_ut1, jd_tt)
        r_gcrf_back = pef_to_gcrf(r_pef, jd_ut1, jd_tt)

        assert_allclose(r_gcrf_back, r_gcrf, rtol=1e-10)

    def test_pef_magnitude_preserved(self):
        """Magnitude should be preserved in GCRF -> PEF transformation."""
        from pytcl.astronomical.reference_frames import gcrf_to_pef

        r_gcrf = np.array([5102.5096, 6123.01152, 6378.1363])
        jd_ut1 = JD_J2000 + 100
        jd_tt = jd_ut1 + 32.184 / 86400

        r_pef = gcrf_to_pef(r_gcrf, jd_ut1, jd_tt)

        assert_allclose(np.linalg.norm(r_pef), np.linalg.norm(r_gcrf), rtol=1e-10)


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_propagate_and_transform(self):
        """Propagate orbit and transform to Earth-fixed frame."""
        # Initial orbit
        elements = OrbitalElements(a=7000, e=0.01, i=0.5, raan=0.2, omega=0.1, nu=0.0)
        state = orbital_elements_to_state(elements)

        # Propagate
        dt = 1800
        new_state = kepler_propagate_state(state, dt)

        # Transform to ECEF
        gmst = np.pi / 6
        r_ecef = eci_to_ecef(new_state.r, gmst)

        # Should have same magnitude
        assert_allclose(np.linalg.norm(r_ecef), np.linalg.norm(new_state.r), rtol=1e-10)

    def test_leo_to_geo_transfer(self):
        """Test realistic LEO to GEO transfer scenario."""
        # LEO parking orbit
        r_leo = 6678  # 300 km altitude

        # GEO target
        r_geo = 42164

        # Hohmann transfer
        dv1, dv2, tof = hohmann_transfer(r_leo, r_geo)

        # Create initial state at LEO
        v_leo = circular_velocity(r_leo)
        r1 = np.array([r_leo, 0.0, 0.0])
        v1 = np.array([0.0, v_leo + dv1, 0.0])  # After first burn
        state1 = StateVector(r=r1, v=v1)

        # Propagate through transfer
        state2 = kepler_propagate_state(state1, tof)

        # Should arrive at GEO radius
        r2_mag = np.linalg.norm(state2.r)
        assert_allclose(r2_mag, r_geo, rtol=0.01)
