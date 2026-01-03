"""Unit tests for special orbit cases (parabolic and hyperbolic).

Tests cover:
- Parabolic orbit anomaly conversions
- Hyperbolic orbit anomaly conversions
- Orbit type classification
- Radius and velocity computations
- Energy-based calculations
- Escape trajectories
"""

import numpy as np
import pytest

from pytcl.astronomical.special_orbits import (
    OrbitType,
    classify_orbit,
    eccentricity_vector,
    escape_velocity_at_radius,
    hyperbolic_anomaly_to_true_anomaly,
    hyperbolic_asymptote_angle,
    hyperbolic_deflection_angle,
    hyperbolic_excess_velocity,
    mean_to_parabolic_anomaly,
    mean_to_true_anomaly_parabolic,
    parabolic_anomaly_to_true_anomaly,
    radius_parabolic,
    semi_major_axis_from_energy,
    true_anomaly_to_hyperbolic_anomaly,
    true_anomaly_to_parabolic_anomaly,
    velocity_parabolic,
)


class TestOrbitClassification:
    """Test orbit type classification."""

    def test_classify_circular(self):
        """Test circular orbit classification."""
        orbit_type = classify_orbit(0.0)
        assert orbit_type == OrbitType.CIRCULAR

    def test_classify_elliptical(self):
        """Test elliptical orbit classification."""
        for e in [0.1, 0.5, 0.9]:
            orbit_type = classify_orbit(e)
            assert orbit_type == OrbitType.ELLIPTICAL

    def test_classify_parabolic(self):
        """Test parabolic orbit classification."""
        orbit_type = classify_orbit(1.0)
        assert orbit_type == OrbitType.PARABOLIC

    def test_classify_parabolic_near(self):
        """Test near-parabolic orbit (within tolerance)."""
        orbit_type = classify_orbit(1.0 + 1e-10, tol=1e-9)
        assert orbit_type == OrbitType.PARABOLIC

    def test_classify_hyperbolic(self):
        """Test hyperbolic orbit classification."""
        for e in [1.1, 1.5, 2.0]:
            orbit_type = classify_orbit(e)
            assert orbit_type == OrbitType.HYPERBOLIC

    def test_classify_invalid(self):
        """Test invalid eccentricity."""
        with pytest.raises(ValueError):
            classify_orbit(-0.1)

        with pytest.raises(ValueError):
            classify_orbit(np.nan)


class TestParabolicAnomalies:
    """Test parabolic anomaly conversions."""

    def test_parabolic_anomaly_round_trip(self):
        """Test round-trip conversion for parabolic anomaly."""
        D = 0.5
        nu = parabolic_anomaly_to_true_anomaly(D)
        D_back = true_anomaly_to_parabolic_anomaly(nu)
        assert np.isclose(D, D_back)

    def test_mean_to_parabolic_anomaly(self):
        """Test mean to parabolic anomaly conversion."""
        M = 0.5
        D = mean_to_parabolic_anomaly(M)

        # Verify Kepler's equation: M = D + (1/3)*D^3
        M_check = D + (1.0 / 3.0) * D**3
        assert np.isclose(M, M_check)

    def test_mean_to_true_anomaly_parabolic(self):
        """Test mean to true anomaly for parabolic orbit."""
        M = 0.3
        nu = mean_to_true_anomaly_parabolic(M)

        # True anomaly should be reasonable
        assert -np.pi < nu < np.pi

    def test_parabolic_anomaly_zero(self):
        """Test parabolic anomaly at zero mean anomaly."""
        D = mean_to_parabolic_anomaly(0.0)
        assert np.isclose(D, 0.0)

    def test_parabolic_anomaly_convergence(self):
        """Test parabolic anomaly convergence for various M values."""
        for M in np.linspace(-2, 2, 5):
            D = mean_to_parabolic_anomaly(M)
            M_check = D + (1.0 / 3.0) * D**3
            assert np.isclose(M, M_check, atol=1e-12)


class TestParabolicOrbitRadius:
    """Test parabolic orbit radius calculations."""

    def test_radius_parabolic_periapsis(self):
        """Test radius at periapsis (nu=0)."""
        rp = 6678.0  # Earth surface + 300 km
        r = radius_parabolic(rp, 0.0)

        # At periapsis, r = 2*rp / (1 + cos(0)) = 2*rp / 2 = rp
        assert np.isclose(r, rp)

    def test_radius_parabolic_90deg(self):
        """Test radius at 90 degrees true anomaly."""
        rp = 6678.0
        r = radius_parabolic(rp, np.pi / 2)

        # r = 2*rp / (1 + cos(pi/2)) = 2*rp / 1 = 2*rp
        assert np.isclose(r, 2 * rp)

    def test_radius_parabolic_increasing(self):
        """Test that radius increases with true anomaly."""
        rp = 6678.0
        r1 = radius_parabolic(rp, np.pi / 6)
        r2 = radius_parabolic(rp, np.pi / 3)
        r3 = radius_parabolic(rp, np.pi / 2)

        assert r1 < r2 < r3

    def test_radius_parabolic_undefined_at_escape(self):
        """Test that radius is undefined near escape angle."""
        rp = 6678.0

        # At nu = pi (opposite side), denominator = 1 + cos(pi) = 0
        with pytest.raises(ValueError):
            radius_parabolic(rp, np.pi - 1e-10)


class TestParabolicOrbitVelocity:
    """Test parabolic orbit velocity calculations."""

    def test_velocity_parabolic_periapsis(self):
        """Test velocity at periapsis."""
        mu = 398600.4418  # Earth GM
        rp = 6678.0
        v = velocity_parabolic(mu, rp, 0.0)

        # For parabolic: v = sqrt(2*mu/r)
        # At periapsis: r = rp
        v_expected = np.sqrt(2 * mu / rp)
        assert np.isclose(v, v_expected)

    def test_velocity_parabolic_decreases(self):
        """Test that velocity decreases as orbit recedes."""
        mu = 398600.4418
        rp = 6678.0

        v1 = velocity_parabolic(mu, rp, 0.0)
        v2 = velocity_parabolic(mu, rp, np.pi / 4)
        v3 = velocity_parabolic(mu, rp, np.pi / 2)

        assert v1 > v2 > v3

    def test_velocity_parabolic_energy(self):
        """Test parabolic orbit specific energy is zero."""
        mu = 398600.4418
        rp = 6678.0

        # Specific energy for parabolic: epsilon = -mu / (2*a) = 0
        # This means: v^2/2 - mu/r = 0, or v = sqrt(2*mu/r)

        nu_test = np.pi / 3
        r = radius_parabolic(rp, nu_test)
        v = velocity_parabolic(mu, rp, nu_test)

        # Specific energy: epsilon = v^2/2 - mu/r
        epsilon = 0.5 * v**2 - mu / r
        assert np.isclose(epsilon, 0.0, atol=1e-6)


class TestHyperbolicAnomalies:
    """Test hyperbolic anomaly conversions."""

    def test_hyperbolic_anomaly_round_trip(self):
        """Test round-trip conversion for hyperbolic anomaly."""
        e = 1.5
        H = 0.3
        nu = hyperbolic_anomaly_to_true_anomaly(H, e)
        H_back = true_anomaly_to_hyperbolic_anomaly(nu, e)

        assert np.isclose(H, H_back)

    def test_hyperbolic_anomaly_zero(self):
        """Test hyperbolic anomaly at zero (periapsis)."""
        e = 1.5
        H = 0.0
        nu = hyperbolic_anomaly_to_true_anomaly(H, e)

        # At H=0: nu = 2*arctan(sqrt((e+1)/(e-1))*tanh(0)) = 0
        assert np.isclose(nu, 0.0)

    def test_hyperbolic_asymptote_angle(self):
        """Test hyperbolic asymptote angle."""
        e = 2.0

        nu_inf = hyperbolic_asymptote_angle(e)

        # For e=2: cos(nu_inf) = -1/e = -0.5
        # nu_inf = arccos(-0.5) = 2*pi/3
        assert np.isclose(nu_inf, 2 * np.pi / 3)

    def test_hyperbolic_deflection_angle(self):
        """Test deflection angle for hyperbolic orbit."""
        e = 2.0
        delta = hyperbolic_deflection_angle(e)

        # delta = pi - 2*arccos(-1/e)
        # For e=2: delta = pi - 2*arccos(-0.5) = pi - 4*pi/3 = -pi/3
        # The angle is the absolute deflection from asymptotic direction
        assert np.abs(delta) < np.pi
        assert np.abs(delta) > 0


class TestHyperbolicEnergy:
    """Test hyperbolic orbit energy calculations."""

    def test_excess_velocity(self):
        """Test hyperbolic excess velocity."""
        mu = 398600.4418
        a = -10000.0  # Negative semi-major axis for hyperbolic

        v_inf = hyperbolic_excess_velocity(mu, a)

        # v_inf = sqrt(-mu/a)
        v_inf_expected = np.sqrt(-mu / a)
        assert np.isclose(v_inf, v_inf_expected)

    def test_excess_velocity_invalid(self):
        """Test excess velocity with invalid semi-major axis."""
        mu = 398600.4418

        # Positive a is invalid for hyperbolic
        with pytest.raises(ValueError):
            hyperbolic_excess_velocity(mu, 10000.0)

    def test_semi_major_axis_from_energy_elliptical(self):
        """Test semi-major axis from energy for elliptical orbit."""
        mu = 398600.4418
        epsilon = -5.0  # Negative energy for bound orbit

        a = semi_major_axis_from_energy(mu, epsilon)

        # a = -mu / (2*epsilon)
        a_expected = -mu / (2 * epsilon)
        assert np.isclose(a, a_expected)
        assert a > 0  # Elliptical: a > 0

    def test_semi_major_axis_from_energy_hyperbolic(self):
        """Test semi-major axis from energy for hyperbolic orbit."""
        mu = 398600.4418
        epsilon = 5.0  # Positive energy for unbound orbit

        a = semi_major_axis_from_energy(mu, epsilon)

        # a = -mu / (2*epsilon)
        a_expected = -mu / (2 * epsilon)
        assert np.isclose(a, a_expected)
        assert a < 0  # Hyperbolic: a < 0

    def test_semi_major_axis_parabolic_invalid(self):
        """Test semi-major axis from energy for parabolic (invalid)."""
        mu = 398600.4418

        with pytest.raises(ValueError):
            semi_major_axis_from_energy(mu, 0.0)


class TestEscapeVelocity:
    """Test escape velocity calculations."""

    def test_escape_velocity_earth_surface(self):
        """Test escape velocity at Earth's surface."""
        mu = 398600.4418
        r = 6371.0  # Earth radius

        v_esc = escape_velocity_at_radius(mu, r)

        # v_esc = sqrt(2*mu/r)
        v_esc_expected = np.sqrt(2 * mu / r)
        assert np.isclose(v_esc, v_esc_expected)

        # Earth escape velocity ≈ 11.2 km/s
        assert np.isclose(v_esc, 11.2, atol=0.1)

    def test_escape_velocity_decreases(self):
        """Test that escape velocity decreases with altitude."""
        mu = 398600.4418

        v_surface = escape_velocity_at_radius(mu, 6371.0)
        v_orbit = escape_velocity_at_radius(mu, 6678.0)
        v_far = escape_velocity_at_radius(mu, 10000.0)

        assert v_surface > v_orbit > v_far


class TestEccentricityVector:
    """Test eccentricity vector computation."""

    def test_eccentricity_vector_circular(self):
        """Test eccentricity vector for circular orbit."""
        mu = 398600.4418
        r_orbit = 6678.0

        # Circular orbit: v is perpendicular to r, |v| = sqrt(mu/r)
        r = np.array([r_orbit, 0.0, 0.0])
        v = np.array([0.0, np.sqrt(mu / r_orbit), 0.0])

        e_vec = eccentricity_vector(r, v, mu)

        # For circular orbit, eccentricity vector should be near zero
        assert np.linalg.norm(e_vec) < 1e-10

    def test_eccentricity_vector_elliptical(self):
        """Test eccentricity vector magnitude for elliptical orbit."""
        mu = 398600.4418
        a = 7000.0
        e_target = 0.3

        # Periapsis: r = a(1-e)
        r_p = a * (1 - e_target)
        r = np.array([r_p, 0.0, 0.0])

        # At periapsis: v_p = sqrt(mu*(1+e)/(a*(1-e)))
        v_p = np.sqrt(mu * (1 + e_target) / (a * (1 - e_target)))
        v = np.array([0.0, v_p, 0.0])

        e_vec = eccentricity_vector(r, v, mu)
        e_computed = np.linalg.norm(e_vec)

        # Should match target eccentricity
        assert np.isclose(e_computed, e_target, atol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
