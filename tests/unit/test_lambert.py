"""
Comprehensive tests for Lambert's problem solver.

Oracles: the Stumpff functions against their closed forms, Hohmann and
bi-elliptic transfers against vis-viva closed forms, Curtis Example 5.2
for the Lambert velocities, and the independent two-body propagator
``kepler_propagate_state`` (a different function) for every Lambert
solution: propagating the returned ``v1`` for the time of flight must reach
``r2`` and ``v2``.
"""

import numpy as np
import pytest

from pytcl.astronomical.lambert import (
    _stumpff_c2,
    _stumpff_c3,
    bi_elliptic_transfer,
    hohmann_transfer,
    lambert_izzo,
    lambert_universal,
    minimum_energy_transfer,
)
from pytcl.astronomical.orbital_mechanics import (
    GM_EARTH,
    StateVector,
    kepler_propagate_state,
)

MU = 3.986004418e5


def _assert_reaches_r2(solution, r1, r2, tof, mu=GM_EARTH):
    """Propagate (r1, v1) for tof with the Kepler propagator; expect (r2, v2)."""
    end = kepler_propagate_state(
        StateVector(r=np.asarray(r1, dtype=float), v=solution.v1), tof, mu
    )
    assert np.linalg.norm(end.r - r2) / np.linalg.norm(r2) < 1e-6
    assert np.linalg.norm(end.v - solution.v2) / np.linalg.norm(solution.v2) < 1e-6


class TestStumpffFunctions:
    """Stumpff c2, c3 against their closed forms and Taylor series.

    The 1e-12 closed-form checks cover |psi| >= 0.5 only. Just outside the
    Taylor branch (1e-6 < |psi| < 0.5) the implementation's 1 - cos cancellation
    costs up to ~6e-12 relative accuracy, so those points are tested at 1e-10.
    """

    @pytest.mark.parametrize("psi", [0.5, 2.0, 9.0])
    def test_positive_argument_closed_form(self, psi):
        z = np.sqrt(psi)
        assert _stumpff_c2(psi) == pytest.approx(
            (1 - np.cos(z)) / psi, rel=1e-12, abs=0
        )
        assert _stumpff_c3(psi) == pytest.approx(
            (z - np.sin(z)) / z**3, rel=1e-12, abs=0
        )

    @pytest.mark.parametrize("psi", [-0.5, -2.0, -9.0])
    def test_negative_argument_closed_form(self, psi):
        z = np.sqrt(-psi)
        assert _stumpff_c2(psi) == pytest.approx(
            (np.cosh(z) - 1) / (-psi), rel=1e-12, abs=0
        )
        assert _stumpff_c3(psi) == pytest.approx(
            (np.sinh(z) - z) / z**3, rel=1e-12, abs=0
        )

    @pytest.mark.parametrize("psi", [-1e-5, -1e-6, -1e-7, 0.0, 1e-7, 1e-6, 1e-5])
    def test_small_argument_taylor_series(self, psi):
        # Beyond |psi| = 1e-6 the implementation switches to the closed form,
        # whose 1 - cos cancellation costs about 6e-12 relative accuracy.
        c2 = 1 / 2 - psi / 24 + psi**2 / 720
        c3 = 1 / 6 - psi / 120 + psi**2 / 5040
        assert _stumpff_c2(psi) == pytest.approx(c2, rel=1e-10, abs=0)
        assert _stumpff_c3(psi) == pytest.approx(c3, rel=1e-10, abs=0)


class TestLambertUniversal:
    """Lambert's problem via universal variables."""

    R1 = np.array([5000.0, 10000.0, 2100.0])
    R2 = np.array([-14600.0, 2500.0, 7000.0])

    def test_curtis_example_5_2(self):
        """Curtis, Orbital Mechanics for Engineering Students, Example 5.2."""
        solution = lambert_universal(self.R1, self.R2, 3600.0)

        # Curtis lists four decimals, so agreement is bounded by the rounding
        assert solution.v1 == pytest.approx([-5.9925, 1.9254, 3.2456], rel=2e-5, abs=0)
        assert solution.v2 == pytest.approx(
            [-3.3125, -4.1966, -0.38529], rel=2e-5, abs=0
        )

    def test_orbit_elements_match_state(self):
        """a and e of the returned orbit follow from the returned (r1, v1)."""
        solution = lambert_universal(self.R1, self.R2, 3600.0)
        v1 = solution.v1
        r1_mag = np.linalg.norm(self.R1)

        a = 1 / (2 / r1_mag - np.dot(v1, v1) / MU)
        e_vec = (
            (np.dot(v1, v1) - MU / r1_mag) * self.R1 - np.dot(self.R1, v1) * v1
        ) / MU

        assert solution.a == pytest.approx(a, rel=1e-9, abs=0)
        assert solution.e == pytest.approx(np.linalg.norm(e_vec), rel=1e-9, abs=0)
        assert solution.tof == 3600.0

    @pytest.mark.parametrize(
        "r1,r2,tof,prograde",
        [
            ([5000.0, 10000.0, 2100.0], [14600.0, 2500.0, 7000.0], 3600.0, True),
            ([5000.0, 10000.0, 2100.0], [14600.0, 2500.0, 7000.0], 3600.0, False),
            ([5000.0, 10000.0, 2100.0], [-14600.0, 2500.0, 7000.0], 3600.0, False),
            ([6400.0, 0.0, 0.0], [6400.0, 6400.0, 0.0], 1800.0, True),
            ([7000.0, 0.0, 100.0], [0.0, 7500.0, 500.0], 2000.0, True),
        ],
    )
    @pytest.mark.parametrize("low_path", [True, False])
    def test_solution_reaches_r2_under_kepler_propagation(
        self, r1, r2, tof, prograde, low_path
    ):
        r1, r2 = np.array(r1), np.array(r2)
        solution = lambert_universal(r1, r2, tof, prograde=prograde, low_path=low_path)

        _assert_reaches_r2(solution, r1, r2, tof)

    @pytest.mark.parametrize("r_mag", [1000.0, 5000.0, 10000.0, 20000.0, 40000.0])
    def test_solution_reaches_r2_across_scales(self, r_mag):
        r1 = np.array([r_mag, 0.0, 0.0])
        r2 = np.array([r_mag * 0.9, r_mag * 0.436, 0.0])

        solution = lambert_universal(r1, r2, 1000.0)

        _assert_reaches_r2(solution, r1, r2, 1000.0)

    @pytest.mark.parametrize("tof", [600.0, 1200.0, 1800.0, 3600.0])
    def test_solution_reaches_r2_across_times_of_flight(self, tof):
        r1 = np.array([6378.0, 0.0, 0.0])
        r2 = np.array([6378.0, 6378.0, 0.0])

        solution = lambert_universal(r1, r2, tof)

        _assert_reaches_r2(solution, r1, r2, tof)

    def test_low_path_false_selects_the_other_transfer(self):
        low = lambert_universal(self.R1, self.R2, 3600.0, low_path=True)
        high = lambert_universal(self.R1, self.R2, 3600.0, low_path=False)

        assert np.linalg.norm(high.v1 - low.v1) > 1.0

    def test_custom_mu_changes_solution_consistently(self):
        mu = 3.986004418e5 * 1.5
        solution = lambert_universal(self.R1, self.R2, 3600.0, mu=mu)

        _assert_reaches_r2(solution, self.R1, self.R2, 3600.0, mu=mu)

    def test_identical_positions_rejected(self):
        r = np.array([6378.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="transfer angle is zero"):
            lambert_universal(r, r, 3600.0)

    def test_negative_time_of_flight_rejected(self):
        with pytest.raises(ValueError, match="did not converge"):
            lambert_universal(
                np.array([6378.0, 0.0, 0.0]), np.array([7000.0, 1000.0, 0.0]), -5.0
            )

    def test_mismatched_vector_lengths_rejected(self):
        with pytest.raises(ValueError, match="must be 3-vectors"):
            lambert_universal(np.array([1.0, 2.0]), self.R2, 100.0)

    @pytest.mark.xfail(
        strict=True,
        reason="lambert_universal returns zero velocities (e=1, a=3189 km) for "
        "antipodal positions instead of raising as lambert_izzo does",
    )
    def test_antipodal_positions_rejected(self):
        with pytest.raises(ValueError):
            lambert_universal(
                np.array([6378.0, 0.0, 0.0]), np.array([-6378.0, 0.0, 0.0]), 3600.0
            )

    @pytest.mark.xfail(
        strict=True,
        reason="lambert_universal returns inf/nan velocities for tof == 0 "
        "instead of raising",
    )
    def test_zero_time_of_flight_rejected(self):
        with pytest.raises(ValueError):
            lambert_universal(
                np.array([6378.0, 0.0, 0.0]), np.array([7000.0, 1000.0, 0.0]), 0.0
            )


class TestLambertIzzo:
    """Lambert's problem via Izzo's method."""

    R1 = np.array([5000.0, 10000.0, 2100.0])
    R2 = np.array([-14600.0, 2500.0, 7000.0])

    def test_curtis_example_5_2(self):
        solution = lambert_izzo(self.R1, self.R2, 3600.0)

        assert solution.v1 == pytest.approx([-5.9925, 1.9254, 3.2456], rel=2e-5, abs=0)
        assert solution.v2 == pytest.approx(
            [-3.3125, -4.1966, -0.38529], rel=2e-5, abs=0
        )

    @pytest.mark.parametrize("prograde", [True, False])
    def test_solution_reaches_r2_under_kepler_propagation(self, prograde):
        r1 = np.array([5000.0, 10000.0, 2100.0])
        r2 = np.array([14600.0, 2500.0, 7000.0])

        solution = lambert_izzo(r1, r2, 3600.0, prograde=prograde)

        _assert_reaches_r2(solution, r1, r2, 3600.0)

    def test_collinear_positions_rejected(self):
        with pytest.raises(ValueError, match="collinear positions"):
            lambert_izzo(
                np.array([6378.0, 0.0, 0.0]), np.array([-6378.0, 0.0, 0.0]), 3600.0
            )

    def test_agrees_with_universal_variables(self):
        universal = lambert_universal(self.R1, self.R2, 3600.0)
        izzo = lambert_izzo(self.R1, self.R2, 3600.0)

        assert izzo.v1 == pytest.approx(universal.v1, rel=1e-9, abs=0)
        assert izzo.v2 == pytest.approx(universal.v2, rel=1e-9, abs=0)


class TestMinimumEnergyTransfer:
    """Minimum-energy transfer: a = s/2 with s the semi-perimeter."""

    def test_minimum_energy_semi_major_axis_and_propagation(self):
        r1 = np.array([6678.0, 0.0, 0.0])
        r2 = np.array([0.0, 7000.0, 1000.0])

        tof_min, solution = minimum_energy_transfer(r1, r2)

        chord = np.linalg.norm(r2 - r1)
        s = (np.linalg.norm(r1) + np.linalg.norm(r2) + chord) / 2
        assert solution.a == pytest.approx(s / 2, rel=1e-6, abs=0)
        assert solution.tof == tof_min
        _assert_reaches_r2(solution, r1, r2, tof_min)

    def test_identical_positions_rejected(self):
        r = np.array([6678.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="transfer angle is zero"):
            minimum_energy_transfer(r, r)

    def test_scalar_radii_rejected(self):
        with pytest.raises(ValueError, match="zero dimension"):
            minimum_energy_transfer(6678, 7000)


class TestHohmannTransfer:
    """Hohmann transfer against the closed forms of Curtis section 6.3."""

    @pytest.mark.parametrize(
        "r1,r2", [(6678.0, 42164.0), (6578.0, 6878.0), (6678.0, 26560.0)]
    )
    def test_closed_form(self, r1, r2):
        dv1, dv2, tof = hohmann_transfer(r1, r2)

        assert dv1 == pytest.approx(
            np.sqrt(MU / r1) * (np.sqrt(2 * r2 / (r1 + r2)) - 1), rel=1e-10, abs=0
        )
        assert dv2 == pytest.approx(
            np.sqrt(MU / r2) * (1 - np.sqrt(2 * r1 / (r1 + r2))), rel=1e-10, abs=0
        )
        assert tof == pytest.approx(
            np.pi * np.sqrt(((r1 + r2) / 2) ** 3 / MU), rel=1e-12, abs=0
        )

    def test_equal_radii_cost_nothing(self):
        dv1, dv2, _ = hohmann_transfer(6678.0, 6678.0)

        assert abs(dv1) < 1e-12
        assert abs(dv2) < 1e-12


class TestBiEllipticTransfer:
    """Bi-elliptic transfer against vis-viva closed forms."""

    @staticmethod
    def _closed_form(r1, r2, rb):
        a1, a2 = (r1 + rb) / 2, (rb + r2) / 2

        def speed(r, a):
            return np.sqrt(MU) * np.sqrt(2 / r - 1 / a)

        dv1 = speed(r1, a1) - np.sqrt(MU / r1)
        dv2 = abs(speed(rb, a2) - speed(rb, a1))
        dv3 = abs(np.sqrt(MU / r2) - speed(r2, a2))
        tof = np.pi * (np.sqrt(a1**3 / MU) + np.sqrt(a2**3 / MU))
        return dv1, dv2, dv3, tof

    @pytest.mark.parametrize(
        "r1,r2,rb",
        [
            (6678.0, 42164.0, 100000.0),
            (6678.0, 20000.0, 42164.0),
            (7000.0, 30000.0, 60000.0),
        ],
    )
    def test_closed_form(self, r1, r2, rb):
        got = bi_elliptic_transfer(r1, r2, rb)

        assert got == pytest.approx(self._closed_form(r1, r2, rb), rel=1e-10, abs=0)

    def test_beats_hohmann_above_radius_ratio_15_58(self):
        """Curtis section 6.5: for r2/r1 > 15.58 a far enough apoapsis wins."""
        r1, r2 = 7000.0, 7000.0 * 20
        hohmann_total = sum(hohmann_transfer(r1, r2)[:2])
        bi_total = sum(bi_elliptic_transfer(r1, r2, r2 * 10)[:3])

        assert bi_total < hohmann_total

    def test_loses_to_hohmann_below_radius_ratio_11_94(self):
        """Curtis section 6.5: for r2/r1 < 11.94 Hohmann is always cheaper."""
        r1, r2 = 6678.0, 42164.0
        hohmann_total = sum(hohmann_transfer(r1, r2)[:2])
        bi_total = sum(bi_elliptic_transfer(r1, r2, 200000.0)[:3])

        assert bi_total > hohmann_total

    def test_intermediate_radius_below_orbits_rejected(self):
        with pytest.raises(ValueError, match="must be greater than both"):
            bi_elliptic_transfer(6678.0, 42164.0, 30000.0)
