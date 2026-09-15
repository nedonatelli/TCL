"""Reference/property tests for `pytcl.astronomical.lambert`.

Covers the `minimum_energy_transfer` beta-branch defect found in the
v2.11.1 audit: for a transfer angle above pi, beta was not sign-flipped,
so both the minimum-energy time of flight and the semi-major axis of the
orbit `lambert_universal` converges to at that time of flight were wrong.
"""

import numpy as np
import pytest

from pytcl.astronomical.lambert import minimum_energy_transfer
from pytcl.astronomical.orbital_mechanics import GM_EARTH


class TestMinimumEnergySemiMajorAxis:
    """PROPERTY: a_min = s/2 identically, by definition of the
    minimum-energy ellipse -- independent of which side of the transfer
    angle's pi boundary the geometry falls on."""

    @pytest.mark.parametrize("angle_deg", [60.0, 120.0, 200.0, 300.0])
    @pytest.mark.parametrize("prograde", [True, False])
    def test_semi_major_axis_is_half_the_semiperimeter(self, angle_deg, prograde):
        r1 = np.array([7000.0, 0.0, 0.0])
        th = np.radians(angle_deg)
        r2 = 8000.0 * np.array([np.cos(th), np.sin(th), 0.0])

        c = np.linalg.norm(r2 - r1)
        s = 0.5 * (np.linalg.norm(r1) + np.linalg.norm(r2) + c)

        _, solution = minimum_energy_transfer(r1, r2, GM_EARTH, prograde=prograde)

        assert solution.a == pytest.approx(s / 2.0, rel=1e-9)


class TestMinimumEnergyTimeOfFlightRegression:
    """REFERENCE: pins the audit's measured geometry so a future
    regression reproduces the exact numbers that were wrong.

    r1 = [7000, 0, 0] km, r2 = [0, 8000, 0] km, prograde=False puts the
    transfer angle at 270 deg (long way, dnu > pi). Before the fix this
    returned tof=2471.66 s and a=6425.35 km; the correct minimum-energy
    values are tof=2632.78 s and a = s/2 = 6407.54 km.
    """

    def test_long_way_transfer_matches_audit_values(self):
        r1 = np.array([7000.0, 0.0, 0.0])
        r2 = np.array([0.0, 8000.0, 0.0])

        tof_min, solution = minimum_energy_transfer(r1, r2, GM_EARTH, prograde=False)

        assert tof_min == pytest.approx(2632.7758244120946, rel=1e-9)
        assert solution.a == pytest.approx(6407.536453183662, rel=1e-9)

    def test_short_way_transfer_is_unaffected(self):
        r1 = np.array([7000.0, 0.0, 0.0])
        r2 = np.array([0.0, 8000.0, 0.0])

        tof_min, solution = minimum_energy_transfer(r1, r2, GM_EARTH, prograde=True)

        assert tof_min == pytest.approx(2471.6583651129085, rel=1e-9)
        assert solution.a == pytest.approx(6407.536453183662, rel=1e-9)
