"""The gh-25 items that changed behavior rather than only documentation.

That issue is mostly about writing down quantified approximation limits, and
documentation has no test. Four of its bullets turned out to be defects rather
than approximations, and those are covered here:

- ``mot_metrics.num_fragmentations`` was initialized and never incremented, so
  it always reported 0;
- ``mercator`` used the global ``WGS84_E2`` in its scale factor whatever ``e``
  the caller passed, and ``transverse_mercator`` took its semi-minor axis from
  the global ``WGS84_B`` whatever ``a`` and ``e2`` were given -- each silently
  mixing two ellipsoids;
- the MHT track score carried an overall factor of 0.5 that its own
  missed-detection branch did not, so hits and misses accumulated on different
  scales;
- the SRIF initialization recipe in the docstrings held only for diagonal
  ``P0``.

The remaining bullets are genuine bounded approximations whose numbers were
verified before being written into the docstrings -- the coordinated-turn
Jacobian against numerical differentiation, ``gast`` against ``gmst`` -- and
those checks live here too, because a quantified claim that nothing re-measures
is a claim that will drift.
"""

import numpy as np
import pytest

from pytcl.coordinate_systems.projections.projections import (
    WGS84_A,
    WGS84_B,
    WGS84_E2,
    mercator,
    transverse_mercator,
)
from pytcl.performance_evaluation.track_metrics import mot_metrics


class TestFragmentationsAreCounted:
    """The field existed, was documented, and was always 0."""

    @staticmethod
    def _single_track(coverage):
        """One ground-truth track; ``coverage`` says which frames see it."""
        truth = [[np.array([0.0, 0.0])] for _ in coverage]
        estimates = [[np.array([0.1, 0.0])] if seen else [] for seen in coverage]
        return truth, estimates

    def test_an_uninterrupted_track_has_no_fragmentations(self):
        result = mot_metrics(*self._single_track([True] * 5))
        assert result.num_fragmentations == 0

    def test_one_interruption_is_one_fragmentation(self):
        result = mot_metrics(*self._single_track([True, True, False, True, True]))
        assert result.num_fragmentations == 1

    def test_each_resumption_counts_once(self):
        result = mot_metrics(*self._single_track([True, False, True, False, True]))
        assert result.num_fragmentations == 2

    def test_a_track_picked_up_late_is_not_a_fragmentation(self):
        """The first time coverage starts is not an interruption of anything.

        This is why the counter consults "has this been tracked before" rather
        than only the previous frame.
        """
        result = mot_metrics(*self._single_track([False, False, True, True]))
        assert result.num_fragmentations == 0

    def test_a_track_that_is_never_covered_has_none(self):
        result = mot_metrics(*self._single_track([False] * 4))
        assert result.num_fragmentations == 0

    def test_a_track_lost_at_the_end_is_not_a_fragmentation(self):
        """Losing a track and never regaining it is a miss, not a break."""
        result = mot_metrics(*self._single_track([True, True, False, False]))
        assert result.num_fragmentations == 0


class TestProjectionsHonorTheirEllipsoidArguments:
    """Both took ellipsoid parameters and then used global constants."""

    LAT, LON = np.radians(45.0), np.radians(10.0)

    def test_mercator_defaults_are_unchanged(self):
        """The fix must be invisible to every existing caller.

        WGS84 is the only ellipsoid used in-repo, so if the defaults moved,
        this would be a silent behavior change rather than a bug fix.
        """
        expected = np.sqrt(1 - WGS84_E2 * np.sin(self.LAT) ** 2) / np.cos(self.LAT)
        assert mercator(self.LAT, self.LON).scale == pytest.approx(expected, rel=1e-15)

    def test_mercator_scale_follows_the_supplied_eccentricity(self):
        """On a sphere the scale factor is exactly ``sec(lat)``.

        The old code returned the WGS84 value here regardless of ``e``, so this
        is the assertion that separates the two.
        """
        spherical = mercator(self.LAT, self.LON, e=0.0)
        assert spherical.scale == pytest.approx(1.0 / np.cos(self.LAT), rel=1e-14)

    @pytest.mark.parametrize("e", [0.0, 0.05, 0.0818191908426, 0.15])
    def test_mercator_scale_varies_with_eccentricity(self, e):
        expected = np.sqrt(1 - e**2 * np.sin(self.LAT) ** 2) / np.cos(self.LAT)
        assert mercator(self.LAT, self.LON, e=e).scale == pytest.approx(
            expected, rel=1e-14
        )

    def test_transverse_mercator_defaults_are_unchanged(self):
        """The derived semi-minor axis must reproduce the global constant."""
        derived = WGS84_A * np.sqrt(1 - WGS84_E2)
        assert derived == pytest.approx(WGS84_B, abs=1e-6)

    def test_transverse_mercator_follows_a_custom_ellipsoid(self):
        """A different ``a`` must change the result through the flattening too.

        Taking ``b`` from the global constant while ``a`` came from the caller
        described an ellipsoid that was neither.
        """
        default = transverse_mercator(self.LAT, self.LON, lon0=self.LON)
        smaller = transverse_mercator(
            self.LAT, self.LON, lon0=self.LON, a=WGS84_A * 0.99, e2=WGS84_E2
        )
        assert smaller.y != pytest.approx(default.y, rel=1e-9)

    def test_a_spherical_transverse_mercator_is_self_consistent(self):
        """With ``e2=0`` the third flattening is zero, so the meridian arc is
        simply ``a * lat``. Under the old code ``b`` stayed ellipsoidal and this
        identity did not hold."""
        radius = 6_371_000.0
        result = transverse_mercator(
            self.LAT, self.LON, lon0=self.LON, a=radius, e2=0.0
        )
        assert result.y == pytest.approx(radius * self.LAT, rel=1e-9)


class TestCoordinatedTurnJacobianClaim:
    """The docstring now states this is not the Jacobian, with numbers."""

    @staticmethod
    def _propagate(state, dt):
        x, y, psi, v, w = state
        if abs(w) < 1e-12:
            return np.array(
                [x + v * dt * np.cos(psi), y + v * dt * np.sin(psi), psi, v, w]
            )
        return np.array(
            [
                x + (v / w) * (np.sin(psi + w * dt) - np.sin(psi)),
                y - (v / w) * (np.cos(psi + w * dt) - np.cos(psi)),
                psi + w * dt,
                v,
                w,
            ]
        )

    def _true_jacobian(self, state, dt):
        jac = np.zeros((5, 5))
        step = 1e-6
        for k in range(5):
            up, down = state.copy(), state.copy()
            up[k] += step
            down[k] -= step
            jac[:, k] = (self._propagate(up, dt) - self._propagate(down, dt)) / (
                2 * step
            )
        return jac

    def test_the_turn_rate_column_disagrees_with_the_true_jacobian(self):
        """Pins the claim the docstring makes.

        If someone later makes the matrix heading-dependent and correct, this
        fails and the docstring caveat has to come out with it -- which is the
        point of testing a documented limitation.
        """
        from pytcl.dynamic_models.discrete_time.coordinated_turn import (
            f_coord_turn_polar,
        )

        dt, speed, turn_rate = 1.0, 100.0, 0.1
        state = np.array([0.0, 0.0, 0.0, speed, turn_rate])

        returned = f_coord_turn_polar(T=dt, omega=turn_rate, speed=speed)
        truth = self._true_jacobian(state, dt)

        assert truth[0, 4] == pytest.approx(-3.33, abs=0.01)
        assert truth[1, 4] == pytest.approx(49.88, abs=0.01)
        assert returned[0, 4] == pytest.approx(-49.96, abs=0.01)
        assert returned[1, 4] == pytest.approx(-1.67, abs=0.01)

    def test_the_matrix_does_not_depend_on_heading(self):
        """Which is why it cannot be the Jacobian: the true one does."""
        import inspect

        from pytcl.dynamic_models.discrete_time.coordinated_turn import (
            f_coord_turn_polar,
        )

        assert "psi" not in inspect.signature(f_coord_turn_polar).parameters
        assert "heading" not in inspect.signature(f_coord_turn_polar).parameters

        dt, speed, turn_rate = 1.0, 100.0, 0.1
        at_zero = self._true_jacobian(np.array([0.0, 0.0, 0.0, speed, turn_rate]), dt)
        at_ninety = self._true_jacobian(
            np.array([0.0, 0.0, np.pi / 2, speed, turn_rate]), dt
        )
        assert not np.allclose(at_zero[:2, 4], at_ninety[:2, 4])


class TestSiderealTimeDefault:
    """``gast`` with its default arguments returns GMST."""

    def test_the_default_equals_gmst_exactly(self):
        from pytcl.astronomical.time_systems import gast, gmst

        for offset in (0.0, 1234.5, -5000.25):
            jd = 2451545.0 + offset
            assert float(np.atleast_1d(gast(jd))[0]) == pytest.approx(
                float(np.atleast_1d(gmst(jd))[0]), rel=1e-15
            )

    def test_supplying_nutation_separates_them(self):
        """So the caveat is about the default, not about the function."""
        from pytcl.astronomical.time_systems import gast, gmst

        jd = 2451545.0 + 1234.5
        apparent = float(np.atleast_1d(gast(jd, dpsi=1e-5, eps=0.409))[0])
        mean = float(np.atleast_1d(gmst(jd))[0])
        assert apparent != pytest.approx(mean, abs=1e-12)


class TestSrifInitializationRecipe:
    """``inv(cholesky(P0)).T`` holds only for diagonal ``P0``."""

    def test_the_documented_recipe_is_correct_for_a_correlated_prior(self):
        """The recipe the docstrings now give, on the case that broke the old
        one. Doctests used only diagonal P0, so both forms passed."""
        P0 = np.array([[4.0, 1.5], [1.5, 9.0]])

        R0 = np.linalg.cholesky(np.linalg.inv(P0)).T

        np.testing.assert_allclose(R0.T @ R0, np.linalg.inv(P0), atol=1e-12)

    def test_the_old_recipe_fails_for_a_correlated_prior(self):
        """Guard the guard: if the two agreed, the fix would be pointless."""
        P0 = np.array([[4.0, 1.5], [1.5, 9.0]])

        old = np.linalg.inv(np.linalg.cholesky(P0)).T

        assert not np.allclose(old.T @ old, np.linalg.inv(P0))

    def test_both_recipes_agree_when_the_prior_is_diagonal(self):
        """Which is exactly why the doctests never caught it."""
        P0 = np.diag([4.0, 9.0])

        old = np.linalg.inv(np.linalg.cholesky(P0)).T
        new = np.linalg.cholesky(np.linalg.inv(P0)).T

        np.testing.assert_allclose(old.T @ old, new.T @ new, atol=1e-12)
