"""
Benchmarks for the v2.10.0 features.

All tests are @pytest.mark.full (nightly/on-main only): the file is new
and unmeasured against benchmark-light.yml's ~2-minute-per-file budget,
and none of these entries has an SLO yet -- per the gate-calibration
doctrine (see .benchmarks/slos.json derivations), SLOs are set from
observed CI history, so these run SLO-less first to accumulate it.

The two NRLMSISE-00 entries are the performance rationale for the
compiled extension made measurable: the compiled single-point
evaluation guards the extension's speed, and the pure-Python fallback
entry keeps the compiled/fallback ratio visible in the history.
"""

import importlib

import numpy as np
import pytest

from pytcl.atmosphere.nrlmsise00 import (
    nrlmsise00,
    nrlmsise00_alt_for_pressure,
    uses_compiled_backend,
)
from pytcl.coordinate_systems.hessians import calc_spher_hessian
from pytcl.coordinate_systems.jacobians import (
    calc_spher_rr_jacob,
    spher_ang_gradient,
)
from pytcl.magnetism.coordinates import trace2earth_mag_apex
from pytcl.performance_evaluation.mospa import mmospa_approx
from pytcl.scheduling import schedule_weighted_intervals

_rng = np.random.default_rng(210)

STATES_100 = np.vstack(
    [
        1e4 * (1.0 + _rng.random((3, 100))),
        50.0 * _rng.standard_normal((3, 100)),
    ]
)
POINTS_1000 = 1e4 * (1.0 + _rng.random((3, 1000)))
L_RX = np.array([4e3, -6e3, 12.0, -4.0, 2.0, 6.0])
L_TX = np.array([-12e3, 8e3, 5e3, 5.0, 3.0, -2.0])

MOSPA_X = 10.0 * _rng.standard_normal((4, 5, 20))
MOSPA_W = np.full(20, 1.0 / 20)

N_IV = 1000
_starts = _rng.uniform(0, 1000, N_IV)
INTERVALS_1000 = np.vstack([_starts, _starts + _rng.uniform(0.1, 50, N_IV)])
WEIGHTS_1000 = _rng.uniform(0.1, 5.0, N_IV)


class TestNrlmsise00Benchmarks:
    """The compiled extension's speed, and the fallback ratio."""

    @pytest.mark.full
    def test_nrlmsise00_compiled_single(self, benchmark):
        assert uses_compiled_backend()
        out = benchmark(nrlmsise00, 172, 29000.0, 400.0, 60.0, -70.0, 16.0)
        assert out.t[0] > 1000.0

    @pytest.mark.full
    def test_nrlmsise00_fallback_single(self, benchmark):
        mod = importlib.import_module("pytcl.atmosphere.nrlmsise00")

        def run_fallback():
            old = mod._c_ext
            mod._c_ext = None
            try:
                return nrlmsise00(172, 29000.0, 400.0, 60.0, -70.0, 16.0)
            finally:
                mod._c_ext = old

        out = benchmark(run_fallback)
        assert out.t[0] > 1000.0

    @pytest.mark.full
    def test_nrlmsise00_alt_for_pressure(self, benchmark):
        alt, _ = benchmark(
            nrlmsise00_alt_for_pressure, 172, 29000.0, 1000.0, 60.0, -70.0, 16.0
        )
        assert 15.0 < alt < 35.0


class TestMeasurementDerivativeBenchmarks:
    """One scan's worth of Jacobians/Hessians for a bistatic radar."""

    @pytest.mark.full
    def test_calc_spher_rr_jacob_100_states(self, benchmark):
        def run():
            return [
                calc_spher_rr_jacob(STATES_100[:, k], 0, False, L_TX, L_RX)
                for k in range(100)
            ]

        out = benchmark(run)
        assert len(out) == 100 and out[0].shape == (4, 6)

    @pytest.mark.full
    def test_spher_ang_gradient_batch_1000(self, benchmark):
        out = benchmark(spher_ang_gradient, POINTS_1000, 0, L_RX[:3])
        assert out.shape == (2, 3, 1000)

    @pytest.mark.full
    def test_calc_spher_hessian_100_points(self, benchmark):
        def run():
            return [
                calc_spher_hessian(STATES_100[:3, k], 0, False, L_TX[:3], L_RX[:3])
                for k in range(100)
            ]

        out = benchmark(run)
        assert len(out) == 100 and out[0].shape == (3, 3, 3)


class TestMagneticApexBenchmarks:
    @pytest.mark.full
    def test_trace2earth_mag_apex_single(self, benchmark):
        apex, _ = benchmark(trace2earth_mag_apex, np.array([6.4e6, 1e5, 2e6]))
        assert np.all(np.isfinite(apex))


class TestMospaSchedulingBenchmarks:
    @pytest.mark.full
    def test_mmospa_approx_20hyp_5tar(self, benchmark):
        est, _ = benchmark(mmospa_approx, MOSPA_X, MOSPA_W, 3)
        assert est.shape == (4, 5)

    @pytest.mark.full
    def test_schedule_weighted_intervals_1000(self, benchmark):
        weight, jobs = benchmark(
            schedule_weighted_intervals, INTERVALS_1000, WEIGHTS_1000
        )
        assert weight > 0 and jobs.size > 0
