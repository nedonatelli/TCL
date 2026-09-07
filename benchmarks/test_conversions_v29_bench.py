"""
Benchmarks for the v2.9.0 measurement conversions with covariances.

All tests are @pytest.mark.full (nightly/on-main only): the file is new
and unmeasured against benchmark-light.yml's ~2-minute-per-file budget,
and none of these entries has an SLO yet -- per the gate-calibration
doctrine (see .benchmarks/slos.json derivations), SLOs are set from
observed CI history, so these run SLO-less first to accumulate it.
"""

import numpy as np
import pytest

from pytcl.coordinate_systems import (
    camera_coords2uv_cubature,
    monostat_ruv2cart_taylor,
    ruv2ruv_cubature,
    uv2spher_ang_cubature,
)

_rng = np.random.default_rng(99)

UV_BATCH = np.clip(0.3 * _rng.standard_normal((2, 100)), -0.6, 0.6)
S_R_UV = np.linalg.cholesky(np.array([[2e-3, 5e-4], [5e-4, 3e-3]]))

RUV_BATCH = np.vstack(
    [
        1e4 * (5.0 + _rng.random(100)),
        np.clip(0.3 * _rng.standard_normal(100), -0.6, 0.6),
        np.clip(0.3 * _rng.standard_normal(100), -0.6, 0.6),
    ]
)
S_RUV = np.linalg.cholesky(np.diag([50.0**2, 1e-6, 1e-6]))
RX2 = 1e4 * np.array([1.0, 2.0, 0.0])

CAM_BATCH = 1e-2 * _rng.standard_normal((2, 100))
CAM_A = np.diag([35e-3, 35e-3, 1.0])
S_R_CAM = np.diag([1e-4, 1e-4])

R_RUV = np.diag([1.0, 1e-6, 25e-6])


class TestCovarianceConversionBenchmarks:
    """One radar scan's worth of measurements through each converter."""

    @pytest.mark.full
    def test_uv2spher_ang_cubature_100_meas(self, benchmark):
        result = benchmark(uv2spher_ang_cubature, UV_BATCH, S_R_UV)
        assert result.z.shape == (2, 100)

    @pytest.mark.full
    def test_ruv2ruv_cubature_100_meas(self, benchmark):
        result = benchmark(
            ruv2ruv_cubature,
            RUV_BATCH,
            S_RUV,
            False,
            None,
            RX2,
            None,
            None,
            None,
            None,
            0,
        )
        assert result.z.shape == (3, 100)

    @pytest.mark.full
    def test_camera_coords2uv_cubature_100_meas(self, benchmark):
        result = benchmark(camera_coords2uv_cubature, CAM_BATCH, S_R_CAM, CAM_A)
        assert result.z.shape == (2, 100)

    @pytest.mark.full
    def test_monostat_ruv2cart_taylor_cm3_100_meas(self, benchmark):
        result = benchmark(
            monostat_ruv2cart_taylor, RUV_BATCH, R_RUV, True, None, None, 2
        )
        assert result.z.shape == (3, 100)
