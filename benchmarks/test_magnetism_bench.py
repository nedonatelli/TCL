"""
Benchmarks for the shared Schmidt Legendre tables and the field syntheses
built on them (v2.12.0 Task 6.4).

``miss`` cases clear the table cache inside the timed call, so they time
the recursion itself; ``hit`` cases repeat one colatitude, as a longitude
sweep or a repeated potential evaluation does.
"""

import importlib

import numpy as np
import pytest

from pytcl.magnetism import _schmidt

emm = importlib.import_module("pytcl.magnetism.emm")
wmm = importlib.import_module("pytcl.magnetism.wmm")

THETA = 0.7


@pytest.mark.benchmark
@pytest.mark.light
@pytest.mark.parametrize("n_max", [13, 133, 790])
def test_schmidt_table_miss(benchmark, n_max):
    cos_t, sin_t = np.cos(THETA), np.sin(THETA)

    def run():
        _schmidt._schmidt_tables.cache_clear()
        return _schmidt._schmidt_legendre_with_derivative(n_max, cos_t, sin_t)

    P, dP = benchmark(run)
    assert P.shape == dP.shape == (n_max + 1, n_max + 1)


@pytest.mark.benchmark
@pytest.mark.light
def test_schmidt_table_hit(benchmark):
    cos_t, sin_t = np.cos(THETA), np.sin(THETA)
    _schmidt._schmidt_legendre(13, cos_t, sin_t)
    P = benchmark(_schmidt._schmidt_legendre, 13, cos_t, sin_t)
    assert P.shape == (14, 14)


@pytest.mark.benchmark
@pytest.mark.light
@pytest.mark.parametrize("n_max", [36, 133])
def test_emm_field_point_miss(benchmark, n_max):
    coef = emm.create_test_coefficients(n_max)

    def run():
        _schmidt._schmidt_tables.cache_clear()
        return emm._high_res_field_spherical(
            np.pi / 2 - THETA, 0.4, 6380.0, 2020.0, coef
        )

    assert np.all(np.isfinite(benchmark(run)))


@pytest.mark.benchmark
@pytest.mark.light
def test_wmm_field_same_latitude(benchmark):
    coef = wmm.create_wmm2020_coefficients()
    result = benchmark(
        wmm._compute_magnetic_field_spherical_impl,
        np.pi / 2 - THETA,
        0.3,
        6380.0,
        2022.0,
        coef,
    )
    assert np.all(np.isfinite(result))
