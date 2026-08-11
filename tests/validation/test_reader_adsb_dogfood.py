"""The Parquet measurement reader against the ADS-B dogfood chain.

``test_adsb_tracking.py`` runs ``geodetic2enu`` -> ``kf_predict`` ->
``kf_update`` directly over 3600 real position reports and scores the
result against each aircraft's own broadcast ground speed. This test
proves ``pytcl.io.readers.read_measurements_parquet`` is a transparent
pipe into that same chain: for every aircraft it takes the exact ENU
measurements ``_track`` would compute, materializes them to a Parquet file
(as a user piping results through this module would), reads them back
through the reader, and reruns the identical ``kf_predict``/``kf_update``
loop with the same ``POSITION_SIGMA_M``/``PROCESS_VAR`` constants (imported
from ``test_adsb_tracking``, not re-derived). The recovered per-aircraft
velocity-vs-broadcast error is asserted to match the direct path's to
1e-9 -- not merely "close", because a reader that reordered rows, dropped
precision, or mis-grouped scans would still often pass a loose check.
"""

import math

import numpy as np
import pytest

from pytcl.coordinate_systems.conversions import geodetic2enu
from pytcl.dynamic_estimation.kalman.linear import kf_predict, kf_update
from pytcl.dynamic_models import q_discrete_white_noise
from pytcl.io.readers import read_measurements_parquet
from pytcl.mathematical_functions.statistics import nis

# Reused rather than reimported as a fixture: test_adsb_tracking's `tracked`
# fixture is scoped to its own module's test collection, but the plain
# functions and constants it's built from (`_load`, `_track`,
# POSITION_SIGMA_M, PROCESS_VAR, KNOTS_TO_MS, FEET_TO_M) are ordinary module
# attributes and import cleanly with no fixture-collection side effects.
from tests.validation import test_adsb_tracking as adsb

pl = pytest.importorskip("polars")


def _track_from_measurement_set(ms):
    """Replay `adsb._track`'s CV-filter chain over an already-converted
    `MeasurementSet` instead of raw reports.

    Mirrors `adsb._track`'s F/H/R/Q construction and initial covariance
    exactly; duplicated (rather than imported) only because `_track` calls
    `geodetic2enu` inline on each raw report, and this variant instead
    consumes the pre-converted (east, north) rows the reader produced.
    """
    F = np.eye(4)
    H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    R = np.diag([adsb.POSITION_SIGMA_M**2, adsb.POSITION_SIGMA_M**2])

    state = cov = previous = None
    scores = []

    for t, scan in zip(ms.times, ms.scans):
        for z in scan:
            if state is None:
                state = np.array([z[0], 0.0, z[1], 0.0])
                cov = np.diag([100.0**2, 300.0**2, 100.0**2, 300.0**2])
                previous = t
                continue

            dt = t - previous
            if not 0.5 < dt < 120.0:  # duplicate, or a gap too long to bridge
                previous = t
                continue
            previous = t

            F[0, 1] = F[2, 3] = dt
            Q = q_discrete_white_noise(dim=2, T=dt, var=adsb.PROCESS_VAR, block_size=2)

            prediction = kf_predict(state, cov, F, Q)
            innovation = z - H @ prediction.x
            S = H @ prediction.P @ H.T + R
            scores.append(float(nis(innovation, S)))

            update = kf_update(prediction.x, prediction.P, z, H, R)
            state, cov = update.x, update.P

    return state, scores


def _materialize_to_parquet(reports, ref, path):
    """Write one aircraft's ENU-converted reports to `path` as Parquet.

    Columns t, east, north -- exactly what `read_measurements_parquet`
    expects back via `time_column`/`measurement_columns`.
    """
    lat0, lon0, alt0 = ref
    ts: list[float] = []
    easts: list[float] = []
    norths: list[float] = []
    for report in reports:
        east, north, _ = geodetic2enu(
            math.radians(report["lat"]),
            math.radians(report["lon"]),
            report["alt_ft"] * adsb.FEET_TO_M,
            lat0,
            lon0,
            alt0,
        )
        ts.append(float(report["t"]))
        easts.append(float(east))
        norths.append(float(north))

    pl.DataFrame({"t": ts, "east": easts, "north": norths}).write_parquet(path)


class TestReaderIsATransparentPipe:
    def test_recovered_velocity_error_matches_direct_path(self, tmp_path):
        capture = adsb._load()
        ref = (
            math.radians(capture["centre"]["lat_deg"]),
            math.radians(capture["centre"]["lon_deg"]),
            0.0,
        )

        direct_errors = []
        reader_errors = []
        for key, reports in capture["aircraft"].items():
            if len(reports) < 5:
                continue

            state, track_scores = adsb._track(reports, ref)
            if state is None or not track_scores:
                continue
            broadcast = [
                r["gs_kt"] * adsb.KNOTS_TO_MS for r in reports[-3:] if r["gs_kt"]
            ]
            if not broadcast:
                continue
            direct_estimate = math.hypot(state[1], state[3])
            direct_errors.append(direct_estimate - float(np.mean(broadcast)))

            path = tmp_path / f"{key}.parquet"
            _materialize_to_parquet(reports, ref, path)
            ms = read_measurements_parquet(
                path, time_column="t", measurement_columns=["east", "north"]
            )
            reader_state, reader_scores = _track_from_measurement_set(ms)
            assert reader_state is not None and reader_scores
            reader_estimate = math.hypot(reader_state[1], reader_state[3])
            reader_errors.append(reader_estimate - float(np.mean(broadcast)))

        # Guard the guard: a loop that silently skipped everything would
        # make the assertion below vacuous.
        assert len(direct_errors) >= 100
        assert len(reader_errors) == len(direct_errors)

        direct_median = np.median(np.abs(np.array(direct_errors)))
        reader_median = np.median(np.abs(np.array(reader_errors)))
        assert reader_median == pytest.approx(direct_median, abs=1e-9)
