# Performance SLOs and Benchmarking

`.benchmarks/slos.json` is the single source of truth for pytcl's performance
Service Level Objectives. This page explains how that file is structured, how
its thresholds were derived, and how to run the benchmarks that check them.
The table below is a snapshot of that file, not a second set of numbers: when
they disagree, the JSON is right.

An earlier version of this page carried a large dashboard of per-dimension
and per-scenario SLO tables for functions that have no SLO entry at all, with
targets up to 30x away from the enforced ones. Those tables were never
generated from a measurement and nothing checked them, so they are gone
rather than corrected — the same reason ROADMAP.md stopped restating closed
performance targets.

## Current SLOs

24 entries, all thresholds in microseconds in the file itself:

| Benchmark | Max mean | Max p99 |
|-----------|----------|---------|
| `test_cfar_ca_1000` | 50 us | 100 us |
| `test_cfar_ca_5000` | 200 us | 400 us |
| `test_cfar_ca_10000` | 400 us | 800 us |
| `test_kf_predict[4]` | 50 us | 100 us |
| `test_kf_update[4-2]` | 100 us | 200 us |
| `test_jpda_update_100_targets_50_meas` | 111.18 ms | 222.37 ms |
| `test_hungarian_dense_500x500` | 16.57 ms | 33.15 ms |
| `test_assign2d_augmented_500x500` | 29.95 ms | 59.89 ms |
| `test_gate_20_tracks_50_meas` | 4.07 ms | 8.13 ms |
| `test_batch_1000_3d` | 4.09 ms | 8.19 ms |
| `test_kmeans_1000_points` | 17.95 ms | 35.91 ms |
| `test_dbscan_1000_points` | 18.52 ms | 37.04 ms |
| `test_smolyak_generation[8]` | 6.66 ms | 13.32 ms |
| `test_quat_rotate_batch_1000` | 9.53 ms | 19.05 ms |
| `test_euler2rotmat_batch_1000` | 17.92 ms | 35.84 ms |
| `test_quat_multiply_batch_1000` | 4.87 ms | 9.73 ms |
| `test_fft_1d_large[65536]` | 1.32 ms | 2.65 ms |
| `test_stft_large` | 445 us | 890 us |
| `test_pulse_compression` | 232 us | 464 us |
| `test_cwt_morlet` | 10.63 ms | 21.26 ms |
| `test_generalized_hypergeometric_3f2_large[1000]` | 259 us | 518 us |
| `test_kf_cycle_with_sql_storage` | 1.43 ms | 2.85 ms |
| `test_store_scenario_10_tracks` | 7.08 ms | 14.15 ms |
| `test_store_detection_batch_100` | 87.56 ms | 175.13 ms |

Regenerate this table from the file rather than editing it by hand:

```bash
uv run python -c "
import json
b = json.load(open('.benchmarks/slos.json'))['benchmarks']
for k, v in b.items():
    print(k, v['max_mean_us'], v['max_p99_us'])
"
```

## File format

```json
{
  "description": "Service Level Objectives for benchmarks",
  "benchmarks": {
    "test_kf_predict[4]": {
      "max_mean_us": 50.0,
      "max_p99_us": 100.0
    }
  }
}
```

Keys are pytest node names (including any parametrization suffix), so an SLO
only binds if its key matches the benchmark's id exactly. Values carry the two
thresholds in microseconds, plus an optional `_derivation` string recording
how the number was arrived at.

## How thresholds are derived

Entries added from v2.5.0 onward follow one doctrine, recorded per entry in
`_derivation`:

```
local median x measured CI/local hardware ratio x 1.5 headroom
```

The CI/local ratio is measured, not assumed — the median of CI-history-vs-fresh-local
ratios across several unrelated calibration benchmarks on the same machine and day.
`max_p99_us` is set to 2x `max_mean_us` by this file's existing convention.
Older entries predate the doctrine and are round numbers.

This matters when reading a threshold: 111 ms for the 100-target JPDA cycle is
not a claim that the operation takes 111 ms. It is a CI-side ceiling built from
a 33.55 ms local median. Local timings live in CHANGELOG.md alongside the change
that produced them.

## Regression detection

`scripts/detect_regressions.py` checks a pytest-benchmark results JSON against
both the SLOs and `.benchmarks/history.jsonl`:

| Threshold | Effective value | Action |
|-----------|-----------------|--------|
| Warning | +15% vs history | Reported in the PR comment |
| Failure | +30% vs history | Non-zero exit under `--strict` |
| Min samples | 5 history records | Below this, no regression verdict |

These are the script's own defaults. `.benchmarks/slos.json` carries no
`regression_thresholds` block, so the defaults are what actually run; adding
that block to the JSON is what would override them.

## Running benchmarks

Light suite, as `benchmark-light.yml` runs it on pull requests — the file list
and the `light` marker must agree, or a file silently contributes nothing:

```bash
uv run pytest benchmarks/test_kalman_bench.py \
              benchmarks/test_gating_bench.py \
              benchmarks/test_rotations_bench.py \
              benchmarks/test_cubature_bench.py \
              benchmarks/test_special_functions_bench.py \
              benchmarks/test_signal_processing_bench.py \
  --benchmark-only \
  --benchmark-json=/tmp/benchmark_results.json \
  --benchmark-warmup=on \
  --benchmark-min-rounds=100 \
  -m "light" -q
```

Full suite, as `benchmark-full.yml` runs it nightly and on main:

```bash
uv run pytest benchmarks/ --benchmark-only
```

Check the results against the SLOs and history:

```bash
uv run python scripts/detect_regressions.py /tmp/benchmark_results.json \
  --slos .benchmarks/slos.json \
  --history .benchmarks/history.jsonl
```

Render the same comparison as the markdown PR comment:

```bash
uv run python scripts/generate_slo_report.py /tmp/benchmark_results.json \
  --slos .benchmarks/slos.json --format markdown -o /tmp/summary.md
```

Append a commit's numbers to the history file:

```bash
uv run python scripts/track_performance.py
```

## Caching

Several geophysical and astronomical paths memoize expensive intermediates
with `functools.lru_cache` — `associated_legendre`'s scaling factors
(`pytcl.gravity.spherical_harmonics`), `precession_matrix_iau76`
(`pytcl.astronomical.reference_frames`), geodesic constants
(`pytcl.navigation.geodesy`, `pytcl.navigation.great_circle`), and the WMM
field evaluation (`pytcl.magnetism.wmm`). Repeated evaluation at the same
inputs therefore costs a dictionary lookup rather than the full computation.

No benchmark measures these speedups, so no multiplier is quoted here. See
[ADR-001](ADR-001-geophysical-caching.md) for the caching decision and
[module-interdependencies.md](module-interdependencies.md) for where the
caches sit. `pytcl.gravity.clear_legendre_cache()` clears the largest one.
