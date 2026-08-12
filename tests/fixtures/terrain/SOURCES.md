# Synthetic GEBCO fixture

`synthetic_gebco.nc` is **not real bathymetry data**. It is a tiny,
deterministically-generated NetCDF file that satisfies
`pytcl.terrain.loaders.parse_gebco_netcdf`'s schema (`lat`, `lon`,
`elevation` variables, GEBCO naming), so the real `load_gebco` loader can
parse it end-to-end with no special-casing.

| File | Contents | Size |
|------|----------|------|
| `synthetic_gebco.nc` | 21 x 21 grid, 10 N-12 N by 20 E-22 E, 0.1-degree (360 arc-second) resolution, elevation in whole meters, seed `20260811` | 7362 bytes |

```
b9c0aabefe019ffe7045b19d3f36a23ecaf0c6deef8516b13284d22669c98933  synthetic_gebco.nc
```

## Why this exists

The real GEBCO grid is ~7.5 GB and is never checked into the repo or present
on any CI runner (`_find_gebco_file` in `pytcl/terrain/loaders.py` raises
`FileNotFoundError` when it is absent, and the terrain test suite skips
gracefully). That has meant `load_gebco`'s diagnostics instrumentation --
the "candidate pattern" / "file found" DEBUG records from `_find_gebco_file`,
the parse start/finish DEBUG records from `parse_gebco_netcdf`, and the
`progress=True` log-pair fallback path -- has never actually executed in CI.
It only ever ran against a developer's local GEBCO download, which made it
easy for that instrumentation to silently rot.

This fixture closes that gap permanently: it is small enough to commit
(well under the repo's 1000 kB prek large-file gate), and the real loader
reads it exactly as it would a real GEBCO file, so
`tests/unit/test_diagnostics.py::TestDataFileInstrumentation::test_gebco_fixture_exercises_found_and_parse_records`
exercises the diagnostics path unconditionally (only `netCDF4` needs to be
installed; the test uses `pytest.importorskip("netCDF4")`).

## Generation

Produced by `scripts/make_synthetic_gebco.py`, which is deterministic (fixed
seed `20260811`) -- re-running it reproduces the same elevation values:

```
python scripts/make_synthetic_gebco.py
```

See that script's docstring for the exact grid parameters and the reasoning
behind the coarse 0.1-degree resolution (real GEBCO2025 is 15 arc-seconds;
this fixture is ~24x coarser purely to stay small).

## Updating

If `parse_gebco_netcdf`'s expected schema ever changes (new required
variable, renamed dimension, etc.), regenerate with the command above and
update the hash in this file. Nothing else in the library reads this fixture
-- only the diagnostics test above does.
