#!/usr/bin/env python3
"""Generate a tiny synthetic GEBCO NetCDF fixture for CI.

``pytcl.terrain.loaders.load_gebco``'s diagnostics path -- the "found" +
parse start/finish DEBUG records, and the ``progress=True`` log-pair
fallback -- has never executed end-to-end in CI: the real GEBCO grid is
~7.5 GB and no CI runner has it, so those code paths only ever run against
a developer's local download. This script writes a minimal NetCDF file with
the exact schema ``parse_gebco_netcdf`` expects (``lat``, ``lon``,
``elevation`` variables, degrees for coordinates), so the real loader parses
it with no special-casing and the diagnostics path finally executes in CI.

Deterministic: fixed seed, so re-running reproduces the same elevation
values every time (NetCDF-level metadata such as creation timestamps is not
written by this script, so the output file is otherwise stable too).

Grid
----
2-degree tile, 10 N-12 N by 20 E-22 E (an arbitrary open-ocean/land-agnostic
patch, chosen only to stay away from the poles and the antimeridian), at
0.1-degree (360 arc-second) resolution -- 21 x 21 points. Real GEBCO2025 is
15 arc-seconds; this fixture is ~24x coarser, chosen purely to keep the file
under the repo's prek large-file gate (< 100 kB, well under the 1000 kB
limit).

Usage
-----
    python scripts/make_synthetic_gebco.py
"""

from pathlib import Path

import numpy as np

SEED = 20260811
LAT_MIN, LAT_MAX = 10.0, 12.0
LON_MIN, LON_MAX = 20.0, 22.0
RESOLUTION_DEG = 0.1  # 360 arc-seconds; real GEBCO2025 is 15 arc-seconds
OUTPUT = (
    Path(__file__).resolve().parent.parent
    / "tests"
    / "fixtures"
    / "terrain"
    / "synthetic_gebco.nc"
)


def build_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build deterministic lat/lon/elevation arrays for the fixture."""
    n = int(round((LAT_MAX - LAT_MIN) / RESOLUTION_DEG)) + 1
    lats = np.linspace(LAT_MIN, LAT_MAX, n)
    lons = np.linspace(LON_MIN, LON_MAX, n)
    rng = np.random.default_rng(SEED)
    # GEBCO stores elevation as whole-meter integers; a uniform +/-50 m
    # spread is enough to exercise the read path without implying any real
    # bathymetric/topographic meaning.
    elevation = np.round(rng.uniform(-50.0, 50.0, size=(n, n))).astype(np.int16)
    return lats.astype(np.float64), lons.astype(np.float64), elevation


def main() -> None:
    import netCDF4 as nc

    lats, lons, elevation = build_grid()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with nc.Dataset(OUTPUT, "w", format="NETCDF4") as dataset:
        dataset.createDimension("lat", lats.size)
        dataset.createDimension("lon", lons.size)
        lat_var = dataset.createVariable("lat", "f8", ("lat",))
        lon_var = dataset.createVariable("lon", "f8", ("lon",))
        elev_var = dataset.createVariable("elevation", "i2", ("lat", "lon"))
        lat_var[:] = lats
        lon_var[:] = lons
        elev_var[:] = elevation
        lat_var.units = "degrees_north"
        lon_var.units = "degrees_east"
        elev_var.units = "m"
        dataset.title = (
            "Synthetic GEBCO-schema fixture for pytcl CI -- not real bathymetry"
        )
        dataset.Conventions = "CF-1.6"

    size = OUTPUT.stat().st_size
    print(f"Wrote {OUTPUT} ({size} bytes)")


if __name__ == "__main__":
    main()
