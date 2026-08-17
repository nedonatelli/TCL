# MATLAB TCL reference fixtures

Value tables captured from the original MATLAB Tracker Component Library,
used to check specific pytcl ports numerically match MATLAB's output at
float64 tolerance. Nothing in the library reads these at runtime; they are
loaded only by the `Test*MatlabFixtures` test classes that pair with each
capture script, which skip gracefully when the corresponding CSV is absent
(and fail loudly instead if `PYTCL_REQUIRE_MATLAB_FIXTURES=1` is set).

This directory holds no golden byte blobs -- every fixture here is a plain
CSV of floating-point values, diffable and regenerable from a documented
MATLAB run.

## seventh_order_alg\<A\>_n\<N\>.csv

Produced by `scripts/matlab_capture/capture_seventh_order.m`, run in MATLAB
against the Tracker Component Library at commit
[`593ce51`](https://github.com/USNavalResearchLaboratory/TrackerComponentLibrary/tree/593ce51)
(the commit `seventh_order_cubature_points` in
`pytcl/mathematical_functions/numerical_integration/cubature_points.py` was
ported from). Consumed by
`TestSeventhOrderMatlabFixtures.test_matches_matlab` in
`tests/unit/test_cubature_points.py`.

**Format:** one row per cubature point; the first `N` columns are the
point's coordinates (matching pytcl's `(num_points, n)` convention -- the
capture script transposes MATLAB's native `n x num_points` layout), the
last column is the point's weight, `%.17g` (17 significant digits, i.e. the
full float64 round-trip precision). MATLAB's `seventhOrderCubPoints`
weights are already normalized to sum to 1 for every algorithm captured
here, matching pytcl's convention; the capture script applies no extra
normalization.

**Coverage:** one file per `(algorithm, n)` pair in
`TestSeventhOrderAlgorithms.CASES`, MINUS algorithms 3 and 8. pytcl's
`_e2_7_2` (algorithm 3) and `_e4_7_1` (algorithm 8) deliberately do not
reproduce MATLAB's numeric output for those two algorithms: MATLAB's
formulas there, even including the corrections documented in
`seventhOrderCubPoints.m`'s own comments, do not actually integrate every
polynomial of total degree <= 7 exactly against the standard normal
(verified with exact symbolic arithmetic -- see the deviation notes in
`_e2_7_2` and `_e4_7_1`'s docstrings for the derivation of the corrected
formulas pytcl uses instead). A MATLAB-parity fixture for those two would
fail by construction and would not indicate a defect in the port, so
`capture_seventh_order.m` does not produce one and
`TestSeventhOrderMatlabFixtures` excludes them from its parametrization.

## lcd_n\<N\>_pts\<P\>\*.csv

Produced by `scripts/matlab_capture/capture_lcd.m`, run in MATLAB against
the Tracker Component Library at commit
[`593ce51`](https://github.com/USNavalResearchLaboratory/TrackerComponentLibrary/tree/593ce51).
Three files per (numDim, numSamples) case: `lcd_n<N>_pts<P>.csv` (coordinates
and weights), `lcd_n<N>_pts<P>_sinit.csv` (seed matrix for optimizer), and
`lcd_n<N>_pts<P>_meta.csv` (metadata including convergence details).

**Format:** `lcd_n<N>_pts<P>.csv` has one row per point; N coordinate columns
followed by a weight column, %.17g (17 significant digits, float64 precision).
`lcd_n<N>_pts<P>_sinit.csv` is the N x floor(P/2) seed matrix in native
(untransposed) layout, %.17g. `lcd_n<N>_pts<P>_meta.csv` contains a header row
and one data row: numDim, numSamples, CvMDistMin, exitCode, and
determinism_max_abs_diff (measuring bit-exact repeatability).

**Status:** Not yet captured -- `TestGaussianLCDSamplesMatlabFixtures` skips
gracefully until a maintainer with MATLAB access (and a compiled
`quasiNewtonLBFGS` MEX binary) runs the capture script and commits the CSVs.

## region_\*\_alg\*\_n\*\*.csv (and region_\*\_order\*.csv)

Produced by `scripts/matlab_capture/capture_region_rules.m`, run in MATLAB
against the Tracker Component Library at commit
[`593ce51`](https://github.com/USNavalResearchLaboratory/TrackerComponentLibrary/tree/593ce51).
Covers four region families (Cube_Space, Simplex, Sphere, Spherical_Surface)
with systematic dimension and algorithm sweeps. File naming pattern varies by
function but follows `region_<region>_<functionName>_n<N>[_alg<A>|_order<O>|...].csv`.

**Format:** One row per cubature point; N coordinate columns followed by a
weight column, %.17g (17 significant digits). Weights are NOT normalized to
sum to 1 -- they encode the true integral over each region's measure (volume
or surface area), per the design spec.

**Status:** Not yet captured -- `TestRegionCubatureMatlabFixtures` skips
gracefully until a maintainer with MATLAB access runs the capture script and
commits the CSVs.

## Regenerating

As of this writing no `seventh_order_alg*_n*.csv`, `lcd_n*_pts*.csv`, or
`region_*.csv` files have been captured yet. To (re)generate the seventh-order
fixtures:

1. Clone the Tracker Component Library and check out the pinned commit:
   ```bash
   git clone https://github.com/USNavalResearchLaboratory/TrackerComponentLibrary
   cd TrackerComponentLibrary && git checkout 593ce51
   ```
2. In MATLAB, run that checkout's path-setup script so
   `seventhOrderCubPoints` and its helpers are on the MATLAB path.
3. From the pytcl repo root, run `scripts/matlab_capture/capture_seventh_order.m`
   in MATLAB (adjust `OUTPUT_DIR` inside the script if running from a
   different working directory).
4. `uv run pytest tests/unit/test_cubature_points.py::TestSeventhOrderMatlabFixtures -q`
   should now exercise the fixtures instead of skipping.
