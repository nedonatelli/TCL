%CAPTURE_REGION_RULES Capture MATLAB TCL reference values for the
%          region-cubature subset (Cube_Space, Simplex, Sphere,
%          Spherical_Surface), for pytcl's future region_cubature.py
%          port fixtures.
%
%   Owner-run, not part of any automated pipeline. Requires MATLAB with
%   the Tracker Component Library on the path, checked out at commit
%   593ce51 (the commit the region-rules subset spec at
%   docs/superpowers/specs/2026-08-16-region-cubature-design.md was
%   written against):
%
%       git clone https://github.com/USNavalResearchLaboratory/TrackerComponentLibrary
%       cd TrackerComponentLibrary && git checkout 593ce51
%
%   Then, with that checkout's top-level init script run (so
%   *NDimCubPoints, *SimplexCubPoints, *SpherCubPoints, and
%   *SpherSurfCubPoints are on the MATLAB path), run this script from
%   MATLAB. Unlike capture_lcd.m, none of these functions depend on a
%   compiled MEX binary -- every one is plain .m source, so there is no
%   preflight compilation check here.
%
%   Run this in the SAME MATLAB session as capture_seventh_order.m and
%   capture_lcd.m (this script and capture_lcd.m were designed together
%   per the spec; all three share conventions and run in one capture
%   session for the whole v2.4.0 cycle).
%
%   It writes one CSV per case to tests/fixtures/matlab/ (relative to
%   the pytcl repo root -- adjust OUTPUT_DIR below if running from
%   elsewhere): each row is one cubature point, coordinates first,
%   weight last column, 17 significant digits (%.17g), matching
%   capture_seventh_order.m's convention exactly.
%
%   UNLIKE capture_seventh_order.m, NO renormalization or reordering is
%   applied to the weights here: every function in this subset returns
%   weights that sum to the region's true measure (cube volume 2^numDim,
%   simplex volume 1/factorial(numDim), ball/sphere-surface volume
%   (2/numDim)*pi^(numDim/2)/gamma(numDim/2) or its alpha-adjusted
%   variant) rather than to 1. This matches the convention
%   region_cubature.py's port pins in the design spec (Section 1): the
%   region-cubature functions report the true integral, not a
%   probability expectation, so no rescale belongs in this capture
%   script or in the port -- see the design spec before "fixing" this to
%   sum to 1.
%
%   Cases below cover: (a) every Tier-1/Tier-2 function the design spec
%   prioritizes for porting, at a representative dimension sweep; (b)
%   three spot checks against MATLAB files ALREADY functionally covered
%   by existing pytcl code (GaussLegendrePoints1D vs
%   quadrature.gauss_legendre; fourteenthOrderSpherSurfCubPoints vs
%   cubature_points._fourteenth_order_unit_sphere_points_3d;
%   arbOrderSpherSurfCubPoints and arbOrder2DSpherSurfCubPoints vs
%   cubature_points._sphere_surface_points) -- these three exist to
%   validate the EXISTING port against real MATLAB output, not to seed
%   new work. See the design spec's Section 3 (overlap table) and
%   Section 6 (this script's case rationale) for the full argument.
%
%August 2026

OUTPUT_DIR = fullfile('tests', 'fixtures', 'matlab');
if ~exist(OUTPUT_DIR, 'dir')
    mkdir(OUTPUT_DIR);
end

writeRegionCase(OUTPUT_DIR, 'region_cube_GaussLegendrePoints1D_n8', ...
    GaussLegendrePoints1DAsRow(8));

%== Cube_Space ============================================================
%
% Two of this section's originally-planned cases were found, during the
% Task-A1 port (region_cubature.py), to be UNCAPTURABLE as literally
% specified: MATLAB's own source at the pinned commit 593ce51 returns an
% [xi, w] pair whose sizes do not agree with each other for these two
% cases, so the `[xi.', w]` horzcat below would error, not merely produce
% a wrong number. Both are documented in detail in region_cubature.py's
% module docstring ("Two corrected MATLAB defects"); summarized here so
% this script does not crash mid-session:
%
%   * firstOrderNDimCubPoints's DEFAULT algorithm (1, 2^numDim points):
%     `w = 1/2^numDim*ones(numDim,1)*V` is a numDim-length vector against
%     an numDim-by-2^numDim `xi`. Captured below at algorithm 0 (the
%     1-point rule) instead, which is not affected.
%   * thirdOrderNDimCubPoints's DEFAULT algorithm (0, 2*numDim points) at
%     ODD numDim: `xi(numDim, i+1)` overruns the declared 2*numDim
%     columns (every other assignment in the same branch uses `i`, not
%     `i+1`). Captured below only at EVEN numDim (2, 4); numDim = 3, 5
%     are skipped for this one function.
for numDim = 2:5
    [xi, w] = firstOrderNDimCubPoints(numDim, 0);
    writeRegionCase(OUTPUT_DIR, sprintf('region_cube_firstOrderNDimCubPoints_n%d_alg0', numDim), [xi.', w]);

    [xi, w] = secondOrderNDimCubPoints(numDim);
    writeRegionCase(OUTPUT_DIR, sprintf('region_cube_secondOrderNDimCubPoints_n%d_alg0', numDim), [xi.', w]);

    [xi, w] = fifthOrderNDimCubPoints(numDim);
    writeRegionCase(OUTPUT_DIR, sprintf('region_cube_fifthOrderNDimCubPoints_n%d_alg0', numDim), [xi.', w]);
end

for numDim = 2:2:4
    [xi, w] = thirdOrderNDimCubPoints(numDim);
    writeRegionCase(OUTPUT_DIR, sprintf('region_cube_thirdOrderNDimCubPoints_n%d_alg0', numDim), [xi.', w]);
end

[xi, w] = seventhOrderNDimCubPoints(2, 0);
writeRegionCase(OUTPUT_DIR, 'region_cube_seventhOrderNDimCubPoints_n2_alg0', [xi.', w]);
[xi, w] = seventhOrderNDimCubPoints(3, 5);
writeRegionCase(OUTPUT_DIR, 'region_cube_seventhOrderNDimCubPoints_n3_alg5', [xi.', w]);

for numDim = 4:5
    [xi, w] = ninthOrderNDimCubPoints(numDim, 0);
    writeRegionCase(OUTPUT_DIR, sprintf('region_cube_ninthOrderNDimCubPoints_n%d_alg0', numDim), [xi.', w]);
end

%== Simplex ================================================================
for numDim = 2:5
    [xi, w] = secondOrderSimplexCubPoints(numDim);
    writeRegionCase(OUTPUT_DIR, sprintf('region_simplex_secondOrderSimplexCubPoints_n%d', numDim), [xi.', w]);

    [xi, w] = thirdOrderSimplexCubPoints(numDim, 0);
    writeRegionCase(OUTPUT_DIR, sprintf('region_simplex_thirdOrderSimplexCubPoints_n%d_alg0', numDim), [xi.', w]);

    % fifthOrderSimplexCubPoints auto-selects its algorithm from numDim
    % when omitted (0 for numDim>=4, 1 for numDim=2, 2 for numDim=3) --
    % matching the design spec's "default algorithm" case list.
    [xi, w] = fifthOrderSimplexCubPoints(numDim);
    writeRegionCase(OUTPUT_DIR, sprintf('region_simplex_fifthOrderSimplexCubPoints_n%d_algdefault', numDim), [xi.', w]);
end

for numDim = 3:5
    [xi, w] = fourthOrderSimplexCubPoints(numDim);
    writeRegionCase(OUTPUT_DIR, sprintf('region_simplex_fourthOrderSimplexCubPoints_n%d', numDim), [xi.', w]);
end

%== Sphere (solid ball) ====================================================
for numDim = 2:4
    [xi, w] = secondOrderSpherCubPoints(numDim);
    writeRegionCase(OUTPUT_DIR, sprintf('region_sphere_secondOrderSpherCubPoints_n%d', numDim), [xi.', w]);

    [xi, w] = thirdOrderSpherCubPoints(numDim, 0);
    writeRegionCase(OUTPUT_DIR, sprintf('region_sphere_thirdOrderSpherCubPoints_n%d_alg0', numDim), [xi.', w]);

    [xi, w] = fifthOrderSpherCubPoints(numDim, 0);
    writeRegionCase(OUTPUT_DIR, sprintf('region_sphere_fifthOrderSpherCubPoints_n%d_alg0', numDim), [xi.', w]);
end

for numDim = 3:4
    [xi, w] = seventhOrderSpherCubPoints(numDim, 0);
    writeRegionCase(OUTPUT_DIR, sprintf('region_sphere_seventhOrderSpherCubPoints_n%d_alg0', numDim), [xi.', w]);
end

% Nonzero-alpha spot check: algorithm 0's V and r formulas both carry an
% explicit "numDim+alpha" term in the .m source (verified by inspection),
% unlike most other algorithms in this file which hard-error if
% alpha~=0.
[xi, w] = fifthOrderSpherCubPoints(3, 0, 1.0);
writeRegionCase(OUTPUT_DIR, 'region_sphere_fifthOrderSpherCubPoints_n3_alg0_alpha1', [xi.', w]);

%== Spherical_Surface ======================================================
for numDim = 2:4
    [xi, w] = thirdOrderSpherSurfCubPoints(numDim, 0);
    writeRegionCase(OUTPUT_DIR, sprintf('region_sphsurf_thirdOrderSpherSurfCubPoints_n%d_alg0', numDim), [xi.', w]);

    [xi, w] = fifthOrderSpherSurfCubPoints(numDim, 0);
    writeRegionCase(OUTPUT_DIR, sprintf('region_sphsurf_fifthOrderSpherSurfCubPoints_n%d_alg0', numDim), [xi.', w]);
end

for numDim = 3:4
    % Algorithm 0 requires numDim>=3 (error('numDim must be >=3 ...') at
    % numDim<3) -- excluded numDim=2 for this function accordingly.
    [xi, w] = seventhOrderSpherSurfCubPoints(numDim, 0);
    writeRegionCase(OUTPUT_DIR, sprintf('region_sphsurf_seventhOrderSpherSurfCubPoints_n%d_alg0', numDim), [xi.', w]);
end

% Already-ported spot checks (design spec Section 3): these three do NOT
% seed new porting work -- they validate cubature_points.py's EXISTING
% _fourteenth_order_unit_sphere_points_3d and _sphere_surface_points
% against real MATLAB output.
[xi, w] = fourteenthOrderSpherSurfCubPoints(3);
writeRegionCase(OUTPUT_DIR, 'region_sphsurf_fourteenthOrderSpherSurfCubPoints_n3', [xi.', w]);

[xi, w] = arbOrderSpherSurfCubPoints(3, 5);
writeRegionCase(OUTPUT_DIR, 'region_sphsurf_arbOrderSpherSurfCubPoints_n3_order5', [xi.', w]);

[xi, w] = arbOrder2DSpherSurfCubPoints(5);
writeRegionCase(OUTPUT_DIR, 'region_sphsurf_arbOrder2DSpherSurfCubPoints_order5', [xi.', w]);

fprintf('Done. All region-cubature fixtures written to %s.\n', OUTPUT_DIR);

function writeRegionCase(outputDir, caseName, outMat)
%WRITEREGIONCASE Write one (numPoints, numDim+1) matrix -- coordinates
%          then weight in the last column -- to
%          <outputDir>/<caseName>.csv at 17 significant digits, matching
%          capture_seventh_order.m's and capture_lcd.m's CSV convention.
%          No renormalization is applied (see this script's header
%          comment): whatever the source function returned is what gets
%          written.
    outFile = fullfile(outputDir, [caseName, '.csv']);
    fid = fopen(outFile, 'w');
    if fid == -1
        error('capture_region_rules:openFailed', ...
            'Could not open %s for writing.', outFile);
    end
    for row = 1:size(outMat, 1)
        fprintf(fid, '%.17g,', outMat(row, 1:end-1));
        fprintf(fid, '%.17g\n', outMat(row, end));
    end
    fclose(fid);
    fprintf('Wrote %s (%d points, %d dims)\n', outFile, size(outMat, 1), size(outMat, 2) - 1);
end

function outMat = GaussLegendrePoints1DAsRow(n)
%GAUSSLEGENDREPOINTS1DASROW Adapt GaussLegendrePoints1D's (1xn xi, nx1 w)
%          output to this script's (numPoints, numDim+1) convention
%          (numDim=1 here) for the already-ported spot check against
%          quadrature.gauss_legendre.
    [xi, w] = GaussLegendrePoints1D(n);
    outMat = [xi.', w];
end
