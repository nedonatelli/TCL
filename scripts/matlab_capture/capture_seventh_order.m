%CAPTURE_SEVENTH_ORDER Capture MATLAB TCL reference values for
%                       seventhOrderCubPoints, for pytcl's
%                       TestSeventhOrderMatlabFixtures parity test.
%
%   Owner-run, not part of any automated pipeline. Requires MATLAB with
%   the Tracker Component Library on the path, checked out at commit
%   593ce51 (the commit pytcl/mathematical_functions/numerical_integration/
%   cubature_points.py's seventh_order_cubature_points was ported from):
%
%       git clone https://github.com/USNavalResearchLaboratory/TrackerComponentLibrary
%       cd TrackerComponentLibrary && git checkout 593ce51
%
%   Then, with that checkout's top-level init script run (so
%   seventhOrderCubPoints and its helpers are on the MATLAB path), run
%   this script from MATLAB. It writes one CSV per (algorithm, n) case
%   to tests/fixtures/matlab/ (relative to the pytcl repo root -- adjust
%   OUTPUT_DIR below if running from elsewhere): each row is one
%   cubature point, coordinates first, weight last column, 17
%   significant digits (%.17g) so the Python-side comparison in
%   TestSeventhOrderMatlabFixtures.test_matches_matlab is limited by
%   float64 precision, not by this capture.
%
%   MATLAB's seventhOrderCubPoints returns xi as numDim x numPoints and
%   w as numPoints x 1 already normalized to sum to 1 for every
%   algorithm exercised here (verified by inspection of the .m source:
%   algorithm 0 explicitly renormalizes by the sphere's surface area;
%   every other algorithm's A/B/C/D coefficients are printed already
%   summing to 1). No extra normalization is applied in this script.
%
%   Algorithms 3 and 8 are DELIBERATELY OMITTED from CASES below. pytcl's
%   port does not reproduce MATLAB's numeric output for those two: their
%   MATLAB formulas (even with the "corrections" documented in
%   seventhOrderCubPoints.m's comments) do not actually integrate every
%   polynomial of total degree <= 7 exactly against the standard normal --
%   verified with exact symbolic arithmetic on the Python side. See the
%   deviation notes in cubature_points.py's _e2_7_2 and _e4_7_1
%   docstrings for the full derivation of the corrected formulas used
%   instead. Capturing MATLAB's algorithm 3 / 8 output here would produce
%   fixtures no test consumes (TestSeventhOrderMatlabFixtures excludes
%   them for the same reason), so this script does not bother.
%
%August 2026

OUTPUT_DIR = fullfile('tests', 'fixtures', 'matlab');
if ~exist(OUTPUT_DIR, 'dir')
    mkdir(OUTPUT_DIR);
end

% (algorithm, n) pairs, matching
% TestSeventhOrderAlgorithms.CASES / TestSeventhOrderMatlabFixtures in
% tests/unit/test_cubature_points.py, minus algorithms 3 and 8.
CASES = [
    1, 3;
    1, 4;
    1, 6;
    1, 7;
    2, 2;
    4, 3;
    5, 3;
    6, 3;
    7, 3;
    9, 1;
];

for k = 1:size(CASES, 1)
    algorithm = CASES(k, 1);
    n = CASES(k, 2);

    [xi, w] = seventhOrderCubPoints(n, algorithm);

    % xi is n x numPoints; transpose to numPoints x n to match pytcl's
    % (num_points, n) convention, then append w as the last column.
    outMat = [xi.', w];

    outFile = fullfile(OUTPUT_DIR, ...
        sprintf('seventh_order_alg%d_n%d.csv', algorithm, n));
    fid = fopen(outFile, 'w');
    if fid == -1
        error('capture_seventh_order:openFailed', ...
            'Could not open %s for writing.', outFile);
    end
    for row = 1:size(outMat, 1)
        fprintf(fid, '%.17g,', outMat(row, 1:end-1));
        fprintf(fid, '%.17g\n', outMat(row, end));
    end
    fclose(fid);

    fprintf('Wrote %s (%d points, %d dims)\n', outFile, size(outMat, 1), n);
end
