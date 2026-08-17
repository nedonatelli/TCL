%CAPTURE_LCD Capture MATLAB TCL reference values for GaussianLCDSamples,
%            for pytcl's future LCD-samples port fixtures.
%
%   Owner-run, not part of any automated pipeline. Requires MATLAB with
%   the Tracker Component Library on the path, checked out at commit
%   593ce51 (the commit the LCD samples port spec at
%   docs/superpowers/specs/2026-08-16-lcd-samples-design.md was written
%   against):
%
%       git clone https://github.com/USNavalResearchLaboratory/TrackerComponentLibrary
%       cd TrackerComponentLibrary && git checkout 593ce51
%
%   UNLIKE capture_seventh_order.m, this script's target depends on a
%   COMPILED MEX function: GaussianLCDSamples.m calls quasiNewtonLBFGS,
%   whose .m file is a stub that unconditionally errors ("This function
%   is only implemented as a mexed C or C++ function. Please run
%   CompileCLibraries.m to compile the function for use."). The real
%   L-BFGS implementation is a MEX wrapper (quasiNewtonLBFGS.c) around
%   the MIT-licensed 3rd_Party_Libraries/liblbfgs C library. Run that
%   checkout's CompileCLibraries.m (or otherwise build quasiNewtonLBFGS's
%   MEX binary) BEFORE running this script -- the preflight check below
%   fails fast with a pointer to this paragraph if it's missing.
%
%   Then, with that checkout's top-level init script run (so
%   GaussianLCDSamples and quasiNewtonLBFGS are on the MATLAB path), run
%   this script from MATLAB. It writes THREE files per (numDim,
%   numSamples) case to tests/fixtures/matlab/ (relative to the pytcl
%   repo root -- adjust OUTPUT_DIR below if running from elsewhere):
%
%     lcd_n<N>_pts<P>.csv       One row per output cubature point,
%                               coordinates first (N columns), weight
%                               last column, %.17g (17 significant
%                               digits -- float64 round-trip precision).
%                               Matches capture_seventh_order.m's
%                               (numSamples, N) row convention (MATLAB's
%                               native xi layout is N x numSamples;
%                               transposed here).
%     lcd_n<N>_pts<P>_sinit.csv The N x floor(P/2) seed matrix passed as
%                               GaussianLCDSamples's sInit argument,
%                               native (untransposed) layout, %.17g.
%                               MATLAB's randn and NumPy's default RNG
%                               are not cross-compatible even given "the
%                               same" integer seed (different
%                               uniform-to-Gaussian conversion), so this
%                               script does not try to make the Python
%                               side regenerate sInit from a seed number
%                               -- it dumps the actual matrix instead, so
%                               a Python port can start its own optimizer
%                               from the literal same point.
%     lcd_n<N>_pts<P>_meta.csv  One header row
%                               (numDim,numSamples,CvMDistMin,exitCode,
%                               determinism_max_abs_diff) plus one data
%                               row.
%
%   determinism_max_abs_diff ("capture twice, diff"): for each case, this
%   script calls GaussianLCDSamples TWICE with byte-identical arguments
%   (same sInit, passed explicitly both times, so this isolates
%   quasiNewtonLBFGS/MEX determinism from randn entirely) and reports
%   max(abs(xi_run1(:) - xi_run2(:))). Expected result: exactly 0 (a
%   compiled, single-threaded, deterministic numerical C library given
%   identical inputs). This is a NECESSARY but not sufficient condition
%   for the port-vs-wrap decision in the spec above (Section 6): it rules
%   out MATLAB-side nondeterminism, but says nothing about whether a
%   different optimizer implementation (e.g. scipy's L-BFGS-B) converges
%   to the same point -- provably it generally will not, for numDim>=2,
%   per the spec's Section 3 rotation-invariance argument. The first
%   (not second) run's xi/w/CvMDistMin/exitCode are what get written to
%   the fixture files; the second run exists purely for this diff.
%
%   The (numDim, numSamples) grid below is the LCD spec's minimum grid:
%   both parities of numDim (1,3 odd; 2,4 even) and of numSamples (10,20
%   even; 5,15 odd), kept to exactly five cases because
%   GaussianLCDSamples.m's own header comment warns the algorithm is
%   "generally too slow ... for real-time systems" -- each case here
%   already does up to 1000 L-BFGS iterations, each with up to 20
%   line-search evaluations, each evaluation doing an O(numSamples^2)
%   pairwise pass plus an adaptive numerical integration, doubled by the
%   determinism re-run.
%
%August 2026

OUTPUT_DIR = fullfile('tests', 'fixtures', 'matlab');
if ~exist(OUTPUT_DIR, 'dir')
    mkdir(OUTPUT_DIR);
end

% Preflight: fail fast if quasiNewtonLBFGS is still the unmexed .m stub.
% exist(...,'file') returns 3 for a MEX-file, 2 for an ordinary .m file.
if exist('quasiNewtonLBFGS', 'file') ~= 3
    error('capture_lcd:optimizerNotCompiled', ...
        ['quasiNewtonLBFGS is not compiled as a MEX function (got ' ...
         'exist(...)=%d, expected 3). Run CompileCLibraries.m from ' ...
         'the TrackerComponentLibrary checkout first -- see this ' ...
         'script''s header comment.'], exist('quasiNewtonLBFGS', 'file'));
end

% (numDim, numSamples, rngSeed) triples. rngSeed is used only to
% deterministically GENERATE sInit via randn; the generated matrix is
% what actually gets written to the fixture and what matters for
% reproducibility, not the seed value itself (see header comment).
CASES = [
    1,  5, 20260816;
    2, 10, 20260817;
    2, 20, 20260818;
    3, 15, 20260819;
    4, 20, 20260820;
];

for k = 1:size(CASES, 1)
    numDim = CASES(k, 1);
    numSamples = CASES(k, 2);
    rngSeed = CASES(k, 3);

    numHalfSamples = fix(numSamples / 2);

    rng(rngSeed, 'twister');
    sInit = randn(numDim, numHalfSamples);

    % First (fixture-producing) run.
    [xi, w, CvMDistMin, exitCode, ~] = GaussianLCDSamples( ...
        numDim, numSamples, true, [], [], [], sInit);

    % Second run, byte-identical arguments including sInit, purely to
    % measure quasiNewtonLBFGS/MEX determinism ("capture twice, diff").
    [xi2, ~, ~, ~, ~] = GaussianLCDSamples( ...
        numDim, numSamples, true, [], [], [], sInit);

    if isempty(xi) || isempty(xi2)
        error('capture_lcd:optimizationFailed', ...
            'GaussianLCDSamples(%d, %d, ...) returned an empty result (exitCode=%d).', ...
            numDim, numSamples, exitCode);
    end

    determinismMaxAbsDiff = max(abs(xi(:) - xi2(:)));
    if determinismMaxAbsDiff ~= 0
        fprintf(['WARNING: (numDim=%d, numSamples=%d) is NOT bit-' ...
            'deterministic across two runs with identical inputs ' ...
            '(max abs diff = %.17g). This invalidates the necessary ' ...
            'condition in the LCD port spec''s Section 6 -- report ' ...
            'this before relying on any port comparison for this case.\n'], ...
            numDim, numSamples, determinismMaxAbsDiff);
    end

    % --- Write the points file: (numSamples, numDim) rows, weight last.
    outMat = [xi.', w];
    pointsFile = fullfile(OUTPUT_DIR, ...
        sprintf('lcd_n%d_pts%d.csv', numDim, numSamples));
    fid = fopen(pointsFile, 'w');
    if fid == -1
        error('capture_lcd:openFailed', 'Could not open %s for writing.', pointsFile);
    end
    for row = 1:size(outMat, 1)
        fprintf(fid, '%.17g,', outMat(row, 1:end-1));
        fprintf(fid, '%.17g\n', outMat(row, end));
    end
    fclose(fid);

    % --- Write the sInit file: native (numDim, numHalfSamples) layout.
    sInitFile = fullfile(OUTPUT_DIR, ...
        sprintf('lcd_n%d_pts%d_sinit.csv', numDim, numSamples));
    fid = fopen(sInitFile, 'w');
    if fid == -1
        error('capture_lcd:openFailed', 'Could not open %s for writing.', sInitFile);
    end
    for row = 1:size(sInit, 1)
        fprintf(fid, '%.17g,', sInit(row, 1:end-1));
        fprintf(fid, '%.17g\n', sInit(row, end));
    end
    fclose(fid);

    % --- Write the meta file.
    metaFile = fullfile(OUTPUT_DIR, ...
        sprintf('lcd_n%d_pts%d_meta.csv', numDim, numSamples));
    fid = fopen(metaFile, 'w');
    if fid == -1
        error('capture_lcd:openFailed', 'Could not open %s for writing.', metaFile);
    end
    fprintf(fid, 'numDim,numSamples,CvMDistMin,exitCode,determinism_max_abs_diff\n');
    fprintf(fid, '%d,%d,%.17g,%d,%.17g\n', ...
        numDim, numSamples, CvMDistMin, exitCode, determinismMaxAbsDiff);
    fclose(fid);

    fprintf('Wrote %s (%d points, %d dims, CvMDistMin=%.17g, exitCode=%d, determinism diff=%.17g)\n', ...
        pointsFile, numSamples, numDim, CvMDistMin, exitCode, determinismMaxAbsDiff);
end
