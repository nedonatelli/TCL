%%CAPTURE_GRAVITY_ELLIPS Capture oracle fixtures for the gravity
% ellipsoidal-parameter conversions:
%   altEllipsParam2Flattening -> pytcl.gravity.alt_ellips_param_to_flattening
%   ellipsGravCoeffs          -> pytcl.gravity.ellips_grav_coeffs
% MATLAB TCL tree: /Users/nedonatelli/Documents/Local Repositories/matlab-tcl
% at commit a9acd8f. Inputs are literal below and mirrored verbatim in
% tests/validation/test_gravity_ellips_params.py. Nothing here is
% RNG-dependent.
tclRoot = '/Users/nedonatelli/Documents/Local Repositories/matlab-tcl';
addpath(genpath(fullfile(tclRoot, 'Gravity')));
addpath(genpath(fullfile(tclRoot, 'Physical_Values')));
addpath(genpath(fullfile(tclRoot, 'Container_Classes')));
addpath(genpath(fullfile(tclRoot, 'Mathematical_Functions')));

outDir = '/Users/nedonatelli/Documents/Local Repositories/TCL/tests/fixtures/matlab';

% --- altEllipsParam2Flattening: (omega, a, C20Bar, GM, f) rows ---
% Cases: EGM2008 defining constants; WGS84-like; Mars-like; Moon-like.
% (A fast-rotator case was tried and dropped: the iteration in the
% MATLAB original has no cap and loops forever on it.)
cases = [ ...
    7292115e-11, 6378136.3,  -484.1654767e-6, 3986004.415e8; ...
    7292115e-11, 6378137.0,  -484.16685e-6,   3986005.0e8;   ...
    7.088e-5,    3396190.0,  -876.5e-6,       4.2828372e13;  ...
    2.6617e-6,   1737400.0,  -9.0884e-5,      4.9048695e12   ...
];
rows = zeros(size(cases,1), 5);
for i = 1:size(cases,1)
    f = altEllipsParam2Flattening(cases(i,1), cases(i,2), cases(i,3), cases(i,4));
    rows(i,:) = [cases(i,:), f];
end
writematrix(rows, fullfile(outDir, 'grav_alt_ellips_flattening.csv'));

% --- ellipsGravCoeffs: WGS84 defaults, maxOrder x isNormalized grid ---
% Each row: [maxOrder, isNormalized, n, m, CBar(n+1,m+1)] for the nonzero
% zonal entries only (S is all zero by symmetry and is not captured).
rows = [];
for maxOrder = [2, 4, 8, 20]
    for isNorm = [0, 1]
        [CBar, ~, aOut, cOut] = ellipsGravCoeffs(maxOrder, logical(isNorm));
        C = CountingClusterSet(CBar);
        for n = 0:2:maxOrder
            rows(end+1,:) = [maxOrder, isNorm, n, 0, C(n+1, 0+1)]; %#ok<AGROW>
        end
    end
end
writematrix(rows, fullfile(outDir, 'grav_ellips_coeffs.csv'));
fprintf('gravity ellips fixtures written (aOut=%.17g, cOut=%.17g)\n', aOut, cOut);
