% Capture oracle fixtures for the measurement-Jacobian tier
% (Component_Gradients + top-level and converted Jacobians).
tclRoot = '/Users/nedonatelli/Documents/Local Repositories/matlab-tcl';
addpath(genpath(fullfile(tclRoot, 'Coordinate_Systems')));
addpath(genpath(fullfile(tclRoot, 'Mathematical_Functions')));
addpath(fullfile(tclRoot, 'Physical_Values'));
addpath(fullfile(tclRoot, 'Misc'));
outDir = '/Users/nedonatelli/Documents/Local Repositories/TCL/tests/fixtures/matlab';

% Fixed geometry shared with the Python tests.
t   = [-3e3; -2e3; -1e3];
lTx = [-12e3; 8e3; 5e3];
lRx = [4e3; -6e3; 12];
M   = expm([0, -0.3, 0.2; 0.3, 0, -0.1; -0.2, 0.1, 0]);
xs  = [-3e3; -2e3; -1e3; 30; -20; 10];
sTx = [lTx; 5; 3; -2];
sRx = [lRx; -4; 2; 6];
p2  = [3e3; 4e3];
lTx2 = [-2e3; 1e3];
lRx2 = [500; -300];
M2  = [cos(0.3), -sin(0.3); sin(0.3), cos(0.3)];
xs2 = [3e3; 4e3; 10; -5];
sTx2 = [lTx2; 2; -1];
sRx2 = [lRx2; -3; 4];
lRef = [1e3; -2e3; 300];

fid = fopen(fullfile(outDir, 'measurement_jacobians.csv'), 'w');
fprintf(fid, 'label,vals\n');
dump = @(label, J) fprintf(fid, ['%s', repmat(',%.17e', 1, numel(J)), '\n'], label, J(:).');

% Component gradients.
for uhr = [false, true]
    dump(sprintf('rangeGradient_uhr%d_bi', uhr), rangeGradient(t, uhr, lTx, lRx));
    dump(sprintf('rangeGradient_uhr%d_mono', uhr), rangeGradient(t, uhr, [], []));
    dump(sprintf('rangeRateGradient_uhr%d_bi', uhr), rangeRateGradient(xs, uhr, sTx, sRx));
    dump(sprintf('rangeRateGradient_uhr%d_mono', uhr), rangeRateGradient(xs, uhr, [], []));
    dump(sprintf('rangeRateGradient2D_uhr%d_bi', uhr), rangeRateGradient(xs2, uhr, sTx2, sRx2));
end
for st = 0:3
    dump(sprintf('spherAngGradient_st%d', st), spherAngGradient(t, st, lRx, M));
    dump(sprintf('spherAngGradient_st%d_plain', st), spherAngGradient(t, st, [], []));
end
for st = 0:1
    dump(sprintf('polAngGradient_st%d', st), polAngGradient(p2, st, lRx2));
end
dump('uvGradient', uvGradient(t, lRx, M, false));
dump('uvGradient_w', uvGradient(t, lRx, M, true));
dump('uGradient2D', uGradient2D(p2, lRx2, M2, false));
dump('uGradient2D_v', uGradient2D(p2, lRx2, M2, true));
dump('uGradient3D', uGradient3D(t, lRx, M));
dump('TDOAGradient', TDOAGradient(t, lRef, lRx, []));
dump('normVecJacob', normVecJacob(t));

% Top-level Jacobians.
for st = 0:3
    dump(sprintf('calcSpherJacob_st%d_bi', st), calcSpherJacob(t, st, false, lTx, lRx, M));
    dump(sprintf('calcSpherJacob_st%d_mono', st), calcSpherJacob(t, st));
    dump(sprintf('calcSpherInvJacob_st%d', st), calcSpherInvJacob([9e3; 0.5; 0.2], st));
    dump(sprintf('calcSpherRRJacob_st%d', st), calcSpherRRJacob(xs, st, false, sTx, sRx, M));
end
for st = 0:1
    dump(sprintf('calcPolarJacob_st%d_bi', st), calcPolarJacob(p2, st, false, lTx2, lRx2));
    dump(sprintf('calcPolarJacob_st%d_mono', st), calcPolarJacob(p2, st));
    dump(sprintf('calcPolarRRJacob_st%d', st), calcPolarRRJacob(xs2, st, false, sTx2, sRx2));
end
dump('calcRuvJacob', calcRuvJacob(t, false, lTx, lRx, M, false));
dump('calcRuvJacob_w', calcRuvJacob(t, false, lTx, lRx, M, true));
dump('calcRuvRRJacob', calcRuvRRJacob(xs, false, sTx, sRx, M, false));
dump('calcRuvRRJacob_w', calcRuvRRJacob(xs, false, sTx, sRx, M, true));
dump('calcCartRRJacob_c0', calcCartRRJacob(0, xs, false, sTx, sRx));
dump('calcCartRRJacob_c1', calcCartRRJacob(1, xs));

% Converted Jacobians (measurements from the same geometry).
zSpher = Cart2Sphere(t, 0, false, lTx, lRx, M);
dump('calcSpherConvJacob_st0_bi', calcSpherConvJacob(zSpher, 0, false, lTx, lRx, M));
zSpherM = Cart2Sphere(t, 0, true);
dump('calcSpherConvJacob_st0_mono', calcSpherConvJacob(zSpherM, 0));
zPol = Cart2Pol(p2, 0, false, lTx2, lRx2, M2);
dump('calcPolarConvJacob_st0_bi', calcPolarConvJacob(zPol, 0, false, lTx2, lRx2, M2));
dump('calcPolarRRConvJacob_st0_bi', calcPolarRRConvJacob([zPol; 12], 0, false, lTx2, lRx2, M2));
zRuv = Cart2Ruv(t, false, lTx, lRx, M);
dump('calcRuvConvJacob_bi', calcRuvConvJacob(zRuv, false, lTx, lRx, M));
dump('calcRuvRRConvJacob_bi', calcRuvRRConvJacob([zRuv; 25], false, lTx, lRx, M));

fclose(fid);
disp('capture done');
