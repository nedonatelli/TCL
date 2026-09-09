% Capture oracle fixtures for the measurement-Hessian tier
% (Component_Hessians, top-level/converted Hessians, cross
% gradients/Hessians). Note: uPolar2DCrossGrad's rotation arguments
% cannot be captured -- its MATLAB source calls rotMat2D2Angle, which
% is not defined anywhere in the library.
tclRoot = '/Users/nedonatelli/Documents/Local Repositories/matlab-tcl';
addpath(genpath(fullfile(tclRoot, 'Coordinate_Systems')));
addpath(genpath(fullfile(tclRoot, 'Mathematical_Functions')));
outDir = '/Users/nedonatelli/Documents/Local Repositories/TCL/tests/fixtures/matlab';

% Same geometry as capture_measurement_jacobians.m.
t   = [-3e3; -2e3; -1e3];
lTx = [-12e3; 8e3; 5e3];
lRx = [4e3; -6e3; 12];
M   = expm([0, -0.3, 0.2; 0.3, 0, -0.1; -0.2, 0.1, 0]);
p2  = [3e3; 4e3];
lRx2 = [500; -300];
M2  = [cos(0.3), -sin(0.3); sin(0.3), cos(0.3)];
Ms  = M;
Muv = expm([0, 0.1, -0.2; -0.1, 0, 0.15; 0.2, -0.15, 0]);
azEl = [0.5; 0.2];
uvPt = [0.1; 0.2];
uvwPt = [0.1; 0.2; sqrt(1 - 0.1^2 - 0.2^2)];
zSph = [9e3; 0.5; 0.2];

fid = fopen(fullfile(outDir, 'measurement_hessians.csv'), 'w');
fprintf(fid, 'label,vals\n');
dump = @(label, H) fprintf(fid, ['%s', repmat(',%.17e', 1, numel(H)), '\n'], label, H(:).');

% Component Hessians.
for uhr = [false, true]
    dump(sprintf('rangeHessian_uhr%d_bi', uhr), rangeHessian(t, uhr, lTx, lRx));
    dump(sprintf('rangeHessian_uhr%d_mono', uhr), rangeHessian(t, uhr, [], []));
end
dump('rangeHessian_2d', rangeHessian(p2, false, [-2e3; 1e3], lRx2));
for st = 0:3
    dump(sprintf('spherAngHessian_st%d', st), spherAngHessian(t, st, lRx, M));
    dump(sprintf('spherAngHessian_st%d_plain', st), spherAngHessian(t, st, [], []));
end
dump('uvHessian', uvHessian(t, lRx, M, false));
dump('uvHessian_w', uvHessian(t, lRx, M, true));
dump('uHessian2D', uHessian2D(p2, lRx2, M2, false));
dump('uHessian2D_v', uHessian2D(p2, lRx2, M2, true));
dump('uHessian3D', uHessian3D(t, lRx, M));

% Top-level and converted Hessians.
for st = 0:3
    dump(sprintf('calcSpherHessian_st%d_bi', st), calcSpherHessian(t, st, false, lTx, lRx, M));
    dump(sprintf('calcSpherHessian_st%d_mono', st), calcSpherHessian(t, st));
    dump(sprintf('calcSpherInvHessian_st%d', st), calcSpherInvHessian(zSph, st));
end
zSpher = Cart2Sphere(t, 0, false, lTx, lRx, M);
dump('calcSpherConvHessian_st0_bi', calcSpherConvHessian(zSpher, 0, false, lTx, lRx, M));

% Helpers.
HuvW = uvHessian(t, lRx, M, true);
% Square map: the MATLAB function assigns M*H back in place, so a
% rectangular output map errors upstream.
MM = [1, 2, 0; 0, 1, -1; 0.5, 0, 1];
dump('HessianOfAffineTransFun', HessianOfAffineTransFun(HuvW, MM));
Hf = calcSpherHessian(t, 0, false, lTx, lRx, M);
Jf = calcSpherJacob(t, 0, false, lTx, lRx, M);
Hg = uvHessian(t, lRx, M, true);
Jg = uvGradient(t, lRx, M, true);
dump('HessianChainRule', HessianChainRule(Hf, Hg, Jf, Jg));

% Cross gradients and Hessians.
for st = 0:3
    dump(sprintf('uvSpherAngCrossGrad_st%d', st), uvSpherAngCrossGrad(azEl, st, false, Ms, Muv));
    dump(sprintf('uvSpherAngCrossGrad_st%d_w', st), uvSpherAngCrossGrad(azEl, st, true, Ms, Muv));
    dump(sprintf('spherAngUvCrossGrad_st%d', st), spherAngUvCrossGrad(uvPt, st, Ms, Muv));
    dump(sprintf('spherAngUvCrossGrad_st%d_w', st), spherAngUvCrossGrad(uvwPt, st, Ms, Muv));
    dump(sprintf('uvSpherAngCrossHessian_st%d', st), uvSpherAngCrossHessian(azEl, st, false, Ms, Muv));
    dump(sprintf('uvSpherAngCrossHessian_st%d_w', st), uvSpherAngCrossHessian(azEl, st, true, Ms, Muv));
    dump(sprintf('spherAngUvCrossHessian_st%d', st), spherAngUvCrossHessian(uvPt, st, Ms, Muv));
end
for st = 0:1
    dump(sprintf('uPolar2DCrossGrad_st%d', st), uPolar2DCrossGrad([0.3, 0.7, 1.4], st, [], [], false));
    dump(sprintf('uPolar2DCrossGrad_st%d_v', st), uPolar2DCrossGrad([0.3, 0.7, 1.4], st, [], [], true));
    dump(sprintf('polarU2DCrossGrad_st%d_u', st), polarU2DCrossGrad([0.2, 0.5, 0.8], st));
    dump(sprintf('polarU2DCrossGrad_st%d_uv', st), polarU2DCrossGrad([0.2, 0.5; 0.6, 0.3], st));
    dump(sprintf('uPolar2DCrossHessian_st%d', st), uPolar2DCrossHessian([0.3, 0.7, 1.4], st, false));
    dump(sprintf('uPolar2DCrossHessian_st%d_v', st), uPolar2DCrossHessian([0.3, 0.7, 1.4], st, true));
    dump(sprintf('polarU2DCrossHessian_st%d_u', st), polarU2DCrossHessian([0.2, 0.5, 0.8], st));
    dump(sprintf('polarU2DCrossHessian_st%d_uv', st), polarU2DCrossHessian([0.2, 0.5; 0.6, 0.3], st));
end

fclose(fid);
disp('capture done');
