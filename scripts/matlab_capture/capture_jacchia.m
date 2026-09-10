%%CAPTURE_JACCHIA Capture oracle fixtures for the Jacchia 1971 model:
%   jacchiaAtmosParam -> pytcl.atmosphere.jacchia.jacchia_atmos_param
% MATLAB TCL tree at commit a9acd8f.
%
% THE ORACLE IS A PATCHED COPY. jacchiaAtmosParam.m as shipped is
% unrunnable: line 89 references deltaTTUT1, xpyp, dXdY, which are never
% assigned, and its astro chain (GCRS2ITRS, TT2LAT) is MEX-only. The
% patch below excises the solar-geometry computation (UTC2TT through the
% lha assignment) and takes the solar declination `dec` and local hour
% angle `lha` as explicit inputs; everything downstream -- the actual
% Jacchia 1971 physics -- is untouched MATLAB source, executed here.
% The Python port computes dec/lha with its own (SOFA-derived) chain,
% tested separately against astropy; these fixtures pin the physics
% bit-tight given identical solar geometry.
%
% CSV rows: [jul1, jul2, lat, lon, altMeters, F10, F10b, Kp, dec, lha,
%            rho, P, T, Te] with P in MATLAB's output unit (numerically
% kPa; the Python port returns true Pa = 1000x, and its docstring says
% so).
tclRoot = '/Users/nedonatelli/Documents/Local Repositories/matlab-tcl';
addpath(genpath(fullfile(tclRoot, 'Physical_Values')));
addpath(genpath(fullfile(tclRoot, 'Mathematical_Functions')));
outDir = '/Users/nedonatelli/Documents/Local Repositories/TCL/tests/fixtures/matlab';

% Build the patched copy in a temp folder on the path.
src = fileread(fullfile(tclRoot, 'Atmosphere_and_Refraction', 'jacchiaAtmosParam.m'));
src = strrep(src, ...
    'function [rho,P,T,Te]=jacchiaAtmosParam(Jul1,Jul2,point,F10,F10b,Kp)', ...
    'function [rho,P,T,Te]=jacchiaAtmosParamPatched(Jul1,Jul2,point,F10,F10b,Kp,dec,lha)');
iStart = strfind(src, '[TT1,TT2]=UTC2TT(Jul1,Jul2);');
iEnd = strfind(src, 'lha=LAT-pi;%local hour angle');
lhaLine = 'lha=LAT-pi;%local hour angle';
assert(~isempty(iStart) && ~isempty(iEnd));
src = [src(1:iStart-1), '%PATCH: dec and lha are inputs (see capture_jacchia.m).', ...
       newline, src(iEnd+length(lhaLine):end)];
patchDir = fullfile(tempdir, 'jacchia_patch');
if ~exist(patchDir, 'dir'); mkdir(patchDir); end
fid = fopen(fullfile(patchDir, 'jacchiaAtmosParamPatched.m'), 'w');
fwrite(fid, src); fclose(fid);
addpath(patchDir);

% Case grid: altitudes across all eight coefficient-table branches,
% latitudes incl. both hemispheres (equator excluded: upstream NaN),
% two dates for the semiannual/seasonal phases, flux/Kp spanning the
% low- and high-Te regimes.
alts = [95, 150, 250, 450, 700, 1500] * 1e3;
lats = [-60, -20, 35, 70] * pi/180;
cases = [];
juls = [2451545.0, 0.25; 2460000.5, 0.6];
fluxes = [90, 100, 1.0; 150, 150, 3.0; 220, 190, 7.0];  % F10, F10b, Kp
decs = [-0.35, 0.1, 0.4];
lhas = [-2.5, 0.3, 1.8];
rows = [];
idx = 0;
for a = alts
  for iL = 1:numel(lats)
    idx = idx + 1;
    % Deterministic cycling through the other parameter sets so the
    % Python test can reconstruct the grid exactly.
    jd = juls(mod(idx, 2) + 1, :);
    fl = fluxes(mod(idx, 3) + 1, :);
    dec = decs(mod(idx, 3) + 1);
    lha = lhas(mod(idx + 1, 3) + 1);
    point = [lats(iL); 0.5; a];
    try
        [rho, P, T, Te] = jacchiaAtmosParamPatched(jd(1), jd(2), point, ...
            fl(1), fl(2), fl(3), dec, lha);
        rows(end+1, :) = [jd, lats(iL), 0.5, a, fl, dec, lha, rho, P, T, Te]; %#ok<AGROW>
    catch e
        fprintf('skipped alt=%g lat=%g: %s\n', a, lats(iL), e.message);
    end
  end
end
writematrix(rows, fullfile(outDir, 'jacchia_atmos.csv'));
fprintf('jacchia fixtures written: %d rows\n', size(rows, 1));
