% Capture oracle fixtures for the magnetic coordinate systems
% (centered dipole, headings, apex/QD field-line tracing).
tclRoot = '/Users/nedonatelli/Documents/Local Repositories/matlab-tcl';
addpath(genpath(fullfile(tclRoot, 'Magnetism')));
addpath(genpath(fullfile(tclRoot, 'Coordinate_Systems')));
addpath(genpath(fullfile(tclRoot, 'Mathematical_Functions')));
addpath(fullfile(tclRoot, 'Physical_Values'));
addpath(genpath(fullfile(tclRoot, 'Container_Classes')));
addpath(genpath(fullfile(tclRoot, 'Misc')));
addpath(genpath(fullfile(tclRoot, 'Dynamic_Estimation')));
addpath(genpath(fullfile(tclRoot, 'Mathematical_Functions', 'Differential_Equations')));
outDir = '/Users/nedonatelli/Documents/Local Repositories/TCL/tests/fixtures/matlab';

fid = fopen(fullfile(outDir, 'magnetic_coords.csv'), 'w');
fprintf(fid, 'label,vals\n');
dump = @(label, V) fprintf(fid, ['%s', repmat(',%.17e', 1, numel(V)), '\n'], label, V(:).');

% Centered dipole with explicit degree-1 coefficients (unit-agnostic).
g10 = -29350.0; g11 = -1410.3; h11 = 4545.5;
zTest = [6.4e6; 1e5; 2e6];
dump('itrs2cart_cd_explicit', ITRS2CartCD(zTest, g10, g11, h11));
dump('cart_cd2itrs_explicit', CartCD2ITRS(zTest, g10, g11, h11));
zSph = [6.4e6; 0.5; 0.2];
dump('spher_itrs2cd_explicit', spherITRS2SpherCD(zSph, g10, g11, h11));
dump('spher_cd2itrs_explicit', spherCD2SpherITRS(zSph, g10, g11, h11));
dump('spher_itrs2cd_2elem', spherITRS2SpherCD([0.5; 0.2], g10, g11, h11));

% Centered dipole with the IGRF at the reference epoch. The default
% call form ITRS2CartCD(zTest) errors in R2026a (ClusterSet 2-argument
% indexing), so the degree-1 coefficients are extracted linearly, as
% spherITRS2SpherCD itself does.
[CI, SI] = getIGRFCoeffs([], false);
gI10 = CI(2); gI11 = CI(3); hI11 = SI(3);
dump('igrf_degree1', [gI10; gI11; hI11]);
dump('itrs2cart_cd_igrf', ITRS2CartCD(zTest, gI10, gI11, hI11));

% Headings (WMM default and IGRF), geodetic points in radians/meters.
points = [0.7, -1.2, 0.0;   % latitudes (rad)
          -0.4, 2.0, 2.5;   % longitudes (rad)
          100, 5e3, 10e3];  % heights (m)
headings = [0.5; -1.0; 2.5];
dump('geog_heading2mag_wmm', geogHeading2Mag(points, headings, 'WMM', []));
dump('geog_heading2mag_igrf', geogHeading2Mag(points, headings, 'IGRF', []));
dump('mag_heading2geog_wmm', magHeading2Geog(points, headings, 'WMM', []));

% Apex / QD via field-line tracing (IGRF at the reference epoch).
tracePts = [[6.4e6; 1e5; 2e6], [5.8e6; -2.5e6; 1.5e6], [6.2e6; 1.9e6; -2.2e6]];
for k = 1:size(tracePts, 2)
    zC = tracePts(:, k);
    [apexPoints, signVals] = trace2EarthMagApex(zC);
    dump(sprintf('trace_apex_%d', k), apexPoints);
    dump(sprintf('trace_sign_%d', k), signVals);
    zQD = ITRS2QD(zC);
    dump(sprintf('qd_%d', k), zQD);
    zApex = ITRS2MagneticApex(zC);
    dump(sprintf('apex_%d', k), zApex);
    zApexMod = ITRS2MagneticApex(zC, 110e3);
    dump(sprintf('apex_mod_%d', k), zApexMod);
end

fclose(fid);
disp('capture done');
