% Capture oracle fixtures for Constants.gasProp and
% speedOfSoundInAir algorithm 0.
tclRoot = '/Users/nedonatelli/Documents/Local Repositories/matlab-tcl';
addpath(fullfile(tclRoot, 'Physical_Values'));
addpath(fullfile(tclRoot, 'Atmosphere_and_Refraction'));
addpath(genpath(fullfile(tclRoot, 'Mathematical_Functions')));
outDir = '/Users/nedonatelli/Documents/Local Repositories/TCL/tests/fixtures/matlab';

% --- gasProp over each gas's own valid temperature range -------------
gases = {'N2',[250,700]; 'O2',[250,400]; 'Ar',[250,1024]; ...
         'CO2',[250,1100]; 'Ne',[250,973]; 'Kr',[250,700]; ...
         'CH4',[270,600]; 'He',[250,1400]; 'N2O',[250,423]; ...
         'NO',[250,311]; 'Xe',[250,650]; 'CO',[250,573]; ...
         'H2',[250,400]; 'H2O',[270,363]};
fid = fopen(fullfile(outDir, 'gasprop.csv'), 'w');
fprintf(fid, 'gas,T,AMU,C0p,B,dBdT,d2BdT2\n');
for k = 1:size(gases,1)
    g = gases{k,1};
    r = gases{k,2};
    for T = linspace(r(1), r(2), 7)
        [AMU,C0p,B,dBdT,d2BdT2] = Constants.gasProp(g, T);
        fprintf(fid, '%s,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e\n', ...
                g, T, AMU, C0p, B, dBdT, d2BdT2);
    end
end
fclose(fid);

% --- speedOfSoundInAir algorithm 0 over composition/T/P grids --------
% Compositions as {name, numberDensity} tables. Case 3 includes species
% gasProp does not know (atomic O, N, anomalous O*) which the algorithm
% must silently ignore; case 4 is a humid mix with CO2.
comps = cell(4,1);
comps{1} = {'N2',1.9e25/2.5; 'O2',5.2e24/2.5; 'Ar',2.3e23/2.5};
comps{2} = {'N2',1.6e25; 'O2',4.3e24; 'Ar',2.0e23; 'CO2',1.0e22; 'He',1.4e20};
comps{3} = {'N2',1.6e25; 'O2',4.3e24; 'O',7.5e16; 'N',3.1e14; 'O*',5.0e10; 'Ar',1.9e23};
comps{4} = {'N2',1.5e25; 'O2',4.1e24; 'Ar',1.9e23; 'CO2',9.0e21; 'H2O',4.4e23};
fid = fopen(fullfile(outDir, 'speed_of_sound_gas_table.csv'), 'w');
fprintf(fid, 'case,T,P,c\n');
for k = 1:4
    for T = [275.0, 288.15, 300.0, 330.0]
        for P = [75000.0, 101325.0]
            c = speedOfSoundInAir(0, T, P, comps{k});
            fprintf(fid, '%d,%.17e,%.17e,%.17e\n', k, T, P, c);
        end
    end
end
fclose(fid);
disp('capture done');
