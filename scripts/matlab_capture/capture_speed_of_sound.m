%CAPTURE_SPEED_OF_SOUND Capture MATLAB TCL reference values for
%speedOfSoundInAir algorithms 1 and 2, ported as
%speed_of_sound_ideal_gas / speed_of_sound_cramer in
%pytcl/atmosphere/models.py.

if(~exist('OUTPUT_DIR','var'))
    error('Set OUTPUT_DIR before running.');
end

warning('off','all');%Range warnings are exercised deliberately.

rows=[];
for T=273.15+[0,10,20,30]
    for Rh=[0,0.3,0.7,1]
        c=speedOfSoundInAir(1,T,Rh);
        rows(end+1,:)=[T,Rh,c]; %#ok<SAGROW>
    end
end
writematrix(rows,fullfile(OUTPUT_DIR,'sos_ideal_gas.csv'));

rows=[];
for T=273.15+[0,15,30]
    for P=[75000,90000,101325]
        for xw=[0,0.02,0.06]
            for xc=[0,0.004,0.01]
                c=speedOfSoundInAir(2,T,P,xw,xc);
                rows(end+1,:)=[T,P,xw,xc,c]; %#ok<SAGROW>
            end
        end
    end
end
writematrix(rows,fullfile(OUTPUT_DIR,'sos_cramer.csv'));

disp('capture_speed_of_sound done');
