%CAPTURE_HUMIDITY_REFRAC Capture MATLAB TCL reference values for the
%humidity conversions, dew-point functions and refractivity helpers ported
%in pytcl/atmosphere/{humidity,refraction}.py. Writes CSVs with input
%columns followed by output columns to OUTPUT_DIR.
%
%Run headlessly (see scripts/matlab_capture/README notes in repo memory):
%  matlab -batch "addpath(genpath('<matlab-tcl>')); OUTPUT_DIR='<out>'; run('<this file>')"

if(~exist('OUTPUT_DIR','var'))
    error('Set OUTPUT_DIR before running.');
end

%Dew point pressure and temperature, all three algorithms. Temperature
%grids respect each algorithm's documented validity range.
rows=[];
for alg=0:2
    switch(alg)
        case 1
            TVals=273.15+(-40:10:50);
        case 2
            TVals=273.15+(-80:10:0);
        otherwise
            TVals=193.15:20:333.15;
    end
    for T=TVals
        p=dewPointPres4Temp(T,alg);
        TBack=dewPointTemp4Pres(p,alg);
        rows(end+1,:)=[alg,T,p,TBack]; %#ok<SAGROW>
    end
end
writematrix(rows,fullfile(OUTPUT_DIR,'humidity_dew_point.csv'));

%Relative <-> absolute humidity across RH, T and algorithm.
rows=[];
for alg=0:2
    for Rh=[0.05,0.25,0.5,0.75,0.95]
        for T=273.15+(-30:15:45)
            ah=relHumid2AbsHumid(Rh,T,alg);
            RhBack=absHumid2RelHumid(ah,T,alg);
            rows(end+1,:)=[alg,Rh,T,ah,RhBack]; %#ok<SAGROW>
        end
    end
end
writematrix(rows,fullfile(OUTPUT_DIR,'humidity_rel_abs.csv'));

%Absolute <-> specific humidity for both definitions.
rows=[];
for defChoice=0:1
    for ah=[1e-4,1e-3,5e-3,0.01,0.03]
        for rho=[0.9,1.0,1.225]
            sh=absHumid2SpecHumid(ah,rho,defChoice);
            ahBack=specHumid2AbsHumid(sh,rho,defChoice);
            rows(end+1,:)=[defChoice,ah,rho,sh,ahBack]; %#ok<SAGROW>
        end
    end
end
writematrix(rows,fullfile(OUTPUT_DIR,'humidity_abs_spec.csv'));

%Relative <-> specific humidity, both definitions, all algorithms.
rows=[];
for alg=0:2
    for defChoice=0:1
        for Rh=[0.1,0.5,0.9]
            for T=273.15+(-20:20:40)
                rho=1.225;
                sh=relHumid2SpecHumid(Rh,T,rho,defChoice,alg);
                RhBack=specHumid2RelHumid(sh,T,rho,defChoice,alg);
                rows(end+1,:)=[alg,defChoice,Rh,T,rho,sh,RhBack]; %#ok<SAGROW>
            end
        end
    end
end
writematrix(rows,fullfile(OUTPUT_DIR,'humidity_rel_spec.csv'));

%Absolute humidity <-> H2O number density.
rows=[];
for ah=[1e-5,1e-4,1e-3,5e-3,0.01,0.03,0.05]
    nd=absHumid2NumberDensH2O(ah);
    ahBack=numberDensH2O2AbsHumid(nd);
    rows(end+1,:)=[ah,nd,ahBack]; %#ok<SAGROW>
end
writematrix(rows,fullfile(OUTPUT_DIR,'humidity_number_density.csv'));

%ITU-R P.453-11 refractivity approximation.
rows=[];
for T=273.15+(-30:15:45)
    for P=[80000,90000,101325,105000]
        for PwFrac=[0,0.25,0.75,1]
            Pw=PwFrac*dewPointPres4Temp(T,0);
            N=approxRefractivity(T,P,Pw);
            rows(end+1,:)=[T,P,Pw,N]; %#ok<SAGROW>
        end
    end
end
writematrix(rows,fullfile(OUTPUT_DIR,'refrac_approx_refractivity.csv'));

%CRPL exponential-atmosphere decay constant.
rows=[];
for Ns=200:25:450
    [ce,deltaN]=atmosExpDecayConst4Refrac(Ns);
    rows(end+1,:)=[Ns,ce,deltaN]; %#ok<SAGROW>
end
writematrix(rows,fullfile(OUTPUT_DIR,'refrac_exp_decay_const.csv'));

disp('capture_humidity_refrac done');
