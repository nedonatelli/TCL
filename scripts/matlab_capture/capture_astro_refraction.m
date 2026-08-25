%CAPTURE_ASTRO_REFRACTION Capture MATLAB TCL reference values for the
%astronomical-refraction functions ported in pytcl/atmosphere/refraction.py
%(tier 2): simpAstroRefParam (requires the compiled MEX), SinclairAtmos,
%removeAstroRefrac and addAstroRefrac for all three algorithms.
%Writes CSVs with input columns followed by output columns to OUTPUT_DIR.

if(~exist('OUTPUT_DIR','var'))
    error('Set OUTPUT_DIR before running.');
end

deg2rad=pi/180;

%simpAstroRefParam (SOFA refco fit) across weather and wavelength,
%including a radio-band wavelength to hit the non-optical branch.
rows=[];
for Rh=[0,0.5,1]
    for P=[80000,101325]
        for T=[263.15,288.15,308.15]
            for wl=[0.4e-6,0.574e-6,1e-6,0.03]
                [A,B]=simpAstroRefParam(Rh,P,T,wl);
                rows(end+1,:)=[Rh,P,T,wl,A,B]; %#ok<SAGROW>
            end
        end
    end
end
writematrix(rows,fullfile(OUTPUT_DIR,'astro_ref_params.csv'));

%SinclairAtmos over height, observer latitude/altitude and weather.
rows=[];
for lat=[-60,0,35]*deg2rad
    for h0=[0,100,2000]
        plhObs=[lat;0.25;h0];
        for h=[h0,1000,5000,10999,11001,20000,60000]
            for Rh=[0,0.7]
                [n,dndr,T,P]=SinclairAtmos(h,plhObs,Rh,101325,288.15,0.574e-6,11000);
                rows(end+1,:)=[lat,h0,h,Rh,n,dndr,T,P]; %#ok<SAGROW>
            end
        end
    end
end
writematrix(rows,fullfile(OUTPUT_DIR,'astro_sinclair.csv'));

%removeAstroRefrac and addAstroRefrac for each algorithm across zenith
%distance and weather. Algorithm-specific zenith limits keep every row
%inside the validity region.
rows=[];
for alg=0:2
    switch(alg)
        case 1
            zVals=[0.01,10,30,50,65]*deg2rad;
        otherwise
            zVals=[0.01,10,30,50,70,85]*deg2rad;
    end
    for z0=zVals
        for Rh=[0,0.5]
            for T=[273.15,288.15]
                plhObs=[0.61;0;100];
                [zTrue,deltaZ]=removeAstroRefrac(alg,plhObs,z0,Rh,101325,T,0.574e-6);
                [zBack,deltaZBack]=addAstroRefrac(alg,plhObs,zTrue,Rh,101325,T,0.574e-6);
                rows(end+1,:)=[alg,z0,Rh,T,zTrue,deltaZ,zBack,deltaZBack]; %#ok<SAGROW>
            end
        end
    end
end
writematrix(rows,fullfile(OUTPUT_DIR,'astro_remove_add.csv'));

%Algorithm 0 at a second observer (high altitude, southern latitude).
rows=[];
plhObs=[-33*deg2rad;1.1;2500];
for z0=[5,25,45,65,80]*deg2rad
    [zTrue,deltaZ]=removeAstroRefrac(0,plhObs,z0,0.3,75000,278.15,0.5e-6);
    rows(end+1,:)=[z0,zTrue,deltaZ]; %#ok<SAGROW>
end
writematrix(rows,fullfile(OUTPUT_DIR,'astro_remove_alt_observer.csv'));

disp('capture_astro_refraction done');
