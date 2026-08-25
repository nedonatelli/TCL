%CAPTURE_EXP_REFRACTION Capture MATLAB TCL reference values for the
%standard-exponential-model refraction functions ported in
%pytcl/atmosphere/refraction.py (tier 3): osculatingSpher4LatLon,
%Cart2RuvStdRefrac, ruv2CartStdRefrac, stdRefracBiasApprox,
%reduceStdRefrac2Spher and the two cubature wrappers.

if(~exist('OUTPUT_DIR','var'))
    error('Set OUTPUT_DIR before running.');
end

deg2rad=pi/180;

%osculatingSpher4LatLon over a latitude/longitude grid.
rows=[];
for lat=[-75,-30,0,30,60]*deg2rad
    for lon=[-2.5,0,1.2]
        [rE,spherCent]=osculatingSpher4LatLon([lat;lon]);
        rows(end+1,:)=[lat,lon,rE,spherCent(:).']; %#ok<SAGROW>
    end
end
writematrix(rows,fullfile(OUTPUT_DIR,'exp_osculating_sphere.csv'));

%Geometry: receiver on the WGS-84 surface; monostatic and bistatic.
zRx=ellips2Cart([0.61;0.25;100]);
zTx=ellips2Cart([0.62;0.27;50]);%~70 km away, for the bistatic cases.

%Cart2RuvStdRefrac, monostatic: targets over range/bearing/height,
%including a near-vertical target to hit the closed-form branch.
uENU=getENUAxes([0.61;0.25;100]);
rows=[];
for Ns=[280,313,360]
    for grd=[20e3,100e3,300e3]
        for up=[2e3,10e3,30e3]
            zTar=zRx+grd/sqrt(2)*(uENU(:,1)+uENU(:,2))+up*uENU(:,3);
            [z,uTx,uTarRx,uTarTx]=Cart2RuvStdRefrac(zTar,true,zRx,zRx,eye(3),Ns);
            rows(end+1,:)=[Ns,grd,up,zTar(:).',z(:).',uTx(:).',uTarRx(:).',uTarTx(:).']; %#ok<SAGROW>
        end
    end
end
%Near-vertical target (closed-form branch).
zTar=zRx+40e3*uENU(:,3)+0.5*uENU(:,1);
[z,uTx,uTarRx,uTarTx]=Cart2RuvStdRefrac(zTar,true,zRx,zRx,eye(3),313);
rows(end+1,:)=[313,0.5,40e3,zTar(:).',z(:).',uTx(:).',uTarRx(:).',uTarTx(:).'];
writematrix(rows,fullfile(OUTPUT_DIR,'exp_cart2ruv_mono.csv'));

%Cart2RuvStdRefrac, bistatic.
rows=[];
for grd=[50e3,150e3]
    for up=[5e3,20e3]
        zTar=zRx+grd/sqrt(2)*(uENU(:,1)+uENU(:,2))+up*uENU(:,3);
        [z,uTx,uTarRx,uTarTx]=Cart2RuvStdRefrac(zTar,false,zTx,zRx,eye(3),313);
        rows(end+1,:)=[grd,up,zTar(:).',z(:).',uTx(:).',uTarRx(:).',uTarTx(:).']; %#ok<SAGROW>
    end
end
writematrix(rows,fullfile(OUTPUT_DIR,'exp_cart2ruv_bistatic.csv'));

%ruv2CartStdRefrac: round-trip the monostatic measurements.
rows=[];
for grd=[20e3,100e3,300e3]
    for up=[2e3,10e3,30e3]
        zTar=zRx+grd/sqrt(2)*(uENU(:,1)+uENU(:,2))+up*uENU(:,3);
        z=Cart2RuvStdRefrac(zTar,true,zRx,zRx,eye(3),313);
        zCart=ruv2CartStdRefrac(z,true,zRx,zRx,eye(3),313);
        rows(end+1,:)=[grd,up,z(:).',zCart(:).']; %#ok<SAGROW>
    end
end
writematrix(rows,fullfile(OUTPUT_DIR,'exp_ruv2cart_mono.csv'));

%ruv2CartStdRefrac, bistatic round trip.
rows=[];
for grd=[50e3,150e3]
    for up=[5e3,20e3]
        zTar=zRx+grd/sqrt(2)*(uENU(:,1)+uENU(:,2))+up*uENU(:,3);
        z=Cart2RuvStdRefrac(zTar,false,zTx,zRx,eye(3),313);
        zCart=ruv2CartStdRefrac(z,false,zTx,zRx,eye(3),313);
        rows(end+1,:)=[grd,up,z(:).',zCart(:).']; %#ok<SAGROW>
    end
end
writematrix(rows,fullfile(OUTPUT_DIR,'exp_ruv2cart_bistatic.csv'));

%stdRefracBiasApprox for both algorithms across L, elevation and height,
%plus the near-vertical closed-form branch of algorithm 1.
rows=[];
for L=[30e3,100e3,300e3]
    for thetaEl=[0.5,5,20,45]*deg2rad
        for radarHeight=[0,500]
            for alg=0:1
                [dR,dTheta]=stdRefracBiasApprox(L,thetaEl,radarHeight,313,[],[],alg);
                rows(end+1,:)=[alg,L,thetaEl,radarHeight,dR,dTheta]; %#ok<SAGROW>
            end
        end
    end
end
%Algorithm 1 only: high elevations (alg 0 errors above 49 degrees).
for thetaEl=[60,89.99]*deg2rad
    [dR,dTheta]=stdRefracBiasApprox(100e3,thetaEl,100,313,[],[],1);
    rows(end+1,:)=[1,100e3,thetaEl,100,dR,dTheta]; %#ok<SAGROW>
end
writematrix(rows,fullfile(OUTPUT_DIR,'exp_bias_approx.csv'));

%reduceStdRefrac2Spher: rows padded with NaN to two solution slots.
rows=[];
for NMeas=[250,280,300,320,350]
    for height=[0,500,1000,3000,10000]
        NsVals=reduceStdRefrac2Spher(NMeas,height);
        sol=nan(1,2);
        sol(1:numel(NsVals))=NsVals(:).';
        rows(end+1,:)=[NMeas,height,numel(NsVals),sol]; %#ok<SAGROW>
    end
end
writematrix(rows,fullfile(OUTPUT_DIR,'exp_reduce_refrac.csv'));

%Cubature wrappers, one monostatic case each.
[xi,w]=fifthOrderCubPoints(3);
zTar=zRx+50e3/sqrt(2)*(uENU(:,1)+uENU(:,2))+10e3*uENU(:,3);
SR=diag([100;100;100]);
[zRuv,RRuv]=Cart2RuvStdRefracCubature(zTar,SR,true,zRx,zRx,eye(3),313,xi,w);
row=[zTar(:).',zRuv(:).',RRuv(:).'];
writematrix(row,fullfile(OUTPUT_DIR,'exp_cubature_c2r.csv'));

zMeas=Cart2RuvStdRefrac(zTar,true,zRx,zRx,eye(3),313);
SRruv=diag([10;1e-4;1e-4]);
[zCart,RCart]=ruv2CartStdRefracCubature(zMeas,SRruv,true,zRx,zRx,eye(3),313,xi,w);
row=[zMeas(:).',zCart(:).',RCart(:).'];
writematrix(row,fullfile(OUTPUT_DIR,'exp_cubature_r2c.csv'));

disp('capture_exp_refraction done');
