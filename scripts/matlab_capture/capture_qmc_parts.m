%CAPTURE_QMC_PARTS Capture MATLAB TCL reference values for the
%deterministic parts of the QMC Kalman family, ported in
%pytcl/dynamic_estimation/kalman/qmc.py: QMCKalUpdateWithPred
%(qmc_kf_update_with_pred) and calcQMCKalmanGain
%(calc_qmc_kalman_gain). The sampling functions (discQMCKalPred,
%QMCKalUpdate, QMCKalMeasPred) draw from MATLAB's global RNG with no
%injection point, so they are validated statistically on the Python
%side instead; these two consume a precomputed otherInfo struct and are
%fully deterministic, so the struct is built here by hand with fixed
%values mirrored in tests/validation/test_qmc_kalman.py.

if(~exist('OUTPUT_DIR','var'))
    error('Set OUTPUT_DIR before running.');
end

%Two components, xDim=3, zDim=2.
xPred=[1.0, -0.6;
      -0.4,  0.9;
       0.3,  1.2];
PPred=zeros(3,3,2);
PPred(:,:,1)=[0.50, 0.10, 0.00;
              0.10, 0.80, 0.05;
              0.00, 0.05, 0.30];
PPred(:,:,2)=[0.90, -0.20, 0.10;
             -0.20,  0.60, 0.00;
              0.10,  0.00, 0.40];
zPred=[0.95, -0.55;
      -0.35,  0.85];
PzPred=zeros(2,2,2);
PzPred(:,:,1)=[0.45, 0.08;
               0.08, 0.70];
PzPred(:,:,2)=[0.85, -0.15;
              -0.15,  0.55];
Pxz=zeros(3,2,2);
Pxz(:,:,1)=[0.40, 0.05;
            0.12, 0.65;
            0.02, 0.08];
Pxz(:,:,2)=[0.80, -0.10;
           -0.18,  0.50;
            0.09,  0.03];

otherInfo=struct('innovTrans',@(a,b)bsxfun(@minus,a,b),...
                 'stateTrans',@(x)x,...
                 'xPred',xPred,'PPred',PPred,'Pxz',Pxz,...
                 'zPred',zPred,'PzPred',PzPred);

z=[1.15;-0.25];
R=0.2*eye(2);

[xUpdate,PUpdate,innov,Pzz,W]=QMCKalUpdateWithPred(z,R,otherInfo);
writematrix(xUpdate,fullfile(OUTPUT_DIR,'qmc_upwp_x.csv'));
writematrix([PUpdate(:,:,1);PUpdate(:,:,2)],fullfile(OUTPUT_DIR,'qmc_upwp_P.csv'));
writematrix(innov,fullfile(OUTPUT_DIR,'qmc_upwp_innov.csv'));
writematrix([Pzz(:,:,1);Pzz(:,:,2)],fullfile(OUTPUT_DIR,'qmc_upwp_Pzz.csv'));
writematrix([W(:,:,1);W(:,:,2)],fullfile(OUTPUT_DIR,'qmc_upwp_W.csv'));

%calcQMCKalmanGain works on a single-component otherInfo.
otherInfo1=struct('Pxz',Pxz(:,:,1));
WGain=calcQMCKalmanGain(R,PzPred(:,:,1),otherInfo1);
writematrix(WGain,fullfile(OUTPUT_DIR,'qmc_gain.csv'));

disp('capture_qmc_parts done');
