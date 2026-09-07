%CAPTURE_ENKF_ESRIF Capture MATLAB TCL reference values for the ensemble
%Kalman filter and extended square-root information filter, ported as
%pytcl/dynamic_estimation/kalman/ensemble.py (enkf_predict/enkf_update)
%and esrif_predict/esrif_update in
%pytcl/dynamic_estimation/information_filter.py.
%
%EnKF: explicit vSamp/wSamp noise arrays make both steps fully
%deterministic, so bitwise fixtures work despite the stochastic
%algorithm. ESRIF: QR sign conventions differ between LAPACK builds, so
%fixtures store the sign-invariant quantities (recovered state x and the
%information matrix R'*R), not the raw factors.
%
%EnKFDiscPred's multi-output form calls the undefined stateAvgFun (an
%upstream bug the port fixes), so the prediction fixture captures the
%single-output ensemble and derives mean/covariance here.
%
%Inputs are mirrored verbatim in tests/validation/test_enkf_esrif.py.

if(~exist('OUTPUT_DIR','var'))
    error('Set OUTPUT_DIR before running.');
end

%% Deterministic pseudo-noise (fixed, not RNG-drawn).
xDim=3;
zDim=2;
numSamp=6;

%Ensemble: columns are samples.
xEnsemb=[ 1.0,  1.2, 0.8,  1.1,  0.9, 1.05;
         -0.5, -0.4,-0.6, -0.45,-0.55,-0.5;
          2.0,  2.1, 1.9,  2.05, 1.95, 2.0];
vSamp=[ 0.05,-0.03, 0.02,-0.04, 0.01,-0.01;
       -0.02, 0.04,-0.01, 0.03,-0.03,-0.01;
        0.01, 0.02,-0.02,-0.01, 0.03,-0.03];
wSamp=[ 0.06,-0.02, 0.03,-0.05, 0.02,-0.04;
       -0.03, 0.05,-0.02, 0.01,-0.04, 0.03];

F=[1, 0.5, 0.125;
   0, 1,   0.5;
   0, 0,   1];
f=@(x)F*x;
SQ=0.1*eye(xDim);

%EnKF prediction (single-output form; the multi-output form is broken
%upstream).
xEnsembPred=EnKFDiscPred(xEnsemb,f,SQ,0,[],vSamp);
writematrix(xEnsembPred,fullfile(OUTPUT_DIR,'enkf_pred_ensemble.csv'));

%EnKF update on the predicted ensemble, filter types 0 and 2.
H=[1, 0, 0;
   0, 1, 0];
h=@(x)H*x;
z=[1.6;-0.7];
SR=0.2*eye(zDim);
for ft=[0,2]
    [xEnsembUp,xUpdate,PUpdate,~,Pzz,W]=EnKFUpdate(xEnsembPred,z,SR,h,ft,[],[],[],[],[],wSamp);
    writematrix(xEnsembUp,fullfile(OUTPUT_DIR,sprintf('enkf_up_ensemble_ft%d.csv',ft)));
    writematrix(xUpdate',fullfile(OUTPUT_DIR,sprintf('enkf_up_x_ft%d.csv',ft)));
    writematrix(PUpdate,fullfile(OUTPUT_DIR,sprintf('enkf_up_P_ft%d.csv',ft)));
    writematrix(Pzz,fullfile(OUTPUT_DIR,sprintf('enkf_up_Pzz_ft%d.csv',ft)));
    writematrix(W,fullfile(OUTPUT_DIR,sprintf('enkf_up_W_ft%d.csv',ft)));
end

%% ESRIF: linear-consistency scene (nonlinear f/h via matrices so the
%Jacobians are exact and the port comparison is at machine precision).
x0=[1.0;-0.4;0.3];
P0=diag([0.5;0.8;0.3]);
RInfo=chol(inv(P0));%Upper-triangular, PInv=RInfo'*RInfo
ySqrt=RInfo*x0;
fJacob=@(x)F;
u=[0.1;-0.2;0.05];
Gamma=[1,0,0;0,1,0;0,0,0.5];

[ySqrtPred,RPred,RwPred,RwxPred]=ESRIFDiscPred(ySqrt,RInfo,f,fJacob,SQ,u,Gamma);
xPred=RPred\ySqrtPred;
PInvPred=RPred'*RPred;
writematrix([xPred';PInvPred],fullfile(OUTPUT_DIR,'esrif_pred.csv'));

hJacob=@(x)H;
[ySqrtUp,RUp]=ESRIFUpdate(ySqrtPred,RPred,z,SR,h,hJacob);
xUp=RUp\ySqrtUp;
PInvUp=RUp'*RUp;
writematrix([xUp';PInvUp],fullfile(OUTPUT_DIR,'esrif_update.csv'));

disp('capture_enkf_esrif done');
