%CAPTURE_STATIC_EST_FINAL Capture MATLAB TCL reference values for the
%final Static_Estimation ports in pytcl/static_estimation/localization.py:
%  computePolyMeasFIM -> poly_meas_fim
%  directionOnlyStaticLocEst -> direction_only_static_loc_est
%Inputs are mirrored verbatim in
%tests/validation/test_static_localization.py (with zero-based indices
%and row-convention cubature points on the Python side).
%
%directionOnlyStaticLocEst caveats: its triangulateKnownR subfunction
%overwrites RInv with eye(3) (a bug the port fixes), so every fixture
%here uses identity RInv, where the overwrite is a no-op and MATLAB
%remains a valid oracle. The quasi-Newton cases are captured for
%optimum comparison; the port uses SciPy BFGS, so agreement is to
%optimizer tolerance, not bitwise.

if(~exist('OUTPUT_DIR','var'))
    error('Set OUTPUT_DIR before running.');
end

%% computePolyMeasFIM: two TDOA pairs + one bistatic range, 3D.
x=[1e3;2e3;3e3];
sensorLocs=[0,    8e3, -6e3, 2e3;
            0,    1e3,  5e3, -7e3;
            0,   -2e3,  1e3,  4e3];
c=299792458;
measTypes=[0;0;1];
sigma2List=[1e-14;1e-14;100];
sensorIdxLists=[1,1,3;
                2,3,4];
[xi,w]=fifthOrderCubPoints(3);
FIM=computePolyMeasFIM(x,sigma2List,[],measTypes,sensorIdxLists,sensorLocs,c,xi,w);
writematrix(FIM,fullfile(OUTPUT_DIR,'se_fim_tdoa_range.csv'));

%% computePolyMeasFIM: two range rates + two frequency measurements.
sensorVel=[ 50, -80,  20,  90;
            30,  60, -70, -40;
           -20,  10,  50,  25];
sensorStates=[sensorLocs;sensorVel];
fTx=1e9;
measTypes2=[2;2;3;3];
sigma2List2=[0.25;0.25;400;400];
sensorIdxLists2=[1,2,3,4;
                 0,0,0,0];
[xi4,w4]=fifthOrderCubPoints(4);
FIM=computePolyMeasFIM(x,sigma2List2,fTx,measTypes2,sensorIdxLists2,sensorStates,c,xi4,w4);
writematrix(FIM,fullfile(OUTPUT_DIR,'se_fim_rr_freq.csv'));

%% directionOnlyStaticLocEst scene: four 3D sensors, one target.
t=[500;800;1200];
lRx=[0,   1000, -200, 400;
     0,    100,  900, -300;
     0,   -200,  300,  700];
u=bsxfun(@minus,t,lRx);
u=bsxfun(@rdivide,u,sqrt(sum(u.*u,1)));

%Algorithm 0, unweighted, one refinement iteration (default).
[tEst,exitCode]=directionOnlyStaticLocEst(u,lRx,0);
assert(all(exitCode==0));
writematrix(tEst',fullfile(OUTPUT_DIR,'se_dironly_alg0.csv'));

%Algorithm 0 with a weighted suboptimal LS stage (W used only in the
%LS stage; the refinement stays identity-weighted either way).
W=zeros(3,3,4);
for k=1:4
    W(:,:,k)=diag([1;2;4]*k);
end
params1=struct('W',W);
[tEst,exitCode]=directionOnlyStaticLocEst(u,lRx,0,params1,struct('numIter',0));
assert(all(exitCode==0));
writematrix(tEst',fullfile(OUTPUT_DIR,'se_dironly_alg0_weighted.csv'));

%Algorithm 0 with the constrained LS stage (convexQuadProg).
params1=struct('useConstAlg',true);
[tEst,exitCode]=directionOnlyStaticLocEst(u,lRx,0,params1,struct('numIter',0));
assert(all(exitCode==0));
writematrix(tEst',fullfile(OUTPUT_DIR,'se_dironly_alg0_const.csv'));

%Algorithm 2: explicit solution with known ranges, numIter=1.
r=sqrt(sum(bsxfun(@minus,t,lRx).^2,1));
params1=struct('r',r,'numIter',1);
[tEst,exitCode]=directionOnlyStaticLocEst(u,lRx,2,params1);
assert(all(exitCode==0));
writematrix(tEst',fullfile(OUTPUT_DIR,'se_dironly_alg2.csv'));

%Algorithm 1: LS then quasi-Newton, with noisy directions so the
%optimum is nontrivial. Fixed perturbation for determinism.
uNoisy=u+[ 0.01, -0.02, 0.015, -0.01;
          -0.02,  0.01, -0.01,  0.02;
           0.01,  0.02, -0.02, -0.015];
uNoisy=bsxfun(@rdivide,uNoisy,sqrt(sum(uNoisy.*uNoisy,1)));
[tEst,exitCode]=directionOnlyStaticLocEst(uNoisy,lRx,1);
assert(all(exitCode==0));
writematrix(tEst',fullfile(OUTPUT_DIR,'se_dironly_alg1.csv'));
%Also record the noisy directions so Python needn't re-derive them.
writematrix(uNoisy,fullfile(OUTPUT_DIR,'se_dironly_unoisy.csv'));

disp('capture_static_est_final done');
