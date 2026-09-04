%CAPTURE_BATCH_LS Capture MATLAB TCL reference values for the batch
%least-squares estimators, ported in
%pytcl/dynamic_estimation/batch_estimation.py. The Gauss-Newton and
%closed-form estimators are deterministic (machine-precision
%fixtures); the LM variants go through LSEstLMarquardt, which the port
%replaces with SciPy's LM, so their fixtures are compared at optimizer
%tolerance on the Python side. MATLAB's 1-based kD maps to the port's
%zero-based k_d = kD - 1. Inputs are mirrored verbatim in
%tests/validation/test_batch_ls.py.

if(~exist('OUTPUT_DIR','var'))
    error('Set OUTPUT_DIR before running.');
end

F=[1, 1; 0, 1];
H=[1, 0];
R=0.01;
Q=[0.02, 0.01; 0.01, 0.03];
z=[0.1, 1.05, 2.02, 2.95];%Four scalar position measurements.

%Linear closed form, kD=2 (python k_d=1), with process noise.
[xEst,PEst]=batchLSLinMeasLinDyn(z,H,F,R,2,Q);
writematrix(xEst',fullfile(OUTPUT_DIR,'batchls_lin_x.csv'));
writematrix(PEst,fullfile(OUTPUT_DIR,'batchls_lin_P.csv'));

%Nonlinear measurement h(x)=x1^2, linear dynamics, kD=1.
h=@(x)x(1)^2;
HJacob=@(x)[2*x(1), 0];
zN=[4.1, 6.2, 9.1, 12.2];
xInit=[1.8;0.4];
[xEst,PEst]=batchLSNonlinMeasLinDyn(xInit,zN,h,F,R,1,HJacob,10);
writematrix(xEst',fullfile(OUTPUT_DIR,'batchls_nlm_x.csv'));
writematrix(PEst,fullfile(OUTPUT_DIR,'batchls_nlm_P.csv'));

%Nonlinear-dynamics form: per-step h fold in powers of F.
hCells=cell(4,1);
HJCells=cell(4,1);
for k=1:4
    Fk=F^(k-1);
    hCells{k}=@(x)(Fk(1,:)*x)^2;
    HJCells{k}=@(x)2*(Fk(1,:)*x)*Fk(1,:);
end
[xEst,PEst]=batchLSNonlinMeasNonlinDyn(xInit,zN,hCells,R,HJCells,10);
writematrix(xEst',fullfile(OUTPUT_DIR,'batchls_nnl_x.csv'));
writematrix(PEst,fullfile(OUTPUT_DIR,'batchls_nnl_P.csv'));

%LM variant, no process noise, kD=1. Three outputs must be requested:
%the original's two-output form crashes (its covariance block reads
%xBatchEst, which is only built when nargout>2 — an upstream defect
%the port avoids by always building the trajectory).
[xEst,PEst,~]=batchLSNonlinMeasLinDynLM(xInit,zN,h,F,R,1,[],HJacob);
writematrix(xEst',fullfile(OUTPUT_DIR,'batchls_lm_x.csv'));
writematrix(PEst,fullfile(OUTPUT_DIR,'batchls_lm_P.csv'));

%LM variant with process noise (trajectory mode).
[xEst,PEst,xBatchEst]=batchLSNonlinMeasLinDynLM(xInit,zN,h,F,R,1,Q,HJacob);
writematrix(xEst',fullfile(OUTPUT_DIR,'batchls_lmq_x.csv'));
writematrix(xBatchEst,fullfile(OUTPUT_DIR,'batchls_lmq_xbatch.csv'));

%Nonlinear-dynamics LM (x only; its PEst is inconsistent upstream —
%it inverts the stacked Cholesky factors instead of R — and the port
%deliberately computes the R-inverse form instead).
[xEst,~]=batchLSNonlinMeasNonlinDynLM(xInit,zN,hCells,R,1);
writematrix(xEst',fullfile(OUTPUT_DIR,'batchls_nnlm_x.csv'));

%Two-point differencing, batch of two pairs, with process noise bias.
T=2;
zTp=zeros(2,2,2);
zTp(:,:,1)=[0, 2; 1, 0];
zTp(:,:,2)=[1, 1.5; -1, -0.5];
RTp=[0.04, 0.01; 0.01, 0.09];
[x,P]=twoPointDiffInit(T,zTp,RTp,0.5);
writematrix(x,fullfile(OUTPUT_DIR,'twopoint_x.csv'));
writematrix([P(:,:,1);P(:,:,2)],fullfile(OUTPUT_DIR,'twopoint_P.csv'));

disp('capture_batch_ls done');
