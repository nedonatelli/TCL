%CAPTURE_SMOOTHERS Capture MATLAB TCL reference values for the batch and
%interval smoothers ported in pytcl/dynamic_estimation/batch_smoothers.py
%and the square-root cubature steps in
%pytcl/dynamic_estimation/kalman/sqrt_cubature.py. All algorithms here are
%deterministic given the inputs, so fixtures are machine-precision except
%where square-root factors are compared (S*S' on the Python side, since QR
%sign conventions differ). sqrtInfoBatchSmoother is NOT captured: its
%backward pass reads the stored Rw/Rwx factors one index below where the
%forward pass stores them (the first smoothed step consumes never-written
%zeros), and it also injects the control input into the residual-domain
%smoothing step where the control cancels identically; the port fixes both
%defects, validating instead against the exact RTS optimum. EKalman
%capture needs a DiscEKFPred->discEKFPred case shim on the path (the
%library ships the lowercase file but the smoother calls the capitalized
%name, so it cannot run unshimmed). MATLAB's 1-based kD maps to the port's
%k_d = kD - 1. Inputs are mirrored verbatim in
%tests/validation/test_batch_smoothers.py.

if(~exist('OUTPUT_DIR','var'))
    error('Set OUTPUT_DIR before running.');
end

F=[1, 1; 0, 1];
H=[1, 0];
R=0.01;
Q=[0.02, 0.01; 0.01, 0.03];
z=[0.05, 1.1, 1.95, 3.02];
xInit=[0;1];
PInit=[1, 0.2; 0.2, 0.5];
u=[0.1, -0.05, 0.02; 0.05, 0.1, -0.1];
N=4;

%Fraser-Potter delegation (useFP=true), full batch, with control inputs.
[xS,PS]=KalmanBatchSmoother(xInit,PInit,z,u,H,F,R,Q);
writematrix(xS,fullfile(OUTPUT_DIR,'sm_kbs_fp_x.csv'));
writematrix([PS(:,:,1);PS(:,:,2);PS(:,:,3);PS(:,:,4)],fullfile(OUTPUT_DIR,'sm_kbs_fp_P.csv'));

%RTS forward-backward (useFP=false), full batch and single step kD=2.
[xS,PS]=KalmanBatchSmoother(xInit,PInit,z,u,H,F,R,Q,[],false);
writematrix(xS,fullfile(OUTPUT_DIR,'sm_kbs_rts_x.csv'));
writematrix([PS(:,:,1);PS(:,:,2);PS(:,:,3);PS(:,:,4)],fullfile(OUTPUT_DIR,'sm_kbs_rts_P.csv'));
[xS2,PS2]=KalmanBatchSmoother(xInit,PInit,z,u,H,F,R,Q,2,false);
writematrix(xS2',fullfile(OUTPUT_DIR,'sm_kbs_rts_x2.csv'));
writematrix(PS2,fullfile(OUTPUT_DIR,'sm_kbs_rts_P2.csv'));

%Fraser-Potter smoother with an uninformative prior.
[yS,PInvS,xS,PS]=FPInfoBatchSmoother([],[],z,u,H,F,R,Q);
writematrix(yS,fullfile(OUTPUT_DIR,'sm_fp_y.csv'));
writematrix([PInvS(:,:,1);PInvS(:,:,2);PInvS(:,:,3);PInvS(:,:,4)],fullfile(OUTPUT_DIR,'sm_fp_Pinv.csv'));
writematrix(xS,fullfile(OUTPUT_DIR,'sm_fp_x.csv'));
writematrix([PS(:,:,1);PS(:,:,2);PS(:,:,3);PS(:,:,4)],fullfile(OUTPUT_DIR,'sm_fp_P.csv'));

%Extended Kalman batch smoother, quadratic measurement, linear dynamics
%expressed as function handles. numIter=0 and numIter=2.
zN=[1.05, 4.2, 8.85, 16.4];
xInitE=[1;1];
h=@(x)x(1)^2;
HJacob=@(x)[2*x(1), 0];
f=@(x)F*x;
FJacob=@(x)F;
RE=0.04;
[xS,PS,xU,PU]=EKalmanBatchSmoother(xInitE,PInit,zN,h,HJacob,f,FJacob,RE,Q);
writematrix(xS,fullfile(OUTPUT_DIR,'sm_ekbs_x.csv'));
writematrix([PS(:,:,1);PS(:,:,2);PS(:,:,3);PS(:,:,4)],fullfile(OUTPUT_DIR,'sm_ekbs_P.csv'));
writematrix(xU,fullfile(OUTPUT_DIR,'sm_ekbs_xupd.csv'));
[xS,~,~,~]=EKalmanBatchSmoother(xInitE,PInit,zN,h,HJacob,f,FJacob,RE,Q,[],2);
writematrix(xS,fullfile(OUTPUT_DIR,'sm_ekbs_it_x.csv'));

%Square-root cubature Kalman single prediction and update, then the batch
%smoother, all with fifth-order cubature points (positive weights for
%xDim=2). Nonlinear measurement and dynamics. Root factors are captured as
%S*S' since QR sign conventions differ between implementations.
[xi,w]=fifthOrderCubPoints(2);
SR=0.1;
SQ=chol(Q,'lower');
SInit=chol(PInit,'lower');
hC=@(x)x(1)+0.05*x(2)^2;
fC=@(x)[x(1)+x(2)+0.01*x(1)^2; x(2)-0.02*x(1)];
[xPred,SPred]=sqrtDiscCubKalPred(xInit,SInit,fC,SQ,xi,w);
writematrix(xPred',fullfile(OUTPUT_DIR,'sm_sckf_xpred.csv'));
writematrix(SPred*SPred',fullfile(OUTPUT_DIR,'sm_sckf_Ppred.csv'));
[xUpd,SUpd]=sqrtCubKalUpdate(xPred,SPred,z(2),SR,hC,xi,w);
writematrix(xUpd',fullfile(OUTPUT_DIR,'sm_sckf_xupd.csv'));
writematrix(SUpd*SUpd',fullfile(OUTPUT_DIR,'sm_sckf_Pupd.csv'));

[xS,SS,xU,SU]=sqrtCubKalBatchSmoother(xInit,SInit,z,hC,fC,SR,SQ,xi,w);
writematrix(xS,fullfile(OUTPUT_DIR,'sm_scks_x.csv'));
PP=zeros(2*4,2);
for k=1:4
    PP((2*k-1):(2*k),:)=SS(:,:,k)*SS(:,:,k)';
end
writematrix(PP,fullfile(OUTPUT_DIR,'sm_scks_P.csv'));
writematrix(xU,fullfile(OUTPUT_DIR,'sm_scks_xupd.csv'));

%Interval Kalman smoother: grow a length-3 interval from a single
%posterior, then slide it once. Mirrors the documented usage pattern.
xFwdPred=[];
PFwdPred=[];
xFwdPost=xInit;
PFwdPost=PInit;
for curStep=2:4
    [xInt,PInt,xFwdPred,PFwdPred,xFwdPost,PFwdPost]=KalmanIntervalSmoother(xFwdPred,PFwdPred,xFwdPost,PFwdPost,3,z(curStep),R,H,F,Q);
    writematrix(xInt,fullfile(OUTPUT_DIR,sprintf('sm_kis_x%d.csv',curStep)));
    PP=zeros(2*size(xInt,2),2);
    for k=1:size(xInt,2)
        PP((2*k-1):(2*k),:)=PInt(:,:,k);
    end
    writematrix(PP,fullfile(OUTPUT_DIR,sprintf('sm_kis_P%d.csv',curStep)));
end

%Interval Fraser-Potter information smoother, same grow-then-slide
%pattern with an uninformative prior. The forward info estimate feeding
%each call is the previous call's yFwdEnd output.
[yPrev,PInvPrev]=infoFilterUpdate(zeros(2,1),zeros(2,2),z(1),R,H);
yFwdPred=zeros(2,1);
PInvFwdPred=zeros(2,2);
for curStep=2:4
    zWin=z(:,max(1,curStep-2):curStep);
    [yInt,PInvInt,yFwdPred,PInvFwdPred,yPrev,PInvPrev]=FPInfoIntervalSmoother(yFwdPred,PInvFwdPred,yPrev,PInvPrev,3,zWin,R,H,F,Q);
    writematrix(yInt,fullfile(OUTPUT_DIR,sprintf('sm_fpis_y%d.csv',curStep)));
    PP=zeros(2*size(yInt,2),2);
    for k=1:size(yInt,2)
        PP((2*k-1):(2*k),:)=PInvInt(:,:,k);
    end
    writematrix(PP,fullfile(OUTPUT_DIR,sprintf('sm_fpis_Pinv%d.csv',curStep)));
end

%FIR smoother at kD=1, 2 and 4 (1-based) with control inputs.
HStack=repmat(H,[1,1,N]);
FStack=repmat(F,[1,1,N-1]);
RStack=repmat(R,[1,1,N]);
QStack=repmat(Q,[1,1,N-1]);
for kD=[1,2,4]
    [xEst,PEst]=KalmanFIRSmoother(z,u,HStack,FStack,RStack,QStack,kD);
    writematrix(xEst',fullfile(OUTPUT_DIR,sprintf('sm_fir_x%d.csv',kD)));
    writematrix(PEst,fullfile(OUTPUT_DIR,sprintf('sm_fir_P%d.csv',kD)));
end

disp('capture_smoothers done');
