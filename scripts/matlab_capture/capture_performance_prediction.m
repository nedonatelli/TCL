%CAPTURE_PERFORMANCE_PREDICTION Capture MATLAB TCL reference values for the
%Performance_Prediction ports in
%pytcl/dynamic_estimation/performance_prediction.py. All functions are
%deterministic given the inputs. The Riccati/FIM functions seed from
%RiccatiSolveD (qz-based); the port seeds from SciPy's DARE solver, whose
%stabilizing solution is the same matrix, so fixtures are compared at
%~1e-8 rather than machine precision. Two upstream defects shape this
%capture: trackPurityLinApprox's initial update parses as (H'*R)\H and
%crashes whenever zDim~=xDim, so the rectangular case is captured through
%trackPurityLinApproxFixed.m (a shim with the one-character precedence
%fix that the loop body itself uses) while the square H=eye case, where
%the two parses coincide, exercises the unmodified original;
%PCRLBPredAdd ignores user-supplied cubature points (nargin<8 in a
%seven-argument function), so its cubature fixture uses the default
%fifth-order points, which the port also defaults to. Inputs are mirrored
%verbatim in tests/validation/test_performance_prediction.py.

if(~exist('OUTPUT_DIR','var'))
    error('Set OUTPUT_DIR before running.');
end

%A 6-state, 3-measurement NCV model, the example used in the MATLAB
%docstrings themselves.
T=1;
F=FPolyKal(T,6,1);
q0=1;
Q=QPolyKal(T,6,1,q0);
R=diag([10;10;10]);
H=[1,0,0,0,0,0;
   0,1,0,0,0,0;
   0,0,1,0,0,0];

%Riccati solutions at PD=1 (pure DARE) and PD=0.5 (iterated).
P=RiccatiPredNoClutter(H,F,R,Q,1);
writematrix(P,fullfile(OUTPUT_DIR,'pp_ricpred_pd1.csv'));
P=RiccatiPredNoClutter(H,F,R,Q,0.5);
writematrix(P,fullfile(OUTPUT_DIR,'pp_ricpred_pd05.csv'));
P=RiccatiPostNoClutter(H,F,R,Q,1);
writematrix(P,fullfile(OUTPUT_DIR,'pp_ricpost_pd1.csv'));
P=RiccatiPostNoClutter(H,F,R,Q,0.5);
writematrix(P,fullfile(OUTPUT_DIR,'pp_ricpost_pd05.csv'));

%Asymptotic FIMs, explicit (nonsingular Q) branch.
J=FIMPostNoClutter(H,F,R,Q,1);
writematrix(J,fullfile(OUTPUT_DIR,'pp_fimpost_pd1.csv'));
J=FIMPostNoClutter(H,F,R,Q,0.5);
writematrix(J,fullfile(OUTPUT_DIR,'pp_fimpost_pd05.csv'));
J=FIMPredNoClutter(H,F,R,Q,0.5);
writematrix(J,fullfile(OUTPUT_DIR,'pp_fimpred_pd05.csv'));

%FIM with a singular Q (iterative branch): the discrete white noise
%acceleration model's rank-1 Q, a standard tracking model whose
%asymptotic FIM exists.
F2=[1, 1; 0, 1];
QSing=0.5*[1/4, 1/2; 1/2, 1];
H2=[1, 0];
R2=4;
J=FIMPostNoClutter(H2,F2,R2,QSing,0.9);
writematrix(J,fullfile(OUTPUT_DIR,'pp_fimpost_singq.csv'));

%PCRLB recursion, constant matrices: three predict/update cycles from
%zero information.
Q2=[0.02, 0.01; 0.01, 0.03];
J=zeros(2,2);
for k=1:3
    J=PCRLBPredAdd(J,[],[],Q2,F2);
    J=PCRLBUpdateAddNoClutter(J,[],[],R2,0.8,H2);
end
writematrix(J,fullfile(OUTPUT_DIR,'pp_pcrlb_const.csv'));

%PCRLB recursion with Jacobian functions averaged over a Gaussian prior
%via the default fifth-order cubature points (user-provided points are
%ignored upstream; see the header comment).
xPrior=[1;0.5];
PPrior=[0.09, 0.02; 0.02, 0.06];
FJacob=@(x)[1, 1; -0.1*x(1), 1];
HJacob=@(x)[2*x(1), 0.2];
J=PCRLBPredAdd(0.5*eye(2),xPrior,PPrior,Q2,FJacob);
writematrix(J,fullfile(OUTPUT_DIR,'pp_pcrlb_cubpred.csv'));
J=PCRLBUpdateAddNoClutter(0.5*eye(2),xPrior,PPrior,R2,0.8,HJacob);
writematrix(J,fullfile(OUTPUT_DIR,'pp_pcrlb_cubupd.csv'));
%Jacobian handle with a zero covariance evaluates at the mean only.
J=PCRLBUpdateAddNoClutter(0.5*eye(2),xPrior,zeros(2,2),R2,0.8,HJacob);
writematrix(J,fullfile(OUTPUT_DIR,'pp_pcrlb_jacmean.csv'));

%Correct-association probability and track purity. The square H=eye case
%runs the unmodified original (both parses of the initial update
%coincide there); the rectangular case needs the precedence-fixed shim.
Pc=correctAssocProbApprox(3,1e-4,det(R));
writematrix(Pc,fullfile(OUTPUT_DIR,'pp_assoc_prob.csv'));

[Pc,PInv]=trackPurityLinApprox(2e-3,eye(2),F2,[4, 0; 0, 9],Q2,10,eye(2));
writematrix(Pc,fullfile(OUTPUT_DIR,'pp_purity_sq_pc.csv'));
writematrix(PInv,fullfile(OUTPUT_DIR,'pp_purity_sq_pinv.csv'));
[Pc,PInv]=trackPurityLinApproxFixed(2e-3,H2,F2,R2,Q2,10,eye(2));
writematrix(Pc,fullfile(OUTPUT_DIR,'pp_purity_rect_pc.csv'));
writematrix(PInv,fullfile(OUTPUT_DIR,'pp_purity_rect_pinv.csv'));

%Untrackability, both input forms, on both sides of the threshold.
S=diag([100;100]);
vals=zeros(3,1);
vals(1)=linTargetIsUntrackable(S,0.5,1e-8);
vals(2)=linTargetIsUntrackable(S,0.5,10);
targetParams=struct('H',H2,'F',F2,'R',R2,'Q',Q2);
vals(3)=linTargetIsUntrackable(targetParams,0.5,1e-6);
writematrix(vals,fullfile(OUTPUT_DIR,'pp_untrackable.csv'));

%Prior model: generic form and the 3-, 6- and 9-state polynomial forms.
[xHat,P]=DiscPriorPModel(3,[0;1],F2,Q2,1);
writematrix(xHat',fullfile(OUTPUT_DIR,'pp_prior_gen_x.csv'));
writematrix(P,fullfile(OUTPUT_DIR,'pp_prior_gen_P.csv'));
xInit6=(1:6)';
[xHat,P]=DiscPriorPModel(4,xInit6,0.5,2,0);
writematrix(xHat',fullfile(OUTPUT_DIR,'pp_prior_ncv_x.csv'));
writematrix(P,fullfile(OUTPUT_DIR,'pp_prior_ncv_P.csv'));
xInit9=(1:9)';
[xHat,P]=DiscPriorPModel(2,xInit9,0.5,2,0);
writematrix(xHat',fullfile(OUTPUT_DIR,'pp_prior_nca_x.csv'));
writematrix(P,fullfile(OUTPUT_DIR,'pp_prior_nca_P.csv'));
xInit3=[1;2;3];
[xHat,P]=DiscPriorPModel(5,xInit3,0.5,2,0);
writematrix(xHat',fullfile(OUTPUT_DIR,'pp_prior_pos_x.csv'));
writematrix(P,fullfile(OUTPUT_DIR,'pp_prior_pos_P.csv'));

disp('capture_performance_prediction done');
