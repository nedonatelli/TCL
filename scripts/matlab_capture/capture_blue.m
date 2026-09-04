%CAPTURE_BLUE Capture MATLAB TCL reference values for the BLUE
%polar/spherical measurement updates, ported as
%blue_polar_meas_update / blue_spher_meas_update in
%pytcl/dynamic_estimation/kalman/blue.py. Inputs are mirrored verbatim
%in tests/validation/test_blue_updates.py.

if(~exist('OUTPUT_DIR','var'))
    error('Set OUTPUT_DIR before running.');
end

xP=[1000;500;10;-5];
PP=[100,10,5,0; 10,120,0,8; 5,0,25,3; 0,8,3,30];
z2=[hypot(1010,495); atan2(495,1010)];
R2=diag([25;1e-4]);
[xU,PU,innov,Pzz,W]=BLUEPolarMeasUpdateApprox(xP,PP,z2,R2);
writematrix(xU',fullfile(OUTPUT_DIR,'blue_polar_x.csv'));
writematrix(PU,fullfile(OUTPUT_DIR,'blue_polar_P.csv'));
writematrix(innov',fullfile(OUTPUT_DIR,'blue_polar_innov.csv'));
writematrix(Pzz,fullfile(OUTPUT_DIR,'blue_polar_Pzz.csv'));
writematrix(W,fullfile(OUTPUT_DIR,'blue_polar_W.csv'));
xP6=[2000;1000;500;10;-5;2];
PP6=diag([100;110;90;25;30;20]);
PP6(1,2)=15; PP6(2,1)=15; PP6(1,4)=5; PP6(4,1)=5; PP6(3,6)=4; PP6(6,3)=4;
truePos=[2010;995;505];
rng_=norm(truePos);
z3=[rng_; atan2(995,2010); asin(505/rng_)];
R3=diag([25;1e-4;1e-4]);
[xU,PU,innov,S,W]=BLUESpherMeasUpdateApprox(xP6,PP6,z3,R3);
writematrix(xU',fullfile(OUTPUT_DIR,'blue_spher_x.csv'));
writematrix(PU,fullfile(OUTPUT_DIR,'blue_spher_P.csv'));
writematrix(innov',fullfile(OUTPUT_DIR,'blue_spher_innov.csv'));
writematrix(S,fullfile(OUTPUT_DIR,'blue_spher_S.csv'));
writematrix(W,fullfile(OUTPUT_DIR,'blue_spher_W.csv'));
disp('capture_blue done');
