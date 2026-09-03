%CAPTURE_STATIC_ESTIMATION Capture MATLAB TCL reference values for the
%polynomial-free Static_Estimation localization functions, ported in
%pytcl/static_estimation/localization.py:
%  TDOAOnlyStaticLocEst -> tdoa_only_static_loc_est
%  rangeOnlyStaticLocEstNP -> range_only_static_loc_est_np
%  RROnlyStaticVelEst -> rr_only_static_vel_est
%  getAdHocCartCov -> ad_hoc_cart_cov
%plus rotAxis2Vec (method 0) -> rot_axis_to_vec in
%pytcl/coordinate_systems/rotations. Every input here is mirrored
%verbatim in tests/validation/test_static_localization.py.

if(~exist('OUTPUT_DIR','var'))
    error('Set OUTPUT_DIR before running.');
end

%% TDOAOnlyStaticLocEst
S1=[9;39;100];S2=[65;10;-60];S3=[64;71;43];S4=[-128;6;12];S5=[0;-20;4];
c=341;
t=[27;0;-42];
lRx2=[S2,S3,S4,S5];
TDOA=zeros(4,1);
for k=1:4
    TDOA(k)=(norm(t-lRx2(:,k))-norm(t-S1))/c;
end
out=TDOAOnlyStaticLocEst(TDOA,S1,lRx2,c);
writematrix(out',fullfile(OUTPUT_DIR,'se_tdoa_only_case1.csv'));

%Perturbed delays (fixed, deterministic).
TDOANoisy=TDOA+[2e-4;-1e-4;3e-4;-2e-4];
out=TDOAOnlyStaticLocEst(TDOANoisy,S1,lRx2,c);
writematrix(out',fullfile(OUTPUT_DIR,'se_tdoa_only_case2.csv'));

%Two reference receivers, cell-array form. Ref 1 is S1 with {S2,S3,S4};
%ref 2 is S5 with {S2,S3}.
delays1=zeros(3,1);
nonRef1=[S2,S3,S4];
for k=1:3
    delays1(k)=(norm(t-nonRef1(:,k))-norm(t-S1))/c;
end
delays2=zeros(2,1);
nonRef2=[S2,S3];
for k=1:2
    delays2(k)=(norm(t-nonRef2(:,k))-norm(t-S5))/c;
end
out=TDOAOnlyStaticLocEst({delays1;delays2},[S1,S5],{nonRef1;nonRef2},c);
writematrix(out',fullfile(OUTPUT_DIR,'se_tdoa_only_case3.csv'));

%% rangeOnlyStaticLocEstNP
tLoc=[4e3;-2e3;3e3];
zRx=[100;200;-50];
zTx5=[0,   8e3, -6e3, 2e3, -3e3;
      0,   1e3,  5e3, -7e3, 2e3;
      0,  -2e3,  1e3,  4e3, 9e3];
rB5=zeros(5,1);
for k=1:5
    rB5(k)=norm(tLoc-zTx5(:,k))+norm(tLoc-zRx);
end
%Spherical intersection (method 1), overdetermined.
out=rangeOnlyStaticLocEstNP(rB5,zTx5,zRx,1);
writematrix(out',fullfile(OUTPUT_DIR,'se_range_only_case1.csv'));
%Spherical interpolation (method 0), overdetermined.
out=rangeOnlyStaticLocEstNP(rB5,zTx5,zRx,0);
writematrix(out',fullfile(OUTPUT_DIR,'se_range_only_case2.csv'));
%Minimal system (3 measurements): two solution columns.
zTx3=zTx5(:,1:3);
rB3=rB5(1:3);
out=rangeOnlyStaticLocEstNP(rB3,zTx3,zRx,1);
writematrix(out,fullfile(OUTPUT_DIR,'se_range_only_case3.csv'));
%Noisy overdetermined with covariance outputs.
rB5n=rB5+[3;-5;2;-1;4];
RCov=diag([9;25;4;1;16]);
[xEst,PTaylor,PCRLB]=rangeOnlyStaticLocEstNP(rB5n,zTx5,zRx,1,RCov);
writematrix([xEst',reshape(PTaylor(:,:,1),1,9),reshape(PCRLB(:,:,1),1,9)],...
    fullfile(OUTPUT_DIR,'se_range_only_case4.csv'));

%% RROnlyStaticVelEst
zTar=[0;40e3;40e3];
vTar=[400;-200;100];
xTx1=[100;10e3;3e3;50;50;-50];
xTx2=[0;0;0;0;0;-20];
xTx3=[10e3;10e3;3e3;100;-100;100];
xRx1=[-10e3;0;3e3;100;100;100];
xRx2=[0;10e3;30;-80;-200;-20];
xRx3=xRx2;
states1=[xTx1,xTx2,xTx3];
states2=[xRx1,xRx2,xRx3];
xTar=[zTar;vTar];
rr=zeros(3,1);
rr(1)=getRangeRate(xTar,false,xTx1,xRx1);
rr(2)=getRangeRate(xTar,false,xTx2,xRx2);
rr(3)=getRangeRate(xTar,false,xTx3,xRx3);
out=RROnlyStaticVelEst(rr,states1,states2,zTar,false);
writematrix([rr',out'],fullfile(OUTPUT_DIR,'se_rr_only_case1.csv'));
%Same geometry, useHalfRange=true (halve the range rates).
out=RROnlyStaticVelEst(rr/2,states1,states2,zTar,true);
writematrix(out',fullfile(OUTPUT_DIR,'se_rr_only_case2.csv'));

%2D case.
zTar2=[0;40e3];
vTar2=[400;-200];
xTx2d=[[100;10e3;50;50],[0;0;0;0],[10e3;10e3;100;-100]];
xRx2d=[[-10e3;0;100;100],[0;10e3;-80;-200],[0;10e3;-80;-200]];
xTar2=[zTar2;vTar2];
rr2=zeros(3,1);
for k=1:3
    rr2(k)=getRangeRate(xTar2,false,xTx2d(:,k),xRx2d(:,k));
end
out=RROnlyStaticVelEst(rr2,xTx2d,xRx2d,zTar2,false);
writematrix([rr2',out'],fullfile(OUTPUT_DIR,'se_rr_only_case3.csv'));

%Target is the transmitter (empty xTx), 5 receivers, fixed states.
zTarE=[1.5;-0.4;2.2];
vTarE=[0.3;1.1;-0.7];
xRxE=[ 0.5, -1.2,  2.0,  0.0, -0.8;
       1.0,  0.3, -1.5,  2.2,  0.7;
      -0.6,  1.8,  0.4, -1.0,  1.3;
       0.1, -0.5,  0.7,  0.2, -0.3;
      -0.2,  0.4,  0.1, -0.6,  0.5;
       0.3,  0.2, -0.4,  0.1, -0.1];
xTarE=[zTarE;vTarE];
rrE=zeros(5,1);
for k=1:5
    rrE(k)=getRangeRate(xTarE,false,xTarE,xRxE(:,k));
end
out=RROnlyStaticVelEst(rrE,[],xRxE,zTarE,false);
writematrix([rrE',out'],fullfile(OUTPUT_DIR,'se_rr_only_case4.csv'));

%% getAdHocCartCov
V=getAdHocCartCov(5e6,[deg2rad(2),deg2rad(10)],10,[1e3;1e3;1e3]);
writematrix(V,fullfile(OUTPUT_DIR,'se_adhoc_cov_case1.csv'));
V=getAdHocCartCov(2e6,deg2rad(3),15,[-2e3;500;1e3]);
writematrix(V,fullfile(OUTPUT_DIR,'se_adhoc_cov_case2.csv'));
V=getAdHocCartCov(5e6,deg2rad(2),10,[3e3;-4e3]);
writematrix(V,fullfile(OUTPUT_DIR,'se_adhoc_cov_case3.csv'));

%% rotAxis2Vec, method 0 (the default Householder path)
u=[1;2;3];
for ax=1:3
    R=rotAxis2Vec(u,ax);
    writematrix(R,fullfile(OUTPUT_DIR,sprintf('rot_axis2vec_3d_ax%d.csv',ax)));
end
R=rotAxis2Vec([-1;0.5;-2],1);
writematrix(R,fullfile(OUTPUT_DIR,'rot_axis2vec_3d_neg.csv'));
R=rotAxis2Vec([3;-4],2);
writematrix(R,fullfile(OUTPUT_DIR,'rot_axis2vec_2d.csv'));
uVec=[53;183;-225;86;31;-130;-43;34];
R=rotAxis2Vec(uVec,5);
writematrix(R,fullfile(OUTPUT_DIR,'rot_axis2vec_8d.csv'));

disp('capture_static_estimation done');
