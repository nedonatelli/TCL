%CAPTURE_POLY_ESTIMATORS Capture MATLAB TCL reference values for the
%polynomial-solver-based Static_Estimation functions, ported in
%pytcl/static_estimation/localization.py:
%  TDOA2Cart -> tdoa_to_cart
%  rangeRate2StaticPos -> range_rate_to_static_pos
%  rangeRateRatio2StaticPos2D -> range_rate_ratio_to_static_pos_2d
%Each fixture holds the filtered real solution set (dim rows, numSol
%columns); the Python tests compare as sets, order-independently.
%Inputs are mirrored verbatim in
%tests/validation/test_static_localization.py.

if(~exist('OUTPUT_DIR','var'))
    error('Set OUTPUT_DIR before running.');
end

%% TDOA2Cart, 3D docstring example.
S1=[9;39;100];S2=[65;10;-60];S3=[64;71;43];S4=[-128;6;12];
c=341;
t=[27;0;-42];
lRx1=[S1,S1,S4];
lRx2=[S2,S3,S3];
TDOA=zeros(3,1);
TDOA(1)=(norm(t-S2)-norm(t-S1))/c;
TDOA(2)=(norm(t-S3)-norm(t-S1))/c;
TDOA(3)=(norm(t-S3)-norm(t-S4))/c;
[zCart,exitCode]=TDOA2Cart(TDOA,lRx1,lRx2,c);
assert(exitCode==0);
writematrix(zCart,fullfile(OUTPUT_DIR,'se_tdoa2cart_3d.csv'));

%% TDOA2Cart, 2D with a shared reference (single lRx1 column).
t2=[350;-125];
R1=[0;0];R2=[800;200];R3=[-300;700];
TDOA2=zeros(2,1);
TDOA2(1)=(norm(t2-R2)-norm(t2-R1))/c;
TDOA2(2)=(norm(t2-R3)-norm(t2-R1))/c;
[zCart,exitCode]=TDOA2Cart(TDOA2,R1,[R2,R3],c);
assert(exitCode==0);
writematrix(zCart,fullfile(OUTPUT_DIR,'se_tdoa2cart_2d.csv'));

%% rangeRate2StaticPos, 2D docstring example.
uTrue=[1e3;5e3];
s=[500, 1100; 2500, 2500];
sDot=[300, 300; 0, 0];
stateRx=[s;sDot];
rr=zeros(2,1);
for m=1:2
    rr(m)=-sDot(:,m)'*(uTrue-s(:,m))/norm(uTrue-s(:,m));
end
[zCart,exitCode]=rangeRate2StaticPos(rr,stateRx);
assert(exitCode==0);
writematrix(zCart,fullfile(OUTPUT_DIR,'se_rr2pos_2d.csv'));

%% rangeRate2StaticPos, 3D docstring example.
uTrue3=[1e3;5e3;2e3];
s3=[500, 1100, 5000; 3000, 1000, -1000; 0, 2000, 1000];
sDot3=[300, 100, 0; 0, 100, 300; 0, 0, 0];
stateRx3=[s3;sDot3];
rr3=zeros(3,1);
for m=1:3
    rr3(m)=-sDot3(:,m)'*(uTrue3-s3(:,m))/norm(uTrue3-s3(:,m));
end
[zCart,exitCode]=rangeRate2StaticPos(rr3,stateRx3);
assert(exitCode==0);
writematrix(zCart,fullfile(OUTPUT_DIR,'se_rr2pos_3d.csv'));

%% rangeRateRatio2StaticPos2D, docstring example.
lRx1e=[1000;3000];
lRx1Dot=[150;-150];
lRxe=[500, 1100; 2500, 2500];
lRxDot=[300, 300; 0, 0];
sRRef=[lRx1e;lRx1Dot];
sRx=[lRxe;lRxDot];
rr1=-lRx1Dot'*(uTrue-lRx1e)/norm(uTrue-lRx1e);
rrK=zeros(2,1);
for m=1:2
    rrK(m)=-lRxDot(:,m)'*(uTrue-lRxe(:,m))/norm(uTrue-lRxe(:,m));
end
cLight=Constants.speedOfLight;
fRat=(1-rr1/cLight)./(1-rrK/cLight);
[zCart,exitCode]=rangeRateRatio2StaticPos2D(fRat,sRRef,sRx);
assert(exitCode==0);
writematrix(zCart,fullfile(OUTPUT_DIR,'se_rrratio_2d.csv'));

disp('capture_poly_estimators done');
