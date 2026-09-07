%CAPTURE_COVARIANCE_CONVERSIONS Capture MATLAB TCL reference values for
%the measurement conversions with covariances ported in
%pytcl/coordinate_systems/conversions/covariance_conversions.py. All
%conversions are deterministic given the cubature points. MATLAB's
%fifthOrderCubPoints and pytcl's fifth_order_cubature_points implement
%DIFFERENT degree-5 rules (a tensor Gauss-Hermite special case at n=2 and
%a 2^n+2n rule at n>=3 versus Stroud's 2n^2+1 rule), so default-point
%results agree only to the rules' truncation error on non-polynomial
%conversions; the cubature fixtures therefore pass pytcl's exported
%points (cc_xi2/cc_w2, cc_xi3/cc_w3) explicitly to both sides, making
%them machine-precision. One upstream defect shapes this capture: with several measurements and
%useHalfRange=false, monostatRuv2CartTaylor halves only the FIRST
%measurement's range while scaling every covariance, so the
%useHalfRange=false fixtures are captured one measurement at a time
%(where the original is correct) and the port's batched path is
%validated against those stacked single calls. Inputs are mirrored
%verbatim in tests/validation/test_covariance_conversions.py.

if(~exist('OUTPUT_DIR','var'))
    error('Set OUTPUT_DIR before running.');
end

xi2=readmatrix(fullfile(OUTPUT_DIR,'cc_xi2.csv'));
w2=readmatrix(fullfile(OUTPUT_DIR,'cc_w2.csv'));
xi3=readmatrix(fullfile(OUTPUT_DIR,'cc_xi3.csv'));
w3=readmatrix(fullfile(OUTPUT_DIR,'cc_w3.csv'));

%uv -> spherical angles, all four system types, with rotated frames.
zUV=[0.3, -0.2, 0.05;
     0.4,  0.1, -0.35];
RUV=[2e-3, 5e-4; 5e-4, 3e-3];
SRUV=chol(RUV,'lower');
%Fixed (non-random) rotations: about z by 0.3 rad and about x by -0.2 rad.
Ms=[cos(0.3), -sin(0.3), 0; sin(0.3), cos(0.3), 0; 0, 0, 1];
Muv=[1, 0, 0; 0, cos(-0.2), -sin(-0.2); 0, sin(-0.2), cos(-0.2)];
for sysType=0:3
    [azEl,RAzEl]=uv2SpherAngCubature(zUV,SRUV,sysType,Ms,Muv,xi2,w2);
    writematrix(azEl,fullfile(OUTPUT_DIR,sprintf('cc_uv2sph_z%d.csv',sysType)));
    PP=zeros(2*3,2);
    for k=1:3
        PP((2*k-1):(2*k),:)=RAzEl(:,:,k);
    end
    writematrix(PP,fullfile(OUTPUT_DIR,sprintf('cc_uv2sph_R%d.csv',sysType)));
end

%Bistatic ruv -> ruv, from the function's own example geometry, for
%includeW=0, 1 and 2.
pointCart=1e4*[1.173024396049598;
              -4.844345843776918;
               3.969150579064054];
RRuv=[50^2, 0, 0; 0, 1e-3^2, 0; 0, 0, 1e-3^2];
SRuv=chol(RRuv,'lower');
xTx1=[0;0;0];
xRx1=1e4*[1;2;0];
xTx2=xTx1;
xRx2=xTx1;
M1=eye(3,3);
M2=[cos(0.4), sin(0.4), 0; -sin(0.4), cos(0.4), 0; 0, 0, 1];
zRuv1=Cart2Ruv(pointCart,false,xTx1,xRx1,M1);
for includeW=0:2
    [zConv,RConv]=ruv2RuvCubature(zRuv1,SRuv,false,xTx1,xRx1,M1,xTx2,xRx2,M2,includeW,xi3,w3);
    writematrix(zConv,fullfile(OUTPUT_DIR,sprintf('cc_ruv2ruv_z%d.csv',includeW)));
    writematrix(RConv(:,:,1),fullfile(OUTPUT_DIR,sprintf('cc_ruv2ruv_R%d.csv',includeW)));
end
%Two measurements with per-measurement covariances and the two-element
%useHalfRange form (source channel two-way, destination one-way),
%includeW=0.
zRuv2=Cart2Ruv(1e4*[0.5;-2.1;1.7],false,xTx1,xRx1,M1);
zBoth=[zRuv1,zRuv2];
SBoth=cat(3,SRuv,chol(diag([30^2;2e-3^2;2e-3^2]),'lower'));
[zConv,RConv]=ruv2RuvCubature(zBoth,SBoth,[false;true],xTx1,xRx1,M1,xTx2,xRx2,M2,0,xi3,w3);
writematrix(zConv,fullfile(OUTPUT_DIR,'cc_ruv2ruv_zmix.csv'));
writematrix([RConv(:,:,1);RConv(:,:,2)],fullfile(OUTPUT_DIR,'cc_ruv2ruv_Rmix.csv'));

%Camera image-plane coordinates -> uv.
f=35e-3;
A=diag([f,f,1]);
zCam=[1e-2, -7e-3;
      -2e-2, 1.4e-2];
SRCam=diag([1e-4,1e-4]);
[zUVc,RUVc]=cameraCoords2UVCoordsCubature(zCam,SRCam,A,xi2,w2);
writematrix(zUVc,fullfile(OUTPUT_DIR,'cc_cam2uv_z.csv'));
writematrix([RUVc(:,:,1);RUVc(:,:,2)],fullfile(OUTPUT_DIR,'cc_cam2uv_R.csv'));

%Monostatic ruv -> Cartesian Taylor conversions, all four algorithms
%(3 is the uncorrected CM2, present but undocumented upstream), with a
%rotated receiver frame and offset. useHalfRange=true batch works
%upstream; useHalfRange=false is captured per measurement (see header).
zRuvM=[100e3, 80e3;
       0.5,  -0.3;
       0.2,   0.4];
RM=diag([1;1e-3^2;5e-3^2]);
M=[cos(0.25), 0, -sin(0.25); 0, 1, 0; sin(0.25), 0, cos(0.25)];
zRx=[100;-200;50];
for alg=0:3
    [zC,RC]=monostatRuv2CartTaylor(zRuvM,RM,true,zRx,M,alg);
    writematrix(zC,fullfile(OUTPUT_DIR,sprintf('cc_ruv2cart_z%d.csv',alg)));
    writematrix([RC(:,:,1);RC(:,:,2)],fullfile(OUTPUT_DIR,sprintf('cc_ruv2cart_R%d.csv',alg)));
    %Two-way-range inputs, one measurement at a time.
    zTW1=zRuvM(:,1); zTW1(1)=2*zTW1(1);
    zTW2=zRuvM(:,2); zTW2(1)=2*zTW2(1);
    [zC1,RC1]=monostatRuv2CartTaylor(zTW1,RM,false,zRx,M,alg);
    [zC2,RC2]=monostatRuv2CartTaylor(zTW2,RM,false,zRx,M,alg);
    writematrix([zC1,zC2],fullfile(OUTPUT_DIR,sprintf('cc_ruv2cart_tw_z%d.csv',alg)));
    writematrix([RC1;RC2],fullfile(OUTPUT_DIR,sprintf('cc_ruv2cart_tw_R%d.csv',alg)));
end

disp('capture_covariance_conversions done');
