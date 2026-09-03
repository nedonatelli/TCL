%CAPTURE_UV_COORDS Capture MATLAB TCL reference values for the u-v
%direction-cosine coordinate functions, ported in
%pytcl/coordinate_systems/conversions/uv.py:
%  uv2SpherAng -> uv2spher_ang        spherAng2Uv -> spher_ang2uv
%  ruv2Cart -> ruv2cart_bistatic      Cart2Ruv -> cart2ruv_bistatic
%  ruv2Ruv -> ruv2ruv                 stateRuv2Cart -> state_ruv2cart
%  cameraCoords2UVCoords -> camera_coords2uv
%Inputs are mirrored verbatim in tests/validation/test_uv_coordinates.py.

if(~exist('OUTPUT_DIR','var'))
    error('Set OUTPUT_DIR before running.');
end

%% uv2SpherAng / spherAng2Uv across all four system types, with rotations.
uvIn=[0.3, -0.5, 0.1,  0.62;
      0.4,  0.2, -0.7, -0.05];
Ms=rotAxis2Vec([1;2;3]/norm([1;2;3]),'z');
Muv=rotAxis2Vec([-1;1;4]/norm([-1;1;4]),'z');
for st=0:3
    azEl=uv2SpherAng(uvIn,st,Ms,Muv);
    writematrix(azEl,fullfile(OUTPUT_DIR,sprintf('uv2spher_st%d.csv',st)));
    uvBack=spherAng2Uv(azEl,st,true,Ms,Muv);
    writematrix(uvBack,fullfile(OUTPUT_DIR,sprintf('spher2uv_st%d.csv',st)));
end

%% Bistatic ruv2Cart / Cart2Ruv with offsets and rotation.
M=rotAxis2Vec([0.2;-0.1;0.97],'z');
zC=[100, 2000, -500;
     50, -300,  900;
    800, 1200,  300];
zTx=[10;20;5];
zRx=[-15;8;2];
z=Cart2Ruv(zC,false,zTx,zRx,M,true);
writematrix(z,fullfile(OUTPUT_DIR,'cart2ruv_bistatic.csv'));
zC2=ruv2Cart(z,false,zTx,zRx,M);
writematrix(zC2,fullfile(OUTPUT_DIR,'ruv2cart_bistatic.csv'));
%Half-range monostatic without w.
zHalf=Cart2Ruv(zC,true,zRx,zRx,M,false);
writematrix(zHalf,fullfile(OUTPUT_DIR,'cart2ruv_half.csv'));
zC3=ruv2Cart(zHalf,true,zRx,zRx,M);
writematrix(zC3,fullfile(OUTPUT_DIR,'ruv2cart_half.csv'));

%% ruv2Ruv between two displaced, rotated bistatic pairs.
M2=rotAxis2Vec([0.5;0.5;0.7071],'z');
zTx2=[-30;12;9];
zRx2=[25;-18;4];
zNew=ruv2Ruv(z,false,zTx,zRx,M,zTx2,zRx2,M2);
writematrix(zNew,fullfile(OUTPUT_DIR,'ruv2ruv_pair.csv'));

%% stateRuv2Cart, 6- and 9-element states.
x6=[1000; 0.3; -0.4; 12; 1e-3; -2e-3];
writematrix(stateRuv2Cart(x6),fullfile(OUTPUT_DIR,'state_ruv2cart_6.csv'));
x9=[1000; 0.3; -0.4; 12; 1e-3; -2e-3; 0.5; 1e-5; 2e-5];
writematrix(stateRuv2Cart(x9),fullfile(OUTPUT_DIR,'state_ruv2cart_9.csv'));

%% cameraCoords2UVCoords with a skewed intrinsics matrix and rotation.
A=[500, 2, 320;
     0, 480, 240;
     0, 0, 1];
zCam=[320, 100, 500;
      240,  50, 400];
MCam=rotAxis2Vec([0.1;0.2;0.97],'z');
d=cameraCoords2UVCoords(zCam,A,MCam,true);
writematrix(d,fullfile(OUTPUT_DIR,'camera2uv.csv'));

disp('capture_uv_coords done');
