%CAPTURE_INFO_FILTER_PRED Capture MATLAB TCL reference values for the
%information-filter prediction step from a singular information matrix,
%ported as the singular-Y branch of information_filter in
%pytcl/dynamic_estimation/information_filter.py.
%
%infoFilterDiscPred (Dynamic_Estimation/State_Propagation/Discrete_Time/
%infoFilterDiscPred.m:47-58) needs only F invertible; neither Q nor PInvPrev
%is inverted, so the cases cover exactly singular Q and Y.
%
%Format: one file per case, %.17g. Row 1 is yPred', rows 2..end are
%PInvPred. Inputs are mirrored verbatim in
%tests/validation/test_information_filter_guards.py.

if(~exist('OUTPUT_DIR','var'))
    error('Set OUTPUT_DIR before running.');
end

F2=[1,1;0,1];
Y2=diag([1,0]);
y2=[0.5;0];

names={'q0','q1e-6','q1e-12','q1e-15','q1e-2','qdiag'};
Qs={zeros(2),1e-6*eye(2),1e-12*eye(2),1e-15*eye(2),1e-2*eye(2),diag([0,1e-2])};
for k=1:numel(names)
    [yPred,PInvPred]=infoFilterDiscPred(y2,Y2,F2,Qs{k});
    writeCase(OUTPUT_DIR,names{k},yPred,PInvPred);
end

[yPred,PInvPred]=infoFilterDiscPred(zeros(2,1),zeros(2),F2,1e-2*eye(2));
writeCase(OUTPUT_DIR,'y0',yPred,PInvPred);

%4-state constant velocity, rank-2 information, rank-1 process noise.
F4=[1,0,0.5,0;
    0,1,0,0.5;
    0,0,1,0;
    0,0,0,1];
A=[1,0.2;
   0.3,1.5;
   0.1,0;
   0,0.4];
Y4=A*A';
y4=Y4*[1;-2;0.5;0.3];
g=[0.125;0.125;0.5;0.5];
Q4=0.1*(g*g');
[yPred,PInvPred]=infoFilterDiscPred(y4,Y4,F4,Q4);
writeCase(OUTPUT_DIR,'4state',yPred,PInvPred);

function writeCase(outDir,name,yPred,PInvPred)
    M=[yPred';PInvPred];
    fid=fopen(fullfile(outDir,['info_pred_',name,'.csv']),'w');
    for r=1:size(M,1)
        fprintf(fid,'%.17g',M(r,1));
        fprintf(fid,',%.17g',M(r,2:end));
        fprintf(fid,'\n');
    end
    fclose(fid);
end
