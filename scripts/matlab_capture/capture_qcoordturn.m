%CAPTURE_QCOORDTURN Capture MATLAB TCL reference values for the
%coordinated-turn process noise covariance, ported as q_coord_turn_2d and
%q_coord_turn_3d in pytcl/dynamic_models/process_noise/coordinated_turn.py.
%
%Captured with MATLAB R2026a against matlab-tcl a9acd8f.
%QCoordTurn (Dynamic_Models/Discrete_Time/Process_Noise_Covariance_Matrices/
%QCoordTurn.m) dispatches on length(x): case 5 is 2-D [x;y;xdot;ydot;omega],
%case 7 is 3-D [x;y;z;xdot;ydot;zdot;omega]. Both build Q=G*G'. The linear
%acceleration cases (6 and 8) have no pytcl counterpart and are not captured.
%
%Writes qcoordturn_case5.csv and qcoordturn_case7.csv. Each data row is
%T, sigmaV2, sigmaTurn2, then Q in MATLAB's own state order flattened
%row-major, %.17g. Lines starting with # are provenance.

if(~exist('OUTPUT_DIR','var'))
    error('Set OUTPUT_DIR before running.');
end

Ts=[0.1,1,3.7];
sigmaV2s=[1,4];
sigmaTurn2s=[0,0.01];

for n=[5,7]
    fid=fopen(fullfile(OUTPUT_DIR,sprintf('qcoordturn_case%d.csv',n)),'w');
    fprintf(fid,'# QCoordTurn(T,zeros(%d,1),sigmaV2,sigmaTurn2), matlab-tcl a9acd8f, MATLAB %s\n',n,version);
    fprintf(fid,'# captured by scripts/matlab_capture/capture_qcoordturn.m (matlab -batch)\n');
    fprintf(fid,'# columns: T,sigmaV2,sigmaTurn2,Q(1,1:%d),...,Q(%d,1:%d) in MATLAB state order\n',n,n,n);
    for T=Ts
        for sigmaV2=sigmaV2s
            for sigmaTurn2=sigmaTurn2s
                Q=QCoordTurn(T,zeros(n,1),sigmaV2,sigmaTurn2);
                row=[T,sigmaV2,sigmaTurn2,reshape(Q.',1,[])];
                fprintf(fid,'%.17g',row(1));
                fprintf(fid,',%.17g',row(2:end));
                fprintf(fid,'\n');
            end
        end
    end
    fclose(fid);
end
