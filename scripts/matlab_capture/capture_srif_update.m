%CAPTURE_SRIF_UPDATE Capture MATLAB TCL reference values for the linear
%square-root information filter measurement update, ported as srif_update
%in pytcl/dynamic_estimation/information_filter.py.
%
%Captured with MATLAB R2026a against matlab-tcl a9acd8f.
%sqrtInfoFilterUpdate (Dynamic_Estimation/Measurement_Update/
%Complete_Measurement_Updates/Square_Root_Filters/sqrtInfoFilterUpdate.m:
%67-75) whitens the measurement with SR\H and SR\z, SR the lower Cholesky
%factor of R. Both cases use a correlated (non-diagonal) R.
%
%QR sign conventions differ between LAPACK builds, so each file stores the
%sign-invariant quantities: row 1 is the updated state x' (T\ySqrt), rows
%2..end the updated information matrix T'*T. %.17g. Inputs are mirrored
%verbatim in tests/validation/test_information_filter_guards.py.

if(~exist('OUTPUT_DIR','var'))
    error('Set OUTPUT_DIR before running.');
end

%2-state, full-state measurement.
P0=2*eye(2);
x0=[1;0.5];
H=eye(2);
z=[1.2;0.4];
R=[2,1;1,1];
captureCase(OUTPUT_DIR,'2state',P0,x0,z,R,H);

%3-state, two correlated measurements of a correlated prior.
P0=[1,0.2,0.1;0.2,2,0.3;0.1,0.3,0.5];
x0=[1;-0.4;0.3];
H=[1,0,0.5;0,1,-0.2];
z=[1.6;-0.7];
R=[0.5,0.3;0.3,0.4];
captureCase(OUTPUT_DIR,'3state',P0,x0,z,R,H);

function captureCase(outDir,name,P0,x0,z,R,H)
    PInvSqrt=chol(inv(P0));
    ySqrt=PInvSqrt*x0;
    SR=chol(R,'lower');
    [ySqrtUp,PInvSqrtUp]=sqrtInfoFilterUpdate(ySqrt,PInvSqrt,z,SR,H);
    M=[(PInvSqrtUp\ySqrtUp)';PInvSqrtUp'*PInvSqrtUp];
    fid=fopen(fullfile(outDir,['srif_update_',name,'.csv']),'w');
    for r=1:size(M,1)
        fprintf(fid,'%.17g',M(r,1));
        fprintf(fid,',%.17g',M(r,2:end));
        fprintf(fid,'\n');
    end
    fclose(fid);
end
