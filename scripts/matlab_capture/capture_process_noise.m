%CAPTURE_PROCESS_NOISE Capture MATLAB TCL reference values for the
%polynomial and Gauss-Markov process noise covariances, ported in
%pytcl/dynamic_models/process_noise/{polynomial,singer}.py:
%  QPolyKal           -> q_poly_kal, q_continuous_white_noise
%  QPolyKalDirectDisc -> q_discrete_white_noise, q_constant_velocity,
%                        q_constant_acceleration
%  QGaussMarkov       -> q_singer (order 2)
%
%Captured with MATLAB R2026a against matlab-tcl a9acd8f. All files use
%numDim=2, so MATLAB's state is derivative-major [x;y;vx;vy;...] (kron(Q1,
%diag(q0))). Each data row is T, q, then Q in MATLAB's own state order
%flattened row-major, %.17g (QGaussMarkov rows: T, tau, q, Q). Lines
%starting with # are provenance.

if(~exist('OUTPUT_DIR','var'))
    error('Set OUTPUT_DIR before running.');
end

numDim=2;
Ts=[0.1,1,3.7];
qs=[0.5,4];

for order=0:3
    fid=open_fixture(OUTPUT_DIR,sprintf('qpolykal_order%d.csv',order));
    fprintf(fid,'# QPolyKal(T,%d,%d,q0), numDim=2, matlab-tcl a9acd8f, MATLAB %s\n',(order+1)*numDim,order,version);
    fprintf(fid,'# captured by scripts/matlab_capture/capture_process_noise.m (matlab -batch)\n');
    fprintf(fid,'# columns: T,q0,Q(1,1:n),...,Q(n,1:n) in MATLAB state order\n');
    for T=Ts
        for q0=qs
            write_row(fid,[T,q0],QPolyKal(T,(order+1)*numDim,order,q0));
        end
    end
    fclose(fid);
end

for order=1:3
    fid=open_fixture(OUTPUT_DIR,sprintf('qpolykaldirectdisc_order%d.csv',order));
    fprintf(fid,'# QPolyKalDirectDisc(T,%d,%d,sigmaV2), numDim=2, matlab-tcl a9acd8f, MATLAB %s\n',(order+1)*numDim,order,version);
    fprintf(fid,'# captured by scripts/matlab_capture/capture_process_noise.m (matlab -batch)\n');
    fprintf(fid,'# columns: T,sigmaV2,Q(1,1:n),...,Q(n,1:n) in MATLAB state order\n');
    for T=Ts
        for sigmaV2=qs
            write_row(fid,[T,sigmaV2],QPolyKalDirectDisc(T,(order+1)*numDim,order,sigmaV2));
        end
    end
    fclose(fid);
end

%T/tau >= 0.05 only: below that q_singer switches to a Taylor series, and
%QGaussMarkov's closed form loses ~3 digits to cancellation (its header warns
%of this), so it is not a usable oracle there.
cases=[1,10;1,20;3.7,5;2,20;0.5,5];
fid=open_fixture(OUTPUT_DIR,'qgaussmarkov_order2.csv');
fprintf(fid,'# QGaussMarkov(T,6,q,tau,2), q=2*sigmam^2/tau, numDim=2, matlab-tcl a9acd8f, MATLAB %s\n',version);
fprintf(fid,'# captured by scripts/matlab_capture/capture_process_noise.m (matlab -batch)\n');
fprintf(fid,'# columns: T,tau,sigmam,Q(1,1:6),...,Q(6,1:6) in MATLAB state order\n');
for k=1:size(cases,1)
    T=cases(k,1);tau=cases(k,2);
    for sigmam=[0.5,3]
        q=2*sigmam^2/tau;
        write_row(fid,[T,tau,sigmam],QGaussMarkov(T,3*numDim,q,tau,2));
    end
end
fclose(fid);

function fid=open_fixture(dirName,name)
    fid=fopen(fullfile(dirName,name),'w');
end

function write_row(fid,params,Q)
    row=[params,reshape(Q.',1,[])];
    fprintf(fid,'%.17g',row(1));
    fprintf(fid,',%.17g',row(2:end));
    fprintf(fid,'\n');
end
