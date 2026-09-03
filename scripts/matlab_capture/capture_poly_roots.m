%CAPTURE_POLY_ROOTS Capture MATLAB TCL reference roots from
%polyRootsMultiDim, ported as
%pytcl/mathematical_functions/polynomials/multivariate.py. Roots are
%complex; each fixture holds [real; imag] stacked (2n rows, numSol
%columns). Inputs are mirrored verbatim in
%tests/validation/test_poly_roots_multivariate.py.

if(~exist('OUTPUT_DIR','var'))
    error('Set OUTPUT_DIR before running.');
end

%Dreesen Section 2.1 two-variable system (4 real roots).
p=zeros(3,3);
p(0+1,0+1)=-4;p(2+1,0+1)=-1;p(1+1,1+1)=2;p(0+1,2+1)=1;p(1+1,0+1)=5;p(0+1,1+1)=-3;
q=zeros(3,3);
q(0+1,0+1)=-1;q(2+1,0+1)=1;q(1+1,1+1)=2;q(0+1,2+1)=1;
polyCoeffMats={p;q};
[rootVals,exitCode]=polyRootsMultiDim(polyCoeffMats);
assert(exitCode==0);
writematrix([real(rootVals);imag(rootVals)],...
    fullfile(OUTPUT_DIR,'poly_roots_2var.csv'));

%Same system through the Motzkin null-space path.
[rootVals,exitCode]=polyRootsMultiDim(polyCoeffMats,[],true);
assert(exitCode==0);
writematrix([real(rootVals);imag(rootVals)],...
    fullfile(OUTPUT_DIR,'poly_roots_2var_motzkin.csv'));

%Dreesen p.94 three-variable cubic system. The MATLAB docstring says 18
%roots are found, but on R2026a (Apple Silicon LAPACK) the original's
%matrixRank algorithm-0 tolerance misjudges the augmented-block nullity
%and polyRootsMultiDim FAILS with exitCode 2. We record the exit code
%rather than asserting success; the pytcl port (which uses the
%algorithm-1 tolerance) is validated on this system by residuals
%instead of by MATLAB output.
p=zeros(4,4,4);
p(2+1,0+1,0+1)=1;p(1+1,1+1,0+1)=5;p(0+1,1+1,1+1)=4;p(0+1,0+1,0+1)=-10;
q=zeros(4,4,4);
q(0+1,3+1,0+1)=1;q(2+1,1+1,0+1)=3;q(0+1,0+1,0+1)=-12;
k=zeros(4,4,4);
k(0+1,0+1,3+1)=1;k(1+1,1+1,1+1)=4;k(0+1,0+1,0+1)=-8;
polyCoeffMats={p;q;k};
[rootVals,exitCode]=polyRootsMultiDim(polyCoeffMats);
writematrix(exitCode,fullfile(OUTPUT_DIR,'poly_roots_3var_matlab_exitcode.csv'));
if(exitCode==0)
    writematrix([real(rootVals);imag(rootVals)],...
        fullfile(OUTPUT_DIR,'poly_roots_3var.csv'));
end

%The 2D range-rate localization polynomial system that
%rangeRate2StaticPos builds internally (its docstring example scene).
uTrue=[1e3;5e3];
s=[500, 1100; 2500, 2500];
sDot=[300, 300; 0, 0];
rr=zeros(2,1);
for curMeas=1:2
    rr(curMeas)=-sDot(:,curMeas)'*(uTrue-s(:,curMeas))/norm(uTrue-s(:,curMeas));
end
polyCoeffMats=cell(2,1);
for curMeas=1:2
    l=s(:,curMeas);
    lDot=sDot(:,curMeas);
    rDot=rr(curMeas);
    lTilde=2*(lDot*(l'*lDot)-rDot^2*l);
    cTilde=rDot^2*norm(l)^2-(l'*lDot)^2;
    coeffs=zeros(3,3);
    coeffs(2+1,0+1)=(rDot^2-lDot(1)^2);
    coeffs(0+1,2+1)=(rDot^2-lDot(2)^2);
    coeffs(1+1,1+1)=-2*lDot(1)*lDot(2);
    coeffs(1+1,0+1)=lTilde(1);
    coeffs(0+1,1+1)=lTilde(2);
    coeffs(0+1,0+1)=cTilde;
    polyCoeffMats{curMeas}=coeffs;
end
[rootVals,exitCode]=polyRootsMultiDim(polyCoeffMats);
assert(exitCode==0);
writematrix([real(rootVals);imag(rootVals)],...
    fullfile(OUTPUT_DIR,'poly_roots_rrloc2d.csv'));

disp('capture_poly_roots done');
