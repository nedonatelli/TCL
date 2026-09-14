%%CAPTURE_MIXTURE_REDUCTION Capture oracle fixtures for Gaussian mixture
% reduction:
%   RunnalsGaussMixRed -> pytcl.clustering.gaussian_mixture.reduce_mixture_runnalls
%   WestGaussReduction -> pytcl.clustering.gaussian_mixture.reduce_mixture_west
% MATLAB TCL tree: /Users/nedonatelli/Documents/Local Repositories/matlab-tcl
% at commit a9acd8f. Cases: both functions' docstring examples (r1, w1) plus
% independent 2-D/3-D/4-D mixtures (mv2, mv3, mv4). Inputs are literal
% below and are also written out (mixred_<case>_in_*.csv) so the tests
% read them instead of retyping them. Nothing here is RNG-dependent.
% Consumed by tests/validation/test_mixture_reduction_matlab.py.
tclRoot = '/Users/nedonatelli/Documents/Local Repositories/matlab-tcl';
addpath(genpath(fullfile(tclRoot, 'Clustering_and_Mixture_Reduction')));
addpath(genpath(fullfile(tclRoot, 'Mathematical_Functions')));
outDir = '/Users/nedonatelli/Documents/Local Repositories/TCL/tests/fixtures/matlab';

% r1: RunnalsGaussMixRed example 1 (10 -> 6)
w=[0.03, 0.18, 0.12, 0.19, 0.02, 0.16, 0.06, 0.1, 0.08, 0.06];
mu=[1.45, 2.20, 0.67, 0.48, 1.49, 0.91, 1.01, 1.42, 2.77, 0.89];
P=reshape([0.0487, 0.0305, 0.1171, 0.0174, 0.0295,0.0102, 0.0323, 0.0380, 0.0115, 0.0679],[1,1,10]);
saveMix(outDir,'r1_in',w,mu,P);
[wRed,muRed,PRed]=RunnalsGaussMixRed(w,mu,P,6); saveMix(outDir,'r1_runnalls',wRed,muRed,PRed);

% w1: WestGaussReduction example (9 -> 6)
w=[0.03,0.18,0.12,0.19,0.16,0.06,0.1,0.08,0.06]; w=w/sum(w);
mu=[1.45,2.20,0.67,0.48,0.91,1.01,1.42,2.77,0.89];
P=reshape([0.0487,0.0305,0.1171,0.0174,0.0102, 0.0323, 0.0380, 0.0115, 0.0679],[1,1,9]);
saveMix(outDir,'w1_in',w,mu,P);
[wRed,muRed,PRed]=WestGaussReduction(w,mu,P,6,0,0); saveMix(outDir,'w1_west_kl',wRed,muRed,PRed);
[wRed,muRed,PRed]=WestGaussReduction(w,mu,P,6,0,1); saveMix(outDir,'w1_west_kl_enh',wRed,muRed,PRed);
[wRed,muRed,PRed]=WestGaussReduction(w,mu,P,6,1,0); saveMix(outDir,'w1_west_ise',wRed,muRed,PRed);
[wRed,muRed,PRed]=RunnalsGaussMixRed(w,mu,P,6); saveMix(outDir,'w1_runnalls',wRed,muRed,PRed);

% mv2: 2-D, 8 -> 4
w=[0.05 0.15 0.1 0.2 0.12 0.08 0.18 0.12];
mu=[0 0; 0.3 0.1; 2 1.5; 2.2 1.3; -1 2; -1.2 2.3; 4 -1; 0.5 -0.4].';
P=zeros(2,2,8);
P(:,:,1)=[0.2 0.05;0.05 0.3]; P(:,:,2)=[0.15 -0.02;-0.02 0.1]; P(:,:,3)=[0.3 0.1;0.1 0.25]; P(:,:,4)=[0.2 0;0 0.2];
P(:,:,5)=[0.4 0.12;0.12 0.35]; P(:,:,6)=[0.25 -0.05;-0.05 0.2]; P(:,:,7)=[0.5 0.2;0.2 0.6]; P(:,:,8)=[0.1 0.03;0.03 0.12];
saveMix(outDir,'mv2_in',w,mu,P);
[wRed,muRed,PRed]=RunnalsGaussMixRed(w,mu,P,4); saveMix(outDir,'mv2_runnalls',wRed,muRed,PRed);
[wRed,muRed,PRed]=WestGaussReduction(w,mu,P,4,0,0); saveMix(outDir,'mv2_west_kl',wRed,muRed,PRed);

% mv3: 3-D, 7 -> 3 (RunnalsGaussMixRed's explicit 3x3 determinant branch)
w=[0.1 0.2 0.15 0.05 0.25 0.1 0.15];
mu=[0 0 0; 0.5 0.2 -0.1; 3 3 3; 3.2 2.8 3.1; -2 1 0.5; -2.3 1.2 0.4; 1 -3 2].';
s=[0.2 0.3 0.25 0.4 0.35 0.15 0.5]; rho=[0.1 -0.2 0.3 0 0.25 -0.3 0.15];
Moff=[0 1 0;1 0 1;0 1 0];
P=zeros(3,3,7);
for i=1:7, P(:,:,i)=s(i)*(eye(3)+rho(i)*Moff); end
saveMix(outDir,'mv3_in',w,mu,P);
[wRed,muRed,PRed]=RunnalsGaussMixRed(w,mu,P,3); saveMix(outDir,'mv3_runnalls',wRed,muRed,PRed);
[wRed,muRed,PRed]=WestGaussReduction(w,mu,P,3,0,0); saveMix(outDir,'mv3_west_kl',wRed,muRed,PRed);

% mv4: 4-D, 6 -> 3 (general det branch)
w=[0.2 0.1 0.25 0.15 0.2 0.1];
mu=[0 0 0 0; 0.4 -0.2 0.1 0.3; 5 5 5 5; 5.3 4.8 5.2 4.9; -3 2 1 0; 2 -2 3 -1].';
P=zeros(4,4,6);
for i=1:6
    Ai=reshape(sin((1:16)*i*0.37),4,4);
    P(:,:,i)=0.1*eye(4)+0.05*(Ai*Ai.');
end
saveMix(outDir,'mv4_in',w,mu,P);
[wRed,muRed,PRed]=RunnalsGaussMixRed(w,mu,P,3); saveMix(outDir,'mv4_runnalls',wRed,muRed,PRed);
[wRed,muRed,PRed]=WestGaussReduction(w,mu,P,3,0,0); saveMix(outDir,'mv4_west_kl',wRed,muRed,PRed);

% ISE fixtures from a PATCHED WestGaussReduction: the original assigns a
% merged component's refreshed ISE self-term to the local scalar JrrCur
% instead of the Jrr cache, so later distance evaluations use a stale
% value. pytcl deliberately refreshes it; these fixtures pin that. The
% patch is applied to a temporary copy placed ahead of the TCL on the path.
patchDir = fullfile(tempdir, 'pytcl_west_patched');
if ~exist(patchDir, 'dir'), mkdir(patchDir); end
orig = fileread(fullfile(tclRoot, 'Clustering_and_Mixture_Reduction', 'WestGaussReduction.m'));
patched = strrep(orig, 'JrrCur(minIdxFull)=((4*pi)^xDim*detCur).^(-1/2);', ...
                       'Jrr(minIdxFull)=((4*pi)^xDim*detCur).^(-1/2);');
assert(~strcmp(orig, patched), 'patch target line not found -- upstream changed?');
fid = fopen(fullfile(patchDir, 'WestGaussReduction.m'), 'w'); fwrite(fid, patched); fclose(fid);
addpath(patchDir);
% mv3 and mv4 inputs as above
w=[0.1 0.2 0.15 0.05 0.25 0.1 0.15];
mu=[0 0 0; 0.5 0.2 -0.1; 3 3 3; 3.2 2.8 3.1; -2 1 0.5; -2.3 1.2 0.4; 1 -3 2].';
P=zeros(3,3,7); for i=1:7, P(:,:,i)=s(i)*(eye(3)+rho(i)*Moff); end
[wRed,muRed,PRed]=WestGaussReduction(w,mu,P,2,1,0); saveMix(outDir,'mv3_west_ise_patched',wRed,muRed,PRed);
w=[0.2 0.1 0.25 0.15 0.2 0.1];
mu=[0 0 0 0; 0.4 -0.2 0.1 0.3; 5 5 5 5; 5.3 4.8 5.2 4.9; -3 2 1 0; 2 -2 3 -1].';
P=zeros(4,4,6); for i=1:6, Ai=reshape(sin((1:16)*i*0.37),4,4); P(:,:,i)=0.1*eye(4)+0.05*(Ai*Ai.'); end
[wRed,muRed,PRed]=WestGaussReduction(w,mu,P,2,1,0); saveMix(outDir,'mv4_west_ise_patched',wRed,muRed,PRed);
rmpath(patchDir);
disp('mixture reduction fixtures written');

function saveMix(outDir,name,w,mu,P)
    writeMat(fullfile(outDir,['mixred_' name '_w.csv']),w(:).');
    writeMat(fullfile(outDir,['mixred_' name '_mu.csv']),mu.');
    d=size(mu,1); N=size(mu,2);
    Pf=zeros(N,d*d);
    for i=1:N, Pf(i,:)=reshape(P(:,:,i).',1,[]); end
    writeMat(fullfile(outDir,['mixred_' name '_P.csv']),Pf);
end
function writeMat(fn,M)
    fid=fopen(fn,'w'); [~,c]=size(M);
    fmt=[repmat('%.17g,',1,c-1) '%.17g\n']; fprintf(fid,fmt,M.'); fclose(fid);
end
