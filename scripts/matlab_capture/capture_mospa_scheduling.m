% Capture oracle fixtures for the MOSPA family and interval
% scheduling. Inputs are sin-formula generated so the Python tests
% can reconstruct them exactly.
tclRoot = '/Users/nedonatelli/Documents/Local Repositories/matlab-tcl';
addpath(genpath(fullfile(tclRoot, 'Scheduling')));
addpath(genpath(fullfile(tclRoot, 'Performance_Evaluation')));
addpath(genpath(fullfile(tclRoot, 'Mathematical_Functions')));
addpath(genpath(fullfile(tclRoot, 'Assignment_Algorithms')));
outDir = '/Users/nedonatelli/Documents/Local Repositories/TCL/tests/fixtures/matlab';

fid = fopen(fullfile(outDir, 'mospa_scheduling.csv'), 'w');
fprintf(fid, 'label,vals\n');
dump = @(label, V) fprintf(fid, ['%s', repmat(',%.17e', 1, numel(V)), '\n'], label, V(:).');

% Scheduling: 12 intervals, sin-formula starts/durations.
n = 12;
k = 1:n;
starts = 5 * (1 + sin(1.7 * k));
ends = starts + 2 * (1.05 + sin(3.1 * k));
intervals = [starts; ends];
weights = 1.5 + sin(2.3 * k);
dump('schedule_intervals', scheduleIntervals(intervals));
dump('partition_intervals', partitionIntervals(intervals));
[wv, jobs] = scheduleWeightedIntervals(intervals, weights);
dump('weighted_val', wv);
dump('weighted_jobs', jobs);
deadlines = 4 * (1.1 + sin(1.3 * k));
durations = 0.55 + 0.5 * sin(0.9 * k);
[mlJobs, tStarts] = scheduleMinLatenessDense(deadlines, durations, 0.25);
dump('min_lateness_jobs', mlJobs);
dump('min_lateness_starts', tStarts);

% MOSPA: 4-D states, 3 targets, 6 hypotheses.
xDim = 4; numTar = 3; numHyp = 6;
x = zeros(xDim, numTar, numHyp);
for h = 1:numHyp
    for tt = 1:numTar
        for d = 1:xDim
            x(d, tt, h) = 10 * sin(0.7 * d + 1.9 * tt + 3.7 * h);
        end
    end
end
w = 0.5 + 0.5 * sin(1.1 * (1:numHyp));
w = w / sum(w);
xEst = zeros(xDim, numTar);
for tt = 1:numTar
    for d = 1:xDim
        xEst(d, tt) = 10 * sin(0.3 * d + 2.9 * tt);
    end
end
dump('calc_mospa_error', calcMOSPAError(xEst, x, w));
for scans = [1, 3]
    [est, orders] = MMOSPAApprox(x, w, scans);
    dump(sprintf('mmospa_approx_est_s%d', scans), est);
    dump(sprintf('mmospa_approx_orders_s%d', scans), orders);
end

% MMOSPA2Tar2D: 40 particles of stacked 2-D pairs.
np2 = 40;
j = 1:np2;
particles = [2 + sin(1.3 * j); sin(2.1 * j); -2 + sin(0.7 * j); sin(2.9 * j)];
dump('mmospa_2tar_2d', MMOSPA2Tar2D(particles));

fclose(fid);
disp('capture done');
