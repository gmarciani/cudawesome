%==========================================================================
% RESPONSE TIME ANALYSIS
%==========================================================================
% DATA
t = readtable("../out/profile/timing_float.csv");

inputDimensions = {'2048', '4096', '8192', '16384', '32768', '65536'};
%timePerformance = [10 12 14 ; 16 18 20; 22 24 26; 28 30 32; 34 36 38; 40 42 44];
blockDimensions = {'32', '64', '128', '256', '512', '1024'};

n_blockDimensions = 6;
replications = 10;

timePerformance = mean(reshape(t{:,'time'}, replications, []));
timePerformance = reshape(timePerformance, n_blockDimensions, []);
timePerformance = timePerformance.';

maxTime = max(max(timePerformance));
minTime = min(min(timePerformance));

scaleMax = 1.02;
scaleMin = 0.95;

% PLOT
figure(1)
bar(timePerformance);

title({'Prefix-Sum (CUDA)';'Response Time Analysis'});

xlabel('Array Dimension');
xlim([0.5 6.5]);
set(gca, 'XTick', 1:6, 'XTickLabel', inputDimensions)

ylabel('Time (ms)');
ylim([minTime*scaleMin maxTime*scaleMax]);

yyaxis right
plot([ 0.5 1 2 3 4 5 6 7 7.5 ], [1 1 1 1 1 1 1 1 1] * minTime, '--r')
hold on
plot([ 0.5 1 2 3 4 5 6 7 7.5 ], [1 1 1 1 1 1 1 1 1] * maxTime, '--r')
hold off

ylim([minTime*scaleMin maxTime*scaleMax]);
set(gca, 'YTick', [round(minTime) round(maxTime)]);

hleg = legend(blockDimensions, 'Location', 'northeast', 'Orientation', 'horizontal');
title(hleg, 'Block Dimension');
