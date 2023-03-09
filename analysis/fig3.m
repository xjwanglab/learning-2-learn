clear
close all

suff = '0.000100_0.001000_0.000100_0.000500_0.100000_10.000000';
dir = '../data';

% Load trials to criterion data for trained nets of each condition
nTypes = {'ControlManifPert','ControlManifPert','DSManifPert','SSManifPert'};
C1 = zeros(10, 4);
for i = 1:10
    for j = 1:4
        c = load(sprintf('%s/conv_%d_%s_runType.%s.txt',dir,i-1,suff, nTypes{j}));
        if j == 1
            C1(i, 1) = c(1);
        else
            C1(i, j) = mean(c(51:100));
        end
    end
end

% Plot trials to criterion
figure;
h = boxplot(C1);
set(h,{'linew'},{2});
box off
set(gca,'fontsize',20)
set(gca,'linewidth',2)
set(gca,'xtick',1:4)
set(gca,'xticklabel',{'Prob. 1', 'Frozen Readout','D->S Manif. Pert.', 'S->S Manif. Pert.'})
ylabel('Trials to criterion')
ylim([0 4000])
xlim([0.5 4.5])
