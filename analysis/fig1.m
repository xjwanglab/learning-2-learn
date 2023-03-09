clear
close all

useSaved = true;

suff = '0.000100_0.001000_0.000100_0.000500_0.100000_10.000000_runType.Full';
dir = '../data';

% Load trials to criterion data for trained nets
C1 = zeros(1001,10);
for i = 1:10
  c = load(sprintf('%s/conv_%d_%s.txt',dir,i-1,suff));
  C1(:,i) = c(1:1001);
end

if useSaved == false
    % Fit exponential curve to quantify l2l performance
    figure;
    timeCnsts= [];
    scales= [];
    asymps= [];
    clf
    for i = 1:10
        Y1 = C1(2:1001,i);
        if sum(isnan(Y1))
            print 'problem'
            continue;
        end
        % Initial parameter value
        asymptote = mean(Y1((end-30):end));
        scale = (Y1-asymptote);
        scale = mean(scale(1:30));
        if scale < 0
            scale = 100;
        end
        fo = fitoptions('Method','NonlinearLeastSquares',...
            'Lower',[0,-1/5,0],...
            'Upper',[2*scale,-1/200,2*asymptote],'StartPoint',[scale -1/50 asymptote],'MaxFunEvals',5000,'MaxIter',2000,'TolFun',10^-20,'TolX',10^-20);
        ft1 = fittype('a*exp(b*x)+c','options',fo);
        
        X = [0:999]';
        
        [fC,~,op] = fit(X,Y1,ft1);
        
        % Plot fit
        subplot(5,6,i)
        plot(fC,X,Y1)
        hold on
        plot(movmean(Y1,30),'k','linewidth',3)
        ylim([0 500])
        title(sprintf('%d, Time const = %f',i, -1/fC.b))
        timeCnsts = [timeCnsts -1/fC.b];
        scales = [scales fC.a];
        
        asymps = [asymps fC.c];
        
    end
    
    % Save l2l performance measurements
    TC = timeCnsts;
    AS = asymps;
    SC = scales;
    % figure;
    % subplot(2,2,1)
    % boxplot(TC);
    % subplot(2,2,2)
    % boxplot(AS);
    % subplot(2,2,3)
    % boxplot(SC);
    save('../results/f1_perfFit', 'TC','AS','SC');
end

% Plot l2l performance measurements
load('../results/f1_perfFit');
figure; 
subplot(1,3,1)
sNet = 4;
Y1 = C1(1:1001,sNet);
X = 1:length(Y1);
pRange = 2:50:1001; 
D = zeros(50,length(pRange));
for i = 1:length(pRange)
    D(:,i) = Y1(pRange(i):(pRange(i)+49));
end
hold on
h = boxplot(D, 'positions', [25:50:1000]);
set(h,{'linew'},{2});
plot(0,Y1(1),'o','linewidth',5)
plot(movmean(Y1(2:end),30),'k','linewidth',3)
X = [0:999];
plot(X+1,SC(sNet)*exp((-1/TC(sNet)).*X)+AS(sNet),'color',[0,0.7,0],'linewidth',3)
xlim([-1,1000])
% ylim([-5, Y1(1)+100])
set(gca,'xtick',[100:200:1000])
set(gca,'xticklabel',[100:200:1000])
xlabel('Problems')
ylabel('# trials to criterion')
title('Sample network performance fit')
box off
set(gca,'fontsize',20)
set(gca,'linewidth',2)

subplot(1,3,2)
h = boxplot(TC);
set(h,{'linew'},{2});
title('Time constants')
box off
set(gca,'fontsize',20)
set(gca,'linewidth',2)
set(gca,'xtick',[])


subplot(1,3,3)
h = boxplot(AS);
set(h,{'linew'},{2});
title('Asymptotes')
box off
set(gca,'fontsize',20)
set(gca,'linewidth',2)
set(gca,'xtick',[])
ylim([0 40])
