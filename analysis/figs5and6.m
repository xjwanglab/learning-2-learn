clear
close all

useSaved = true;

suff = '0.000100_0.001000_0.000100_0.000500_0.100000_10.000000_runType.Full';
dir = '../data';

dt = 1.0;
tau = 100.0;
alphax = dt/tau;
N = 100;
T = 2;

if useSaved == false
    
    corrs = []; % to save correlation coefficients / R^2 between weight change magnitudes and learning performance
    for net = 1:10
        
        VFC = zeros(1000,9); % to save vector field change magnitude measurements
        % to save vector field change dimensionality measurements
        dims_SingProb = zeros(1000,2,2);
        dims_SingEp = zeros(20,2,2);
        dimsT_SingProb = zeros(1000,1);
        dimsT_SingEp = zeros(20,1);
        
        tc=0;
        % loop through problems in groups of 50 at a time
        for ep = 1:20
            
            tasks = ((ep-1)*50+2):(ep*50 +1);
            comps = zeros(2, 2, N, 50, 2, int32(2000/dt));
            compsT = zeros(N, 50, 2, int32(2000/dt));
            t_ec = 0;
            
            % loop through each task in the current 50-problem epoch
            for task = tasks
                tc = tc+1;
                t_ec = t_ec+1;
                [task tc t_ec]
                
                load(sprintf('%s/saved_%d_%s_%d.mat', dir, net-1, suff, task-1));
                IS0 = double(wts_leakyRNN_init_state);
                IW0 = double(wts_RNNin_weights);
                RW0 = double(wts_leakyRNN_weights);
                RB0 = double(wts_leakyRNN_biases);
                
                load(sprintf('%s/saved_%d_%s_%d.mat', dir, net-1, suff, task));
                IS = double(wts_leakyRNN_init_state);
                IW = double(wts_RNNin_weights);
                RW = double(wts_leakyRNN_weights);
                RB = double(wts_leakyRNN_biases);
                images = double(images);
                
                R = repmat(IS,T,1);
                R0x = repmat(IS,T,1);
                
                st = zeros(T,11);
                st(:,11) = 1.0/sqrt(10.0);
                st(1:2,1:10) = images;
                
                mWC = 0;
                
                mW = 0;
                mW_par = 0;
                mW_perp = 0;
                
                mS = 0;
                mS_par = 0;
                mS_perp = 0;
                
                mT = 0;
                
                % loop through time to measure vector field quantity
                % at each time step
                for t = 1:int32(2000/dt)
                    
                    if t == int32(500/dt)+1
                        st = zeros(T,11);
                        st(:,11) = 1.0/sqrt(10.0);
                    elseif t == int32(1500/dt)+1
                        st = zeros(T,11);
                    end
                    
                    Dx = R-R0x;
                    
                    % change in post-syn current magnitudes
                    vfcW_C = st*(IW-IW0) + R*(RW-RW0) + (RB-RB0);
                    mWC = mWC + vecnorm(vfcW_C, 2, 2);
                    
                    % weight-driven vector field change magnitudes
                    vfcW = alphax*(log(1+exp(st*IW + R*RW + RB))-log(1+exp(st*IW0 + R*RW0 + RB0)));
                    mW = mW + vecnorm(vfcW, 2, 2);
                    
                    % state-driven vector field change magnitudes
                    vfcS = alphax*(-R + R0x + log(1+exp(st*IW0 + R*RW0 + RB0))-log(1+exp(st*IW0 + R0x*RW0 + RB0)));
                    mS = mS + vecnorm(vfcS, 2, 2);
                    
                    dZ = vfcW + vfcS; % delta Z
                    mdZ = vecnorm(dZ, 2, 2);
                    mT = mT + mdZ;
                    
                    % normalize dZ vector magnitude
                    dZ = dZ./repmat(mdZ, 1, N);
                    
                    % compute parallel and perp. components of
                    % weight- and state-driven VFCs relative to dZ
                    vfcW_par = repmat(diag(vfcW*dZ'), 1, N).*dZ;
                    vfcW_perp = vfcW - vfcW_par;
                    mW_par = mW_par + vecnorm(vfcW_par, 2, 2);
                    mW_perp = mW_perp - vecnorm(vfcW_perp, 2, 2);
                    comps(1, 1, :, t_ec, : , t) = vfcW_par';
                    comps(1, 2, :, t_ec, : , t) = vfcW_perp';
                    
                    vfcS_par = repmat(diag(vfcS*dZ'), 1, N).*dZ;
                    vfcS_perp = vfcS - vfcS_par;
                    mS_par = mS_par + vecnorm(vfcS_par, 2, 2);
                    mS_perp = mS_perp + vecnorm(vfcS_perp, 2, 2);
                    comps(2, 1, :, t_ec, : , t) = vfcS_par';
                    comps(2, 2, :, t_ec, : , t) = vfcS_perp';
                    
                    compsT(:, t_ec, : , t) = dZ';
                    
                    % simulate forward in time
                    R = (1.0-alphax)*R + alphax*log(1+exp(st*IW + R*RW + RB));
                    R0x = (1.0-alphax)*R0x + alphax*log(1+exp(st*IW0 + R0x*RW0 + RB0));
                    
                    
                end
                % temporal mean of magnitudes
                mWC = mWC./(2000/dt);
                
                mW = mW./(2000/dt);
                mW_par = mW_par./(2000/dt);
                mW_perp = mW_perp./(2000/dt);
                
                mS = mS./(2000/dt);
                mS_par = mS_par./(2000/dt);
                mS_perp = mS_perp./(2000/dt);
                
                mT = mT./(2000/dt);
                
                % dimensionality of VFC quantities within single problems
                for i = 1:2
                    for j = 1:2
                        [COEFF, SCORE, LATENT, TSQUARED, EXPLAINED, MU] = pca(squeeze(comps(i, j, :, t_ec, :))', 'Centered', false);
                        dims_SingProb(tc, i, j) = (sum(LATENT).^2)/sum(LATENT.^2);
                    end
                end
                [COEFF, SCORE, LATENT, TSQUARED, EXPLAINED, MU] = pca(squeeze(compsT(:, t_ec, :))', 'Centered', false);
                dimsT_SingProb(tc) = (sum(LATENT).^2)/sum(LATENT.^2);
                
                
                VFC(tc,1) = mean(mWC);
                
                VFC(tc,2) = mean(mW);
                VFC(tc,3) = mean(mS);
                
                VFC(tc,4) = vecnorm(RW(:)-RW0(:));
                
                VFC(tc,5) = mean(mT);
                
                VFC(tc,6) = mean(mW_par);
                VFC(tc,7) = mean(mW_perp);
                VFC(tc,8) = mean(mS_par);
                VFC(tc,9) = mean(mS_perp);
                
            end
            
            % dimensionality of VFC quantities across 50-problem group
            for i = 1:2
                for j = 1:2
                    [COEFF, SCORE, LATENT, TSQUARED, EXPLAINED, MU] = pca(squeeze(comps(i,j, :, :))', 'Centered', false);
                    dims_SingEp(ep, i, j) = (sum(LATENT).^2)/sum(LATENT.^2);
                end
            end
            [COEFF, SCORE, LATENT, TSQUARED, EXPLAINED, MU] = pca(squeeze(compsT( :, :))', 'Centered', false);
            dimsT_SingEp(ep) = (sum(LATENT).^2)/sum(LATENT.^2);
            
        end
        % Correlate weight change magnitude with learning performance
        C = load(sprintf('%s/conv_%d_%s.txt',dir,net-1,suff));
        C = C(2:end, 1);
        corrs = [corrs; corr(VFC(:,2),C) corr(VFC(:,3),C) corr(VFC(:,4),C)];
        
        %Summarize quantities by problem group for plotting
        VFC_sum = zeros(20, 9);
        dims_SingProb_sum = zeros(20,2,2);
        dimsT_SingProb_sum = zeros(20,1);
        
        for ep = 1:20
            tasks = ((ep-1)*50+1):(ep*50);
            VFC_sum(ep, :) = mean(VFC(tasks,:),1);
            dims_SingProb_sum(ep, :, :) = mean(dims_SingProb(tasks, :, :),1);
            dimsT_SingProb_sum(ep) = mean(dimsT_SingProb(tasks));
        end
        
        % Raw plots
        figure;
        subplot(2, 6, 1)
        plot(dims_SingProb_sum(:, 1, 1)); hold on
        plot(dims_SingEp(:, 1, 1))
        subplot(2, 6, 2)
        plot(dims_SingProb_sum(:, 1, 2)); hold on
        plot(dims_SingEp(:, 1, 2))
        subplot(2, 6, 3)
        plot(dims_SingProb_sum(:, 2, 1)); hold on
        plot(dims_SingEp(:, 2, 1))
        subplot(2, 6, 4)
        plot(dims_SingProb_sum(:, 2, 2)); hold on
        plot(dims_SingEp(:, 2, 2))
        subplot(2, 6, 5)
        plot(dimsT_SingProb_sum); hold on
        plot(dimsT_SingEp)
        
        subplot(2, 6, 6)
        plot(corrs(end,:).^2)
        
        
        subplot(2, 6, 7)
        plot(VFC_sum(:, 1))
        subplot(2, 6, 8)
        plot(VFC_sum(:, [2 3 5]))
        subplot(2, 6, 9)
        plot(VFC_sum(:, 4))
        subplot(2, 6, 10)
        plot(VFC_sum(:, [2 6 7]))
        subplot(2, 6, 11)
        plot(VFC_sum(:, [3 8 9]))
        save(sprintf('../results/f5And6_seed_%d', net-1), 'C', 'VFC', 'VFC_sum', 'dims_SingEp','dimsT_SingEp', 'dims_SingProb_sum', 'dimsT_SingProb_sum');
    end
    save('../results/f5And6_All', 'corrs');
end

% Summarize data for plots
dZDecMag = zeros(10, 2, 3);
dZDecDim = zeros(10, 2, 2);
dW_VFC = zeros(10, 1);
dMag = zeros(10,20,3);
for net = 1:10
    load(sprintf('../results/f5And6_seed_%d', net-1));
    dZDecMag(net, 1, 1) = VFC_sum(1, 5);
    dZDecMag(net, :, 2) = VFC_sum(1, 6:7)';
    dZDecMag(net, :, 3) = VFC_sum(1, 8:9)';
    
    dZDecDim(net, 1, 1:2) = dims_SingProb_sum(1,1,1:2);
    dZDecDim(net, 2, 1:2) = dims_SingEp(1,1,1:2);

    dW_VFC(net) = corr(VFC(1:500,4), C(1:500))^2;
    
    X = VFC_sum(:,[4, 1, 2]);
    X = X./repmat(X(1,:), 20, 1);
    dMag(net,:,:) = X;

end
load('../results/f5And6_All');
% dW_VFC = corrs(:,3);

figure;
subplot(1,6,1)
h = boxplot(squeeze(dZDecMag(:,:,1)));
set(h,{'linew'},{2});
box off
set(gca,'fontsize',15)
set(gca,'linewidth',2)
set(gca,'xtick',1)
set(gca,'xticklabel',{'Total deltaz'})
ylabel('Magnitude (a.u.)')
ylim([-0.015 0.04])
subplot(1,6,2)
h = boxplot(squeeze(dZDecMag(:,:,2)));
set(h,{'linew'},{2});
box off
set(gca,'fontsize',15)
set(gca,'linewidth',2)
set(gca,'xtick',[1 2])
set(gca,'xticklabel',{'WdVFC(par)', 'WdVFC(perp)'})
ylim([-0.015 0.04])
subplot(1,6,3)
h = boxplot(squeeze(dZDecMag(:,:,3)));
set(h,{'linew'},{2});
box off
set(gca,'fontsize',15)
set(gca,'linewidth',2)
set(gca,'xtick',[1 2])
set(gca,'xticklabel',{'SdVFC(par)', 'SdVFC(perp)'})
ylim([-0.015 0.04])


subplot(1,6,5)
h = boxplot(squeeze(dZDecDim(:,:,1)));
set(h,{'linew'},{2});
box off
set(gca,'fontsize',15)
set(gca,'linewidth',2)
set(gca,'xtick',[1 2])
set(gca,'xticklabel',{'Single(par)', 'Group(par)'})
ylabel('Dimensionality')
ylim([0 14])
subplot(1,6,6)
h = boxplot(squeeze(dZDecDim(:,:,2)));
set(h,{'linew'},{2});
box off
set(gca,'fontsize',15)
set(gca,'linewidth',2)
set(gca,'xtick',[1 2])
set(gca,'xticklabel',{'Single(perp)', 'Group(perp)'})
ylim([0 14])


figure;
subplot(1,3,1)
load(sprintf('../results/f5And6_seed_%d', 3));
plot(VFC(1:500,4), C(1:500), 'o')
box off
set(gca,'fontsize',20)
set(gca,'linewidth',2)
xlabel('Trials to Criterion')
ylabel('Change in W_{rec}')
ylim([0 450])
xlim([0 0.43])

subplot(1,3,2)
h = boxplot(dW_VFC.*100);
set(h,{'linew'},{2});
box off
set(gca,'fontsize',20)
set(gca,'linewidth',2)
ylabel('Explained variance')
ylim([40 100])
set(gca,'xtick',[])

subplot(1,3,3)
errorbar(repmat([25:50:1000]',1,3), squeeze(mean(dMag,1)), squeeze(std(dMag,0,1))./sqrt(10), 'linewidth',2)    
box off
set(gca,'fontsize',20)
set(gca,'linewidth',2)
xlabel('Problems')
ylabel('Normalized magnitude')
legend('delta(W_{rec})','delta(Curr.)','W-d VFC')
legend boxoff
