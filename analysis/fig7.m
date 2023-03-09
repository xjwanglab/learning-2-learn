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
    
    dims = zeros(10,8, 500,2); % to measure VFC component dimensionality
    cVFC_mag = zeros(10, 8, 500, 500) + nan; % CVFC magnitudes
    cVFC_sup = zeros(2, 10, 8, 500) + nan; % CVFC suppression magnitudes
    for net = 1:10
        
        tc=0;
        for ep = 1:10 % loop through 50-problem groups
            
            tasks = ((ep-1)*50+2):(ep*50 +1);
            comps = zeros(2, N, 50, 2, int32(2000/dt));
            t_ec = 0;
            
            for task = tasks % loop through problems in current 50-problem group
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
                
                RHist = zeros(N, T, int32(2000/dt)); % to save learned trajectories
                stHist = zeros(11, T, int32(2000/dt)); % to save input sequences
                for t = 1:int32(2000/dt) % simulate model forward in time
                    
                    if t == int32(500/dt)+1
                        st = zeros(T,11);
                        st(:,11) = 1.0/sqrt(10.0);
                    elseif t == int32(1500/dt)+1
                        st = zeros(T,11);
                    end
                    
                    % weight-driven vector field change
                    vfcW = alphax*(log(1+exp(st*IW + R*RW + RB))-log(1+exp(st*IW0 + R*RW0 + RB0)));
                    
                    % state-driven vector field change
                    vfcS = alphax*(-R + R0x + log(1+exp(st*IW0 + R*RW0 + RB0))-log(1+exp(st*IW0 + R0x*RW0 + RB0)));
                    
                    dZ = vfcW + vfcS; % delta Z
                    mdZ = vecnorm(dZ, 2, 2);
                    
                    % normalize dZ vector magnitude
                    dZ = dZ./repmat(mdZ, 1, N);
                    
                    % compute parallel and perp. components of
                    % weight- and state-driven VFCs relative to dZ
                    vfcW_par = repmat(diag(vfcW*dZ'), 1, N).*dZ;
                    vfcW_perp = vfcW - vfcW_par;
                    comps(1, :, t_ec, : , t) = vfcW_par';
                    comps(2, :, t_ec, : , t) = vfcW_perp';
                    
                    % simulate forward in time
                    R = (1.0-alphax)*R + alphax*log(1+exp(st*IW + R*RW + RB));
                    R0x = (1.0-alphax)*R0x + alphax*log(1+exp(st*IW0 + R0x*RW0 + RB0));
                    RHist(:,:,t) = R';
                    
                    stHist(:,:,t) = st';
                    
                end
                
                % divide each trial in 250ms intervals
                for tInt = 1:8
                    intvl = ((tInt-1)*int32(250/dt) + 1):(tInt*int32(250/dt));
                    
                    % calculate 1D parallel and perp. modes/directions per
                    % 250ms interval
                    dat = squeeze(comps(1, :, t_ec, :, intvl));
                    [COEFF, SCORE, LATENT, TSQUARED, EXPLAINED, MU] = pca(squeeze(dat(:,:))', 'Centered', false);
                    dims(net, tInt, tc,1) = (sum(LATENT).^2)/sum(LATENT.^2);
                    parMode = COEFF(:,1);
                    dat = squeeze(comps(2, :, t_ec, :, intvl));
                    [COEFF, SCORE, LATENT, TSQUARED, EXPLAINED, MU] = pca(squeeze(dat(:,:))', 'Centered', false);
                    dims(net, tInt, tc,2) = (sum(LATENT).^2)/sum(LATENT.^2);
                    perpMode = COEFF(:,1);
                    
                    % loop through all previous problems to calculate CVFC
                    ptc = 0;
                    for ptask = (task-1):-1:1
                        ptc = ptc + 1;
                        load(sprintf('%s/saved_%d_%s_%d.mat', dir, net-1, suff, ptask));
                        IWp = double(wts_RNNin_weights);
                        RWp = double(wts_leakyRNN_weights);
                        RBp = double(wts_leakyRNN_biases);
                        
                        
                        Rdat = RHist(:,:,intvl);
                        Rdat = Rdat(:,:)';
                        
                        stdat = stHist(:,:,intvl);
                        stdat = stdat(:,:)';
                        
                        % calculate cvfc and their parallel/perp.
                        % components
                        vfc = alphax*(log(1+exp(stdat*IW + Rdat*RW + RB))-log(1+exp(stdat*IWp + Rdat*RWp + RBp)));
                        vfc_perp = vfc*perpMode;
                        cVFC_mag(net, tInt, tc, ptc) = mean(abs(vfc_perp));
                        
                        % calculate suppressive effect of CVFCs on
                        % Weight-driven VFCs
                        if ptask == 1
                            vfc_perp = reshape(vfc_perp', T, length(intvl));
                            
                            vfc_par = vfc*parMode;
                            vfc_par = reshape(vfc_par', T, length(intvl));
                            
                            
                            load(sprintf('%s/saved_%d_%s_%d.mat', dir, net-1, suff, task-1));
                            IW0 = double(wts_RNNin_weights);
                            RW0 = double(wts_leakyRNN_weights);
                            RB0 = double(wts_leakyRNN_biases);
                            
                            % calculate weight-driven VFC magnitude along
                            % each component (based on their 1D modes
                            % calculated above)
                            vfc0 = alphax*(log(1+exp(stdat*IW + Rdat*RW + RB))-log(1+exp(stdat*IW0 + Rdat*RW0 + RB0)));
                            
                            vfc0_perp = vfc0*perpMode;
                            vfc0_perp = reshape(vfc0_perp', T, length(intvl));
                            
                            vfc0_par = vfc0*parMode;
                            vfc0_par = reshape(vfc0_par', T, length(intvl));
                            
                            % sign correct (see methods in paper)
                            for m = 1:T
                                s = sign(mean(vfc_perp(m,:)));
                                vfc_perp(m, :) = s.*vfc_perp(m,:);
                                
                                s = sign(mean(vfc_par(m,:)));
                                vfc_par(m, :) = s.*vfc_par(m,:);
                                
                                s = sign(mean(vfc0_perp(m,:)));
                                vfc0_perp(m, :) = s.*vfc0_perp(m,:);
                                
                                s = sign(mean(vfc0_par(m,:)));
                                vfc0_par(m, :) = s.*vfc0_par(m,:);
                            end
                            cVFC_sup(1, net, tInt, tc) = mean(vfc_par(:)-vfc0_par(:));
                            cVFC_sup(2, net, tInt, tc) = mean(vfc_perp(:)-vfc0_perp(:));
                            
                        end
                    end
                end
                
            end
            
        end
        
    end
    save('../results/f7', 'cVFC_mag', 'cVFC_sup');
end

% Plot
load('../results/f7')
figure;
subplot(1,2,1)
for ep = [1 2 5 10]
   tasks = ((ep-1)*50 + 1):(ep*50);
   ptmax = max(tasks -1);
   dat = squeeze(nanmean(nanmean(cVFC_mag(:, :, tasks, 1:ptmax), 2), 3));
   errorbar(500:-1:(500-ptmax+1), mean(dat,1), std(dat,0,1)./sqrt(10), 'linewidth', 2);
   hold on;

end
box off
set(gca,'fontsize',20)
set(gca,'linewidth',2)
ylabel('Magnitude of total VFC along z_{perp}')
xlabel('Relative problem #')
set(gca, 'xtick', 100:100:500)
set(gca, 'xticklabel', -400:100:0)
xlim([0 490])
legend({'Problems 2-51', 'Problems 52-101', 'Problems 202-251', 'Problems 452-501'});
legend boxoff

subplot(1,2,2)
mn = [];
ste = [];
for ep = 1:10
   tasks = ((ep-1)*50 + 1):(ep*50);
   dat = squeeze(mean(mean(cVFC_sup(:, :, :, tasks), 3), 4));
   mn = [mn; mean(dat,2)'];
   ste = [ste; std(dat,0,2)'./sqrt(10)];
end
errorbar(repmat([1:10]',1, 2), mn, ste,'linewidth',2);
box off
set(gca,'fontsize',20)
set(gca,'linewidth',2)
ylabel({'Magnitude of net suppression','of WdVFC by CVFC'})
xlabel('Problems')
set(gca, 'xtick', 2:2:10)
set(gca, 'xticklabel', 100:100:500)
legend({'Along dZ', 'Orth. to dZ'});
legend boxoff


