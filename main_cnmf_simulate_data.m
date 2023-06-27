addpath(genpath('cnmf'));
addpath(genpath('utils'));
plot_cvx = false;
seed = 3;
SNRvals = -10:5:30;
noise_vals = 0.1:0.1:0.9;


%% for CARSE dataset
norm_mse_CRsAE = nan(size(SNRvals));
clean_y = load('../sp_rep_data/Simulated_Data/CRsAE/CRsAE_SNR_100.mat', 'y');
clean_y = double(squeeze(clean_y.y));
clean_y = reorganize(clean_y, 4);
train_size = 1;
%round(0.3*size(clean_y, 1));
test_size = 1;%round(0.3*size(clean_y, 1));
train_inds = 1:train_size;
test_inds = (1:test_size) + train_inds(end);
noise_level_CRsAE = nan(size(SNRvals));

for snr_i = 1:length(SNRvals)
    tic;
    disp(snr_i/length(SNRvals));
    dataset = load(['../sp_rep_data/Simulated_Data/CRsAE/CRsAE_SNR_' ...
        num2str(SNRvals(snr_i)) '.mat']);
    Y = double(squeeze(dataset.y_noisy));
    Y = (reorganize(Y, 4));
   
    noise_level_CRsAE(snr_i) = tune_noise_param_cnmf(Y, noise_vals, clean_y, train_inds);
    toc;
    tic;
    disp('test');    
    [norm_mse_CRsAE(snr_i), esty_CRsAE(snr_i, :)] = test_cnmf(Y, test_inds, noise_level_CRsAE(snr_i), clean_y);
    toc;
end

%% for CNMF dataset
norm_mse_CNMF = nan(size(SNRvals));
noise_level_CNMF = nan(size(SNRvals));

clean_y = load(['../sp_rep_data/Simulated_Data/CNMF/CNMF_SNR_100.mat'], 'y');
clean_y = double(squeeze(clean_y.y));
clean_y = abs(reorganize(clean_y, 4));
train_size = 1;%round(0.5*size(clean_y, 1));
test_size = 1;%round(0.5*size(clean_y, 1));
train_inds = 1:train_size;
test_inds = (1:test_size) + train_inds(end);


for snr_i = 1:length(SNRvals)
    dataset = load(['../sp_rep_data/Simulated_Data/CNMF/CNMF_SNR_' ...
        num2str(SNRvals(snr_i)) '.mat']);
     Y = abs(reorganize(double(squeeze(dataset.y_noisy)), 4));  
    noise_level_CNMF(snr_i) = tune_noise_param_cnmf(Y, noise_vals, clean_y, train_inds);
    [norm_mse_CNMF(snr_i), esty_CNMF(snr_i, :)] = test_cnmf(Y,test_inds, noise_level_CNMF(snr_i), clean_y);
end
figure;
plot(SNRvals, norm_mse_CRsAE);
hold all;
plot(SNRvals, norm_mse_CNMF);
xlabel('SNR[dB]');
ylabel('MSE/Var(Sig)');
legend('CRsAE');
save('../sp_rep_results/Simulated_results_cnmf.mat', 'esty_CRsAE', 'esty_CNMF', 'norm_mse_CNMF', 'norm_mse_CRsAE','noise_level_CNMF', 'noise_level_CRsAE', 'SNRvals');