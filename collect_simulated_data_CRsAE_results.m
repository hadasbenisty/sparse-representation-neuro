function collect_simulated_data_CRsAE_results
addpath(genpath('utils'));
SNRvals = -10:5:30;
path2results = '../sp_rep_results/Simulated_Data\CRsAE\';
cleanfile = '../sp_rep_data/Simulated_Data\CRsAE\CRsAE_SNR_100.mat';
[test_mse_CRsAE, selfile_CRsAE] = collect_results(path2results, cleanfile, 1:2400, 2401:4800, SNRvals);

path2results = '../sp_rep_results/Simulated_Data\CNMF\';
cleanfile = '../sp_rep_data/Simulated_Data\CNMF\CNMF_SNR_100.mat';
[test_mse_CNMF, selfile_CNMF] = collect_results(path2results, cleanfile, 1:1000, 1001:2000, SNRvals);

figure;plot(SNRvals, test_mse_CNMF);
hold all;plot(SNRvals, test_mse_CRsAE)
end
function [test_mse, selfile] = collect_results(path2results, cleanfile, trinds, testinds, SNRvals)


r = load(cleanfile);
clean_y_train = r.y(trinds, :, :);
clean_y_test = r.y(testinds, :, :);

test_mse = nan(size(SNRvals));
for snr_i = 1:length(SNRvals)
    files = dir(fullfile(path2results, ['*SNR_' num2str(SNRvals(snr_i)) '*.mat'])); 
    if isempty(files)
        continue;
    end
    norm_mse_test_tmp = nan(length(files),1);
    norm_mse_train_tmp = nan(length(files),1);
    for fi = 1:length(files)
        try
        a = load(fullfile(files(fi).folder, files(fi).name));
        catch
            continue;
        end
    norm_mse_train_tmp(fi) = MSE_norm(clean_y_train(:), a.estimated_signal_tr(:));
    norm_mse_test_tmp(fi) = MSE_norm(clean_y_test(:), a.estimated_signal_test(:));

    end
    [~,sel] = min(norm_mse_train_tmp);
    selfile(snr_i) = files(sel);
    test_mse(snr_i) = norm_mse_test_tmp(sel);
 end
end