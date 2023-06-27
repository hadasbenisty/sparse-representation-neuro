path2data = '../sp_rep_data/Simulated_Data';
folders = dir(path2data);
cleanSNR = 100;
newSNR = -10:5:30;

for fi = 3:length(folders)
    cleanfile = dir(fullfile(folders(fi).folder, folders(fi).name, ['*' num2str(cleanSNR) '*.mat']));
    load(fullfile(cleanfile.folder, cleanfile.name), 'y');
    y = double(y);
    sig_s = std(y(:));
    
    sig_n = sig_s*10.^(-newSNR/20);
    for snr_i = 1:length(newSNR)
        y_noisy = single(sig_n(snr_i)*randn(size(y)) + y);
        save(fullfile(cleanfile.folder, [folders(fi).name, '_SNR_' num2str(newSNR(snr_i)) '.mat']), 'y_noisy');
    end
end
    

