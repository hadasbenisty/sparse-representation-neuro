fid = fopen('arguments_for_Train_simulated_data.txt','w');
for lr = [0.1 0.01 0.001]
    for lam = 0.5:0.5:3
        for C = 1:3
            for snr_i = -10:5:30
                str = (['Simulated_data/CRsAE 200 10 ' num2str(C) ' 15 ' ...
                    num2str(lam) ' ' num2str(lr) ' 256 20 CRsAE_SNR_' ...
                    num2str(snr_i) '.mat CRsAE_SNR_100.mat']);
                fprintf(fid, "%s\n", str);
                str = (['Simulated_data/CNMF 200 10 ' num2str(C) ' 15 ' ...
                    num2str(lam) ' ' num2str(lr) ' 256 20 CNMF_SNR_' ...
                    num2str(snr_i) '.mat CNMF_SNR_100.mat']);
                fprintf(fid, "%s\n", str);
            end
        end
    end
end
fclose(fid);