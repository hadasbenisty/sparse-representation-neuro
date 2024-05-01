%% test modulate for all oasis functions. 
col = {[0 114 178],[0 158 115], [213 94 0],[230 159 0],...
    [86 180 233], [204 121 167], [64 224 208], [240 228 66]}; % colors
plot_cvx = false; 
seed = 3;

%% for CARSE dataset
noise = -1;
dataset = load("C:\Users\yonatan.kle\OneDrive - Technion\Desktop\homework\hadas_proj\data_for_200k_samp.mat");
Y = dataset.dataArray;
Y = Y - mean(Y, 1);

%true_C = double(reshape(dataset.labels,4800,500));

%% for CNMF dataset
%noise = 0.5;
%dataset = load("Simulated_Data\CNMF\CNMF_SNR_50.0.mat");
%Y = double(reshape(dataset.data,2000,500)); 
%true_C = double(reshape(dataset.labels,2000,500));
%% 
y = Y(161,:);
%true_c = true_C(5,:);
[c_oasis, s_oasis, options] = deconvolveCa(y, 'ar2', 'sn', noise, 'thresholded',...
    'optimize_smin','optimize_pars', 'thresh_factor', 1); 
%error = immse(true_c,c_oasis')/(var(true_c)+0.0000001);
%%
figure()
subplot(311)
plot(y)
legend('Noisy signal')
subplot(312)
plot(c_oasis)
legend('estimation')
%subplot(313)
%plot(true_c)
%legend('original signal')
%%%%%%%%%%%%%%  END %%%%%%%%%%%%%%%%%%

save('cnmf_spike_detection_noise01_minus_mean2.mat', 'c_oasis');
