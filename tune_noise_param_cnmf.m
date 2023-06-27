function noise_level = tune_noise_param_cnmf(Y, noise_vals, clean_y, train_inds)

train_res = nan(length(train_inds), length(noise_vals));
for noise_i = 1:length(noise_vals)
    
    for k=train_inds
        c_oasis = deconvolveCa(Y(k, :), 'ar2', 'sn', noise_vals(noise_i), 'thresholded',...
            'optimize_smin','optimize_pars', 'thresh_factor', 1);
        train_res(k, noise_i) = MSE_norm(clean_y(k, :), c_oasis);
    end
    mean_train = mean(train_res, 1, 'omitnan');
    if noise_i>1 && mean_train(noise_i) > mean_train(noise_i-1)
        break
    end
end
mean_train = mean(train_res, 1, 'omitnan');
[~, sel_noise_i] = min(mean_train);
noise_level = noise_vals(sel_noise_i);
end