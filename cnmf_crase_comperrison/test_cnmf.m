function [norm_mse, esty] = test_cnmf(Y, test_inds, noise_val, clean_y)
test_res = nan(length(test_inds), 1);

for k=1:length(test_inds)
    c_oasis = deconvolveCa(Y(test_inds(k), :), 'ar2', 'sn', noise_val, 'thresholded',...
        'optimize_smin','optimize_pars', 'thresh_factor', 1);
    test_res(k) = MSE_norm(clean_y(test_inds(k), :), c_oasis);
    esty(:, k) = c_oasis;
end
norm_mse = mean(test_res, 1, 'omitnan');

end