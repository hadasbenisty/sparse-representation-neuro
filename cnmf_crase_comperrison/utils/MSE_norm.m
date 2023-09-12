function norm_mse = MSE_norm(y, y_hat)

mse = mean((y(:) - y_hat(:)).^2, 'omitnan' );
var_s = var(y, 'omitnan');
if var_s < 1e-6
    norm_mse = nan;
else
norm_mse = mse/var(y, 'omitnan');
end

