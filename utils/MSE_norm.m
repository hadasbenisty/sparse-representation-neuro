function norm_mse = MSE_norm(y, y_hat)

mse = mean((y(:) - y_hat(:)).^2, 'omitnan' );
norm_mse = mse/var(y, 'omitnan');

