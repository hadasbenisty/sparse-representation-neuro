function y = reorganize(y, n)

sz = size(y);
y = y';
y = y(:);
y = reshape(y, sz(2)*n, [])';
