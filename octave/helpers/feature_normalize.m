function [X_norm, mu, sigma] = feature_normalize(X, mu, sigma)
%FEATURE_NORMALIZE Normalizes the features in X

if ~exist('mu', 'var') || isempty(mu)
  mu = mean(X);
end

X = double(X);
X_norm = bsxfun(@minus, X, mu);

if ~exist('sigma', 'var') || isempty(sigma)
  sigma = std(X_norm);
end

X_norm = bsxfun(@rdivide, X_norm, sigma);

X_norm(find(isnan(X_norm))) = 0;

end
