function [error_val] = lrcalculateerror(theta, X, y)
predictions = lrpredict(theta, X);
error_val = sum(predictions != y) / size(X, 1);
end
