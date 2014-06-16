function [error_val] = nncalculateerror(thetas, X, y)
predictions = nnpredict(thetas, X);
error_val = sum(predictions != y) / size(X, 1);
end
