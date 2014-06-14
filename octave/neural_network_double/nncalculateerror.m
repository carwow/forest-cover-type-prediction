function [error_val] = nncalculateerror(Theta1, Theta2, Theta3, X, y)
predictions = nnpredict(Theta1, Theta2, Theta3, X);
error_val = sum(predictions != y) / size(X, 1);
end
