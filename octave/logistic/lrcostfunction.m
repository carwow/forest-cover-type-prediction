function [J, grad] = lrcostfunction(theta, X, y, lambda)
m = length(y); % number of training examples
h = sigmoid(X * theta);
regularization_term = (lambda / (2*m)) * sumsq(theta(2:end));
J = (1/m) * sum( -y .* log(h) - (1-y) .* log(1 - h) ) + regularization_term;

grad = ((1/m) * ((h-y)'*X))' + [0; (lambda/m)*theta(2:end)];
grad = grad(:);
end
