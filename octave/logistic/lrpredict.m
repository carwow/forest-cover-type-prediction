function [predictions] = lrpredict(theta, X)
  predictions = sigmoid(X * theta) >= 0.1;
end
