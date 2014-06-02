function [p] = lrpredict(all_theta, X)
predictions = sigmoid(X * all_theta');
[k, p] = max(predictions, [], 2);
end
