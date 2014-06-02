function [C, sigma, performance_val] = svm_params_finder(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

test_range = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

performance_val = zeros(64, 3);

i = 0;

yval = double(yval);
Xval = double(Xval);

for c = test_range
  for s = test_range
    i += 1;
    fprintf('Training SVM with C = %f and sigma = %f\n', c, s);
    fflush(stdout);
    model = svmtrain(y, X, sprintf('-t 2 -c %f -g %f', c, 1.0/s));
    fprintf('Predicting\n');
    fflush(stdout);
    predictions = svmpredict(yval, Xval, model);
    fprintf('Calculatin\n');
    fflush(stdout);
    %error_val = mean(double((predictions == 1) ~= yval));
    [precision, recall, f_score] = calculate_performance(model, Xval, yval)
    performance_val(i,:) = [c, s, f_score];
  end
end

[min_value, min_index] = max(performance_val(:, 3));

C = performance_val(min_index, 1);
sigma = performance_val(min_index, 2);






% =========================================================================

end
