function [] = lrplotlearningcurve(X_train, y_train, X_val, y_val, num_learning_steps)

m = size(X_train, 1);
learning_step_size = floor(m / num_learning_steps);

[error_train, error_val] = lrlearningcurve(X_train, y_train, X_val, y_val, num_learning_steps, 7, 0)

learning_step_ranges = 1:learning_step_size:(learning_step_size * num_learning_steps);

plot(learning_step_ranges, error_train, learning_step_ranges, error_val);

title(sprintf('Logistic Learning Curve (steps = %i)', num_learning_steps));
xlabel('Number of training examples');
ylabel('F-Score');
legend('Train', 'Cross Validation');
end
