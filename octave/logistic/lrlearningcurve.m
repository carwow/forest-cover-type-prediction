function [error_train, error_val] = lrlearningcurve(X, y, X_val, y_val, learning_steps, num_labels, lambda)
%LEARNING_CURVE returns learning curve data

m = size(X, 1);
step_size = floor(m / learning_steps);

error_train = zeros(learning_steps, 1);
error_val   = zeros(learning_steps, 1);

for i=1:learning_steps
  num_training_examples = i * step_size;
  X_train = X(1:num_training_examples, :);
  y_train = y(1:num_training_examples, :);
  disp(sprintf('Learning with %i training examples', num_training_examples));
  fflush(stdout);
  all_theta = lrtrain(X_train, y_train, num_labels, lambda, 50);
  disp(sprintf('Done learning with %i training examples', num_training_examples));
  fflush(stdout);

  error_train(i) = lrcalculateerror(all_theta, X_train, y_train);
  error_val(i) = lrcalculateerror(all_theta, X_val, y_val);
end

end
