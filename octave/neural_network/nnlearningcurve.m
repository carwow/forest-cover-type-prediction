function [error_train, error_val] = lrlearningcurve(X, y, X_val, y_val, learning_steps, num_labels, lambda)
%LEARNING_CURVE returns learning curve data

[m,n] = size(X);
step_size = floor(m / learning_steps);
input_layer_size = n;
hidden_layer_size = n;

initial_nn_params = calculate_initial_nn_params(input_layer_size, hidden_layer_size, num_labels);

error_train = zeros(learning_steps, 1);
error_val   = zeros(learning_steps, 1);

for i=1:learning_steps
  num_training_examples = i * step_size;
  X_train = X(1:num_training_examples, :);
  y_train = y(1:num_training_examples, :);
  disp(sprintf('Learning with %i training examples', num_training_examples));
  fflush(stdout);
  [Theta1, Theta2] = nntrain(input_layer_size, hidden_layer_size, num_labels, X_train, y_train, lambda, initial_nn_params, 1000);
  disp(sprintf('Done learning with %i training examples', num_training_examples));
  fflush(stdout);

  error_train(i) = nncalculateerror(Theta1, Theta2, X_train, y_train);
  error_val(i) = nncalculateerror(Theta1, Theta2, X_val, y_val);
end

end
