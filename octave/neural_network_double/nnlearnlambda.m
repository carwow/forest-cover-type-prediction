function [errors] = nnlearnlambda(X_train, y_train, X_val, y_val)

[m, n] = size(X_train);
input_layer_size  = n;
hidden_layer_size = round(n*1.25);   % 25 hidden units
num_labels = 7;          % 10 labels, from 1 to 10   
max_iter = 1000;
initial_nn_params = calculate_initial_nn_params(input_layer_size, hidden_layer_size, num_labels);

lambdas = [2, 3, 4, 5];

for i=1:numel(lambdas)
  lambda = lambdas(i);
  [Theta1, Theta2, Theta3, cost] = nntrain(input_layer_size, hidden_layer_size, num_labels, X_train, y_train, lambda, initial_nn_params, max_iter);
  errors(i) = nncalculateerror(Theta1, Theta2, Theta3, X_val, y_val);
end
