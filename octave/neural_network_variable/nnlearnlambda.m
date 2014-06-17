function [errors] = nnlearnlambda(dimensions, X_train, y_train, X_val, y_val)

max_iter = 1000;
initial_nn_params = calculate_initial_nn_params(dimensions);

lambdas = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

for i=1:numel(lambdas)
  lambda = lambdas(i);
  [thetas, cost] = nntrain(dimensions, X_train, y_train, lambda, initial_nn_params, max_iter);
  errors(i) = nncalculateerror(thetas, X_val, y_val);
end
