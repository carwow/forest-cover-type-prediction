function [J grad] = nncostfunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

training_set_y = [];
for n=1:num_labels
  training_set_y = [training_set_y n==y];
end

Delta_1 = 0;
Delta_2 = 0;

for i=1:m
  xi = X(i, :);
  yi = training_set_y(i, :);
  a1 = [1 xi];
  z2 = a1 * Theta1';
  a2 = [1 sigmoid(z2)];
  z3 = a2 * Theta2';
  a3 = sigmoid(z3);

  hypothesis = a3;

  J += 1/m * sum(-yi .* log(hypothesis) - (1 - yi) .* log(1 - hypothesis));

  delta_3 = a3 - yi;
  delta_2 = delta_3 * Theta2;
  delta_2 = delta_2(2:end) .* sigmoidGradient(z2);

  Delta_1 = Delta_1 + delta_2' * a1;
  Delta_2 = Delta_2 + delta_3' * a2;

end

Theta1_grad = Delta_1/m;
Theta2_grad = Delta_2/m;

reg = (lambda / (2 * m)) * (sumsq(Theta1(:, 2:end)(:)) + sumsq(Theta2(:, 2:end)(:)));

J += reg;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
