function [J grad] = nncostfunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)


[Theta1, Theta2, Theta3] = nnreshapethetas(nn_params, input_layer_size, hidden_layer_size, num_labels);

% Setup some useful variables
m = size(X, 1);

% 1. Feed-forward to compute h = a4.
a1 = [ones(1, m); X'];
z2 = Theta1 * a1;
a2 = [ones(1, m); sigmoid(z2)];
z3 = Theta2 * a2;
a3 = [ones(1, m); sigmoid(z3)];
z4 = Theta3 * a3;
a4 = sigmoid(z4);

% Explode y into 10 values with Y[i] := i == y.
Y = zeros(num_labels, m);
Y(sub2ind(size(Y), y', 1:m)) = 1;

% Compute the non-regularized error. Fully vectorized, at the expense of
% having an expanded Y in memory (which is 1/40th the size of X, so it should be
% fine).
J = (1/m) * sum(sum(-Y .* log(a4) - (1 - Y) .* log(1 - a4)));

% Add regularized error. Drop the bias terms in the 1st columns.
J = J + (lambda / (2*m)) * sum(sum(Theta1(:, 2:end) .^ 2));
J = J + (lambda / (2*m)) * sum(sum(Theta2(:, 2:end) .^ 2));
J = J + (lambda / (2*m)) * sum(sum(Theta3(:, 2:end) .^ 2));


% 2. Backpropagate to get gradient information.
d4 = a4 - Y;
d3 = (Theta3' * d4) .* [ones(1, m); sigmoidGradient(z3)];  % 26 x m
d3 = d3(2:end, :);
d2 = (Theta2' * d3) .* [ones(1, m); sigmoidGradient(z2)];  % 26 x m
d2 = d2(2:end, :);

% Vectorized ftw:
Theta3_grad = (1/m) * d4 * a3';
Theta2_grad = (1/m) * d3 * a2';
Theta1_grad = (1/m) * d2 * a1';

% Add gradient regularization.
Theta3_grad = Theta3_grad + ...
              (lambda / m) * ([zeros(size(Theta3, 1), 1), Theta3(:, 2:end)]);
Theta2_grad = Theta2_grad + ...
              (lambda / m) * ([zeros(size(Theta2, 1), 1), Theta2(:, 2:end)]);
Theta1_grad = Theta1_grad + ...
              (lambda / m) * ([zeros(size(Theta1, 1), 1), Theta1(:, 2:end)]);

% Unroll gradients.
grad = [Theta1_grad(:) ; Theta2_grad(:) ; Theta3_grad(:)];

end
