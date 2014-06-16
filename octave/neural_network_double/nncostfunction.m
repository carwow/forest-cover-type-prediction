function [J grad] = nncostfunction(nn_params, ...
                                   dimensions,
                                   X, y, lambda)


thetas = nnreshapethetas(nn_params, dimensions);

% Setup some useful variables
m = size(X, 1);
num_of_layers = numel(dimensions);
num_of_thetas = num_of_layers - 1;

num_labels = dimensions(num_of_layers);
input_layer_size = dimensions(1);

a = cell(num_of_layers, 1);
z = cell(num_of_layers, 1);

% Feed forward
a{1} = [ones(1, m); X'];

for t = 1:(num_of_thetas-1)
  z{t+1} = thetas{t} * a{t};
  a{t+1} = [ones(1, m); sigmoid(z{t+1})];
end

z{num_of_layers} = thetas{num_of_thetas} * a{num_of_layers-1};
a{num_of_layers} = sigmoid(z{num_of_layers});

% Explode y into num_labels values with Y[i] := i == y.
Y = zeros(num_labels, m);
Y(sub2ind(size(Y), y', 1:m)) = 1;

% Compute the non-regularized error. Fully vectorized, at the expense of
% having an expanded Y in memory (which is 1/40th the size of X, so it should be
% fine).
J = (1/m) * sum(sum(-Y .* log(a{num_of_layers}) - (1 - Y) .* log(1 - a{num_of_layers})));

% Add regularized error. Drop the bias terms in the 1st columns.
for t=1:num_of_thetas
  J = J + (lambda / (2*m)) * sum(sum(thetas{t}(:, 2:end) .^ 2));
end


% 2. Backpropagate to get gradient information.
d = cell(num_of_layers, 1);

d{num_of_layers} = a{num_of_layers} - Y;

i = num_of_layers - 1;
while i > 1
  d{i} = (thetas{i}' * d{i+1}) .* [ones(1, m); sigmoidGradient(z{i})];  % 26 x m
  d{i} = d{i}(2:end, :);
  i = i - 1;
end

thetas_grad = cell(num_of_thetas, 1);

grad = [];
% Vectorized ftw:
for t = 1:num_of_thetas
  reg = (lambda / m) * ([zeros(size(thetas{t}, 1), 1), thetas{t}(:, 2:end)]);
  theta_grad = (1/m) * d{t+1} * a{t}' + reg;
  grad = [grad; theta_grad(:)];
end

end
