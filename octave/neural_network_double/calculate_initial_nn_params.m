function [initial_nn_params] = calculate_initial_nn_params(dimensions)

num_of_layers = numel(dimensions);
num_of_thetas = num_of_layers - 1;


thetas = zeros(num_of_thetas, 1);
initial_nn_params = [];

for i = 1:num_of_thetas
  theta = nnrandinitializeweights(dimensions(i), dimensions(i+1)); 
  initial_nn_params = [initial_nn_params ; theta(:)];
end

end
