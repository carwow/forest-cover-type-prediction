function [initial_nn_params] = calculate_initial_nn_params(input_layer_size, hidden_layer_size, num_labels)

initial_Theta1 = nnrandinitializeweights(input_layer_size, hidden_layer_size);
size(initial_Theta1)
initial_Theta2 = nnrandinitializeweights(hidden_layer_size, num_labels);
size(initial_Theta2)

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];
end
