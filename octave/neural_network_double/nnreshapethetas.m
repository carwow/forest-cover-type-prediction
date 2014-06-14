function [Theta1, Theta2, Theta3] = nnreshapethetas(nn_params, input_layer_size, hidden_layer_size, num_labels)

start_Theta1 = 1;
end_Theta1 = hidden_layer_size * (input_layer_size + 1);
Theta1 = reshape(nn_params(start_Theta1:end_Theta1), hidden_layer_size, (input_layer_size + 1));

start_Theta2 = end_Theta1 + 1;
end_Theta2 = start_Theta2 + (hidden_layer_size * (hidden_layer_size + 1)) - 1;
Theta2 = reshape(nn_params(start_Theta2:end_Theta2), ...
                 hidden_layer_size, (hidden_layer_size + 1));

start_Theta3 = end_Theta2 + 1;
end_Theta3 = start_Theta3 + (num_labels * (hidden_layer_size + 1)) - 1;
Theta3 = reshape(nn_params(start_Theta3:end_Theta3), ...
                 num_labels, (hidden_layer_size + 1));

end
