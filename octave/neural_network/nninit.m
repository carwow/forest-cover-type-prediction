function [Theta1, Theta2] = nninit(X_train, y_train)

[m, n] = size(X_train);

input_layer_size  = n;
hidden_layer_size = 25;   % 25 hidden units
num_labels = 7;          % 10 labels, from 1 to 10   


fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = nnrandinitializeweights(input_layer_size, hidden_layer_size);
size(initial_Theta1)
initial_Theta2 = nnrandinitializeweights(hidden_layer_size, num_labels);
size(initial_Theta2)

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

fprintf('\nTraining Neural Network... \n')


[Theta1, Theta2] = nntrain(input_layer_size, hidden_layer_size, num_labels, X_train, y_train, 0, initial_nn_params)

end
