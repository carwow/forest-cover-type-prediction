function [thetas] = nninit(X_train, y_train)

[m, n] = size(X_train);

input_layer_size  = n;
hidden_layer_size = round(50);   % 25 hidden units
num_labels = 7;          % 10 labels, from 1 to 10   

dimensions = [input_layer_size, num_labels];


fprintf('\nInitializing Neural Network Parameters ...\n')

[initial_nn_params] = calculate_initial_nn_params(dimensions);

fprintf('\nTraining Neural Network... \n')

max_iter = 100;
[thetas, cost] = nntrain(dimensions, X_train, y_train, 0, initial_nn_params, max_iter);

%plot(1:max_iter, cost);

end
