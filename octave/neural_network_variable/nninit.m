function [thetas] = nninit(X_train, y_train, dimensions)

fprintf('\nInitializing Neural Network Parameters ...\n')

[initial_nn_params] = calculate_initial_nn_params(dimensions);

fprintf('\nTraining Neural Network... \n')

max_iter = 1000;
[thetas, cost] = nntrain(dimensions, X_train, y_train, 0.3, initial_nn_params, max_iter);

%plot(1:max_iter, cost);

end
