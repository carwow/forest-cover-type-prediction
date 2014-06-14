function [Theta1, Theta2, Theta3, cost] = nntrain(input_layer_size, hidden_layer_size, num_labels, X, y, lambda, initial_nn_params, max_iter)

%  After you have completed the assignment, change the MaxIter to a larger
%  value to see how more training helps.
options = optimset('MaxIter', max_iter);

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nncostfunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 and Theta3 back from nn_params
[Theta1, Theta2, Theta3] = nnreshapethetas(nn_params, input_layer_size, hidden_layer_size, num_labels);

end
