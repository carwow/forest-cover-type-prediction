function [thetas, cost] = nntrain(dimensions, X, y, lambda, initial_nn_params, max_iter)

%  After you have completed the assignment, change the MaxIter to a larger
%  value to see how more training helps.
options = optimset('MaxIter', max_iter);

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nncostfunction(p, ...
                                   dimensions,
                                   X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain thetas back from nn_params
thetas = nnreshapethetas(nn_params, dimensions);

end
