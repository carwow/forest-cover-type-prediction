addpath ("./libsvm/matlab/");
addpath ("./helpers/");
addpath ("./logistic/");
addpath ("./neural_network_variable/");
[X_train, y_train, X_val, y_val, X_test, y_test] = load_data_set();

% Plot learning curve
%lrplotlearningcurve(X_train, y_train, X_val, y_val, 10);

[thetas] = nninit(X_train, y_train);

disp 'Precision Learning';
sum(nnpredict(thetas,  X_train) == y_train) / size(y_train, 1)
disp 'Precision Cross-Validation';
sum(nnpredict(thetas,  X_val) == y_val) / size(y_val, 1)
disp 'Precision Test';
sum(nnpredict(thetas,  X_test) == y_test) / size(y_test, 1)
