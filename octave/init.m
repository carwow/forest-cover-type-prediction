addpath ("./libsvm/matlab/");
addpath ("./helpers/");
addpath ("./logistic/");
[X_train, y_train, X_val, y_val, X_test, y_test] = load_data_set();

% Plot learning curve
%lrplotlearningcurve(X_train, y_train, X_val, y_val, 10);
