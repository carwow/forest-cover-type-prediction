addpath ("./libsvm/matlab/");
addpath ("./helpers/");
addpath ("./logistic/");
addpath ("./neural_network_variable/");
[X_train, y_train, X_val, y_val, X_test, y_test, X, y, X_submission] = load_data_set();

% Plot learning curve
%lrplotlearningcurve(X_train, y_train, X_val, y_val, 10);

[m, n] = size(X_train);

input_layer_size  = n;
num_labels = 7;          % 10 labels, from 1 to 10   
hidden_layers_size = round(input_layer_size * 1.25);

%disp '----------submit-----------'
%dimensions = [input_layer_size, hidden_layers_size, hidden_layers_size, num_labels];
%[thetas] = nninit(X, y, dimensions);
%disp 'Predicting';
%predictions = nnpredict(thetas, X_submission);

%disp '------------ No hidden layer ------------'
%dimensions = [input_layer_size, num_labels];
%[thetas] = nninit(X_train, y_train, dimensions);
%disp 'Precision Learning';
%nncalculateprecision(thetas, X_train, y_train)
%disp 'Precision Cross-Validation';
%nncalculateprecision(thetas, X_val, y_val)
%disp 'Precision Test';
%nncalculateprecision(thetas, X_test, y_test)



%disp '------------ One hidden layer ------------'
%dimensions = [input_layer_size, hidden_layers_size, num_labels];
%[thetas] = nninit(X_train, y_train, dimensions);
%disp 'Precision Learning';
%nncalculateprecision(thetas, X_train, y_train)
%disp 'Precision Cross-Validation';
%nncalculateprecision(thetas, X_val, y_val)
%disp 'Precision Test';
%nncalculateprecision(thetas, X_test, y_test)

%disp '------------ Two hidden layers ------------'
%dimensions = [input_layer_size, hidden_layers_size, hidden_layers_size, num_labels];
%%[errors] = nnlearnlambda(dimensions, X_train, y_train, X_val, y_val)
%[thetas] = nninit(X_train, y_train, dimensions);
%disp 'Precision Learning';
%nncalculateprecision(thetas, X_train, y_train)
%disp 'Precision Cross-Validation';
%nncalculateprecision(thetas, X_val, y_val)
%disp 'Precision Test';
%nncalculateprecision(thetas, X_test, y_test)

%disp '------------ Three hidden layers ------------'
%dimensions = [input_layer_size, hidden_layers_size, hidden_layers_size, hidden_layers_size, num_labels];
%[thetas] = nninit(X_train, y_train, dimensions);
%disp 'Precision Learning';
%nncalculateprecision(thetas, X_train, y_train)
%disp 'Precision Cross-Validation';
%nncalculateprecision(thetas, X_val, y_val)
%disp 'Precision Test';
%nncalculateprecision(thetas, X_test, y_test)

%disp '------------ Four hidden layers ------------'
%dimensions = [input_layer_size, hidden_layers_size, hidden_layers_size, hidden_layers_size, hidden_layers_size, num_labels];
%[thetas] = nninit(X_train, y_train, dimensions);
%disp 'Precision Learning';
%nncalculateprecision(thetas, X_train, y_train)
%disp 'Precision Cross-Validation';
%nncalculateprecision(thetas, X_val, y_val)
%disp 'Precision Test';
%nncalculateprecision(thetas, X_test, y_test)
