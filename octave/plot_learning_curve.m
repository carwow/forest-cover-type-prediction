function [] = plot_learning_curve(X_train, y_train, X_val, y_val, num_learning_steps)
%PLOT_LEARNING_CURVE plots the learning curve

m = size(X_train, 1);
learning_step_size = floor(m / num_learning_steps);

[fscore_train, fscore_val] = learning_curve(X_train, y_train, X_val, y_val, num_learning_steps);

learning_step_ranges = 1:learning_step_size:(learning_step_size * num_learning_steps);

plot(learning_step_ranges, fscore_train, learning_step_ranges, fscore_val);

title(sprintf('SVM Learning Curve (steps = %i)', num_learning_steps));
xlabel('Number of training examples');
ylabel('F-Score');
legend('Train', 'Cross Validation');
end
