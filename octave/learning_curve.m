function [fscore_train, fscore_val] = learning_curve(X, y, X_val, y_val, learning_steps)
%LEARNING_CURVE returns learning curve data

m = size(X, 1);
step_size = floor(m / learning_steps);

fscore_train = zeros(learning_steps, 1);
fscore_val   = zeros(learning_steps, 1);

for i=1:learning_steps
  num_training_examples = i * step_size;
  X_train = X(1:num_training_examples, :);
  y_train = y(1:num_training_examples, :);
  disp(sprintf('Learning with %i training examples', num_training_examples));
  fflush(stdout);
  model = svmtrain(y_train, X_train, '-t 2 -c 0.1');
  disp(sprintf('Done learning with %i training examples', num_training_examples));
  fflush(stdout);

  [precision, recall, f_score] = calculate_performance(model, X_train, y_train);
  fscore_train(i, 1) = f_score;

  [precision, recall, f_score] = calculate_performance(model, X_val, y_val);
  fscore_val(i, 1) = f_score;
end

end
