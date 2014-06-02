function [X_train, y_train, X_val, y_val, X_test, y_test] = load_data_set()
  disp 'Loading dataset...';
  load -ascii '../data/train.m';
  disp 'Done!';
  fflush(stdout);

  Xy = train(:, 2:end);

  [m, n] = size(Xy);

  X = Xy(:, 1:n-1);
  y = Xy(:, n);

  X = double(X);
  y = double(y);

  train_range_start = 1;
  train_range_end = floor(m * 0.6);
  val_range_start = train_range_end + 1;
  val_range_end = val_range_start + floor(m * 0.2);
  test_range_start = val_range_end + 1;
  test_range_end = m;

  fprintf('Training range: %i, %i\n', train_range_start, train_range_end);
  fprintf('Validation range: %i, %i\n', val_range_start, val_range_end);
  fprintf('Test range: %i, %i\n', test_range_start, test_range_end);
  fflush(stdout);


  X_train = X(train_range_start:train_range_end, :);
  y_train = double(y(train_range_start:train_range_end, :));
  [X_train, mu, sigma] = feature_normalize(X_train);

  X_val = X(val_range_start:val_range_end, :);
  y_val = double(y(val_range_start:val_range_end, :));
  X_val = feature_normalize(X_val, mu, sigma);

  X_test = X(test_range_start:test_range_end, :);
  y_test = double(y(test_range_start:test_range_end, :));
  X_test = feature_normalize(X_test, mu, sigma);

  X_train = double(add_bias(X_train));
  X_val = double(add_bias(X_val));
  X_test = double(add_bias(X_test));
end
