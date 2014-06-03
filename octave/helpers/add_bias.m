function X = add_bias(X)
  %ADD_BIAS adds biases to input features
  m = size(X, 1);

  X = [ones(m, 1) X];
end
