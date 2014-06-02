function [all_theta] = lrtrain(X, y, num_labels, lambda, max_iter)
[m, n] = size(X);

all_theta = zeros(num_labels, n);

initial_theta = zeros(n, 1);
options = optimset('GradObj', 'on', 'MaxIter', max_iter);

for l = 1:num_labels
  all_theta(l, :) = fmincg (@(t)(lrcostfunction(t, X, (y == l), lambda)), initial_theta, options);
end

end
