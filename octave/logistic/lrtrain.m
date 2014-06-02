function [theta] = lrtrain(X_train, y_train, lambda)
m = size(X_train, 1);
n = size(X_train, 2);

theta = zeros(n, 1);
options = optimset('GradObj', 'on', 'MaxIter', 50);
[theta] = fminunc (@(t)(lrcostfunction(t, X_train, y_train, lambda)), theta, options);

end

