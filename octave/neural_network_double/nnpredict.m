function p = nnpredict(thetas, X)
% Useful values
m = size(X, 1);
num_of_thetas = size(thetas, 1);
num_labels = size(thetas{num_of_thetas}, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

h = cell(num_of_thetas, 1);

h{1} = sigmoid([ones(m, 1) X] * thetas{1}');

for t = 2:num_of_thetas
  h{t} = sigmoid([ones(m, 1) h{t-1}] * thetas{t}');
end

[dummy, p] = max(h{num_of_thetas}, [], 2);

% =========================================================================


end
