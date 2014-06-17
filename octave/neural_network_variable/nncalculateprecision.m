function [precision] = nncalculateprecision(thetas, X, y)
  precision = sum(nnpredict(thetas,  X) == y) / size(y, 1)
end
