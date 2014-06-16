function [thetas] = nnreshapethetas(nn_params, dimensions)

num_of_thetas = numel(dimensions) - 1;

thetas = cell(num_of_thetas, 1);

last_theta_end = 0;

for i = 1:num_of_thetas
  start_theta = last_theta_end + 1;
  end_theta = (dimensions(i) + 1) * dimensions(i+1) + start_theta - 1; 
  last_theta_end = end_theta;
  thetas{i} = reshape(nn_params(start_theta:end_theta), dimensions(i+1), dimensions(i) + 1); 
end

end
