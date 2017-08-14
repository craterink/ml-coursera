function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

% A. Compute cost
hyp = X*Theta';
diff = hyp - Y;
masked_diff = R.*diff;
J = 1/2*sum(sum(masked_diff.^2));

% B. Compute cost gradient w.r.t. movie features
masked_diff_3d = permute(repmat(masked_diff, 1, 1, num_features), [1 3 2]);
theta_rep_3d = permute(repmat(Theta, 1, 1, num_movies), [3 2 1]);
mult_pre_sum = masked_diff_3d .* theta_rep_3d;
X_grad = sum(mult_pre_sum, 3);

% C. Compute cost gradient w.r.t. user features
masked_diff_3d = permute(masked_diff_3d, [3 2 1]);
x_rep_3d = permute(repmat(X, 1, 1, num_users), [3 2 1]);
mult_pre_sum = masked_diff_3d .* x_rep_3d;
Theta_grad = sum(mult_pre_sum, 3);


% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
