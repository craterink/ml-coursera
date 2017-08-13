function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%
[num_examples, num_dims] = size(X);
for i = 1:num_examples
   X_i = X(i, :);
   best_cen_dist = Inf;
   best_cen = NaN;
   for j = 1:K
       cen_j = centroids(j, :);
       dist = sum((X_i - cen_j).^2);
       if dist < best_cen_dist
           best_cen_dist = dist;
           best_cen = j;
       end
   end
   idx(i) = best_cen;
end





% =============================================================

end

