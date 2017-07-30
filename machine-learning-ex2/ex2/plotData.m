function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

pos_ex = find(y);
neg_ex = find(1-y);

pos_axis_1 = X(pos_ex, 1);
pos_axis_2 = X(pos_ex, 2);

neg_axis_1 = X(neg_ex, 1);
neg_axis_2 = X(neg_ex, 2);

plot(pos_axis_1, pos_axis_2, 'k+','LineWidth', 2, 'MarkerSize', 7);
plot(neg_axis_1, neg_axis_2, 'ko', 'MarkerFaceColor', 'y','MarkerSize', 7);


% =========================================================================



hold off;

end
