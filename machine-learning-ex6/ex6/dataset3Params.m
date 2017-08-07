function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
C = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma =[ 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
params = combvec(C, sigma);

best_pred_error = inf;
best_params = [0;0];

for i = 1:size(params, 2)
    params_i = params(:, i);
    C_i = params_i(1);
    sigma_i = params_i(2);
    
    model_i = svmTrain(X,y, C_i, ...
        @(x1,x2) gaussianKernel(x1,x2,sigma_i)); % let tol, max-passes be default
    pred_i = svmPredict(model_i, Xval);
    pred_err_i = mean(double(pred_i ~= yval));
    
    if(pred_err_i < best_pred_error)
        best_pred_error = pred_err_i;
        best_params = params_i;
    end
end

C = best_params(1);
sigma = best_params(2);
% =========================================================================

end
