function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

    % A. feed forward the NN
        % i. add bias unit column to X
        X = addBiasColumn(X);
        % ii. compute hidden layer outputs (rows are the output vectors)
        hlo = sigmoid(X * Theta1');
        hlo = addBiasColumn(hlo);
        % iii. compute outputs
        outputs = sigmoid(hlo * Theta2');
        
    % B. compute regularized cost
        % i. generate binary label matrix
        y_mat = repmat(y, 1, num_labels) == repmat(1:num_labels, m, 1);
        % ii. compute cost for each training example
        cost = sum(logCost(y_mat, outputs), 2);
        % iii. average that shit
        J = mean(cost);

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.


del_1 = zeros(size(Theta1));
del_2 = zeros(size(Theta2));
for t = 1:m
    % A. init column vectors pertinent to example t
    X_t = X(t, :)';
    ymat_t = y_mat(t, :)';
    output_t = outputs(t, :)';
    hlo_t = hlo(t, :)';
    
    % B. compute error terms
    output_err = output_t - ymat_t;
    hli =  Theta1 * X_t;
    hlo_err = Theta2' * output_err .* sigmoidGradient([0; hli]);
    hlo_err = hlo_err(2:end);
    
    % C. compute del by accumulating error terms
    delta3 = output_err;
    a2 = hlo_t;
    delta2 = hlo_err;
    a1 = X_t;
    del_2 = del_2 + delta3 * a2';
    del_1 = del_1 + delta2 * a1';
    
end

Theta1_grad = (1/m)*del_1;
Theta2_grad = (1/m)*del_2;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.

t1_reg = removeBiasColumn(Theta1);
t2_reg = removeBiasColumn(Theta2);
t1_mag = t1_reg .* t1_reg;
t2_mag = t2_reg .* t2_reg;
J = J + lambda/2/m*(sum(sum(t1_mag)) + sum(sum(t2_mag)));

















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients

grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
