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
%
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
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Feedforward computation
a_1 = [ones(m, 1) X];
z_2 = a_1 * Theta1';
a_2 = sigmoid(z_2);
a_2 = [ones(m, 1) a_2];
z_3 = a_2 * Theta2';
a_3 = sigmoid(z_3);

% One hot encoding y output - Method #1
% y_encoded = double(bsxfun(@eq, 1:num_labels, y));
% One hot encoding y output - Method #2
y_encoded = eye(num_labels)(y,:);

% Cost computation - Method #1
% tmp = (log(a_3).*y_encoded) + (log(1-a_3).*(1-y_encoded));
% J = -sum(tmp(:))/m;

% Cost computation - Method #2
tmp = (log(a_3')*y_encoded) + (log(1-a_3')*(1-y_encoded));
J = -trace(tmp)/m;

% Add regularization to cost
Theta1_squared = Theta1(:,2:end).^2;
Theta2_squared = Theta2(:,2:end).^2;
regularization_value = sum(Theta1_squared(:)) + sum(Theta2_squared(:));
J += regularization_value*lambda/(2*m);

% BACK-PROPAGATION
% Method #1 - for loop
% accumulated_gradient_1 = zeros(size(Theta1));
% accumulated_gradient_2 = zeros(size(Theta2));
% for i = 1:m
%     delta_3 = (a_3(i,:) - y_encoded(i,:))';
%     delta_2 = (Theta2(:,2:end)' * delta_3) .* sigmoidGradient(z_2(i,:)');
%     accumulated_gradient_1 += delta_2 * a_1(i,:);
%     accumulated_gradient_2 += delta_3 * a_2(i,:);
% end
% Theta1_grad = accumulated_gradient_1/m;
% Theta2_grad = accumulated_gradient_2/m;

% Method #2 - Vectorization
delta_3 = a_3 - y_encoded;  % 5000x10
delta_2 = delta_3 * Theta2(:,2:end) .* sigmoidGradient(z_2); % 5000x25
Theta1_grad = delta_2' * a_1 / m; % 25x401
Theta2_grad = delta_3' * a_2 / m; % 10x26

% Add regularization to gradient of thetas
Theta1_grad(:,2:end) += Theta1(:,2:end)*lambda/m;
Theta2_grad(:,2:end) += Theta2(:,2:end)*lambda/m;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
