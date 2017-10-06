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


% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

%     1) Adding the 1's row to the X matrix
X = [ones(m, 1) X]; % Dont forget to add your one's

%     2) Converting vector 'y' to component (i think that's what it's called) matrix 'Y'

rows_in_y = size(y, 1);
Y = zeros(rows_in_y, num_labels); % Dimention 5000x10

% Put a 1 in the proper row and column for the 1-9 + 0 digits
for i = 1 : rows_in_y
  Y(i, y(i)) = 1;
end

% size(Y) % Dimention 5000x10, still, but now every row had a one in the column which represents the decimal value of it's y


% 3)  calculate a2 - I dont really under stand where this comes from, i didnt understand it in the lecture.  I found it in a student's explanatin of how he solved it

z2 = X * Theta1';
a2 = sigmoid(z2); % a2 is 5000x25

% size(a2)

% 4) calculate a3

a2 = [ones(m, 1) a2]; % Dont forget to add your one's 5000x26
z3 = a2 * Theta2';
a3 = sigmoid(z3); % 5000x10

% size(a3)

% 5) calculate J
h = a3;

% There was a conversation about sum(sum()) and using matricies to solve this.  Appearently this solution is simplier, but slower.


% Trevis, remember that no matter which matrix you transpose, the result is the same.  Just so happens that transposing Y results in a smaller matrix. (The sizes of the two products are 10x10, instead of 5000x5000)
% J = 1/m * trace((-Y * log(h)' ) - (1 - Y) * log(1 - h)');

J = 1/m * trace((-Y' * log(h) ) - (1 - Y)' * log(1 - h));

% J = 1/m * sum(sum((-Y .* log(h) ) - (1 - Y) .* log(1 - h)));

% The three above methods of calculating J give the same result.  Read this thing about the math to understand why: https://www.coursera.org/learn/machine-learning/discussions/all/threads/AzIrrO7wEeaV3gonaJwAFA

Theta1_ = Theta1(:,2:end); % 25x400
Theta2_ = Theta2(:,2:end); % 10x25

% 6)  Calculate the regularization term

regularized = (lambda / (2 * m)) * ((sum(sum(Theta1_ .^ 2))) +  (sum(sum(Theta2_ .^ 2))));

J = J + regularized;



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



d3 = a3 - Y; % 5000x10

%% δ2 or d2 is tricky. It uses the (:,2:end) columns of Theta2. d2 is the product of d3 and Theta2(no bias), then element-wise scaled by sigmoid gradient of z2. The size is (m x r) ⋅ (r x h) --> (m x h). The size is the same as z2, as must be.

d2 = d3 * Theta2_ .* sigmoidGradient(z2); % 5000x25

% Δ1 or Delta1 is the product of d2 and a1. The size is (h x m) ⋅ (m x n) --> (h x n)
% a1 = X(:,2:end);  % Just removing the bias.  a1 is 5000x400

a1 = X; % 5000x401

D1 = d2' * a1; % 25x401

% Δ2 or Delta2 is the product of d3 and a2. The size is (r x m) ⋅ (m x [h+1]) --> (r x [h+1])

D2 = d3' * a2; % 10x26


% Theta1_grad and Theta2_grad are the same size as their respective Deltas, just scaled by 1/m.

Theta1_grad_ = D1 .* (1/m);
Theta2_grad_ = D2 .* (1/m);

% keyboard % For debugging!

% To regularize the gradients, for each Delta matrix add its Theta matrix scaled by lambda/m. --WTF?
% Trevis, you never made heads of tails out of the above.  You went back to the instructions and figured out that you needed to add the lambda/m scaling to the unregularized Theta_grad!  Then, you figured out from a comment that you could zero the first col, to get rid of the bias from the calcultion

Theta1_z = Theta1;
Theta2_z = Theta2;

%Set the first col to zeros! This is how you adhere to "not regularizing the bias term"
Theta1_z(:,1) = 0;
Theta2_z(:,1) = 0;

Theta1_grad = Theta1_grad_ + ((lambda/m) .* Theta1_z);
Theta2_grad = Theta2_grad_ + ((lambda/m) .* Theta2_z);


% size(Theta1_grad) % 25x401
% size(Theta2_grad) % 10x26




















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients

grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
