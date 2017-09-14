function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

s = size(z);

for row = 1:s(1,1)
  for col = 1:s(1,2)
    g(row,col) = 1 / ( 1 + e^-z(row,col));
  end
end

% This is the simple function.  But
% the version implemented here handles matricies too. g = 1 / ( 1 + e^-z)

% =============================================================

end
