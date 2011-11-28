function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples
n = size(X, 2);

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
h = X * theta;

diff = h - y;
J1 = 1 / (2 * m) * sum(diff .^ 2);
J2 = 1 / (2 * m) * lambda * sum(theta(2:end) .^ 2);
J = J1 + J2;

grad(1) = 1 / m * sum(diff .* X(:, 1));
for i = 2:n
  grad(i) = 1 / m * sum(diff .* X(:, i)) + lambda / m * theta(i);
endfor
% =========================================================================

grad = grad(:);

end
