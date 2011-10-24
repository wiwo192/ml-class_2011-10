function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.
% Method 1:
% h = zeros(m,1);
% for i = 1:m
%   % h(i) = h(i) + X(i, 0) * theta(0) + X(i, 1) * theta(1);
%   h(i) = h(i) + X(i, :) * theta;
% endfor
% Method 2:
h = X * theta;

% % d: diff (mx1 vector)
% d = h - y;
% % d_sq: diff (mx1 vector)
% d_sq = d .^ 2;
% % J: cost (1x1 value)
% J = sum(d_sq) / (2 * m);

J = sum((h - y) .^ 2, 1) / (2 * m);
% =========================================================================

end
