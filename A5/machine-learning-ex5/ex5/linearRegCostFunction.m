function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

size(X);%    12x2
size(theta);%2x1
size(y);%    12x1
size(grad);% 2x1



h = X * theta;
unregCost = (1/(2*m)) * sum((h-y).^ 2);
regTerm = (lambda/(2*m)) * sum(theta(2:end).^ 2);
J = unregCost + regTerm;

%Theta1_grad = Theta1_grad/m;
%Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda/m) * Theta1(:,2:end);
derivative = X' * (h-y)/m;
grad = derivative + (lambda/m) * [0;theta(2:end)];
% =========================================================================

grad = grad(:);

end
