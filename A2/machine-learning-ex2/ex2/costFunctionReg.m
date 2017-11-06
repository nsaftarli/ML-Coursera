function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta



hypothesis = sigmoid(X * theta);
n = length(theta);

%size(hypothesis)%          size = 118 X 1
%size(log(hypothesis))%      size = 118 X 1

%size(y)%                    size = 118 X 1
%size(X)%                    size = 118 X 28
%size(theta)%                size =   28 X 1

costReg = (lambda/(2*m)) * sum(theta(2:end) .^ 2)

J = (-1/m) * sum((y .* log(hypothesis)) + (1-y) .* log(1-hypothesis)) + costReg;

%Two lines of code below are the same, only transposes of each other

%'X hypothesis y: '
%(X' * (hypothesis - y))/m

%'hypothesis y .* X'
%(1/m) * sum((hypothesis-y) .* X)
size((X' * (hypothesis - y))./m);
grads = ((X' * (hypothesis - y))./m);

grad(1) = grads(1);
grad(2:end) = grads(2:end) + (lambda/m) * theta(2:end);



% =============================================================

end
