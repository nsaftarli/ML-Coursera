function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%

hypothesis = sigmoid(X * theta);

size(hypothesis); %          size = 100 X 1
size(log(hypothesis));%      size = 100 X 1

size(y);%                    size = 100 X 1
size(X);%                    size = 100 X 3
size(theta);%                size =   3 X 1



J = (1/m) * sum((-y .* log(hypothesis)) - (1-y) .* log(1-hypothesis));

%Two lines of code below are the same, only transposes of each other

%'X hypothesis y: '
%(X' * (hypothesis - y))/m

%'hypothesis y .* X'
%(1/m) * sum((hypothesis-y) .* X)


   grad = (X' * (hypothesis - y))/m;

   

   
%dJ = 1/m * (X' * sum(hypothesis - y))
% theta = theta - alpha * dJ
%grad = theta - (alpha * dJ);







% =============================================================

end
