function B = l2ls_learn_basis_dual(X, S, c, Binit)
% Learning basis using Lagrange dual (with basis normalization)
%
% This code solves the following problem:
% 
%    minimize_B   0.5*||X - B*S||^2											% X= training data, B= dict, S= coef
%    subject to   ||B(:,j)||_2 <= c, forall j=1...size(S,1)			%number of items in B

%%% number of items in X = L * N
L = size(X,1);
N = size(X,2);
M = size(S, 1);

tic
SSt = S*S';  
XSt = X*S';

if exist('Binit', 'var')
    dual_lambda = diag(Binit\XSt - SSt);
else
    dual_lambda = 10*abs(rand(M,1)); % any arbitrary initialization should be ok.
end

%c = 1;
trXXt = sum(sum(X.^2));

lb=zeros(size(dual_lambda));
options = optimset('GradObj','on', 'Hessian','on');
%  options = optimset('GradObj','on', 'Hessian','on', 'TolFun', 1e-7);

[x, fval, exitflag, output] = fmincon(@(x) fobj_basis_dual(x, SSt, XSt, X, c, trXXt), dual_lambda, [], [], [], [], lb, [], [], options);
% output.iterations
fval_opt = -0.5*N*fval;
dual_lambda= x;

Bt = (SSt+diag(dual_lambda)) \ XSt';
B_dual= Bt';
fobjective_dual = fval_opt;


B= B_dual;
fobjective = fobjective_dual;
toc

return;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [f,g,H] = fobj_basis_dual(dual_lambda, SSt, XSt, X, c, trXXt)			%函數值、gradient、hessian matrix
% Compute the objective function value at x
L= size(XSt,1);
M= length(dual_lambda);

SSt_inv = inv(SSt + diag(dual_lambda));

% trXXt = sum(sum(X.^2));
if L>M
    % (M*M)*((M*L)*(L*M)) => MLM + MMM = O(M^2(M+L))
    f = -trace(SSt_inv*(XSt'*XSt))+trXXt-c*sum(dual_lambda);
    
else
    % (L*M)*(M*M)*(M*L) => LMM + LML = O(LM(M+L))
    f = -trace(XSt*SSt_inv*XSt')+trXXt-c*sum(dual_lambda);
end
f= -f;

if nargout > 1   % fun called with two output arguments
    % Gradient of the function evaluated at x
    g = zeros(M,1);
    temp = XSt*SSt_inv;
    g = sum(temp.^2) - c;
    g= -g;
    
    
    if nargout > 2
        % Hessian evaluated at x
        % H = -2.*((SSt_inv*XSt'*XSt*SSt_inv).*SSt_inv);
        H = -2.*((temp'*temp).*SSt_inv);
        H = -H;
    end
end

return