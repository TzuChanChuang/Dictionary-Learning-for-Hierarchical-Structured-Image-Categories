function [grad,grad2] = Gradient_Comp(Xi,Xa,classn,index,eta2,eta3,eta_2,...
    trls,drls,newpar,BAI,CJ,SharedD_nClass)
% ========================================================================
% IPM's Gradient computation of FDDL, Version 1.0
% Copyright(c) 2011  Meng YANG, Lei Zhang, Xiangchu Feng and David Zhang
% All Rights Reserved.
%
% -----------------------------------------------------------------------
%   
% Input :   (1) Xi:   the coefficient matrix of this class
%           (2) Xa:   the coefficient matrix of the whole class
%           (3) classn:   the number of class
%           (4) index: label of class being processed
%           (5) eta2 : parameter of within-class scatter
%           (6) eta3 : parameter of between-class scatter
%           (7) eta_2: parameter of between-class scatter
%           (8) trls: labels of training samples
%           (9) drls: labels of dictionary's column
%           (10) newpar, BAI, CJ: the precomputed data 
%           (11) SharedD_nClass:  number of column of shared dict
%
% Outputs : (1) grad  :    the gradient vector of coding model
%
%------------------------------------------------------------------------

n_d             =      newpar.n_d;                % the sample number of i-th training data
n               =      newpar.n;
C_j             =      newpar.C_j;
C_line          =      newpar.C_line;
DD              =      newpar.DD;
DAi             =      newpar.DAi;
BiBi            =      newpar.BiBi;
BaiBai          =      newpar.BaiBai;
BaiGxi          =      newpar.BaiGxi;
CjCj            =      newpar.CjCj;
m               =      newpar.m;
for k = 1:classn
    Z(k).Matrix   =   Xa(:,trls==k)*BAI(k).M-Xa*C_line+Xi*C_j+Xa(:,trls==k)*CJ(k).M;
end
X0T      =   (Xi(1:SharedD_nClass,:))';
XiT      =   (Xi(SharedD_nClass+1:end,:))';

tem      =   2*DD*Xi-2*DAi;     %D*Ai-Xi
grad1    =   tem(:);




tem       = [2*eta_2*BiBi*X0T 2*eta2*BiBi*XiT];
grad4     = tem(:);

tem       =  [-eta_2*(2*BaiBai*X0T-2*BaiGxi(:,1:SharedD_nClass)) -eta3*(2*BaiBai*XiT-2*BaiGxi(:,SharedD_nClass+1:end))];
grad5     = tem(:);

grad6 = zeros(size(grad5));
for k = 1:classn
    temz  =  Z(k).Matrix;
    if k~=index
        tem = [-eta_2*(2*CjCj*X0T-2*C_j*(temz(1:SharedD_nClass,:))') -eta3*(2*CjCj*XiT-2*C_j*(temz(SharedD_nClass+1:end,:))')];
        grad6 = grad6+tem(:);
    end
end

tem       =  [2*eta_2*X0T 2*eta2*XiT];
grad10     =  tem(:);


grad456 = reshape(grad4+grad5+grad6+grad10,[n_d m])';
grad456 = grad456(:);

grad = grad1+grad456;

