function [gap] = Total_Energy(X,A,A0,classn,DL_par,DL_ipts)
% ========================================================================
% Total energy computation of FDDL, Version 1.0
% Copyright(c) 2011  Meng YANG, Lei Zhang, Xiangchu Feng and David Zhang
% All Rights Reserved.
%
% -----------------------------------------------------------------------
%   
% Input :   (1) X:          the data matrix of all class
%           (2) A:          the coefficient matrix of the whole class
%           (3) A0:         the shared coefficient matrix of the whole class
%           (4) classn:     the number of class
%           (5) DL_par
%                      .dls     labels of dictionary's column
%                      .lambda  parameter of l1-norm energy of coefficient
%                      .eta     parameter of within-class scatter
%           (6) DL_ipts
%                      .D       The dictioanry
%                      .trls    labels of training samples
% 
% Outputs : (1) gap  :    the total energy
%
%------------------------------------------------------------------------
 D  = DL_ipts.D;
 drls    = DL_par.dls;
 trls    = DL_ipts.trls;
 lambda1 = DL_par.lambda;
 eta2 = DL_par.eta;
 eta3 = eta2;
 eta4 = eta2;
 
gap3  =   0;
gap4  =   0;
GAP2  =   lambda1*sum(abs(A(:)));%
tem_A = A;
tem_A0 = A0;
for i_c = 1:classn
    t_A_ic  = tem_A(:,trls==i_c);
    t_A0_ic = tem_A0(:,trls==i_c);
    gap3 = gap3+norm(t_A_ic-repmat(mean(t_A_ic,2),[1 size(t_A_ic,2)]),'fro')^2;
    gap4 = gap4+size(t_A0_ic,2)*(mean(t_A0_ic,2)-mean(tem_A0,2))'*(mean(t_A0_ic,2)-mean(tem_A0,2));
end
      
GAP3 = eta2*gap3-eta3*gap4;

GAP1  =   0;
for i = 1:classn
    Xi = X(:,trls==i);
    Ai = A(:,trls==i);
    GAP1  = GAP1 + norm(Xi-D(:,drls==i)*Ai,'fro')^2;% only for one class, no effect
end
    
gap = GAP1+GAP2+GAP3;