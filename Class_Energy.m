function [gap] = Class_Energy(Xi,D,A0,Ai,Aa,drls,trls,index,lambda1,eta2,eta3,eta4,classn)
% ========================================================================
% Class energy computation of FDDL, Version 1.0
% Copyright(c) 2011  Meng YANG, Lei Zhang, Xiangchu Feng and David Zhang
% All Rights Reserved.
%
% -----------------------------------------------------------------------
%   
% Input :   (1) Xi :  the data matrix of this class
%           (2) D :   the whole dictionary
%           (3) Ai:   the coefficient matrix of this class
%           (4) Aa:   the coefficient matrix of the whole class
%           (5) drls: labels of dictionary's column
%           (6) trls: labels of training samples
%           (7) index: label of class being processed
%           (8) lambda1 : parameter of l1-norm energy of coefficient
%           (9) eta2 : parameter of within-class scatter
%           (10) eta3 : parameter of between-class scatter
%%          (11) eta4:  parameter of l2-norm energy of coefficient
%           (12) classn:   the number of class
% 
% Outputs : (1) gap  :    the total energy of some class
%
%------------------------------------------------------------------------

gap3  =   0;
gap4  =   0;
GAP1  =   norm((Xi-D(:, drls==index)*Ai),'fro')^2;% only for one class, no effect
GAP2  =   lambda1*sum(abs(Ai(:)));%
    
Aa(:,trls==index)  =  Ai;
tem_XA             =  Aa;
mean_xa            =  mean(tem_XA,2);       %% column vector containing the mean of each row.
tem_A0             =  A0;                   %% shared coef
mean_A0            =  mean(tem_A0,2);       
    
n_c                =  size(Ai,2);
for i_c = 1:classn
    t_A0_ic   = tem_A0(:,trls==i_c);
    n_ic      = size(t_A0_ic,2);
    mean_a0ic = mean(t_A0_ic,2);
    gap4 = gap4+n_ic*(mean_a0ic-mean_A0)'*(mean_a0ic-mean_A0);
end
    
gap3 = norm(Ai-repmat(mean(Ai,2),[1 n_c]),'fro')^2;

GAP3 = eta2*gap3-eta3*gap4;

    
gap = GAP1+GAP2+GAP3;