function [gap] = Class_Energy(Xi,D,A0,MUA0,HA,Ai,Aa,drls,trls,index,lambda1,eta2,eta3,eta_2,classn)
% ========================================================================
% Class energy computation of FDDL, Version 1.0
% Copyright(c) 2011  Meng YANG, Lei Zhang, Xiangchu Feng and David Zhang
% All Rights Reserved.
%
% -----------------------------------------------------------------------
%   
% Input :   (1) Xi :  the data matrix of this class
%           (2) D :   the whole dictionary
%           (3) A0:   the shared coefficient matrix of all the classes
%           (4) MUA0:   the mean of the shared coefficient matrix of the upper classes
%           (5) HA:   the head coefficient matrix of all the classes
%           (6) Ai:   the coefficient matrix of this class
%           (7) Aa:   the coefficient matrix of all the classes
%           (8) drls: labels of dictionary's column
%           (9) trls: labels of training samples
%           (10) index: label of class being processed
%           (11) lambda1 : parameter of l1-norm energy of coefficient
%           (12) eta2 : parameter of within-class scatter
%           (13) eta3 : parameter of between-class scatter
%           (14) eta3 : parameter of upper within and between-class scatter
%           (15) classn:   the number of class
% 
% Outputs : (1) gap  :    the total energy of some class
%
%------------------------------------------------------------------------

gap3  =   0;
gap4  =   0;
GAP1  =   norm((Xi-D(:, drls==index)*Ai),'fro')^2;      % only for one class, no effect
GAP2  =   lambda1*sum(abs(Ai(:)));%
    
Aa(:,trls==index)  =  Ai;
tem_Aa             =  Aa;
mean_Aa            =  mean(tem_Aa,2);       
tem_A0             =  A0;                   %% shared coef
mean_A0            =  mean(tem_A0,2);
tem_HA			   =  HA;					%% head coef 
mean_HA			   =  mean(tem_HA,2)      	
    
for i_c = 1:classn
	t_HA_ic   = tem_HA(:,trls==i_c);
    n_hic     = size(t_HA_ic,2);
    mean_haic = mean(t_HA_ic,2);
    gap4      = gap4+n_hic*(mean_haic-mean_HA)'*(mean_haic-mean_HA);		%Sb

    t_A0_ic   = tem_A0(:,trls==i_c);
    n_ic0     = size(t_A0_ic,2);
    mean_a0ic = mean(t_A0_ic,2);
    gap6 	  = gap6+n_ic0*(mean_a0ic-MUA0)'*(mean_a0ic-MUA0);				%Sb'
end

n_c  =  size(Ai,2);
n_c0 =  size(A0,2);
gap3 = norm(Ai-repmat(mean(Ai,2),[1 n_c]),'fro')^2;		%Sw
gap5 = norm(A0-repmat(mean(A0,2),[1 n_c]),'fro')^2;		%Sw'

GAP3 = eta2*gap3-eta3*gap4; %eta*(Sw-Sb)
GAP4 = eta_2*gap5-eta_2*gap6; %eta*(Sw'-Sb')
    
gap = GAP1+GAP2+GAP3+GAP4;