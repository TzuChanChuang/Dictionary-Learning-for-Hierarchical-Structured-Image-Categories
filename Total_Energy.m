function [gap] = Total_Energy(X,A,A0,HA,classn,DL_par,DL_ipts)
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
%           (4) HA:         the head coefficient matrix of the whole class
%           (5) classn:     the number of class
%           (6) DL_par
%                      .dls     labels of dictionary's column
%                      .tau     parameter of l1-norm energy of coefficient
%                      .eta     parameter of within-class scatter
%                      .eta_2   parameter of upper within-class scatter
%           (7) DL_ipts
%                      .D       The dictioanry
%                      .trls    labels of training samples
%                      .MUSA    the mean of the shared coefficient matrix of the upper classes
% 
% Outputs : (1) gap  :    the total energy
%
%------------------------------------------------------------------------
 D  = DL_ipts.D;
 MUSA = DL_ipts.MUSA;
 drls    = DL_par.dls;
 trls    = DL_ipts.trls;
 lambda1 = DL_par.tau;
 eta2 = DL_par.eta;
 eta3 = eta2;
 eta4 = eta2;
 eta_2 = DL_par.eta_2;
 
gap3  =   0;
gap4  =   0;
gap5  =   0;
gap6  =   0;
GAP2  =   lambda1*sum(abs(A(:)));%
tem_A = A;
tem_A0 = A0;
tem_HA = HA;
for i_c = 1:classn
    t_A_ic  = tem_A(:,trls==i_c);       %Ai of i_cth class
    t_A0_ic = tem_A0(:,trls==i_c);      %A0 of i_cth class
    t_HA_ic = tem_HA(:,trls==i_c);      %Ai^ of i_cth class
    gap3 = gap3+norm(t_A_ic-repmat(mean(t_A_ic,2),[1 size(t_A_ic,2)]),'fro')^2;                         %Sw
    gap4 = gap4+size(t_HA_ic,2)*(mean(t_HA_ic,2)-mean(tem_HA,2))'*(mean(t_HA_ic,2)-mean(tem_HA,2));     %Sb
    gap5 = gap5+norm(t_A0_ic-repmat(mean(t_A0_ic,2),[1 size(t_A0_ic,2)]),'fro')^2;
    gap6 = gap6+size(t_A0_ic,2)*(mean(t_A0_ic,2)-MUSA)'*(mean(t_A0_ic,2)-MUSA);;
end
      
GAP3 = eta2*gap3-eta3*gap4;         %eta*(Sw-Sb)
GAP4 = eta_2*gap5-eta_2*gap6;       %eta_28(Sw'-Sb')


GAP1  =   0;
for i = 1:classn
    Xi = X(:,trls==i);
    Ai = A(:,trls==i);
    GAP1  = GAP1 + norm(Xi-D(:,drls==i)*Ai,'fro')^2;% only for one class, no effect
end
    
gap = GAP1+GAP2+GAP3;