function grad = Gradient_Comp(Xi,Xa,X0,X0_up,X0_up_label,classn,index,index_h,eta2,eta3,eta_2...
    trls,drls,newpar,BAI,CJ,BAI_up,CJ_up)
% ========================================================================
% IPM's Gradient computation of FDDL, Version 1.0
% Copyright(c) 2011  Meng YANG, Lei Zhang, Xiangchu Feng and David Zhang
% All Rights Reserved.
%
% -----------------------------------------------------------------------
%   
% Input :   (1) Xi:   the coefficient matrix of this class
%           (2) Xa:   the coefficient matrix of the whole class
%           (3) X0:   the shared coefficient matrix of the whole class
%           (4) X0_up:   the shared coefficient matrix of the whole class
%           (5) X0_up_label:   the shared coefficient matrix of the whole class
%           (6) classn:   the number of class
%           (7) index: label of class being processed
%           (8) index_h:  label of the upper class being processed
%           (9) eta2 : parameter of within-class scatter
%           (10) eta3 : parameter of between-class scatter
%           (11) eta_2: parameter of between-class scatter
%           (12) trls: labels of training samples
%           (13) drls: labels of dictionary's column4433
%           (14) newpar, BAI, CJ, BAI_up, CJ_up: the precomputed data        
%
% Outputs : (1) grad  :    the gradient vector of coding model
%
%------------------------------------------------------------------------

n_d             =      newpar.n_d;                % the sample number of i-th training data
n               =      newpar.n;
B_line_i        =      newpar.B_line_i;
C_j             =      newpar.C_j;
C_j_up          =      newpar.C_j_up
C_line          =      newpar.C_line;
C_line_up       =      newpar.C_line_up
DD              =      newpar.DD;
DAi             =      newpar.DAi; 
Di0Di0          =      newpar.Di0Di0;
Di0Ai           =      newpar.Di0Ai;
BiBi            =      newpar.BiBi;
BiBi_up   		= 	   newpar.BiBi_up;
BaiBai          =      newpar.BaiBai;
BaiBai_up       =      newpar.BaiBai_up;
BaiGxi          =      newpar.BaiGxi;
BaiGxi_up       =      newpar.BaiGxi_up;
CjCj            =      newpar.CjCj;
CjCj_up         =      newpar.CjCj_up;
m               =      newpar.m;
m_up            =      newpar.m_up;
DoiDoi          =      newpar.DoiDoi;
B_angle_i       =      newpar.Bai;

for k = 1:classn
%     Z(k).Matrix   =   Xa(:,trls==k)*B_line_i-Xa*C_line+Xi*C_j;
%     Z(k).Matrix   =   Xa(:,trls==k)*B_angle_i-Xa*C_line+Xi*C_j+Xa(:,trls==k)*C_j;
    Z(k).Matrix   =   Xa(:,trls==k)*BAI(k).M-Xa*C_line+Xi*C_j+Xa(:,trls==k)*CJ(k).M;
end
for k = 1:2
    Z_up(k).Matrix   =   X0_up(:,X0_up_label==k)*BAI_up(k).M-X0_up*C_line_up+X0*C_j_up+X0_up(:,X0_up_label==k)*CJ_up(k).M;
end

XiT      =   Xi';
X0T      =   X0';

tem      =  2*Di0Di0*Xi-2*Di0Ai;
grad1    =  tem(:);


tem       = 2*eta2*BiBi*XiT;
grad4     = tem(:);

tem       =  -eta3*(2*BaiBai*XiT-2*BaiGxi);
grad5     = tem(:);

grad6 = zeros(size(grad5));
for k = 1:classn
    temz  =  Z(k).Matrix;
    if k~=index
        tem = -eta3*(2*CjCj*XiT-2*C_j*temz');
        grad6 = grad6+tem(:);
    end
end

grad456 = reshape(grad4+grad5+grad6,[n_d m])';
grad456 = grad456(:);

tem       = 2*eta_2*BiBi_up*X0T;
grad7     = tem(:);

tem       =  -eta_2*(2*BaiBai_up*X0T-2*BaiGxi_up);
grad8     = tem(:);

grad9 = zeros(size(grad8));
for k = 1:2
    temz  =  Z_up(k).Matrix;
    if k~=index_h
        tem = -eta_2*(2*CjCj_up*X0T-2*C_j_up*temz');
        grad9 = grad9+tem(:);
    end
end


grad789 = reshape(grad47+grad8+grad9,[n m_up])';
grad789 = grad789(:);

grad = grad1+grad456+grad789;
