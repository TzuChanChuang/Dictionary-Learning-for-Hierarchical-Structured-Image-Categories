function [opts] = UpdateCoef(ipts,par)
% ========================================================================
% Coefficient updating of FDDL, Version 1.0
% Copyright(c) 2011  Meng YANG, Lei Zhang, Xiangchu Feng and David Zhang
% All Rights Reserved.
%
% ----------------------------------------------------------------------- 
% Permission to use, copy, or modify this software and its documentation
% for educational and research purposes only and without fee is here
% granted, provided that this copyright notice and the original authors'
% names appear on all copies and supporting documentation. This program
% shall not be used, rewritten, or adapted as the basis of a commercial
% software or hardware product without first obtaining permission of the
% authors. The authors make no representations about the suitability of
% this software for any purpose. It is provided "as is" without express
% or implied warranty.
%----------------------------------------------------------------------
%
% This is an implementation of the algorithm for updating the
% Coefficient matrix of FDDL (fix the dictionary)
%
% Please refer to the following paper
%
% Meng Yang, Lei Zhang, Xiangchu Feng, and David Zhang,"Fisher Discrimination 
% Dictionary Learning for Sparse Representation", In IEEE Int. Conf. on
% Computer Vision, 2011.
% L. Rosasco, A. Verri, M. Santoro, S. Mosci, and S. Villa. Iterative
% Projection Methods for Structured Sparsity Regularization. MIT Technical
% Reports, MIT-CSAIL-TR-2009-050,CBCL-282, 2009.
% J. Bioucas-Dias, M. Figueiredo, ?A new TwIST: two-step iterative shrinkage
% /thresholding  algorithms for image restoration?, IEEE Transactions on 
% Image Processing, December 2007.
%----------------------------------------------------------------------
%
%  Inputs :   (1) ipts :    the structre of input data
%                    .D     the total dictionary  (this upper class)                                                    
%                    .X     the ith training data
%                    .A     the coefficient matrix in the last iteration(this upper class)
%                    .SA    the shared coefficient matrix                                       
%                    .MUSA  the mean of all the shared coefficient matrix
%                    .HA    the head coefficient matrix in the last iteration                                      
%                    .trls  the labels of training data
%                    .totalX     the training data of all the upper classes
%                    .SA_up      all the upper classes' coefficient matrix
%                    .SA_up_l    the label of all the upper classes' coefficient matrix
%             (2) par :     the struture of input parameters
%                    .tau   the parameter of sparse constraint of coef
%                    .eta   the parameter of within-class scatter
%                    .eta_2   the parameter of upper within-class scatter
%                    .dls     the labels of dictionary's columns
%                    .index   the label of the class being processed
%                    .index_h     the label of the upper class being processed
%                    .SharedD_nClass    number of column of shared dict
% Outputs:    (1) opts :    the structure of output data
%                    .A     the coefficient matrix
%                    .ert   the total energy sequence
%
%---------------------------------------------------------------------

par.nIter    =     200;   % maximal iteration number
par.isshow   =     false;
par.citeT    =     1e-6;  % stop criterion
par.cT       =     1e+10; % stop criterion



drls            =    par.dls;  
D               =    ipts.D;       %the whole dictionary
X               =    ipts.X;       %the training data of the class
X_up            =    ipts.totalX;
A               =    ipts.A;       %the whole coef(this upper class)
SA              =    ipts.SA;      %the whole shared coef(this upper class)
SA_up           =    ipts.SA_up;
SA_up_l         =    ipts.SA_up_l;
MUSA            =    ipts.MUSA;    %the mean of upper classes' coef
HA              =    ipts.HA;
tau             =    par.tau;
lambda1         =    par.tau;
eta2            =    par.eta;
eta3            =    par.eta;
eta_2           =    par.eta_2;
trls            =    ipts.trls;
classn          =    length(unique(trls));
nIter           =    par.nIter;
c               =    par.c;
sigma           =    c;
tau1            =    tau/2;
index           =    par.index;
index_h         =    par.index_h;
Di              =    D(:, drls==index);
m               =    size(Di,2);
m_up            =    size(Di,1);
SharedD_nClass  =    par.SharedD_nClass;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%TWIST parameter
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for_ever           =         1;
IST_iters          =         0;
TwIST_iters        =         0;
sparse             =         1;
verbose            =         1;
enforceMonotone    =         1;
lam1               =         1e-4;   %default minimal eigenvalues
lamN               =         1;      %default maximal eigenvalues
rho0               =         (1-lam1/lamN)/(1+lam1/lamN); 
alpha              =         2/(1+sqrt(1-rho0^2));        %default,user can set
beta               =         alpha*2/(lam1+lamN);         %default,user can set


%%%%%%%%%%%%%%%%%%%%%%%%
%preprocessing
%%%%%%%%%%%%%%%%%%%%%%%%
Ai                 =          X;  % the i-th training data
A_up               =          X_up;
Xa                 =          A;  
X0                 =          SA;
X0_up              =          SA_up;
X0_up_label        =          SA_up_l;
X0_MU              =          MUSA;
Xh                 =          HA;                    
Xi                 =          A(:,trls==index);
Xt_now             =          A(:,trls==index);

newpar.n_d          =   size(Ai,2);             % the sample number of i-th(this) training data
newpar.n            =   size(Xa,2);             % the total sample number of training data
n                   =   newpar.n;
%newpar.n_up         =   size(A_up,2) * size(A_up,3);             % the total sample number of upper classes' training data
%n_up                =   newpar.n_up;


for ci = 1:classn
    t_n_d = sum(trls==ci);
    t_b_line_i     =   ones(t_n_d,newpar.n_d)./t_n_d;
    t_c_j = ones(t_n_d,newpar.n_d)./newpar.n;
    t_b_angle_i           =   t_b_line_i-t_c_j;   
    CJ(ci).M  = t_c_j;
    BAI(ci).M = t_b_angle_i;   
end
%{
for ci = 1:2
    t_n_d = sum(SA_up_l==ci);
    t_b_line_i     =   ones(t_n_d,newpar.n)./t_n_d;
    t_c_j = ones(t_n_d,newpar.n)./newpar.n_up;
    t_b_angle_i           =   t_b_line_i-t_c_j;   
    CJ_up(ci).M  = t_c_j;
    BAI_up(ci).M = t_b_angle_i;   
end
%}
newpar.B_line_i     =   ones(newpar.n_d,newpar.n_d)./newpar.n_d;
%newpar.B_line_i_up  =   ones(newpar.n,newpar.n)./newpar.n;
newpar.C_j          =   ones(newpar.n_d,newpar.n_d)./newpar.n;
%newpar.C_j_up       =   ones(newpar.n,newpar.n)./newpar.n_up;
newpar.CjCj         =   (newpar.C_j)*(newpar.C_j)';
%newpar.CjCj_up      =   (newpar.C_j_up)*(newpar.C_j_up)';
newpar.C_line       =   ones(newpar.n,newpar.n_d)./newpar.n;
%newpar.C_line_up    =   ones(newpar.n_up,newpar.n)./newpar.n_up;
B_i                 =   eye(newpar.n_d,newpar.n_d)-newpar.B_line_i;
%B_i_up              =   eye(newpar.n,newpar.n)-newpar.B_line_i_up;
newpar.BiBi         =   B_i*(B_i)';
%newpar.BiBi_up      =   B_i_up*(B_i_up)';
B_angle_i           =   newpar.B_line_i-newpar.C_j;
%B_angle_i_up        =   newpar.B_line_i_up-newpar.C_j_up;
newpar.Bai          =   B_angle_i;
%newpar.Bai_up       =   B_angle_i_up;
newpar.BaiBai       =   B_angle_i*(B_angle_i)';
%newpar.BaiBai_up    =   B_angle_i_up*(B_angle_i_up)';
Xo                  =   Xa;
Xo(:,trls==index)   =   0;
%Xo_up               =   X0_up;
%Xo_up(:,X0_up_label==index_h) =   0;
G_X_i               =   Xo*newpar.C_line;
%G_X_i_up            =   Xo_up*newpar.C_line_up;
newpar.BaiGxi       =   B_angle_i*(G_X_i)';
%newpar.BaiGxi_up    =   B_angle_i_up*(G_X_i_up)';

newpar.DD           =   Di'*Di;
newpar.DAi          =   Di'*Ai;

newpar.m            =   m;                 % the number of dictionary column atoms
%newpar.m_up         =   m_up;              
%fprintf(['newpar.m_up:' num2str(newpar.m_up) '\n']);      
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%main loop
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Xa(:,trls==index)  =  Xi;
xm2       =      Xi;%A(:,trls==index);
xm1       =      Xi;%A(:,trls==index); % now



[gap] = Class_Energy(Ai,D,X0,X0_MU,Xh,xm1,Xa,drls,trls,index,...                           
        lambda1,eta2,eta3,eta_2,classn);
prev_f   =   gap;
ert(1) = gap;
for n_it = 2 : nIter;     
   Xa(:,trls==index)  =  Xi;
      
   while for_ever
        % IPM estimate
         
        [grad] = Gradient_Comp(xm1,Xa,classn,index,...
        eta2,eta3,eta_2,trls,drls,newpar,...
        BAI,CJ,SharedD_nClass); 
        v        =   xm1(:)-grad./(2*sigma);
        tem      =   soft(v,tau1/sigma);
        x_temp   =   reshape(tem,[size(D(:,drls==index),2),size(xm1,2)]);
        
        if (IST_iters >= 2) | ( TwIST_iters ~= 0)
            % set to zero the past when the present is zero
            % suitable for sparse inducing priors
            if sparse
                mask    =   (x_temp ~= 0);
                xm1     =   xm1.* mask;
                xm2     =   xm2.* mask;
            end
            % two-step iteration
            xm2    =   (alpha-beta)*xm1 + (1-alpha)*xm2 + beta*x_temp;
            % compute residual
           [gap] = Class_Energy(Ai,D,X0,X0_MU,Xh,xm2,Xa,drls,trls,index,...                           
        lambda1,eta2,eta3,eta_2,classn);

           f   =   gap;
          
            if (f > prev_f) & (enforceMonotone)
                TwIST_iters   =  0;  % do a IST iteration if monotonocity fails
            else
                TwIST_iters =   TwIST_iters+1; % TwIST iterations
                IST_iters   =    0;
                x_temp      =   xm2;
                if mod(TwIST_iters,10000) ==0
                   c = 0.9*c; 
                   sigma = c;
                end
                break;  % break loop while
            end
        else
          
        
        [gap] = Class_Energy(Ai,D,X0,X0_MU,Xh,x_temp,Xa,drls,trls,index,...                           
        lambda1,eta2,eta3,eta_2,classn);
    
        f   =   gap;
         
        if f > prev_f
            % if monotonicity  fails here  is  because
            % max eig (A'A) > 1. Thus, we increase our guess
            % of max_svs
            c         =    2*c;  
            sigma     =    c;
            if verbose
%               fprintf('Incrementing c=%2.2e\n',c);
            end
            if  c > par.cT
                break;  % break loop while    
            end
            IST_iters = 0;
            TwIST_iters = 0;
            else
                TwIST_iters = TwIST_iters + 1;
                break;  % break loop while
            end
        end
        
    end

    citerion      =   abs(f-prev_f)/abs(prev_f);
    if citerion < par.citeT | c > par.cT
%        fprintf('Stop!\n c=%2.2e\n citerion=%2.2e\n',c,citerion);
       break;
    end
    
    xm2           =   xm1;
    xm1           =   x_temp;
    Xt_now        =   x_temp;
    Xi            =   Xt_now; 
    prev_f        =   f;
    ert(n_it)     =   f;
%     fprintf('Iteration:%f  Total gap:%f\n',n_it,ert(n_it-1));
end  


opts.A     =       Xt_now;
opts.ert   =       ert;