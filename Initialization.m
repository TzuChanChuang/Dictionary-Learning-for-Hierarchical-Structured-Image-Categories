function [Dict,Drls,CoefM,CMlabel] = FDDL(TrainDat,TrainLabel,opts)
%----------------------------------------------------------------------
%
%Input : (1) TrainDat: the training data matrix. 
%                      Each column is a training sample
%        (2) TrainDabel: the training data labels
%        (3) opts      : the struture of parameters
%               .nClass   the number of classes
%               .wayInit  the way to initialize the dictionary
%               .lambda1  the parameter of l1-norm energy of coefficient
%               .lambda2  the parameter of l2-norm of Fisher Discriminative
%               coefficient term
%               .nIter    the number of FDDL's iteration
%               .show     sign value of showing the gap sequence
%
%Output: (1) Dict:  the learnt dictionary via FDDL
%        (2) Drls:  the labels of learnt dictionary's columns
%        (2) CoefM: Mean Coefficient Matrix. Each column is a mean coef
%                   vector
%        (3) CMlabel: the labels of CoefM's columns.
%
%-----------------------------------------------------------------------
%%%%%%%%%%%%%%%%%%
% normalize energy
%%%%%%%%%%%%%%%%%%
TrainDat = TrainDat*diag(1./sqrt(sum(TrainDat.*TrainDat)));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%initialize dict and coefficient
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Dict_ini  =  []; 
Dlabel_ini = [];				

for ci = 1:opts.nClass
    cdat          			=    TrainDat(:,TrainLabel==ci);
    [dict, output_ini]		=    K_SVD(cdat);
    Dict_ini      			=    [Dict_ini dict];
    Dlabel_ini    			=    [Dlabel_ini repmat(ci,[1 size(dict,2)])];
    coef(:,TrainLabel ==ci) = output_ini.CoefMatrix;
end

%%%%%%%%%%%%%%%%%%%%%%%
%initialize shared dict
%%%%%%%%%%%%%%%%%%%%%%%
SharedDict_ini = [];	%%若shared dict label是放相對應的dict's label那還需要令一個matrix放shared dict嗎？	
SharedDlabel_ini = [];	%%shared dict label放相對應的dict的label？
SharedCoef_ini= [];
%SharedCLabel_ini = [];	%%同shared dict的問題 %%needed?


%%Dict_head??2. 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Main loop of 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
DL_par.dls        =     Dlabel_ini;
DL_par.sdls        =    SharedDlabel_ini;
DL_ipts.D         =     Dict_ini;
DL_ipts.SD 		  =		SharedDict_ini;
DL_ipts.trls      =     TrainLabel;
DL_par.tau        =     opts.lambda1;
DL_par.lambda2    =     opts.lambda2;
 
DL_nit            =     1;
drls              =     Dlabel_ini;
while DL_nit<=opts.nIter  
	if size(DL_ipts.D,1)>size(DL_ipts.D,2)
		DL_par.c        =    1.05*eigs(DL_ipts.D'*DL_ipts.D,1);
    else
      	DL_par.c        =    1.05*eigs(DL_ipts.D*DL_ipts.D',1);
    end
    %-------------------------
    %updating the coefficient
    %-------------------------
    for ci = 1:opts.nClass
        fprintf(['Updating coefficients, class: ' num2str(ci) '\n'])
        DL_ipts.X         			=  TrainDat(:,TrainLabel==ci);
        DL_ipts.A         			=  coef;
        DL_ipts.SA         			=  SharedCoef_ini;
        DL_par.index      			=  ci; 
        [Copts]             		=  FDDL_SpaCoef (DL_ipts,DL_par);
        coef(:,TrainLabel==ci)    	=  Copts.A;
        CMlabel(ci)        			=  ci;
        CoefM(:,ci)         		=  mean(Copts.A,2);
    end
    [GAP_coding(Fish_nit)]  =  FDDL_FDL_Energy(TrainDat,coef,opts.nClass,Fish_par,Fish_ipts)