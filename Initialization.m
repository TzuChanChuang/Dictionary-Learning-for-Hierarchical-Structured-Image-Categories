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

%%%%%%%%%%%%%%%%%%
%initialize dict
%%%%%%%%%%%%%%%%%%
Dict_ini  =  []; 
Dlabel_ini = [];				%%first initialization
for ci = 1:opts.nClass
    cdat          =    TrainDat(:,TrainLabel==ci);
    dict          =    K-SVD(cdat, 1, '');
    Dict_ini      =    [Dict_ini dict];
    Dlabel_ini    =    [Dlabel_ini repmat(ci,[1 size(dict,2)])];
end

for i= 1: iteration
	for ci = 1:opts.nClass
	    cdat          =    TrainDat(:,TrainLabel==ci);
	    dict          =    K-SVD(cdat, 2, Dict_ini(:,Dlabel_ini==ci));
	    Dict_ini(:,Dlabel_ini==ci)      =    dict;
	    %%Dlabel_ini    =    [Dlabel_ini repmat(ci,[1 size(dict,2)])];
	end
end
