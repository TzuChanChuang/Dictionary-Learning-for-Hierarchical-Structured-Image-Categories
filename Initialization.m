function [Dict,Drls,CoefM,CMlabel] = FDDL(TrainDat,TrainLabel,opts)
%----------------------------------------------------------------------
%
%Input : (1) TrainDat: the training data matrix. 
%                      Each column is a training sample
%        (2) TrainLabel: the training data labels
%        (3) opts      : the struture of parameters
%               .nClass   the number of classes
%               .wayInit  the way to initialize the dictionary
%               .lambda  the parameter of l1-norm energy of coefficient
%               .eta  the parameter of l2-norm of coefficient term
%               .nIter    the number of iteration
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
printf('initializing dict & coef');

Dict_ini  =  []; 
Dlabel_ini = [];				

for ci = 1:opts.nClass
    cdat          			=    TrainDat(:,TrainLabel==ci);
    [dict, output_ini]		=    K_SVD(cdat);
    Dict_ini      			=    [Dict_ini dict];
    Dlabel_ini    			=    [Dlabel_ini repmat(ci,[1 size(dict,2)])];
    coef(:,TrainLabel ==ci) = 	 output_ini.CoefMatrix;
end

%%%%%%%%%%%%%%%%%%%%%%%
%initialize shared dict
%%%%%%%%%%%%%%%%%%%%%%%
printf('initializing shared dict');

SharedDict_ini = [];	%%若shared dict label是放相對應的dict's label那還需要令一個matrix放shared dict嗎？	
SharedDlabel_ini = [];	
SharedDlabel_oriDic_ini = []; %%shared dict label放相對應的dict的label

HeadDict_ini = []; % Di^
HeadDictLabel_ini = []; %Di^ label
%SharedCoef_ini = [];
%SharedCLabel_ini = [];	%%同shared dict的問題 %%needed?

%%每個Dict random取第一個column、比較兩者之間的inner product是否超過ξ(threshold)= 0.8
threshold= 0.9;
SharedD_nClass = 0;
flag_last = 0;		%check for the last class
for i= 1:opts.nClass-1
	If_i_In_SD = 0; 	%if class i in shared dict, Di^ = Di - D0自己的部分 
	for j= i+1: opts.nClass
		temp_dic_i = Dict_ini(:, drls==i);
		temp_dic_j = Dict_ini(:, drls==j);
		%size_temp= size(temp_dic_i,1)*size(temp_dic_i,2);
		%item_num= 20;
		%item_selected= randperm(size_temp, item_num);
		item_num = size(temp_dic_i, 1);  %取第一行的column的item數
		inner_ans=0;
		%counting inner product
		for k= 1: item_num
			inner_ans+=temp_dic_i(k)* temp_dic_j(k);
		end
		
		if inner_ans > threshold
			If_i_In_SD = 1;
			if (i==opts.nClass-1) & (j== opts.nClass)
				flag_last = 1;
			end
			if isempty(SharedDlabel_oriDic_ini(1, i))		%%never been put in
				SharedD_nClass +=1;
				SharedDict_ini= [SharedDict_ini temp_dic_i(:,1)];		%put the first column of the 
				SharedDlabel_ini = [SharedDlabel_ini repmat(SharedD_nClass,[1 size(temp_dic_i,2)])];
				SharedDlabel_oriDic_ini = [SharedDlabel_oriDic_ini repmat(i,[1 size(temp_dic_i,2)])];
			end
			if isempty(SharedDlabel_oriDic_ini(1, j))		%%never been put in
				SharedD_nClass +=1;
				SharedDict_ini= [SharedDict_ini temp_dic_j(:,1)];
				SharedDlabel_ini = [SharedDlabel_ini repmat(SharedD_nClass,[1 size(temp_dic_j,2)])];
				SharedDlabel_oriDic_ini = [SharedDlabel_oriDic_ini repmat(j,[1 size(temp_dic_j,2)])];
			end	
		end
	end

	num_col = size(temp_dic_i, 2);			%%store di^
	if If_i_In_SD == 1
		HeadDict_ini = [HeadDict_ini temp_dic_i(:, 2:num_col)];
		HeadDictLabel_ini = [HeadDictLabel_ini repmat(i, [1 num_col-1])];
	else
		HeadDict_ini = [HeadDict_ini temp_dic_i];
		HeadDictLabel_ini = [HeadDictLabel_ini repmat(i, [1 num_col])];
	end
end

%%for the last class
Dict_lastClass = Dict_ini(:, drls==opts.nClass);
num_col = size(Dict_lastClass, 2);
if flag_last==1
	HeadDict_ini = [HeadDict_ini Dict_lastClass(:, 2:num_col)];
	HeadDictLabel_ini = [HeadDictLabel_ini repmat(opts.nClass, [1 num_col-1])];
else
	HeadDict_ini = [HeadDict_ini Dict_lastClass];
	HeadDictLabel_ini = [HeadDictLabel_ini repmat(opts.nClass, [1 num_col])];
end


%%Total Dict Di
TotalDict_ini = [];
TotalDictLabel_ini = [];
for i = 1:opts.nClass
	temp_totaldict = [SharedDict_ini HeadDict_ini(:, HeadDictLabel_ini==i)];
	TotalDict_ini = [TotalDict_ini temp_totaldict];
	TotalDictLabel_ini = [TotalDictLabel_ini repmat(i, [1 size(TotalDict_ini, 2)])];
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%use CVX to update Ai=[Aj0, Aj^],  ||Xi-[D0, Di^]Ai||2 + lambda*|||Ai||1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

nIter_CVX = 15;
for i = 1:nIter_CVX
	for ci = 1:opts.nClass
	    X  =   TrainDat(:,TrainLabel==ci);
	    A  =   coef(:,TrainLabel ==ci);
	    D  =   TotalDict_ini(:,TotalDict_ini ==ci);
	    p  =   size(A,1);
		cvx_begin quiet
			variable A(p);
			minimize (norm(X-D*A) + gamma*norm(A,1));
		cvx_end
	end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Main loop of 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
DL_par.dls        =    	TotalDictLabel_ini;
%DL_par.sdls       =     SharedDlabel_ini;
DL_ipts.D         =     TotalDict_ini;
%DL_ipts.SD 		  =		SharedDict_ini;
DL_ipts.trls      =     TrainLabel;
DL_par.tau        =     opts.lambda;
DL_par.eta    	  =     opts.eta;
 
DL_nit            =     1;
drls              =     TotalDictLabel_ini;
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
        %DL_ipts.SA         			=  SharedCoef_ini;
        DL_par.index      			=  ci; 
        [Copts]             		=  UpdateCoef(DL_ipts,DL_par);
        coef(:,TrainLabel==ci)    	=  Copts.A;
        CMlabel(ci)        			=  ci;
        CoefM(:,ci)         		=  mean(Copts.A,2);
    end
    [GAP_coding(Fish_nit)]  =  Class_Energy(TrainDat,coef,opts.nClass,Fish_par,Fish_ipts) 	 %%%%%%%%要改