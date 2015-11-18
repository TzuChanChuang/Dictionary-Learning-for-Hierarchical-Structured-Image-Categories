function [Dict,Drls,CoefM,CMlabel] = Initialization(TrainDat,TrainLabel,opts)
%----------------------------------------------------------------------
%
%Input : (1) TrainDat: the training data matrix. 
%                      Each column is a training sample
%        (2) TrainLabel: the training data labels
%        (3) opts      : the struture of parameters
%               .nClass   the number of classes
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
fprintf(['initializing dict & coef']);

Dict_ini  =  []; 
Dlabel_ini = [];				

for ci = 1:opts.nClass
    cdat          			=    TrainDat(:,TrainLabel==ci);
    [dict, coef_ini]		=    K_SVD(cdat, opts.nIter);
    Dict_ini      			=    [Dict_ini dict];
    Dlabel_ini    			=    [Dlabel_ini repmat(ci,[1 size(dict,2)])];
    %%scoef(:,TrainLabel ==ci) = 	 coef_ini;
end

%%%%%%%%%%%%%%%%%%%%%%%
%initialize shared dict
%%%%%%%%%%%%%%%%%%%%%%%
fprintf(['initializing shared dict']);

SharedDict_ini = [];	
SharedDlabel_oriDic_ini = []; 

HeadDict_ini = Dict_ini; % Di^
HeadDictLabel_ini = Dlabel_ini; %Di^ label

%pick columns from every two dictionaries and count their inner product, if > ξ = 0.9, put the columns into shared dictionary
threshold= 0.9;
SharedD_nClass = 0;	
%Shared_class_exist = zeros(1, opts.nClass);		%to check if the class already exist in the D0
size_col = size(Dict_ini(:, Dlabel_ini==1),2);
num_class_upperbound = size_col/2; 			 %number of D0's upper bound = 1/2 Di^
num_class_lowerbound = size_col/4; 			%number of D0's lower bound = 1/4 Di^
column_now  = 0;
Shared_class_full = 0;

while(SharedD_nClass < num_class_lowerbound && Shared_class_full ==0)
	column_now += 1;
	fprintf(['column_now = ' num2str(column_now) '\n']);
	Shared_class_exist = zeros(1, opts.nClass);		%to check if the class already exist in the D0

	for i= 1:opts.nClass-1
		fprintf(['Now class = ' num2str(i) '\n']);
		if(size(Dict_ini(:, Dlabel_ini==i),2) < column_now)
			break;
		end
		temp_dic_i = Dict_ini(:, Dlabel_ini==i);
		for j= i+1: opts.nClass
			if(size(Dict_ini(:, Dlabel_ini==j),2) >= column_now)
				temp_dic_j = Dict_ini(:, Dlabel_ini==j);

				%counting inner product
				inner_ans=sum(temp_dic_i(:,column_now).*temp_dic_j(:,column_now));

				if (inner_ans > threshold)					%%put into D0
					if (Shared_class_exist(1,i)==0)			%%never been put in
						SharedD_nClass +=1;
						SharedDict_ini= [SharedDict_ini temp_dic_i(:,column_now)];		
						SharedDlabel_oriDic_ini = [SharedDlabel_oriDic_ini repmat(i,[1 1])];
						Shared_class_exist(1,i)=1;
					end
					if (Shared_class_exist(1,j)==0)	
						SharedD_nClass +=1;
						SharedDict_ini= [SharedDict_ini temp_dic_j(:,column_now)];
						SharedDlabel_oriDic_ini = [SharedDlabel_oriDic_ini repmat(j,[1 1])];
						Shared_class_exist(1,j)=1;
					end	
				end
				if(SharedD_nClass >= num_class_upperbound)
					Shared_class_full = 1;
					break;
				end
			end
		end

		%%store di^
		if (Shared_class_exist(1,i)==1)
			col_delete = temp_dic_i(:,column_now);
			num_col_delete = find(ismember(HeadDict_ini',col_delete','rows'),1);
			HeadDict_ini(:,num_col_delete) = []; 
			HeadDictLabel_ini(:,num_col_delete) = [];
		end

		if(Shared_class_full==1)
			break;
		end
	end
	fprintf(['SharedD_nClass = ' num2str(SharedD_nClass) '\n'])

	%%for the last class
	Dict_lastClass = Dict_ini(:, Dlabel_ini==opts.nClass);
	if (Shared_class_exist(1,opts.nClass)==1)	
		col_delete = Dict_lastClass(:,column_now);
		num_col_delete = find(ismember(HeadDict_ini',col_delete','rows'),1);
		HeadDict_ini(:,num_col_delete) = []; 
		HeadDictLabel_ini(:,num_col_delete) = [];
	end
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
%use CVX to initialize Ai=[Ai0, Ai^],  ||Xi-[D0, Di^]Ai||2 + lambda*||Ai||1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
coef = [];
HeadCoef = [];
SharedCoef = [];
fprintf(['Initalize coefficients \n']);
for ci = 1:opts.nClass
	fprintf(['Initalize coefficients, class:' num2str(ci) '\n']);
	X  =    TrainDat(:,TrainLabel==ci);
	D  =    TotalDict_ini(:,TotalDictLabel_ini ==ci);
	A  =    zeros(size(D,2),size(X,2));
	m  =	size(A,1);
	n  =	size(A,2);
	for j=1:n;
		cvx_begin quiet
			variable a(p);
			minimize (norm(X(:,j)-D*a) + opts.lambda*norm(a,1));
		cvx_end
		A(:,j) = a;
	end
	coef = [coef A];
	SharedCoef = [SharedCoef A(1:SharedD_nClass, :)];
	HeadCoef = [HeadCoef A(SharedD_nClass+1:m, :)];
end
%%A = coef, coef_Label = TrainLaebel
%%Ai^ = HeadCoef
%%Ai0 = SharedCoef

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Main loop of 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
DL_par.dls        =    	TotalDictLabel_ini;
DL_ipts.D         =     TotalDict_ini;   
DL_ipts.trls      =     TrainLabel;
DL_par.tau        =     opts.lambda;
DL_par.eta    	  =     opts.eta;
DL_nit            =     1;
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
        DL_ipts.SA         			=  SharedCoef;
        DL_par.index      			=  ci; 
        [Copts]             		=  UpdateCoef(DL_ipts,DL_par);
        coef(:,TrainLabel==ci)    	=  Copts.A;
        CMlabel(ci)        			=  ci;
        CoefM(:,ci)         		=  mean(Copts.A,2);
    end
    [GAP_coding(Fish_nit)]  =  Class_Energy(TrainDat,coef,opts.nClass,Fish_par,Fish_ipts) 	 

    %------------------------------------------------------------
    %updating the dictionary Di^ : min||Xi - D0*Ai0 - Di^*Ai^||2
    %------------------------------------------------------------
    for ci = 1:opts.nClass
 		Xi = TrainDat(:, TrainLabel==ci) - SharedDict_ini * SharedCoef(:, TrainLabel==ci);
    	c = 1;
    	Dinit_ci = HeadDict_ini(:, HeadDictLabel_ini==ci);
    	Ai = HeadCoef(:, TrainLabel==ci);
    	HeadDict(:, HeadDictLabel_ini==ci)   =  learn_basis_dual(TrainDat(:,TrainLabel==ci), Ai, c, Dinit_ci);
    end

    %------------------------------------------------------------
    %updating the dictionary D0 : min||X0 - D0*Ai0||2
    %------------------------------------------------------------
    fprintf(['Updating D0 \n'])
    A0 = HeadCoef;
    Dinit_shared = SharedDict_ini;
    c = 1;
    X0 = [];
    for ci = 1:opts.nClass
    	Xi = TrainDat(:, TrainLabel==ci) - HeadDict(:, HeadDictLabel_ini==ci) * HeadCoef(:, TrainLabel==ci);
    	X0 = [X0 Xi];
    end
	SharedDict   = learn_basis_dual(X0, A0, c, Dinit_shared);
end










