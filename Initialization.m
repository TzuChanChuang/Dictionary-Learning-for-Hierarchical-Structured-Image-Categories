function [Dict,Drls,CoefM,CMlabel] = Initialization(TrainDat_cat,TrainLabel_cat, TrainDat_dog, TrainLabel_dog, opts)
%----------------------------------------------------------------------
%
%Input : (1) TrainDat*2: 		the training data matrix. 
%                      			Each column is a training sample
%        (2) TrainLabel*2: 		the training data labels
%        (3) opts      : 		the struture of parameters
%               .nClass*2   	the number of classes
%               .lambda   		the parameter of l1-norm energy of coefficient
%               .eta  	  		the parameter of l2-norm of coefficient term
%               .nIter    		the number of iteration
%
%Output: (1) Dict:  the learnt dictionary 
%        (2) Drls:  the labels of learnt dictionary's columns
%        (3) CoefM: Mean Coefficient Matrix. Each column is a mean coef
%                   vector
%        (4) CMlabel: the labels of CoefM's columns.
%
%-----------------------------------------------------------------------
%%%%%%%%%%%%%%%%%%
% normalize energy
%%%%%%%%%%%%%%%%%%
TrainDat(:,:,1) = TrainDat_cat*diag(1./sqrt(sum(TrainDat_cat.*TrainDat_cat)));
TrainDat(:,:,2) = TrainDat_dog*diag(1./sqrt(sum(TrainDat_dog.*TrainDat_dog)));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%initialize dict and coefficient
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
TrainLabel(:,:,1) = TrainLabel_cat;
TrainLabel(:,:,2) = TrainLabel_dog;

for h = 1:2
	fprintf(['initializing dict & coef, h=' num2str(h) '\n']);

	Dict  =  []; 
	Dlabel = [];				

	for ci = 1:opts.nClass(h)
	    cdat          			=    TrainDat(:,TrainLabel(:,:,h)==ci,h);
	    [d, coef_ini]			=    K_SVD(cdat, opts.nIter);
	    Dict      				=    [Dict d];
	    Dlabel    				=    [Dlabel repmat(ci,[1 size(d,2)])];
	end

	Dict_ini(:,:,h) = Dict;
	Dlabel_ini(:,:,h) = Dlabel;
end

%%%%%%%%%%%%%%%%%%%%%%%
%initialize shared dict
%%%%%%%%%%%%%%%%%%%%%%%
for h = 1:2
	fprintf(['initializing shared dict, h = ' num2str(h) '\n']);
	[sdClass, sd, sdl, hd, hdl, td, tdl] = Ini_ShareD(Dict_ini(:,:,h), DLabel_ini(:,:,h), opts.nClass(h));
	SharedD_nClass(h) = sdClass;
	SharedDict_ini(:,:,h) = sd;
	SharedDlabel_oriDic_ini(:,:,h) = sdl;
	HeadDict_ini(:,:,h) = hd;
	HeadDictLabel_ini(:,:,h) = hdl;
	TotalDict_ini(:,:,h) = td;
	TotalDictLabel_ini(:,:,h) = tdl;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%use CVX to initialize Ai=[Ai0, Ai^],  ||Xi-[D0, Di^]Ai||2 + lambda*||Ai||1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for h= 1:2
	fprintf(['Initalize coefficients, h = ' num2str(h) '\n']);
	coef_cvx = [];
	HeadCoef_cvx = [];
	SharedCoef_cvx = [];
	for ci = 1:opts.nClass(h)
		fprintf(['Initalize coefficients, class:' num2str(ci) '\n']);
		X  =    TrainDat(:,TrainLabel(:,:,h)==ci,h);
		D  =    TotalDict_ini(:,TotalDictLabel_ini(:,:,h) ==ci,h);
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
		coef_cvx = [coef_cvx A];
		SharedCoef_cvx = [SharedCoef_cvx A(1:SharedD_nClass(h), :)];
		HeadCoef_cvx = [HeadCoef_cvx A(SharedD_nClass(h)+1:m, :)];
	end

	coef(:,:,h) = coef_cvx;
	HeadCoef(:,:,h) = HeadCoef_cvx;
	SharedCoef(:,:,h) = SharedCoef_cvx;
end
%A = coef, coef_Label = TrainLaebel
%Ai^ = HeadCoef
%Ai0 = SharedCoef

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Main loop 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
DL_par.dls        =    	TotalDictLabel_ini;
DL_ipts.D         =     TotalDict_ini;   
DL_ipts.trls      =     TrainLabel;
DL_par.tau        =     opts.lambda;
DL_par.eta    	  =     opts.eta;
DL_nit            =     1;
SharedDict 		  =     SharedDict_ini;
HeadDict 		  = 	HeadDict_ini;
HeadDictLabel 	  = 	HeadDictLabel_ini;
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
        SharedCoef(:,TrainLabel==ci)=  Copts.A(1:SharedD_nClass, :);
        HeadCoef(:,TrainLabel==ci)  =  Copts.A(SharedD_nClass:end, :);
        CMlabel(ci)        			=  ci;
        CoefM(:,ci)         		=  mean(Copts.A,2);
    end
    [GAP_coding(DL_nit)]  =  Total_Energy(TrainDat,coef,SharedCoef,opts.nClass,DL_par,DL_ipts);	 

    %------------------------------------------------------------
    %updating the dictionary Di^ : min||Xi - D0*Ai0 - Di^*Ai^||2
    %------------------------------------------------------------
    for ci = 1:opts.nClass
    	fprintf(['Updating Di^, class: ' num2str(ci) '\n'])
 		Xi = TrainDat(:, TrainLabel==ci) - SharedDict * SharedCoef(:, TrainLabel==ci);
    	c = 1;
    	Dinit_ci = HeadDict(:, HeadDictLabel==ci);
    	Ai = HeadCoef(:, TrainLabel==ci);
    	HeadDict(:, HeadDictLabel==ci)   =  learn_basis_dual(TrainDat(:,TrainLabel==ci), Ai, c, Dinit_ci);
    end

    %------------------------------------------------------------
    %updating the dictionary D0 : min||X0 - D0*Ai0||2
    %------------------------------------------------------------
    fprintf(['Updating D0 \n'])
    A0 = HeadCoef;
    Dinit_shared = SharedDict;
    c = 1;
    X0 = [];
    for ci = 1:opts.nClass
    	Xi = TrainDat(:, TrainLabel==ci) - HeadDict(:, HeadDictLabel==ci) * HeadCoef(:, TrainLabel==ci);
    	X0 = [X0 Xi];
    end
	SharedDict   = learn_basis_dual(X0, A0, c, Dinit_shared);

	Dict 		= 	[HeadDict;SharedDict];
	DL_ipts.D 	= 	Dict;
	[GAP_Dict(DL_nit)]  =  Total_Energy(TrainDat,coef,SharedCoef,opts.nClass,DL_par,DL_ipts);	

	DL_nit+=1;
end
Drls = Dlabel_ini;

subplot(1,2,1);plot(GAP_coding,'-*');title('GAP_coding');
subplot(1,2,2);plot(GAP_dict,'-o');title('GAP_dict'); 

return;








