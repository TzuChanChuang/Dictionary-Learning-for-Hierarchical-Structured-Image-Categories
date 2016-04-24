function [SharedD_nClass, SharedDict_ini, SharedDlabel_oriDic_ini, new_HeadDict, new_HeadDict_label, TotalDict_ini, TotalDictLabel_ini] = Ini_ShareD(Dict_ini, Dlabel_ini, nClass)

%----------------------------------------------------------------------
%
%Input : (1) Dict_ini: 			the initial dictionary
%        (2) DLabel_ini 		the label of the initial dictionary 
%        (2) nClass 		the number of classes 
%
%Output: (1) SharedD_nClass				the number of columns in shared dictionary
%        (2) SharedDict_ini				the initial shared dictionary D0
%        (3) SharedDlabel_oriDic_ini	the label of the D0 from the original dictionary
%        (4) HeadDict_ini  				Di^
%        (5) HeadDictLabel_ini 			the label of Di^
%        (6) TotalDict_ini 				Di = [D0, Di^]
%        (7) TotalDictLabel_ini  		the label of Di
%
%-----------------------------------------------------------------------

	SharedDict_ini = [];	
	SharedDlabel_oriDic_ini = []; 

	HeadDict_ini = Dict_ini; % Di^
	HeadDictLabel_ini = Dlabel_ini; %Di^ label

	%pick columns from every two dictionaries and count their inner product, if >threshold = 0.9, put the columns into shared dictionary
	threshold= 0.94;
	size_col = size(Dict_ini(:, Dlabel_ini==1),2);
	Shared_class_exist = zeros(1, size_col, nClass);		%to check if the column already exist in D0

	for i=1:nClass-1
		for j=i+1:nClass
			Dict_i = Dict_ini(:,Dlabel_ini==i);
			Dict_j = Dict_ini(:,Dlabel_ini==j);
			t_matrix = Dict_i' *  Dict_j;					% get inner product
			[i_v, j_v] = find(t_matrix>=threshold);
			i_v = unique(i_v);
			j_v = unique(j_v);
			i_v(find(Shared_class_exist(1, i_v, i)==1)) = [];
			j_v(find(Shared_class_exist(1, j_v, j)==1)) = [];
			SharedDict_ini = [SharedDict_ini Dict_i(:,i_v) Dict_j(:,j_v)];	%put in shared dictionary			
			Shared_class_exist(1,i_v,i) = 1;
			Shared_class_exist(1,j_v,j) = 1;
		end
	end
num_inishareddict = size(SharedDict_ini,2);
num_inishareddict
ran_shareddict = randperm(num_inishareddict,20);
    
SharedDict_ini = SharedDict_ini(:,ran_shareddict);


	SharedD_nClass = size(SharedDict_ini,2);
fprintf(['SharedD_nClass num: ' num2str(SharedD_nClass) '\n']);
	a=0;
	for i = 1:nClass
		a = a+size(HeadDict_ini(:,HeadDictLabel_ini==i-1),2)
		HeadDict_ini(:,a+find(Shared_class_exist(1,:,i)==1))=[];%delete columns which are also in shared dictionary from there original dictionary 
		HeadDictLabel_ini(:,a+find(Shared_class_exist(1,:,i)==1))=[];
    end
    
    for i=1:nClass
        fprintf(['HeadDict_ini ' num2str(i) ' num: ' num2str(size(HeadDict_ini(:,HeadDictLabel_ini==i),2)) '\n']);
    end

	%%randomly pick 800 features over each HeadDict_ini(to make number of features in every class the same)
	new_HeadDict = [];
	new_HeadDict_label= [];
	for i = 1:nClass
		temp_HeadDict = HeadDict_ini(:,HeadDictLabel_ini==i);
		temp_HeadDict = temp_HeadDict(:,1:80);					%pick the former 800
		new_HeadDict = [new_HeadDict temp_HeadDict];
		new_HeadDict_label = [new_HeadDict_label repmat(i, [1 80])];
	end
	%Total Dict Di
	TotalDict_ini = [];
	TotalDictLabel_ini = [];
	for i = 1:nClass
		temp_totaldict = [SharedDict_ini new_HeadDict(:, new_HeadDict_label==i)];
		TotalDict_ini = [TotalDict_ini temp_totaldict];
		TotalDictLabel_ini = [TotalDictLabel_ini repmat(i, [1 size(temp_totaldict, 2)])];
	end
