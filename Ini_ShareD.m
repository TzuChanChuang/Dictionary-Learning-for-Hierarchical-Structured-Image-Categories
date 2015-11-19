function [SharedD_nClass, SharedDict_ini, SharedDlabel_oriDic_ini, HeadDict_ini, HeadDictLabel_ini, TotalDict_ini, TotalDictLabel_ini] = Ini_ShareD(Dict_ini, DLabel_ini, opts.nClass)

%----------------------------------------------------------------------
%
%Input : (1) Dict_ini: 			the initial dictionary
%        (2) DLabel_ini 		the label of the initial dictionary 
%        (2) opts.nClass 		the number of classes 
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

					if (inner_ans > threshold)					%put into D0
						if (Shared_class_exist(1,i)==0)			%never been put in
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

			%store di^
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

		%for the last class
		Dict_lastClass = Dict_ini(:, Dlabel_ini==opts.nClass);
		if (Shared_class_exist(1,opts.nClass)==1)	
			col_delete = Dict_lastClass(:,column_now);
			num_col_delete = find(ismember(HeadDict_ini',col_delete','rows'),1);
			HeadDict_ini(:,num_col_delete) = []; 
			HeadDictLabel_ini(:,num_col_delete) = [];
		end
	end


	%Total Dict Di
	TotalDict_ini = [];
	TotalDictLabel_ini = [];
	for i = 1:opts.nClass
		temp_totaldict = [SharedDict_ini HeadDict_ini(:, HeadDictLabel_ini==i)];
		TotalDict_ini = [TotalDict_ini temp_totaldict];
		TotalDictLabel_ini = [TotalDictLabel_ini repmat(i, [1 size(TotalDict_ini, 2)])];
	end


