
close all;  
clear all;
clc;
 
%%

load(['catndog_24_100.mat']);
 
%%%%%%%%%%%%%%%%%%%%%%%%
%parameter
%%%%%%%%%%%%%%%%%%%%%%%%
class_f_num = 100;                              %%%%%%%%%%%%%%%%%
class_f_num_test = 30;
opts.nClass(1)      =   10;                      %%%%%%%%%%%%%%%%%
opts.nClass(2)      =   10;                      %%%%%%%%%%%%%%%%%
opts.num_of_animals =   2;                      %%%%%%%%%%%%%%%%%
opts.lambda         =   0.1;
opts.eta            =   0.1;
opts.eta_2          =   0.1;
opts.nIter          =   20;
 
% normalize energy
cat_24 = cat_24*diag(1./sqrt(sum(cat_24.*cat_24))); %%%%%%%%%%%%%%%%%
dog_24 = dog_24*diag(1./sqrt(sum(dog_24.*dog_24))); %%%%%%%%%%%%%%%%%
 
[Dict,Drls]= Initialization(cat_24, cat_t, dog_24, dog_t, opts);  %%%%%%%%%%%%%%%%%
 
%save('DictwithDrls','Dict','Drls');
%%
%{ 
%%%%%%%%%%%%%%%%%%%%%%%%
%SVM
%%%%%%%%%%%%%%%%%%%%%%%%
X = [cat_24 dog_24];
n = 0;
for h=1:opts.num_of_animals
    for i=1:opts.nClass(h)
        n = n + 1;
        D = Dict(:,Drls(:,:,h)==i,h);
        A = pinv(D)*X;
        fid=fopen(['coef_dat' num2str(n) '.txt'], 'w');
        [w,total_f] = size(A);
        class_num=0;
        for j=1:class_f_num:total_f
           class_num = class_num+1; %?{?b???bclass
           for x = j:j+class_f_num-1 % each data in one class
               fprintf(fid, '%d ',class_num);
              for y=1:w % each feature in one data
                  fprintf(fid, '%d:%.10f ',y,A(y,x));
              end
              fprintf(fid, '\n');
           end
        end
        fclose(fid);
    end % for i=1:opts.nClass(h)
end
 
%%
%train all the models
addpath('libsvm-3.21/matlab');
%for i=1:class_num
for i=1:opts.nClass(1)*opts.num_of_animals
    [trainLabel, trainData]=libsvmread(['coef_dat' num2str(i) '.txt']);
    %model(i) = svmtrain(trainLabel, trainData, '-t 0 -v 5');
    model(i) = svmtrain(trainLabel, trainData, '-t 0');
    filename = ['model.mat'];
    save(filename, 'model');
end

%% 

for label=1:opts.nClass(1)*opts.num_of_animals
  file_num = 0;
  if(label<=opts.nClass(1))
    test_data = cat_24_test(:,cat_test_t==label);
  elseif(label<=opts.nClass(1)*opts.num_of_animals)
    test_data = dog_24_test(:,dog_test_t==label-opts.nClass(1)); %%%%%%%%%%%%%%%
  end
  for i =1:opts.nClass(1) %cat
      test_dict = Dict(:,Drls(:,:,1)==i,1);
      test_coef_dic = pinv(test_dict)*test_data;
        
      % generate testing data file
      [w,total_f] = size(test_coef_dic);
      file_num = file_num+1;
      if(label==1)
        fid = fopen(['coef_dat_testing' num2str(file_num) '.txt'], 'w');
      else
        fid = fopen(['coef_dat_testing' num2str(file_num) '.txt'], 'a');
      end
      for x=1:total_f
          fprintf(fid, '%d ', label);
          for y=1:w
              fprintf(fid, '%d:%.10f ',y, test_coef_dic(y,x));
          end
          fprintf(fid, '\n');
      end
      fclose(fid);
  end
  for i =1:opts.nClass(2) %dog
      test_dict = Dict(:,Drls(:,:,2)==i,2);
      test_coef_dic = pinv(test_dict)*test_data;
      
      % generate testing data file
      [w,total_f] = size(test_coef_dic);
      file_num = file_num+1;
      if(label==1)
        fid = fopen(['coef_dat_testing' num2str(file_num) '.txt'], 'w');
      else
        fid = fopen(['coef_dat_testing' num2str(file_num) '.txt'], 'a');
      end
      for x=1:total_f
          fprintf(fid, '%d ', label);
          for y=1:w
              fprintf(fid, '%d:%.10f ',y, test_coef_dic(y,x));
          end
          fprintf(fid, '\n');
      end
      fclose(fid);
  end
end
 
%%
% predict
for i =1:opts.nClass(1)*opts.num_of_animals
    [testLabel, testData]=libsvmread(['coef_dat_testing' num2str(i) '.txt']);
    [predict_label(:, i), accuracy, prob_values] = svmpredict(testLabel, testData, model(i));
    acc(i) = accuracy(1,1);
end
 
%%
% voting
[n,f] = size(predict_label);
voting_result = zeros(opts.nClass(1)*opts.num_of_animals,1);
data_class_predict = zeros(n, 1);
for i=1:n
   t = tabulate(predict_label(i,:));
   if(size(t,1)~=1)
        max_val = max(t);
        index = find(t(:,3)==max_val(3));
        % ?P???????@??
        voting_result(index(1)) = voting_result(index(1))+1;
        data_class_predict(i) = index(1);
   else
       voting_result(t(1)) =voting_result(t(1))+1;
       data_class_predict(i) = t(1);
   end
end
 
%%
per_class_num = class_f_num_test;      %number of pic per lower class             %%%%%%%%%%%%%%%%%%%%
% accuracy
acc_sum = 0;
for i=1:opts.nClass(1)*opts.num_of_animals
    i
    acc_sum = sum(data_class_predict((i-1)*per_class_num+1:i*per_class_num)==i)+ acc_sum;
    sum(data_class_predict((i-1)*per_class_num+1:i*per_class_num)==i)/per_class_num*100
end
acc=acc_sum/(per_class_num*opts.nClass(1)*opts.num_of_animals)
%}