close all;  
clear all;
clc;

load(['catndog.mat']);

%%%%%%%%%%%%%%%%%%%%%%%%
%parameter
%%%%%%%%%%%%%%%%%%%%%%%%
opts.nClass(1)    	=   10;
opts.nClass(2)   	=   10;
opts.lambda       	=   0.15;
opts.eta       		=   0.1;
opts.eta_2          =   0.1;
opts.nIter         	=   15;
[Dict_ini,Dlabel_ini] = Initialization(cat_f, cat_t, dog_f, dog_t, opts);

%%%%%%%%%%%%%%%%%%%%%%%%
%SVM
%%%%%%%%%%%%%%%%%%%%%%%%
addpath('libsvm-3.21/matlab');