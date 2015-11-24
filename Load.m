close all;
clear all;
clc;

addpath([cd '/utilies']);
load(['catsndogs']);

%%%%%%%%%%%%%%%%%%%%%%%%
%FDDL parameter
%%%%%%%%%%%%%%%%%%%%%%%%
opts.nClass(1)    	=   10;
opts.nClass(2)   	=   12;
opts.lambda       	=   0.15;
opts.eta       		=   0.1;
opts.eta_2          =   0.1;
opts.nIter         	=   15;
[Dict,Drls,CoefM,CMlabel] = Initialization(cats, cat_trls, dogs, dog_trls, opts);