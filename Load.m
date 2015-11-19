close all;
clear all;
clc;

addpath([cd '/utilies']);
load(['cat']);
load(['dog']);

%%%%%%%%%%%%%%%%%%%%%%%%
%FDDL parameter
%%%%%%%%%%%%%%%%%%%%%%%%
opts.nClass_cat    	=   10;
opts.nClass_dog   	=   12;
opts.lambda       	=   0.15;
opts.eta       		=   0.1;
opts.nIter         	=   15;
[Dict,Drls,CoefM,CMlabel] = FDDL(tr_dat,trls,opts);