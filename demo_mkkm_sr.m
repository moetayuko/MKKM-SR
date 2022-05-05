clear; clc;

addpath('funs');

% load dataset
load('wisconsin_Kmatrix');
KH = knorm(kcenter(KH));

n = size(KH, 1);
numclass = numel(unique(Y));
Y_init = Y_Initialize(n, numclass);
lambda = 1;

tic;
[y_pred, alpha, obj] = MKKM_SR(KH, Y_init, lambda);
toc
ClusteringMeasure_new(Y, y_pred)
