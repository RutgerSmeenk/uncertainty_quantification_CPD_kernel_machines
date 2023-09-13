% This study regards a dataset which is learned and where predictions are
% made and the uncertainty in the predictions is quantified 
clear all
rng(19) ;
M = 8;
R = 20;
maxIte = 10;

warning('off','all');

load("X.mat","XTest.mat","Y.mat","YTest.mat")

%% K fold cross validation
[mean_error_all_Bay, conf_error_all_Bay, sigma_folds_Bay, noise_folds_Bay] = cross_Bay_dat(X, Y, M, R, maxIte) ;

% Find the minimum element in the matrix
minValue = min(mean_error_all_Bay);

% Calculate the threshold for the difference
threshold = 0.5 * minValue;

% Find the indices of the minimum elements that meet the condition
index_conf = find(mean_error_all_Bay > minValue + threshold);

conf_error_all_Bay(index_conf) = 10e4 ;

matrix_conf = reshape(conf_error_all_Bay,[length(sigma_folds_Bay), length(noise_folds_Bay)]) ;

min_value = min(matrix_conf(:)) ;

% Find the indices of the minimum value using find
[index_sigma, index_sigmae] = find(matrix_conf == min_value);

% Extract elements
sigma_Bay = sigma_folds_Bay(index_sigma);
sigmae_Bay = noise_folds_Bay(index_sigmae);

%% Making predictions

% Obtaining the weights
[W_mean, W_cov, norms] = CP_ALS_Bay_weights(X, Y, M, R, maxIte, sigma_Bay, sigmae_Bay) ;

% Prediction
[pred_Bay, var_CI, var_PI] = CP_ALS_Bay_predict(XTest, W_mean, W_cov, norms, M, sigmae_Bay) ;

pred_Bay = real(pred_Bay) ;
var_CI = real(var_CI) ;
var_PI = real(var_PI) ;

% Uncertainty intervals
CI_lower_Bay = pred_Bay - (2*sqrt(var_CI)) ;
CI_upper_Bay = pred_Bay + (2*sqrt(var_CI)) ;

%% Plotting

a = [XTest, YTest, CI_lower_Bay, CI_upper_Bay, pred_Bay] ; 

Plot = sortrows(a);

true_model = [Plot(:,1),Plot(:,3)];
pred_train = Plot(:,2);


figure(2)

l = plot(Plot(:,1),Plot(:,2),'go',Plot(:,1),Plot(:,5),'b',Plot(:,1),Plot(:,3),'r--',Plot(:,1),Plot(:,4),'r--');


l(1).MarkerSize = 2; 
%l(2).MarkerSize = 12;
title('Bayesian','interpreter','latex')
xlabel('Feature','interpreter','latex')
ylabel('Function','interpreter','latex')
%ylim([-2 2])
%xlim([-100 100])
legend('Test input','Prediction','Prediction Interval','','Location','southeast','interpreter','latex')
