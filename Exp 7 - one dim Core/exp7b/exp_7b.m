% This study regards a dataset which is learned and where predictions are
% made and the uncertainty in the predictions is quantified 
%clear all
rng(20) ;
M = 8;
R = 20;
maxIte = 10;
sto_core = 3 ;
sigmae_Core = 0 ;

warning('off','all');

load("X.mat","XTest.mat","Y.mat","YTest.mat")


%% K fold cross validation
[mean_error_all_Core, conf_error_all_Core, sigma_folds_Core, lambda_folds_Core] = cross_Core_dat(X, Y, M, R, maxIte, sto_core) ;

% Find the minimum element in the matrix
minValue = min(mean_error_all_Core);

% Calculate the threshold for the difference
threshold = 0.5 * minValue;

% Find the indices of the minimum elements that meet the condition
index_conf = find(mean_error_all_Core > minValue + threshold);

conf_error_all_Core(index_conf) = 10e4 ;

matrix_conf = reshape(conf_error_all_Core,[length(sigma_folds_Core), length(lambda_folds_Core)]) ;

min_value = min(matrix_conf(:)) ;

% Find the indices of the minimum value using find
[index_sigma, index_lambda] = find(matrix_conf == min_value);

% Extract elements
sigma_Core = sigma_folds_Core(index_sigma);
lambda_Core = lambda_folds_Core(index_lambda);

%% Making predictions

[W, norms] = CP_ALS_weights(X, Y, M, R, lambda_Core, maxIte);
[pred_Core, var_CI, var_PI] = CP_ALS_Core_predict(XTest, W, norms, M, sigma_Core, sto_core, Y, X);

pred_Core = real(pred_Core) ;
var_CI = real(var_CI) ;
var_PI = real(var_PI) ;

% Uncertainty intervals
CI_lower_Core = pred_Core - (2*sqrt(var_CI)) ;
CI_upper_Core = pred_Core + (2*sqrt(var_CI)) ;


%% Plotting

a = [XTest, YTest, CI_lower_Core, CI_upper_Core, pred_Core] ; 

Plot = sortrows(a);

true_model = [Plot(:,1),Plot(:,3)];
pred_train = Plot(:,2);


figure(3)

l = plot(Plot(:,1),Plot(:,2),'go',Plot(:,1),Plot(:,5),'b',Plot(:,1),Plot(:,3),'r--',Plot(:,1),Plot(:,4),'r--');


l(1).MarkerSize = 2; 
%l(2).MarkerSize = 12;
title('SBC','interpreter','latex')
xlabel('Feature','interpreter','latex')
ylabel('Function','interpreter','latex')
%ylim([-2 2])
%xlim([-100 100])
legend('Test input','Prediction','Prediction Interval','','Location','southeast','interpreter','latex')
