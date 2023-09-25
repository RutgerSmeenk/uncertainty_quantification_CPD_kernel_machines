% This study regards a dataset which is learned and where predictions are
% made and the uncertainty in the predictions is quantified 
%clear all

load("X.mat","XTest.mat","Y.mat","YTest.mat","X_cross.mat","Y_cross.mat","run_num.mat","Truth_train.mat")

rng(run_num) ;
M = 20;
R = 10;
maxIte = 10;
sto_core = 8;
warning('off','all');


%% K fold cross validation
[mean_error_all_Core, conf_error_all_Core, sigma_folds_Core, lambda_folds_Core] = cross_Bay_dat(X_cross, Y_cross, M, R, maxIte, sto_core) ;
%[mean_error_all_Core, conf_error_all_Core, sigma_folds_Core, lambda_folds_Core] = cross_CI_Core(X_cross, Y_cross, M, R, maxIte, sto_core, Truth_train) ;


% Find the minimum element in the matrix
minValue = min(mean_error_all_Core);

% Calculate the threshold for the difference
threshold = 0.5 * minValue;

% Find the indices of the minimum elements that meet the condition
index_conf = find(mean_error_all_Core > minValue + threshold);

conf_error_all_Core(index_conf) = 10e20 ;

matrix_conf = reshape(conf_error_all_Core,[length(sigma_folds_Core), length(lambda_folds_Core)]) ;

min_value = min(matrix_conf(:)) ;

% Find the indices of the minimum value using find
[index_sigma, index_lambda] = find(matrix_conf == min_value);

% Extract elements
sigma_Core = sigma_folds_Core(index_sigma);
lambda_Core = lambda_folds_Core(index_lambda);

hyper_Core_reg = [lambda_Core, sigma_Core] ;

%% Making predictions

tic ;
[W, norms] = CP_ALS_weights(X, Y, M, R, lambda_Core, maxIte);
[pred, var_CI, var_PI, sigmae_Core_reg] = CP_ALS_Core_predict(XTest, W, norms, M, sigma_Core, sto_core, Y, X);

time_Core_reg = toc ;

pred_Core = real(pred) ;
var_CI = real(var_CI) ;
var_PI = real(var_PI) ;

% Uncertainty intervals
CI_lower_Core = pred - (2*sqrt(var_PI)) ;
CI_upper_Core = pred + (2*sqrt(var_PI)) ;

% variableNames = {'CI_lower_Core', 'CI_upper_Core', 'pred_Core', 'time_Core_reg', 'hyper_Core_reg', 'sigmae_Core_reg'};
% 
% for i = 1:length(variableNames)
%     filePath1 = fullfile('C:\Users\rutge\Documents\MATLAB\Thesis\Exp 3 - multi dim Freq\Exp 5.1 regression', [variableNames{i}, '.mat']);
%     save(filePath1, variableNames{i});
% end


%% Plotting

a = [pred, CI_lower_Core, CI_upper_Core] ;
index = 1:length(pred) ;

Plot = sortrows(a);

figure(1)
%plot(x,noisy_data,'o',x,pred,x,truth,x,CI_min,'--',x,CI_max,'--')
% plot(x,pred,x,truth,x,CI_min,'--',x,CI_max,'--')

%l = plot(true_model(:,1),true_model(:,2),'o',Plot(:,1),Plot(:,2),Plot(:,1),Plot(:,4),'--',Plot(:,1),Plot(:,5),'--');
l = plot(index,Plot(:,1),index,Plot(:,2),'ko',index,Plot(:,3),'ko');


l(2).MarkerSize = 0.8; 
l(3).MarkerSize = 0.8; 
%l(2).MarkerSize = 12;
title('Plot','interpreter','latex')
xlabel('Index','interpreter','latex')
ylabel('Sorted latent variables','interpreter','latex')
%ylim([-4 5])
% ylim([-2 2])
% xlim([-100 100])
%legend('YTest','CPPredict','Truth','CI_{min}','CI_{max}')
% legend('CPPredict','Truth','CI_{min}','CI_{max}')
legend('Prediction','Confidence interval','','Location','southeast','interpreter','latex')

%% Evaluating results

% Evaluating prediction

% For regression: predictive mean squared error
% error = mean((YTest-pred).^2);
% 
% clear c
% % Evaluation prediction interval
% N_Test = length(YTest) ;
% for n = 1:N_Test
%     if (YTest(n) > CI_lower(n)) && (YTest(n) < CI_upper(n))
%     c(n) = 1 ;
%     else
%     c(n) = 0 ;
%     end
% end
% 
% 
% PICP = 1/N_Test * sum(c) ;
% mu = 1 - 0.05 ;
% eta = 50;
% 
% if PICP >= mu 
%     gamma = 0 ;
% else 
%     gamma = 1 ;
% end
% %gamma=0;
% 
% MPIW = 1/N_Test * sum(CI_upper - CI_lower) ;
% CWC = MPIW  + gamma * exp(-eta * (PICP - mu)) ;
