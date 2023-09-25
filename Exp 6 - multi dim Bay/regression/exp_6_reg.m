% This study regards a dataset which is learned and where predictions are
% made and the uncertainty in the predictions is quantified 
%clear all

load("X.mat","XTest.mat","Y.mat","YTest.mat","X_cross.mat","Y_cross.mat","run_num.mat","Truth_train.mat")

rng(run_num) ;
M = 3;
R = 10;
maxIte = 10;

warning('off','all');


%% K fold cross validation
[mean_error_all_Bay, conf_error_all_Bay, sigma_folds_Bay, noise_folds_Bay] = cross_Bay_dat(X_cross, Y_cross, M, R, maxIte) ;
%[mean_error_all_Bay, conf_error_all_Bay, sigma_folds_Bay, noise_folds_Bay] = cross_CI_Bay(X_cross, Y_cross, M, R, maxIte, Truth_train) ;

% Find the minimum element in the matrix
minValue = min(mean_error_all_Bay);

% Calculate the threshold for the difference
threshold = 0.5 * minValue;

% Find the indices of the minimum elements that meet the condition
index_conf = find(mean_error_all_Bay > minValue + threshold);

conf_error_all_Bay(index_conf) = 10e20 ;

matrix_conf = reshape(conf_error_all_Bay,[length(sigma_folds_Bay), length(noise_folds_Bay)]) ;

min_value = min(matrix_conf(:)) ;

% Find the indices of the minimum value using find
[index_sigma, index_sigmae] = find(matrix_conf == min_value);

% Extract elements
sigma_Bay = sigma_folds_Bay(index_sigma);
sigmae_Bay = noise_folds_Bay(index_sigmae);


hyper_Bay_reg = [sigma_Bay, sigmae_Bay] ;

%% Making predictions

% Obtaining the weights
tic ;
[W_mean, W_cov, norms] = CP_ALS_Bay_weights(X, Y, M, R, maxIte, sigma_Bay, sigmae_Bay) ;

% Prediction
[pred_Bay, var_CI, var_PI] = CP_ALS_Bay_predict(XTest, W_mean, W_cov, norms, M, sigmae_Bay) ;
time_Bay_reg = toc ;

pred_Bay = real(pred_Bay) ;
var_CI = real(var_CI) ;
var_PI = real(var_PI) ;

% Uncertainty intervals
CI_lower_Bay = pred_Bay - (2*sqrt(var_PI)) ;
CI_upper_Bay = pred_Bay + (2*sqrt(var_PI)) ;

%[mUT, PUT] = uns(W_mean, W_cov, norms) ;

% variableNames = {'CI_lower_Bay', 'CI_upper_Bay','pred_Bay','time_Bay_reg','hyper_Bay_reg'};
% 
% for i = 1:length(variableNames)
%     filePath1 = fullfile('C:\Users\rutge\Documents\MATLAB\Thesis\Exp 3 - multi dim Freq\Exp 5.1 regression', [variableNames{i}, '.mat']);
%     save(filePath1, variableNames{i});
% end


%% Plotting

a = [pred_Bay, CI_lower_Bay, CI_upper_Bay] ;
index = 1:length(pred_Bay) ;

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
% CWC = MPIW * (1 + gamma * exp(-eta * (PICP - mu))) ;


