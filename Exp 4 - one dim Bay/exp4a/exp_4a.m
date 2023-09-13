
rng(18) ;
M = 8;
R = 20;
maxIte = 10;

load("X.mat","XTest.mat","Y.mat","YTest.mat","Truth_train.mat","Truth_test.mat","sigmae")

%% K fold cross validation
[mean_error_all_Bay, conf_error_all_Bay, sigma_folds_Bay] = cross_Bay_exp(X, Y, M, R, maxIte, sigmae) ;

% Find the minimum element in the matrix
minValue = min(mean_error_all_Bay);

% Calculate the threshold for the difference
threshold = 0.5 * minValue;

% Find the indices of the minimum elements that meet the condition
index = find(mean_error_all_Bay<= minValue + threshold & mean_error_all_Bay >= minValue - threshold);

% Find the minimum element of the other matrix considering only the indicesOtherMatrix
min_val_hyper = min(conf_error_all_Bay(index));

% Find indices for hyperparameters
index_sigma = find(conf_error_all_Bay == min_val_hyper);

% Find hyperparameters
sigma_Bay = sigma_folds_Bay(index_sigma) ;
%sigma = 2 ;

%% Making predictions

% Obtaining the weights

[W_mean, W_cov, norms] = CP_ALS_Bay_weights(X, Y, M, R, maxIte, sigma_Bay, sigmae) ;

% Prediction
[pred_Bay, var_CI, var_PI] = CP_ALS_Bay_predict(XTest, W_mean, W_cov, norms, M, sigmae) ;

pred_Bay = real(pred_Bay) ;
var_CI = real(var_CI) ;
var_PI = real(var_PI) ;

% Uncertainty intervals
CI_lower_Bay = pred_Bay - (2*sqrt(var_CI)) ;
CI_upper_Bay = pred_Bay + (2*sqrt(var_CI)) ;


%% 4. Plotting

a = [XTest, YTest, CI_lower_Bay, CI_upper_Bay, pred_Bay, Truth_test] ;
Plot = sortrows(a);

ref_model = [Plot(:,1),Plot(:,5)];
m_err_1 = Plot(:,2);
m_err_2 = Plot(:,5);


figure(2)

l = plot(Plot(:,1),Plot(:,2),'go',Plot(:,1),Plot(:,5),'b',Plot(:,1),Plot(:,3),'r--',Plot(:,1),Plot(:,4),'r--',Plot(:,1),Plot(:,6),'c');

l(1).MarkerSize = 2; 
%l(2).MarkerSize = 12;
title('Bayesian','interpreter','latex')
xlabel('Feature','interpreter','latex')
ylabel('Function','interpreter','latex')
ylim([-1.6 1.6])
xlim([min(XTest) max(XTest)])
legend('Test input','Prediction','$2 \sigma$ confidence interval','','True function','Location','southeast','interpreter','latex')


