
rng(24) ;
SNR = 20 ;
M = 8;
R = 20;
maxIte = 10;
sto_core = 3 ;

load("X.mat","XTest.mat","Y.mat","YTest.mat","Truth_train.mat","Truth_test.mat","sigmae.mat")


%% K fold cross validation

% [mean_error_all_Core, conf_error_all_Core, sigma_folds_Core, lambda_folds_Core] = cross_Core_dat(X, Y, M, R, maxIte, sto_core) ;
[mean_error_all_Core, conf_error_all_Core, sigma_folds_Core, lambda_folds_Core] = cross_CI_Core(X, Y, M, R, maxIte, sto_core, Truth_train) ;


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
lambda_Core = 1e-8;
%% Making predictions
 
[W, norms] = CP_ALS_weights(X, Y, M, R, lambda_Core, maxIte);
[pred_Core, var_CI, var_PI] = CP_ALS_Core_predict(XTest, W, norms, M, sigma_Core, sto_core, Y, X);

pred_Core = real(pred_Core) ;
var_CI = real(var_CI) ;
var_PI = real(var_PI) ;

% Quantifiying the uncertainty
%[var_CI, var_PI, CI, PI] = lin_uncertainty_one_exp(W_2, X, XTest, Y, M, norms, Truth_train) ;
%[var_CI, var_PI] = delta_method_exp(C_train, C_test, W_2, X, Y, M, norms, Truth_train, hessian_reg, regu) ;

% Uncertainty intervals
CI_lower_Core = pred_Core - (2*sqrt(var_PI)) ;
CI_upper_Core = pred_Core + (2*sqrt(var_PI)) ;

% variableNames = {'CI_lower_Core', 'CI_upper_Core', 'pred_Core'};
% 
% for i = 1:length(variableNames)
%     filePath1 = fullfile('C:\Users\rutge\Documents\MATLAB\Thesis\Exp 1 - one dim Freq\exp_1a', [variableNames{i}, '.mat']);
%     save(filePath1, variableNames{i});
% end

%% 4. Plotting

a = [XTest, YTest, CI_lower_Core, CI_upper_Core, pred_Core, Truth_test] ;
Plot = sortrows(a);

ref_model = [Plot(:,1),Plot(:,5)];
m_err_1 = Plot(:,2);
m_err_2 = Plot(:,5);


figure(3)

l = plot(Plot(:,1),Plot(:,2),'go',Plot(:,1),Plot(:,5),'b',Plot(:,1),Plot(:,3),'r--',Plot(:,1),Plot(:,4),'r--',Plot(:,1),Plot(:,6),'c');

l(1).MarkerSize = 2; 
%l(2).MarkerSize = 12;
title('SBC','interpreter','latex')
xlabel('Feature','interpreter','latex')
ylabel('Function','interpreter','latex')
ylim([-1.6 1.6])
xlim([min(XTest) max(XTest)])
legend('Test input','Prediction','$2 \sigma$ confidence interval','','True function','Location','southeast','interpreter','latex')


