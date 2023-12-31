function [mean_error_all, conf_error_all, lambda_folds] = cross_CI(X, Y, M, R, maxIte, Truth_train)

K = 5 ;
mean_error_all = [] ;
conf_error_all = [] ;
lambda_folds = [1e-2 1e-3 1e-4 1e-5 1e-6 1e-7 1e-8] ;
N_lambda_folds = length(lambda_folds) ;


X_Train = [X,Y] ;
idx = 1:size(X_Train, 1);

Y_Train = X_Train(:,end) ;
X_Train = X_Train(:,1:end-1);

Y_Test = Y_Train ;
X_Test = X_Train ;
Truth_test = Truth_train ;

% Calculate the number of data points in each fold
foldSize = floor(size(X, 1) / K);


for i = 1:N_lambda_folds 
lambda = lambda_folds(i) ;

    
mean_error_folds = [] ;
CWC_folds = [] ;
for k = 1:K

    % Determine the indices for the current fold
    startIdx = (k - 1) * foldSize + 1;
    endIdx = k * foldSize;
    if k == K
        endIdx = size(X_Train, 1);
    end

    % Extract the training and validation data for the current fold
    valIdx = idx(startIdx:endIdx);
    trainIdx = setdiff(idx, valIdx);

    X = X_Train(trainIdx, :);
    Y = Y_Train(trainIdx, :);
    Truth_Train = Truth_train(trainIdx, :);

    XTest = X_Test(valIdx, :);
    YTest = Y_Test(valIdx, :);
    Truth_Test = Truth_test(valIdx, :);


% Obtaining the weights
[W, norms, reg] = CP_ALS_weights(X, Y, M, R, lambda, maxIte) ;

% Making predictions
[pred, C_test, C_train, regu] = CP_ALS_predict(XTest, W, norms, M, X, reg) ;

% Estimating noise variance
sigmae = var(YTest-pred,1) ;

% Quantifiying the uncertainty
%[var_CI, var_PI, CI, PI] = lin_uncertainty_one_dat(W, X, XTest, Y, M, norms) ;
[var_CI, var_PI] = delta_method(C_train, C_test, lambda, sigmae, regu) ;

pred = real(pred) ;
var_CI = real(var_CI) ;
var_PI = real(var_PI) ;

CI_lower = pred - (2*sqrt(var_CI)) ;
CI_upper = pred + (2*sqrt(var_CI)) ;


N_Test = length(YTest) ;
clear c
for n = 1:N_Test
    if (Truth_Test(n) > CI_lower(n)) && (Truth_Test(n) < CI_upper(n))
    c(n) = 1 ;
    else
    c(n) = 0 ;
    end
end


PICP = 1/N_Test * sum(c) ;
mu = 1 - 0.05 ;
eta = 50;

if PICP >= mu 
    gamma = 0 ;
else 
    gamma = 1 ;
end
%gamma=0;

MPIW = 1/N_Test * sum(CI_upper - CI_lower) ;
% CWC = MPIW * (1 + gamma * exp(-eta * (PICP - mu))) ;
CWC = MPIW + gamma * exp(-eta * (PICP - mu)) ;
CWC_folds = [CWC_folds, CWC] ;

%error = mean((YTest - pred_avg).^2) ;

%pred_folds = [pred_folds, pred] ;

%pred_avg = mean(pred_folds,2) ;
mean_error = mean((YTest - pred).^2) ;
mean_error_folds = [mean_error_folds, mean_error] ;
end
mean_error_avg = mean(mean_error_folds) ;

mean_error_all = [mean_error_all, mean_error_avg] ;

CWC_avg = mean(CWC_folds) ;
conf_error_all = [conf_error_all CWC_avg] ;
end
end