function [mean_error_all, conf_error_all, lambda_folds] = cross_dat_reg(X, Y, M, R, maxIte)

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


% Calculate the number of data points in each fold
foldSize = floor(size(X, 1) / K);


for i = 1:N_lambda_folds 
lambda = lambda_folds(i) ;

    
pred_folds = [] ;
mean_error_avg = [] ;
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
    XTest = X_Test(valIdx, :);
    YTest = Y_Test(valIdx, :);


[W, norms, reg] = CP_ALS_weights(X, Y, M, R,lambda, maxIte) ;
[pred, pred_train, C_test, C_train, regu] = CP_ALS_predict(XTest, W, norms, M, X, reg) ;

% Estimating noise variance
var_noise = var(Y - pred_train, 1) ;


% Quantifiying the uncertainty
%[var_CI, var_PI, CI, PI] = lin_uncertainty_multi(W, X, XTest, Y, norms) ;
[var_CI, var_PI] = delta_method(C_train, C_test, lambda, var_noise, regu) ;

pred = real(pred) ;
var_CI = real(var_CI) ;
var_PI = real(var_PI) ;

CI_lower = pred - 2*(sqrt(var_PI)) ;
CI_upper = pred + 2*(sqrt(var_PI)) ;

N_Test = length(YTest) ;
for n = 1:N_Test
    if (YTest(n) > CI_lower(n)) && (YTest(n) < CI_upper(n))
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
CWC = MPIW + gamma * exp(-eta * (PICP - mu)) ;
CWC_folds = [CWC_folds, CWC] ;

mean_error = mean((YTest-pred).^2); 

mean_error_avg = [mean_error_avg, mean_error] ;
end
mean_error_avg = mean(mean_error_avg) ;

mean_error_all = [mean_error_all, mean_error_avg] ;

CWC_avg = mean(CWC_folds) ;
conf_error_all = [conf_error_all CWC_avg] ;
end
end