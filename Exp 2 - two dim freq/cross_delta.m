function [mean_error_all, conf_error_all, lambda_folds, noise_folds] = cross_delta(X, Y, M, R, maxIte)

K = 5 ;
mean_error_all = [] ;
conf_error_all = [] ;
lambda_folds = [1e-2 1e-3 1e-4 1e-5 1e-6 1e-7 1e-8] ;
noise_folds = [0.01 0.1 0.5 1 10 40 80 120] ;
N_lambda_folds = length(lambda_folds) ;
N_noise_folds = length(noise_folds) ;


% X_Train = [X,Y;XTest,YTest] ;
X_Train = [X,Y] ;
%XTest = X_Train ;
idx = 1:size(X_Train, 1);

% perm = randperm(size(X,1));
% X = X(perm,:);

Y_Train = X_Train(:,end) ;
X_Train = X_Train(:,1:end-1);

Y_Test = Y_Train ;
X_Test = X_Train ;



% Calculate the number of data points in each fold
foldSize = floor(size(X, 1) / K);

for j = 1:N_noise_folds
sigmae = noise_folds(j) ;
mean_error_all_lambda = [] ;
conf_error_all_lambda = [] ;

for i = 1:N_lambda_folds 
lambda = lambda_folds(i) ;

    
%pred_folds = [] ;
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
    XTest = X_Test(valIdx, :);
    YTest = Y_Test(valIdx, :);


[W, norms, reg] = CP_ALS_weights(X, Y, M, R,lambda, maxIte) ;
[pred, pred_train, C_test, C_train, regu] = CP_ALS_predict(XTest, W, norms, M, X, reg) ;

[var_CI, var_PI] = delta_method(C_train, C_test, lambda, regu, sigmae) ;

pred = real(pred) ;
var_CI = real(var_CI) ;
var_PI = real(var_PI) ;

CI_low = pred - (2*sqrt(var_PI)) ;
CI_up = pred + (2*sqrt(var_PI)) ;

CI_lower = sign(CI_low) ;
CI_upper = sign(CI_up) ;

N_Test = length(YTest) ;
% for n = 1:N_Test
%     if (YTest(n) == CI_lower(n)) || (YTest(n) == CI_upper(n))
%     c(n) = 1 ;
%     else
%     c(n) = 0 ;
%     end
% end
% PICP = 1/N_Test * sum(c) ;

a = sortrows(real([pred, CI_low, CI_up, YTest, CI_lower, CI_upper])) ;

loc_lower = knnsearch(a(:,3),0) ;
loc_upper = knnsearch(a(:,2),0) ;

prediction_a = [a(1:loc_lower,1); a(loc_upper:end,1)] ;
YTest_a = [a(1:loc_lower,4); a(loc_upper:end,4)] ;

right_class = sum((YTest_a == sign(prediction_a))) ;
wrong_class = sum((YTest_a ~= sign(prediction_a))) ;

PICP = right_class / (right_class + wrong_class) ;




mu = 1 - 0.05 ;
%mu = 0.05 ;
eta = 50 ;

if PICP >= mu 
    gamma = 0 ;
else 
    gamma = 1 ;
end
%gamma=0;


% MPIW = (1/N_Test * sum(CI_upper - CI_lower)) / abs(CI_up - CI_low) ;
MPIW = 1/N_Test * sum(CI_up - CI_low) ;

%CWC = MPIW * (1 + gamma * exp(-eta * (PICP - mu))) ;
CWC = MPIW + gamma * exp(-eta * (PICP - mu)) ;
CWC_folds = [CWC_folds, CWC] ;

%error = mean((YTest - pred_avg).^2) ;

%pred_folds = [pred_folds, pred] ;

%pred_avg = mean(pred_folds,2) ;
%mean_error = mean((YTest - pred).^2) ;
mean_error = mean(YTest~=sign(pred)) ;
mean_error_folds = [mean_error_folds, mean_error] ;
end
mean_error_avg = mean(mean_error_folds) ;

mean_error_all = [mean_error_all, mean_error_avg] ;

CWC_avg = mean(CWC_folds) ;
conf_error_all = [conf_error_all CWC_avg] ;

end
end
