function [mean_error_all, conf_error_all, sigma_folds, noise_folds] = cross_Bay_dat(X, Y, M, R,maxIte)

K = 5 ;
mean_error_all = [] ;
conf_error_all = [] ;
sigma_folds = [1e-4 1e-3 1e-2 1e-1 1 5 10 50 100] ;
noise_folds = [0.001 0.005 0.01 0.05 0.1 0.5] ;
N_sigma_folds = length(sigma_folds) ;
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
mean_error_all_sigma = [] ;
conf_error_all_sigma = [] ;

for i = 1:N_sigma_folds 
sigma = sigma_folds(i) ;

    
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


[W_mean, W_cov, norms] = CP_ALS_Bay_weights(X, Y, M, R, maxIte, sigma, sigmae);
[pred, var_CI, var_PI] = CP_ALS_Bay_predict(XTest, W_mean, W_cov, norms, M, sigmae);

pred = real(pred) ;
var_CI = real(var_CI) ;
var_PI = real(var_PI) ;

CI_lower = pred - (2*sqrt(var_PI)) ;
CI_upper = pred + (2*sqrt(var_PI)) ;

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

















% function [error_sig, sigma_folds, sigmae_folds] = cross_Bay_dat(X, XTest, Y, YTest, M,R,lambda,lengthscale,maxIte, a)
% 
% N_folds = 5 ;
% error_sig = [] ;
% sigma_folds = [10 20 30 40 50 60 70 80 90 100 110 120] ;
% sigmae_folds = [0.001 0.005 0.01 0.05] ;
% 
% 
% for sigmae = sigmae_folds
%     error_sigma_row = [] ;
% for sigma = sigma_folds 
% 
% pred_folds = [] ;
% for fold = 1:N_folds
% X = [X,Y;XTest,YTest] ;
% XTest = X ;
% 
% perm = randperm(size(X,1));
% X = X(perm,:);
% 
% X = X(1:floor(0.6*size(X,1)),:);
% Y = X(:,end) ;
% X = X(:,1:end-1);
% 
% XTest = XTest(floor(0.6*size(XTest,1))+1:end,:);
% YTest = XTest(:,end) ;
% XTest = XTest(:,1:end-1);
% 
% [W_mean, W_cov, C] = CPLS_Bay(X,Y,M,R,lambda,lengthscale,maxIte, a, sigma, sigmae);
% [pred, var_CI, var_PI, weights] = CPPredict_Bay(XTest,W_mean,W_cov,lengthscale,M,sigmae) ;
% pred_folds = [pred_folds, pred] ;
% pred_avg = mean(pred_folds,2) ;
% error = mean((YTest - pred_avg).^2) ;
% end
% 
% error_sigma_row = [error_sigma_row error] ;
% end
% error_sig = [error_sig; error_sigma_row] ;
% end
% end