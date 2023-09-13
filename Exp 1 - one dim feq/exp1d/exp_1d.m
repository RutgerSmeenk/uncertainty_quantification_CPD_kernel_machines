clear all

M = 8;
R = 10;
maxIte = 10;
SNR = 20 ;

CWC_delta = [] ;
CWC_Bay = [] ;
CWC_Core = [] ;

PICP_delta = [] ;
PICP_Bay = [] ;
PICP_Core = [] ;

MPIW_delta = [] ;
MPIW_Bay = [] ;
MPIW_Core = [] ;

time_delta = [] ;
time_Bay = [] ;
time_Core = [] ;

error_delta = [] ;
error_Bay = [] ;
error_Core = [] ;

TotalRuns = 1 ;

for run_num = 1:TotalRuns

    rng(run_num) ;

warning('off','all');


%% Processing data

x1 = linspace(-20, 20, 20);      % Range for dimension 1
x2 = linspace(-100, 100, 20);    % Range for dimension 2
x3 = linspace(-10, 10, 20);         % Range for dimension 3

[X1, X2, X3] = meshgrid(x1, x2, x3);

X = [X1(:), X2(:), X3(:)]  ;

% Randomize weights delta
WCP{1} = randn(M,R);
WCP{2} = randn(M,R);
WCP{3} = randn(M,R);
norms = ones(R,1) ;
pred_data = CP_data_gen(X, WCP) ;

% Adding noise
% noise_norm = norm(pred_data)/(10^(SNR/20));
% noise      = randn(size(pred_data));
% noise      = noise/norm(noise)*noise_norm;
% pred_noise  = pred_data * noise ;
%pred_noise_sign = sign(pred_noise) ;

pred_noise_sign = sign(pred_data) ;

X = [X1(:), X2(:), X3(:), pred_noise_sign];

%X = readmatrix('ripley.csv'); 

XTest = X ;
XMin = min(X(:,1:end-1));  XMax = max(X(:,1:end-1));
%X = (X-XMin)./(XMax-XMin);

perm = randperm(size(X,1));
X = X(perm,:);
% pred_noise = pred_noise(perm,:) ;
% pred_noise = pred_noise(1:floor(0.8*size(pred_noise,1)),:);

X = X(1:floor(0.8*size(X,1)),:);


Y = X(:,end) ;
X = X(:,1:end-1);
% Y = (Y==1)-(Y==2); 
%XMin = min(X);  XMax = max(X);
%YMean = mean(pred_noise);    YStd = std(pred_noise);
%Y_no_sign = (pred_noise-YMean)./YStd ;
X = (X-XMin)./(XMax-XMin);


XTest = XTest(perm,:);
XTest = XTest(floor(0.8*size(XTest,1))+1:end,:);

YTest = XTest(:,end);
XTest = XTest(:,1:end-1);

% YTest = (YTest==1)-(YTest==2); 
XTest = (XTest-XMin)./(XMax-XMin);

% Reducing cross validation data
% X_cross = X(1:ceil(0.3*size(X,1)), :) ;
% Y_cross = Y(1:ceil(0.3*size(Y,1)), :) ;
X_cross = X ;
Y_cross = Y ;


variableNames = {'X', 'XTest', 'Y', 'YTest', 'X_cross', 'Y_cross','run_num'};

for i = 1:length(variableNames)
    filePath1 = fullfile('C:\Users\rutge\Documents\MATLAB\Thesis\Exp 6 - multi dim Bay\classification', [variableNames{i}, '.mat']);
    save(filePath1, variableNames{i});
    filePath2 = fullfile('C:\Users\rutge\Documents\MATLAB\Thesis\Exp 9 - multi dim Core\classification', [variableNames{i}, '.mat']);
    save(filePath2, variableNames{i});
end


%% K fold cross validation
[mean_error_all, conf_error_all, lambda_folds] = cross_dat_class(X, Y, M, R, maxIte) ;

% Find the minimum element in the matrix
minValue = min(mean_error_all);

% Calculate the threshold for the difference
threshold = 0.5 * minValue;

% Find the indices of the minimum elements that meet the condition
index_conf = find(mean_error_all<= minValue + threshold & mean_error_all >= minValue - threshold);

% Find the minimum element of the other matrix considering only the indicesOtherMatrix
min_val_hyper = min(conf_error_all(index_conf));

% Find indices for hyperparameters
index_lambda = find(conf_error_all == min_val_hyper);

% Find hyperparameters
lambda = lambda_folds(index_lambda) ;


%% Evaluating results

tic ;
[W, norms, reg] = CP_ALS_weights(X, Y, M, R, lambda, maxIte) ;
[prediction, pred_train, C_test, C_train, regu] = CP_ALS_predict(XTest, W, norms, M, X, reg) ;

% Estimating noise variance
sigmae = var(Y - pred_train, 1) ;

[var_CI, var_PI] = delta_method(C_train, C_test, lambda, sigmae, regu) ;

time_delta_class = toc ;

prediction = real(prediction) ;
var_CI = real(var_CI) ;
var_PI = real(var_PI) ;

CI_low = real(prediction - (2*sqrt(var_PI))) ;
CI_up = real(prediction + (2*sqrt(var_PI))) ;

CI_lower = sign(CI_low) ;
CI_upper = sign(CI_up) ;


%% Plotting ALL

run("C:\Users\rutge\Documents\MATLAB\Thesis\Exp 6 - multi dim Bay\classification\exp_6_class.m")
run("C:\Users\rutge\Documents\MATLAB\Thesis\Exp 9 - multi dim Core\classification\exp_9_class.m")



%% Evaluating all results


for method = 1:3

if method == 2 

prediction = prediction_Bay ;
CI_lower = CI_lower_Bay ;
CI_upper = CI_upper_Bay ;
CI_low = CI_low_Bay ;
CI_up = CI_up_Bay ;
elseif method == 3
prediction = prediction_Core ;
CI_lower = CI_lower_Core ;
CI_upper = CI_upper_Core ;
CI_low = CI_low_Core ;
CI_up = CI_up_Core ;
end


error = mean(YTest~=sign(prediction));

clear c

right_class = sum((YTest == CI_lower) & (YTest == CI_upper)) ;
wrong_class = sum((YTest ~= CI_lower) & (YTest ~= CI_upper)) ;

PICP = right_class / (right_class + wrong_class) ;


N_Test = length(YTest);


%PICP = 1/N_Test * sum(c) ;
mu = 1 - 0.32 ;
%mu = 0.05 ;
eta = 50 ;

if PICP >= mu 
    gamma = 0 ;
else 
    gamma = 1 ;
end

MPIW = 1/N_Test * sum(CI_up - CI_low) ;

CWC = MPIW + gamma * exp(-eta * (PICP - mu)) ;

if method == 1
error_delta = [error_delta error] ;
CWC_delta = [CWC_delta CWC] ; 
PICP_delta = [PICP_delta PICP] ; 
MPIW_delta = [MPIW_delta MPIW] ; 
elseif method == 2
error_Bay = [error_Bay error] ;
CWC_Bay = [CWC_Bay CWC] ;
PICP_Bay = [PICP_Bay PICP] ;
MPIW_Bay = [MPIW_Bay MPIW] ;
else
error_Core = [error_Core error] ;
CWC_Core = [CWC_Core CWC] ;
PICP_Core = [PICP_Core PICP] ;
MPIW_Core = [MPIW_Core MPIW] ;

end
end

elapsed_times_delta{run_num} = time_delta_class ;
elapsed_times_Bay{run_num} = time_Bay_class ;
elapsed_times_Core{run_num} = time_Core_class ;
var_folds{run_num} = sigmae ; 

hyperparameters_delta{run_num} = lambda ;
hyperparameters_Bay{run_num} = hyper_Bay_class ;
hyperparameters_Core{run_num} = [hyper_Core_class, sigmae_Core_class];

disp(run_num)
end


% Metrics Delta
CWC_delta_Best = min(CWC_delta) ;
CWC_delta_Median = median(CWC_delta) ;
CWC_delta_Std = std(CWC_delta) ;
mean_time_delta = mean(cell2mat(elapsed_times_delta)) ;
error_delta_MSE = mean(error_delta) ;
error_delta_Std = std(error_delta) ;

% Metrics Bayesian
CWC_Bay_Best = min(CWC_Bay) ;
CWC_Bay_Median = median(CWC_Bay) ;
CWC_Bay_Std = std(CWC_Bay) ;
mean_time_Bay = mean(cell2mat(elapsed_times_Bay)) ;
error_Bay_MSE = mean(error_Bay) ;
error_Bay_Std = std(error_Bay) ;

% Metrics SBC
CWC_Core_Best = min(CWC_Core) ;
CWC_Core_Median = median(CWC_Core) ;
CWC_Core_Std = std(CWC_Core) ;
mean_time_Core = mean(cell2mat(elapsed_times_Core)) ;
error_Core_MSE = mean(error_Core) ;
error_Core_Std = std(error_Core) ;



