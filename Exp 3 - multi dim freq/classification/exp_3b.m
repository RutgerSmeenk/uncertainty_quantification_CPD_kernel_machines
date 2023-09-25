clear all

M = 12;
R = 10;
maxIte = 10;

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

TotalRuns = 10 ;

for run_num = 1:TotalRuns

    rng(run_num) ;
    

warning('off','all');


%% Processing data

X = readmatrix('raisin.csv'); 
XTest = X ;
XMin = min(X(:,1:end-1));  XMax = max(X(:,1:end-1));
%X = (X-XMin)./(XMax-XMin);

perm = randperm(size(X,1));
X = X(perm,:);
X = X(1:floor(0.9*size(X,1)),:);
Y = X(:,end) ;
X = X(:,1:end-1);
Y = (Y==1)-(Y==2); 
% XMin = min(X);  XMax = max(X);
X = (X-XMin)./(XMax-XMin);


XTest = XTest(perm,:);
XTest = XTest(floor(0.9*size(XTest,1))+1:end,:);

YTest = XTest(:,end);
XTest = XTest(:,1:end-1);

YTest = (YTest==1)-(YTest==2); 
% XMinTest = min(XTest);  XMaxTest = max(XTest);
XTest = (XTest-XMin)./(XMax-XMin);

% Reducing cross validation data
X_cross = X(1:ceil(1*size(X,1)), :) ;
Y_cross = Y(1:ceil(1*size(Y,1)), :) ;



%% K fold cross validation
[mean_error_all_delta, conf_error_all_delta, lambda_folds_delta, noise_folds_delta] = cross_delta(X, Y, M, R, maxIte) ;

% Find the minimum element in the matrix
minValue = min(mean_error_all_delta);

% Calculate the threshold for the difference
threshold = 0.1 * minValue;

% Find the indices of the minimum elements that meet the condition
index_conf = find(mean_error_all_delta > minValue + threshold);

conf_error_all_delta(index_conf) = 10e20 ;

matrix_conf = reshape(conf_error_all_delta,[length(lambda_folds_delta), length(noise_folds_delta)]) ;

min_value = min(matrix_conf(:)) ;

% Find the indices of the minimum value using find
[index_lambda, index_sigmae] = find(matrix_conf == min_value);

% Extract elements
lambda_delta = lambda_folds_delta(index_lambda);
sigmae_delta = noise_folds_delta(index_sigmae);

variableNames = {'X', 'XTest', 'Y', 'YTest', 'X_cross', 'Y_cross','run_num','lambda_delta'};

for i = 1:length(variableNames)
    filePath1 = fullfile('C:\Users\rutge\Documents\MATLAB\Thesis\Exp 6 - multi dim Bay\classification', [variableNames{i}, '.mat']);
    save(filePath1, variableNames{i});
    filePath2 = fullfile('C:\Users\rutge\Documents\MATLAB\Thesis\Exp 9 - multi dim Core\classification', [variableNames{i}, '.mat']);
    save(filePath2, variableNames{i});
end

%% Evaluating results

tic ;
[W, norms, reg] = CP_ALS_weights(X, Y, M, R, lambda_delta, maxIte) ;
W1=W;
[prediction, pred_train, C_test, C_train, regu] = CP_ALS_predict(XTest, W, norms, M, X, reg) ;
pred1 = prediction ;
[var_CI, var_PI] = delta_method(C_train, C_test, lambda_delta, regu, sigmae_delta) ;

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

a = [prediction, CI_low, CI_up] ;
index = 1:length(prediction) ;

Plot = sortrows(a);

true_model = [Plot(:,1),Plot(:,3)] ;
pred_train = Plot(:,2) ;

loc_lower = knnsearch(Plot(:,3),0) ;
loc_upper = knnsearch(Plot(:,2),0) ;

circle = Plot(:,1) ;
low_limit = Plot(:,2) ;
up_limit = Plot(:,3) ;

figure(1)
%plot(x,noisy_data,'o',x,pred,x,truth,x,CI_min,'--',x,CI_max,'--')
% plot(x,pred,x,truth,x,CI_min,'--',x,CI_max,'--')

%l = plot(true_model(:,1),true_model(:,2),'o',Plot(:,1),Plot(:,2),Plot(:,1),Plot(:,4),'--',Plot(:,1),Plot(:,5),'--');

plot([index(loc_lower) index(loc_upper)],[circle(loc_lower) circle(loc_upper)],'o','MarkerFaceColor', 'k', 'MarkerSize', 10)
hold on
plot(index(1:loc_lower),circle(1:loc_lower),'g', 'LineWidth', 2);
plot(index(1:loc_lower),low_limit(1:loc_lower),'ko', 'MarkerSize', 1.8);
plot(index(1:loc_lower),up_limit(1:loc_lower),'ko', 'MarkerSize', 1.8);
plot(index(loc_lower:loc_upper),circle(loc_lower:loc_upper),'r', 'LineWidth', 2);
plot(index(loc_lower:loc_upper),low_limit(loc_lower:loc_upper),'ko', 'MarkerSize', 1.8);
plot(index(loc_lower:loc_upper),up_limit(loc_lower:loc_upper),'ko', 'MarkerSize', 1.8);
plot(index(loc_upper:end),circle(loc_upper:end),'g', 'LineWidth', 2);
plot(index(loc_upper:end),low_limit(loc_upper:end),'ko', 'MarkerSize', 1.8);
plot(index(loc_upper:end),up_limit(loc_upper:end),'ko', 'MarkerSize', 1.8);

yline(0,'--b');
title('')
xlabel('Index','interpreter','latex','FontSize', 12)
ylabel('Sorted predictions','interpreter','latex','FontSize', 12)
legend('off');
% ylim([-3 3])
% xlim([0 130])



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
N_Test = length(YTest) ;
% for n = 1:N_Test
%     if (YTest(n) == CI_lower(n)) || (YTest(n) == CI_upper(n))
%     c(n) = 1 ;
%     else
%     c(n) = 0 ;
%     end
% end
% PICP = 1/N_Test * sum(c) ;

a = sortrows(real([prediction, CI_low, CI_up, YTest, CI_lower, CI_upper])) ;

loc_lower = knnsearch(a(:,3),0) ;
loc_upper = knnsearch(a(:,2),0) ;

prediction_a = [a(1:loc_lower,1); a(loc_upper:end,1)] ;
YTest_a = [a(1:loc_lower,4); a(loc_upper:end,4)] ;

right_class = sum((YTest_a == sign(prediction_a))) ;
wrong_class = sum((YTest_a ~= sign(prediction_a))) ;

PICP = right_class / (right_class + wrong_class) ;


mu = 1 - 0.05 ;
eta = 50 ;

if PICP >= mu 
    gamma = 0 ;
else 
    gamma = 1 ;
end
%gamma=0;


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

hyperparameters_delta{run_num} = [lambda_delta, sigmae_delta] ;
hyperparameters_Bay{run_num} = [sigma_Bay, sigmae_Bay] ;
hyperparameters_Core{run_num} = [sigma_Core, sigmae_Core];

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



