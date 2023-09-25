clear all 

M = 3;
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

error_delta = [] ;
error_Bay = [] ;
error_Core = [] ;

TotalRuns = 10 ;
for run_num = 1:TotalRuns

rng(run_num) ;
%rng(8) ;


warning('off','all');

%% Processing data


X = readmatrix('concrete.csv'); 
XTest = X ;
XMin = min(X(:,1:end-1));  XMax = max(X(:,1:end-1));
YMean = mean(X(:,end));    YStd = std(X(:,end));

perm = randperm(size(X,1));
X = X(perm,:);
X = X(1:floor(0.9*size(X,1)),:);
Y = X(:,end) ;
X = X(:,1:end-1);

% YMean = mean(Y);    YStd = std(Y);
% XMin = min(X);  XMax = max(X);
Y = (Y-YMean)./YStd ;
X = (X-XMin)./(XMax-XMin);


XTest = XTest(perm,:);
XTest = XTest(floor(0.9*size(XTest,1))+1:end,:);

YTest = XTest(:,end);
XTest = XTest(:,1:end-1);

XTest = (XTest-XMin)./(XMax-XMin);
YTest = (YTest-YMean)./YStd;

X_cross = X(1:ceil(0.6*size(X,1)), :) ;
Y_cross = Y(1:ceil(0.6*size(Y,1)), :) ;


variableNames = {'X', 'XTest', 'Y', 'YTest', 'X_cross', 'Y_cross', 'run_num'};

for i = 1:length(variableNames)
    filePath1 = fullfile('C:\Users\rutge\Documents\MATLAB\Thesis\Exp 6 - multi dim Bay\regression', [variableNames{i}, '.mat']);
    save(filePath1, variableNames{i});
    filePath2 = fullfile('C:\Users\rutge\Documents\MATLAB\Thesis\Exp 9 - multi dim Core\regression', [variableNames{i}, '.mat']);
    save(filePath2, variableNames{i});
end



%% K fold cross validation
[mean_error_all, conf_error_all, lambda_folds] = cross_dat_reg(X_cross, Y_cross, M, R, maxIte) ;

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

tic;
[W, norms, reg] = CP_ALS_weights(X, Y, M, R,lambda, maxIte) ;
[pred, pred_train, C_test, C_train, regu] = CP_ALS_predict(XTest, W, norms, M, X, reg) ;

var_noise = var(Y - pred_train, 1) ;

[var_CI, var_PI] = delta_method(C_train, C_test, lambda, var_noise, regu) ;

time_delta_reg = toc ;

pred = real(pred) ;
var_CI = real(var_CI) ;
var_PI = real(var_PI) ;

CI_lower = pred - (2*sqrt(var_PI)) ;
CI_upper = pred + (2*sqrt(var_PI)) ;

%% Plotting ALL

% load("CI_lower_Bay.mat","CI_upper_Bay.mat","CI_lower_Core.mat","CI_upper_Core.mat","pred_Bay.mat","pred_Core.mat", ...
%     "time_Bay_reg.mat", "time_Core_reg.mat", "hyper_Bay_reg.mat", "hyper_Core_reg.mat", "sigmae_Core_reg.mat")
%a = [pred, pred_Bay, pred_Core, CI_lower, CI_upper, CI_lower_Bay, CI_upper_Bay, CI_lower_Core, CI_upper_Core] ;

run("C:\Users\rutge\Documents\MATLAB\Thesis\Exp 6 - multi dim Bay\regression\exp_6_reg.m")
run("C:\Users\rutge\Documents\MATLAB\Thesis\Exp 9 - multi dim Core\regression\exp_9_reg.m")


% a = [pred, CI_lower, CI_upper] ;
% b = [pred_Bay, CI_lower_Bay, CI_upper_Bay] ;
% c = [pred_Core, CI_lower_Core, CI_upper_Core] ;
% 
% index = 1:length(pred) ;
% 
% Plot1 = sortrows(a);
% Plot2 = sortrows(b);
% Plot3 = sortrows(c);
% 
% figure(1)
% %plot(x,noisy_data,'o',x,pred,x,truth,x,CI_min,'--',x,CI_max,'--')
% % plot(x,pred,x,truth,x,CI_min,'--',x,CI_max,'--')
% hold on
% %l = plot(true_model(:,1),true_model(:,2),'o',Plot(:,1),Plot(:,2),Plot(:,1),Plot(:,4),'--',Plot(:,1),Plot(:,5),'--');
% a1 = plot(index,Plot1(:,1),'b');
% a2 = plot(index,Plot1(:,2),'bo');
% a3 = plot(index,Plot1(:,3),'bo');
% 
% b1 = plot(index,Plot2(:,1),'g');
% b2 = plot(index,Plot2(:,2),'go');
% b3 = plot(index,Plot2(:,3),'go');
% 
% c1 = plot(index,Plot3(:,1),'r');
% c2 = plot(index,Plot3(:,2),'ro');
% c3 = plot(index,Plot3(:,3),'ro');
% hold off
% 
% a2.MarkerSize = 0.8; 
% a3.MarkerSize = 0.8; 
% b2.MarkerSize = 0.8; 
% b3.MarkerSize = 0.8; 
% c2.MarkerSize = 0.8; 
% c3.MarkerSize = 0.8; 
% %l(2).MarkerSize = 12;
% title('Plot','interpreter','latex')
% xlabel('Index','interpreter','latex')
% ylabel('Sorted latent variables','interpreter','latex')
% %ylim([-4 5])
% % ylim([-2 2])
% % xlim([-100 100])
% %legend('YTest','CPPredict','Truth','CI_{min}','CI_{max}')
% % legend('CPPredict','Truth','CI_{min}','CI_{max}')
% legend('Latent variables delta method','Delta $2 \sigma$ confidence interval','','Latent variables Bayesian method','Bayesian $2 \sigma$ confidence interval','','Latent variables SBC method','SBC $2 \sigma$ confidence interval','','Location','southeast','interpreter','latex')
% 




%% Plotting

a = [pred, CI_lower, CI_upper] ;
index = 1:length(pred) ;

Plot = sortrows(a);

figure(2)
%plot(x,noisy_data,'o',x,pred,x,truth,x,CI_min,'--',x,CI_max,'--')
% plot(x,pred,x,truth,x,CI_min,'--',x,CI_max,'--')

%l = plot(true_model(:,1),true_model(:,2),'o',Plot(:,1),Plot(:,2),Plot(:,1),Plot(:,4),'--',Plot(:,1),Plot(:,5),'--');
l1 = plot(index,Plot(:,1),index,Plot(:,2),'ko',index,Plot(:,3),'ko');


l1(2).MarkerSize = 0.8; 
l1(3).MarkerSize = 0.8; 
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

for method = 1:3

if method == 2 

pred = pred_Bay ;
CI_lower = CI_lower_Bay ;
CI_upper = CI_upper_Bay ;
elseif method == 3
pred = pred_Core ;
CI_lower = CI_lower_Core ;
CI_upper = CI_upper_Core ;
end
 
% For classification: missclassification rate
%error = mean(YTest~=sign(CP_ALS_predict(XTest, W, norms)));

% Evaluating prediction

% For regression: predictive mean squared error
error = mean((YTest-pred).^2);

% Evaluation prediction interval
clear c

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

elapsed_times_delta{run_num} = time_delta_reg ;
elapsed_times_Bay{run_num} = time_Bay_reg ;
elapsed_times_Core{run_num} = time_Core_reg ;

hyperparameters_delta{run_num} = [lambda, var_noise] ;
var_folds{run_num} = var_noise ; 
hyperparameters_Bay{run_num} = hyper_Bay_reg ;
hyperparameters_Core{run_num} = [hyper_Core_reg, sigmae_Core_reg] ;

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




