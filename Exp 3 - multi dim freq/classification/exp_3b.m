clear all

M = 8;
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

X = readmatrix('banana.csv'); 
XTest = X ;
XMin = min(X(:,1:end-1));  XMax = max(X(:,1:end-1));
%X = (X-XMin)./(XMax-XMin);

perm = randperm(size(X,1));
X = X(perm,:);
X = X(1:floor(0.9*size(X,1)),:);
Y = X(:,end) ;
X = X(:,1:end-1);
Y = (Y==1)-(Y==2); 
%XMin = min(X);  XMax = max(X);
X = (X-XMin)./(XMax-XMin);


XTest = XTest(perm,:);
XTest = XTest(floor(0.9*size(XTest,1))+1:end,:);

YTest = XTest(:,end);
XTest = XTest(:,1:end-1);

YTest = (YTest==1)-(YTest==2); 
XTest = (XTest-XMin)./(XMax-XMin);

% Reducing cross validation data
X_cross = X(1:ceil(0.3*size(X,1)), :) ;
Y_cross = Y(1:ceil(0.3*size(Y,1)), :) ;
% X_cross = X ;
% Y_cross = Y ;


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

% load("CI_lower_Bay.mat","CI_upper_Bay.mat","CI_lower_Core.mat","CI_upper_Core.mat","pred_Bay.mat","pred_Core.mat",...
%      "prediction_Bay","prediction_Core","CI_low_Bay.mat","CI_up_Bay.mat","CI_low_Core.mat","CI_up_Core.mat",...
%      "time_Bay_class.mat", "time_Core_class.mat", "hyper_Bay_class.mat", "hyper_Core_class.mat", "sigmae_Core_class.mat") 

run("C:\Users\rutge\Documents\MATLAB\Thesis\Exp 6 - multi dim Bay\classification\exp_6_class.m")
run("C:\Users\rutge\Documents\MATLAB\Thesis\Exp 9 - multi dim Core\classification\exp_9_class.m")

% a = real([prediction, CI_low, CI_up]) ;
% b = real([prediction_Bay, CI_low_Bay, CI_up_Bay]) ;
% c = real([prediction_Core, CI_low_Core, CI_up_Core]) ;
% 
% index = 1:length(prediction) ;
% 
% %a = [XTest, YTest, CI_lower, CI_upper, CI_lower_Bay, CI_upper_Bay, CI_lower_Core, CI_upper_Core, pred, pred_Bay, pred_Core] ;
% Plot1 = sortrows(a);
% Plot2 = sortrows(b);
% Plot3 = sortrows(c);
% 
% loc_lower = knnsearch(Plot1(:,3),0) ;
% loc_upper = knnsearch(Plot1(:,2),0) ;
% 
% loc_lower_Bay = knnsearch(Plot2(:,3),0) ;
% loc_upper_Bay = knnsearch(Plot2(:,2),0) ;
% 
% loc_lower_Core = knnsearch(Plot3(:,3),0) ;
% loc_upper_Core = knnsearch(Plot3(:,2),0) ;
% 
% circle = Plot1(:,1) ;
% low_limit = Plot1(:,2) ;
% up_limit = Plot1(:,3) ;
% 
% circle_Bay = Plot2(:,1) ;
% low_limit_Bay = Plot2(:,2) ;
% up_limit_Bay = Plot2(:,3) ;
% 
% circle_Core = Plot3(:,1) ;
% low_limit_Core = Plot3(:,2) ;
% up_limit_Core = Plot3(:,3) ;
% 
% 
% figure(2)
% %plot(x,noisy_data,'o',x,pred,x,truth,x,CI_min,'--',x,CI_max,'--')
% % plot(x,pred,x,truth,x,CI_min,'--',x,CI_max,'--')
% 
% %l = plot(true_model(:,1),true_model(:,2),'o',Plot(:,1),Plot(:,2),Plot(:,1),Plot(:,4),'--',Plot(:,1),Plot(:,5),'--');
% 
% plot([index(loc_lower) index(loc_upper)],[circle(loc_lower) circle(loc_upper)],'o','MarkerFaceColor', 'b')
% hold on
% plot([index(loc_lower_Bay) index(loc_upper_Bay)],[circle_Bay(loc_lower_Bay) circle_Bay(loc_upper_Bay)],'o','MarkerFaceColor', 'g')
% plot([index(loc_lower_Core) index(loc_upper_Core)],[circle_Core(loc_lower_Core) circle_Core(loc_upper_Core)],'o','MarkerFaceColor', 'r')
% 
% a1 = plot(index(1:loc_lower),circle(1:loc_lower),'b',index(1:loc_lower),low_limit(1:loc_lower),'bo',index(1:loc_lower),up_limit(1:loc_lower),'bo','MarkerSize', 1);
% a2 = plot(index(loc_lower:loc_upper),circle(loc_lower:loc_upper),'--b',index(loc_lower:loc_upper),low_limit(loc_lower:loc_upper),'bo',index(loc_lower:loc_upper),up_limit(loc_lower:loc_upper),'bo','MarkerSize', 1);
% a3 = plot(index(loc_upper:end),circle(loc_upper:end),'b',index(loc_upper:end),low_limit(loc_upper:end),'bo',index(loc_upper:end),up_limit(loc_upper:end),'bo','MarkerSize', 1);
% 
% b1 = plot(index(1:loc_lower_Bay),circle_Bay(1:loc_lower_Bay),'g',index(1:loc_lower_Bay),low_limit_Bay(1:loc_lower_Bay),'go',index(1:loc_lower_Bay),up_limit_Bay(1:loc_lower_Bay),'go','MarkerSize', 1);
% b2 = plot(index(loc_lower_Bay:loc_upper_Bay),circle_Bay(loc_lower_Bay:loc_upper_Bay),'--g',index(loc_lower_Bay:loc_upper_Bay),low_limit_Bay(loc_lower_Bay:loc_upper_Bay),'go',index(loc_lower_Bay:loc_upper_Bay),up_limit_Bay(loc_lower_Bay:loc_upper_Bay),'go','MarkerSize', 1);
% b3 = plot(index(loc_upper_Bay:end),circle_Bay(loc_upper_Bay:end),'g',index(loc_upper_Bay:end),low_limit_Bay(loc_upper_Bay:end),'go',index(loc_upper_Bay:end),up_limit_Bay(loc_upper_Bay:end),'go','MarkerSize', 1);
% 
% c1 = plot(index(1:loc_lower_Core),circle_Core(1:loc_lower_Core),'r',index(1:loc_lower_Core),low_limit_Core(1:loc_lower_Core),'ro',index(1:loc_lower_Core),up_limit_Core(1:loc_lower_Core),'ro','MarkerSize', 1);
% c2 = plot(index(loc_lower_Core:loc_upper_Core),circle_Core(loc_lower_Core:loc_upper_Core),'--r',index(loc_lower_Core:loc_upper_Core),low_limit_Core(loc_lower_Core:loc_upper_Core),'ro',index(loc_lower_Core:loc_upper_Core),up_limit_Core(loc_lower_Core:loc_upper_Core),'ro','MarkerSize', 1);
% c3 = plot(index(loc_upper_Core:end),circle_Core(loc_upper_Core:end),'r',index(loc_upper_Core:end),low_limit_Core(loc_upper_Core:end),'ro',index(loc_upper_Core:end),up_limit_Core(loc_upper_Core:end),'ro','MarkerSize', 1);
% 
% 
% hold off
% 
% title('Plot','interpreter','latex')
% xlabel('Index','interpreter','latex')
% ylabel('Sorted latent variables','interpreter','latex')
% yline(0,'--k');
% ylim([-2 2])
% % ylim([-2 2])
% % xlim([-100 100])
% %legend('YTest','CPPredict','Truth','CI_{min}','CI_{max}')
% % legend('CPPredict','Truth','CI_{min}','CI_{max}')
% legend('','','','Prediction','Confidence interval','Location','southeast','interpreter','latex')


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
right_class = sum((YTest == CI_lower) & (YTest == CI_upper)) ;
wrong_class = sum((YTest ~= CI_lower) & (YTest ~= CI_upper)) ;

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



